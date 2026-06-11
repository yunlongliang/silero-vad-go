package speech

// #cgo CFLAGS: -Wall -Werror -std=c99
// #cgo LDFLAGS: -lonnxruntime
// #include "ort_bridge.h"
import "C"

import (
	"fmt"
	"log/slog"
	"math"
	"unsafe"
)

const (
	stateV5Len    = 2 * 1 * 128
	stateV3Len    = 2 * 1 * 64
	maxContextLen = 64 // 16kHz uses 64, 8kHz uses 32
)

type possibleEnd struct {
	pos    int
	silDur int
}

type LogLevel int

func (l LogLevel) OrtLoggingLevel() C.OrtLoggingLevel {
	switch l {
	case LevelVerbose:
		return C.ORT_LOGGING_LEVEL_VERBOSE
	case LogLevelInfo:
		return C.ORT_LOGGING_LEVEL_INFO
	case LogLevelWarn:
		return C.ORT_LOGGING_LEVEL_WARNING
	case LogLevelError:
		return C.ORT_LOGGING_LEVEL_ERROR
	case LogLevelFatal:
		return C.ORT_LOGGING_LEVEL_FATAL
	default:
		return C.ORT_LOGGING_LEVEL_WARNING
	}
}

const (
	LevelVerbose LogLevel = iota + 1
	LogLevelInfo
	LogLevelWarn
	LogLevelError
	LogLevelFatal
)

// DetectorConfig holds the configuration for creating a speech Detector.
type DetectorConfig struct {
	// The path to the ONNX Silero VAD model file to load.
	ModelPath string
	// The sampling rate of the input audio samples. Supported values are 8000 and 16000.
	SampleRate int
	// The probability threshold above which we detect speech. A good default is 0.5.
	Threshold float32
	// The duration of silence (ms) to wait before splitting a speech segment.
	MinSilenceDurationMs int
	// The padding (ms) to add to speech segments to avoid aggressive cutting.
	SpeechPadMs int
	// Minimum speech duration in milliseconds. Speech segments shorter than
	// this value will be discarded. Set to 0 to disable (default).
	MinSpeechDurationMs int
	// Maximum speech duration in seconds. Speech segments longer than this
	// value will be split at the last detected silence boundary.
	// Set to 0 to disable (default, meaning no limit).
	MaxSpeechDurationS float64
	// The log level for the ONNX Runtime environment. Default is LogLevelWarn.
	LogLevel LogLevel
}

func (c DetectorConfig) IsValid() error {
	if c.ModelPath == "" {
		return fmt.Errorf("invalid ModelPath: should not be empty")
	}

	if c.SampleRate != 8000 && c.SampleRate != 16000 {
		return fmt.Errorf("invalid SampleRate: valid values are 8000 and 16000")
	}

	if c.Threshold <= 0 || c.Threshold >= 1 {
		return fmt.Errorf("invalid Threshold: should be in range (0, 1)")
	}

	if c.MinSilenceDurationMs < 0 {
		return fmt.Errorf("invalid MinSilenceDurationMs: should be a non-negative number")
	}

	if c.SpeechPadMs < 0 {
		return fmt.Errorf("invalid SpeechPadMs: should be a non-negative number")
	}

	if c.MinSpeechDurationMs < 0 {
		return fmt.Errorf("invalid MinSpeechDurationMs: should be a non-negative number")
	}

	if c.MaxSpeechDurationS < 0 {
		return fmt.Errorf("invalid MaxSpeechDurationS: should be a non-negative number")
	}

	return nil
}

// Detector performs speech detection on audio samples using Silero VAD.
type Detector struct {
	api         *C.OrtApi
	env         *C.OrtEnv
	sessionOpts *C.OrtSessionOptions
	session     *C.OrtSession
	memoryInfo  *C.OrtMemoryInfo
	cStrings    map[string]*C.char

	inputNames  []*C.char
	outputNames []*C.char

	cfg DetectorConfig

	// v5 model: combined h+c state [2,1,128]
	stateV5 [stateV5Len]float32
	// v3/sherpa model: separate h [2,1,64] and c [2,1,64]
	stateH [stateV3Len]float32
	stateC [stateV3Len]float32

	// true when model uses v3/sherpa format (separate h/c, no sr input)
	separateHC bool

	ctx [maxContextLen]float32

	currSample   int
	triggered    bool
	tempEnd      int
	speechStart  int
	prevEnd      int
	nextStart    int
	possibleEnds []possibleEnd

	residual []float32
}

// Segment contains timing information of a speech segment in milliseconds.
type Segment struct {
	// The relative timestamp in milliseconds of when a speech segment begins.
	SpeechStartAt int
	// The relative timestamp in milliseconds of when a speech segment ends.
	SpeechEndAt int
}

// querySessionNames queries the actual input/output tensor names from the ONNX
// session, so we don't rely on hard-coded names that may differ across
// onnxruntime builds or model versions.
func (sd *Detector) querySessionNames() error {
	var allocator *C.OrtAllocator
	status := C.OrtApiGetAllocatorWithDefaultOptions(sd.api, &allocator)
	if status != nil {
		msg := C.GoString(C.OrtApiGetErrorMessage(sd.api, status))
		C.OrtApiReleaseStatus(sd.api, status)
		return fmt.Errorf("failed to get allocator: %s", msg)
	}

	var inputCount, outputCount C.size_t
	status = C.OrtApiSessionGetInputCount(sd.api, sd.session, &inputCount)
	if status != nil {
		msg := C.GoString(C.OrtApiGetErrorMessage(sd.api, status))
		C.OrtApiReleaseStatus(sd.api, status)
		return fmt.Errorf("failed to get input count: %s", msg)
	}
	status = C.OrtApiSessionGetOutputCount(sd.api, sd.session, &outputCount)
	if status != nil {
		msg := C.GoString(C.OrtApiGetErrorMessage(sd.api, status))
		C.OrtApiReleaseStatus(sd.api, status)
		return fmt.Errorf("failed to get output count: %s", msg)
	}

	slog.Info("silero-vad ONNX session",
		slog.Int("inputCount", int(inputCount)),
		slog.Int("outputCount", int(outputCount)),
		slog.String("model", sd.cfg.ModelPath))

	sd.inputNames = make([]*C.char, int(inputCount))
	for i := 0; i < int(inputCount); i++ {
		var name *C.char
		status = C.OrtApiSessionGetInputName(sd.api, sd.session, C.size_t(i), allocator, &name)
		if status != nil {
			msg := C.GoString(C.OrtApiGetErrorMessage(sd.api, status))
			C.OrtApiReleaseStatus(sd.api, status)
			return fmt.Errorf("failed to get input name %d: %s", i, msg)
		}
		goName := C.GoString(name)
		status = C.OrtApiAllocatorFree(sd.api, allocator, unsafe.Pointer(name))
		if status != nil {
			C.OrtApiReleaseStatus(sd.api, status)
		}
		sd.cStrings[fmt.Sprintf("_input_%d", i)] = C.CString(goName)
		sd.inputNames[i] = sd.cStrings[fmt.Sprintf("_input_%d", i)]
		slog.Info("silero-vad input", slog.Int("index", i), slog.String("name", goName))
	}

	sd.outputNames = make([]*C.char, int(outputCount))
	for i := 0; i < int(outputCount); i++ {
		var name *C.char
		status = C.OrtApiSessionGetOutputName(sd.api, sd.session, C.size_t(i), allocator, &name)
		if status != nil {
			msg := C.GoString(C.OrtApiGetErrorMessage(sd.api, status))
			C.OrtApiReleaseStatus(sd.api, status)
			return fmt.Errorf("failed to get output name %d: %s", i, msg)
		}
		goName := C.GoString(name)
		status = C.OrtApiAllocatorFree(sd.api, allocator, unsafe.Pointer(name))
		if status != nil {
			C.OrtApiReleaseStatus(sd.api, status)
		}
		sd.cStrings[fmt.Sprintf("_output_%d", i)] = C.CString(goName)
		sd.outputNames[i] = sd.cStrings[fmt.Sprintf("_output_%d", i)]
		slog.Info("silero-vad output", slog.Int("index", i), slog.String("name", goName))
	}

	return nil
}

// NewDetector creates a new speech detector with the given configuration.
func NewDetector(cfg DetectorConfig) (*Detector, error) {
	if err := cfg.IsValid(); err != nil {
		return nil, fmt.Errorf("invalid config: %w", err)
	}

	sd := Detector{
		cfg:      cfg,
		cStrings: map[string]*C.char{},
	}

	sd.api = C.OrtGetApi()
	if sd.api == nil {
		return nil, fmt.Errorf("failed to get API")
	}

	sd.cStrings["loggerName"] = C.CString("vad")
	status := C.OrtApiCreateEnv(sd.api, cfg.LogLevel.OrtLoggingLevel(), sd.cStrings["loggerName"], &sd.env)
	defer C.OrtApiReleaseStatus(sd.api, status)
	if status != nil {
		return nil, fmt.Errorf("failed to create env: %s", C.GoString(C.OrtApiGetErrorMessage(sd.api, status)))
	}

	status = C.OrtApiCreateSessionOptions(sd.api, &sd.sessionOpts)
	defer C.OrtApiReleaseStatus(sd.api, status)
	if status != nil {
		return nil, fmt.Errorf("failed to create session options: %s", C.GoString(C.OrtApiGetErrorMessage(sd.api, status)))
	}

	status = C.OrtApiSetIntraOpNumThreads(sd.api, sd.sessionOpts, 1)
	defer C.OrtApiReleaseStatus(sd.api, status)
	if status != nil {
		return nil, fmt.Errorf("failed to set intra threads: %s", C.GoString(C.OrtApiGetErrorMessage(sd.api, status)))
	}

	status = C.OrtApiSetInterOpNumThreads(sd.api, sd.sessionOpts, 1)
	defer C.OrtApiReleaseStatus(sd.api, status)
	if status != nil {
		return nil, fmt.Errorf("failed to set inter threads: %s", C.GoString(C.OrtApiGetErrorMessage(sd.api, status)))
	}

	status = C.OrtApiSetSessionGraphOptimizationLevel(sd.api, sd.sessionOpts, C.ORT_DISABLE_ALL)
	defer C.OrtApiReleaseStatus(sd.api, status)
	if status != nil {
		return nil, fmt.Errorf("failed to set session graph optimization level: %s", C.GoString(C.OrtApiGetErrorMessage(sd.api, status)))
	}

	sd.cStrings["modelPath"] = C.CString(sd.cfg.ModelPath)
	status = C.OrtApiCreateSession(sd.api, sd.env, sd.cStrings["modelPath"], sd.sessionOpts, &sd.session)
	defer C.OrtApiReleaseStatus(sd.api, status)
	if status != nil {
		return nil, fmt.Errorf("failed to create session: %s", C.GoString(C.OrtApiGetErrorMessage(sd.api, status)))
	}

	status = C.OrtApiCreateCpuMemoryInfo(sd.api, C.OrtArenaAllocator, C.OrtMemTypeDefault, &sd.memoryInfo)
	defer C.OrtApiReleaseStatus(sd.api, status)
	if status != nil {
		return nil, fmt.Errorf("failed to create memory info: %s", C.GoString(C.OrtApiGetErrorMessage(sd.api, status)))
	}

	if err := sd.querySessionNames(); err != nil {
		return nil, fmt.Errorf("failed to query session names: %w", err)
	}

	if len(sd.inputNames) < 3 {
		return nil, fmt.Errorf("expected at least 3 inputs, got %d", len(sd.inputNames))
	}
	if len(sd.outputNames) < 2 {
		return nil, fmt.Errorf("expected at least 2 outputs, got %d", len(sd.outputNames))
	}

	// Auto-detect model format: 3 outputs = v3/sherpa (x,h,c -> output,hn,cn),
	// 2 outputs = v5 (input,state,sr -> output,stateN)
	sd.separateHC = len(sd.outputNames) >= 3
	slog.Info("silero-vad model format",
		slog.Bool("separateHC", sd.separateHC),
		slog.Int("inputs", len(sd.inputNames)),
		slog.Int("outputs", len(sd.outputNames)))

	return &sd, nil
}

func (sd *Detector) SetThreshold(t float32) {
	sd.cfg.Threshold = t
}

func (sd *Detector) SetMinSilenceDurationMs(d int) {
	sd.cfg.MinSilenceDurationMs = d
}

func (sd *Detector) SetSpeechPadMs(p int) {
	sd.cfg.SpeechPadMs = p
}

func (sd *Detector) SetMinSpeechDurationMs(d int) {
	sd.cfg.MinSpeechDurationMs = d
}

func (sd *Detector) SetMaxSpeechDurationS(s float64) {
	sd.cfg.MaxSpeechDurationS = s
}

// contextSize returns the number of context samples for the configured sample rate.
// 16kHz -> 64 samples, 8kHz -> 32 samples, matching the official Python/C++ reference.
func (sd *Detector) contextSize() int {
	if sd.cfg.SampleRate == 8000 {
		return 32
	}
	return 64
}

func (sd *Detector) sampleToMs(sample int) int {
	v := sample * 1000 / sd.cfg.SampleRate
	if v < 0 {
		return 0
	}
	return v
}

// Detect runs speech detection on a batch of PCM audio samples and returns
// the detected speech segments. The algorithm matches the official Python
// get_speech_timestamps implementation: collect probabilities, segment with
// hysteresis, then apply padding with overlap resolution.
func (sd *Detector) Detect(pcm []float32) ([]Segment, error) {
	if sd == nil {
		return nil, fmt.Errorf("invalid nil detector")
	}

	windowSize := 512
	if sd.cfg.SampleRate == 8000 {
		windowSize = 256
	}

	if len(pcm) < windowSize {
		return nil, fmt.Errorf("not enough samples")
	}

	slog.Debug("starting speech detection", slog.Int("samplesLen", len(pcm)))

	srPerMs := sd.cfg.SampleRate / 1000
	minSilenceSamples := sd.cfg.MinSilenceDurationMs * srPerMs
	speechPadSamples := sd.cfg.SpeechPadMs * srPerMs
	minSpeechSamples := sd.cfg.MinSpeechDurationMs * srPerMs
	minSilenceSamplesAtMaxSpeech := 98 * srPerMs
	audioLenSamples := len(pcm)

	negThreshold := sd.cfg.Threshold - 0.15
	if negThreshold < 0.01 {
		negThreshold = 0.01
	}

	maxSpeechSamples := math.Inf(1)
	if sd.cfg.MaxSpeechDurationS > 0 {
		maxSpeechSamples = float64(sd.cfg.SampleRate)*sd.cfg.MaxSpeechDurationS -
			float64(windowSize) - float64(2*speechPadSamples)
	}

	// Phase 1: Collect speech probabilities (including zero-padded last partial window).
	var speechProbs []float32
	for i := 0; i < audioLenSamples; i += windowSize {
		end := i + windowSize
		var chunk []float32
		if end > audioLenSamples {
			padded := make([]float32, windowSize)
			copy(padded, pcm[i:])
			chunk = padded
		} else {
			chunk = pcm[i:end]
		}
		prob, err := sd.infer(chunk)
		if err != nil {
			return nil, fmt.Errorf("infer failed: %w", err)
		}
		speechProbs = append(speechProbs, prob)
	}

	// Phase 2: Hysteresis segmentation (matches Python get_speech_timestamps).
	type rawSegment struct{ start, end int }

	var (
		triggered   bool
		speechStart int
		tempEnd     int
		prevEnd     int
		nextStart   int
		posEnds     []possibleEnd
		rawSegs     []rawSegment
	)

	for i, prob := range speechProbs {
		curSample := windowSize * i

		if prob >= sd.cfg.Threshold && tempEnd != 0 {
			silDur := curSample - tempEnd
			if silDur > minSilenceSamplesAtMaxSpeech {
				posEnds = append(posEnds, possibleEnd{tempEnd, silDur})
			}
			tempEnd = 0
			if nextStart < prevEnd {
				nextStart = curSample
			}
		}

		if prob >= sd.cfg.Threshold && !triggered {
			triggered = true
			speechStart = curSample
			continue
		}

		if triggered && float64(curSample-speechStart) > maxSpeechSamples {
			if len(posEnds) > 0 {
				bestIdx := 0
				for j := 1; j < len(posEnds); j++ {
					if posEnds[j].silDur > posEnds[bestIdx].silDur {
						bestIdx = j
					}
				}
				bestEnd := posEnds[bestIdx].pos
				rawSegs = append(rawSegs, rawSegment{speechStart, bestEnd})
				if nextStart < bestEnd {
					triggered = false
				} else {
					speechStart = nextStart
				}
			} else if prevEnd > 0 {
				rawSegs = append(rawSegs, rawSegment{speechStart, prevEnd})
				if nextStart < prevEnd {
					triggered = false
				} else {
					speechStart = nextStart
				}
			} else {
				rawSegs = append(rawSegs, rawSegment{speechStart, curSample})
				triggered = false
			}
			prevEnd = 0
			nextStart = 0
			tempEnd = 0
			posEnds = nil
			continue
		}

		if prob < negThreshold && triggered {
			if tempEnd == 0 {
				tempEnd = curSample
			}
			silDurNow := curSample - tempEnd
			if silDurNow < minSilenceSamples {
				continue
			}
			rawSegs = append(rawSegs, rawSegment{speechStart, tempEnd})
			prevEnd = 0
			nextStart = 0
			tempEnd = 0
			triggered = false
			posEnds = nil
			continue
		}
	}

	if triggered && (audioLenSamples-speechStart) > minSpeechSamples {
		rawSegs = append(rawSegs, rawSegment{speechStart, audioLenSamples})
	}

	// Phase 3: Filter by min speech duration.
	var filtered []rawSegment
	for _, seg := range rawSegs {
		if seg.end-seg.start > minSpeechSamples {
			filtered = append(filtered, seg)
		}
	}

	if len(filtered) == 0 {
		slog.Debug("speech detection done", slog.Int("segmentsLen", 0))
		return nil, nil
	}

	// Phase 4: Apply padding with overlap resolution (matches Python post-processing).
	for i := range filtered {
		if i == 0 {
			filtered[i].start = max(0, filtered[i].start-speechPadSamples)
		}
		if i < len(filtered)-1 {
			silBetween := filtered[i+1].start - filtered[i].end
			if silBetween < 2*speechPadSamples {
				filtered[i].end += silBetween / 2
				filtered[i+1].start = max(0, filtered[i+1].start-silBetween/2)
			} else {
				filtered[i].end = min(audioLenSamples, filtered[i].end+speechPadSamples)
				filtered[i+1].start = max(0, filtered[i+1].start-speechPadSamples)
			}
		} else {
			filtered[i].end = min(audioLenSamples, filtered[i].end+speechPadSamples)
		}
	}

	segments := make([]Segment, len(filtered))
	for i, seg := range filtered {
		segments[i] = Segment{
			SpeechStartAt: sd.sampleToMs(seg.start),
			SpeechEndAt:   sd.sampleToMs(seg.end),
		}
	}

	slog.Debug("speech detection done", slog.Int("segmentsLen", len(segments)))
	return segments, nil
}

// Reset clears all internal state so the detector can be reused for a new audio stream.
func (sd *Detector) Reset() error {
	if sd == nil {
		return fmt.Errorf("invalid nil detector")
	}

	sd.currSample = 0
	sd.triggered = false
	sd.tempEnd = 0
	sd.speechStart = 0
	sd.prevEnd = 0
	sd.nextStart = 0
	sd.possibleEnds = nil
	sd.residual = nil
	for i := 0; i < stateV5Len; i++ {
		sd.stateV5[i] = 0
	}
	for i := 0; i < stateV3Len; i++ {
		sd.stateH[i] = 0
		sd.stateC[i] = 0
	}
	for i := 0; i < maxContextLen; i++ {
		sd.ctx[i] = 0
	}

	return nil
}

// Destroy releases all ONNX Runtime resources. The detector must not be used after calling Destroy.
func (sd *Detector) Destroy() error {
	if sd == nil {
		return fmt.Errorf("invalid nil detector")
	}

	C.OrtApiReleaseMemoryInfo(sd.api, sd.memoryInfo)
	C.OrtApiReleaseSession(sd.api, sd.session)
	C.OrtApiReleaseSessionOptions(sd.api, sd.sessionOpts)
	C.OrtApiReleaseEnv(sd.api, sd.env)
	for _, ptr := range sd.cStrings {
		C.free(unsafe.Pointer(ptr))
	}

	return nil
}
