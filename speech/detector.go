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
	stateLen   = 2 * 1 * 128
	contextLen = 64
)

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

	cfg DetectorConfig

	state [stateLen]float32
	ctx   [contextLen]float32

	currSample  int
	triggered   bool
	tempEnd     int
	speechStart int
	prevEnd     int
	nextStart   int

	residual []float32
}

// Segment contains timing information of a speech segment in milliseconds.
type Segment struct {
	// The relative timestamp in milliseconds of when a speech segment begins.
	SpeechStartAt int
	// The relative timestamp in milliseconds of when a speech segment ends.
	SpeechEndAt int
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

	status = C.OrtApiSetSessionGraphOptimizationLevel(sd.api, sd.sessionOpts, C.ORT_ENABLE_ALL)
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

	sd.cStrings["input"] = C.CString("input")
	sd.cStrings["sr"] = C.CString("sr")
	sd.cStrings["state"] = C.CString("state")
	sd.cStrings["stateN"] = C.CString("stateN")
	sd.cStrings["output"] = C.CString("output")

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

func (sd *Detector) sampleToMs(sample int) int {
	v := sample * 1000 / sd.cfg.SampleRate
	if v < 0 {
		return 0
	}
	return v
}

// Detect runs speech detection on a batch of PCM audio samples and returns
// the detected speech segments. The samples must be float32 mono audio
// at the sample rate specified in the detector config.
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

	maxSpeechSamples := math.Inf(1)
	if sd.cfg.MaxSpeechDurationS > 0 {
		maxSpeechSamples = float64(sd.cfg.SampleRate)*sd.cfg.MaxSpeechDurationS -
			float64(windowSize) - float64(2*speechPadSamples)
	}

	var segments []Segment

	for i := 0; i+windowSize <= len(pcm); i += windowSize {
		speechProb, err := sd.infer(pcm[i : i+windowSize])
		if err != nil {
			return nil, fmt.Errorf("infer failed: %w", err)
		}

		sd.currSample += windowSize

		if speechProb >= sd.cfg.Threshold {
			if sd.tempEnd != 0 {
				sd.tempEnd = 0
				if sd.nextStart < sd.prevEnd {
					sd.nextStart = sd.currSample - windowSize
				}
			}
			if !sd.triggered {
				sd.triggered = true
				sd.speechStart = sd.currSample - windowSize
				slog.Debug("speech start",
					slog.Int("startAtMs", sd.sampleToMs(sd.speechStart-speechPadSamples)))
			}
			continue
		}

		if sd.triggered && float64(sd.currSample-sd.speechStart) > maxSpeechSamples {
			if sd.prevEnd > 0 {
				endAt := sd.sampleToMs(sd.prevEnd + speechPadSamples)
				startAt := sd.sampleToMs(sd.speechStart - speechPadSamples)
				slog.Debug("speech end (max duration, split at prev_end)",
					slog.Int("startAtMs", startAt), slog.Int("endAtMs", endAt))
				segments = append(segments, Segment{
					SpeechStartAt: startAt,
					SpeechEndAt:   endAt,
				})

				if sd.nextStart < sd.prevEnd {
					sd.triggered = false
				} else {
					sd.speechStart = sd.nextStart
				}
				sd.prevEnd = 0
				sd.nextStart = 0
				sd.tempEnd = 0
			} else {
				endAt := sd.sampleToMs(sd.currSample)
				startAt := sd.sampleToMs(sd.speechStart - speechPadSamples)
				slog.Debug("speech end (max duration, force cut)",
					slog.Int("startAtMs", startAt), slog.Int("endAtMs", endAt))
				segments = append(segments, Segment{
					SpeechStartAt: startAt,
					SpeechEndAt:   endAt,
				})

				sd.prevEnd = 0
				sd.nextStart = 0
				sd.tempEnd = 0
				sd.triggered = false
			}
			continue
		}

		if speechProb >= (sd.cfg.Threshold - 0.15) {
			continue
		}

		if sd.triggered {
			if sd.tempEnd == 0 {
				sd.tempEnd = sd.currSample
			}

			if sd.currSample-sd.tempEnd > minSilenceSamplesAtMaxSpeech {
				sd.prevEnd = sd.tempEnd
			}

			if sd.currSample-sd.tempEnd >= minSilenceSamples {
				speechEnd := sd.tempEnd
				speechDuration := speechEnd - sd.speechStart

				if speechDuration > minSpeechSamples {
					endAt := sd.sampleToMs(speechEnd + speechPadSamples)
					startAt := sd.sampleToMs(sd.speechStart - speechPadSamples)
					slog.Debug("speech end",
						slog.Int("startAtMs", startAt), slog.Int("endAtMs", endAt))
					segments = append(segments, Segment{
						SpeechStartAt: startAt,
						SpeechEndAt:   endAt,
					})
				} else {
					slog.Debug("speech segment discarded (too short)",
						slog.Int("durationSamples", speechDuration),
						slog.Int("minSpeechSamples", minSpeechSamples))
				}

				sd.prevEnd = 0
				sd.nextStart = 0
				sd.tempEnd = 0
				sd.triggered = false
			}
		}
	}

	if sd.triggered {
		speechDuration := len(pcm) - sd.speechStart
		if speechDuration > minSpeechSamples {
			startAt := sd.sampleToMs(sd.speechStart - speechPadSamples)
			endAt := sd.sampleToMs(len(pcm))
			slog.Debug("speech end (end of audio)",
				slog.Int("startAtMs", startAt), slog.Int("endAtMs", endAt))
			segments = append(segments, Segment{
				SpeechStartAt: startAt,
				SpeechEndAt:   endAt,
			})
		}

		sd.prevEnd = 0
		sd.nextStart = 0
		sd.tempEnd = 0
		sd.triggered = false
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
	sd.residual = nil
	for i := 0; i < stateLen; i++ {
		sd.state[i] = 0
	}
	for i := 0; i < contextLen; i++ {
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
