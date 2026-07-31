package speech

import (
	"fmt"
	"log/slog"
	"math"
)

// EventType represents the type of speech event during streaming detection.
type EventType int

const (
	// EventSpeechStart is emitted when speech begins.
	EventSpeechStart EventType = iota
	// EventSpeechEnd is emitted when speech ends. The Segment contains both start and end times.
	EventSpeechEnd
)

// SpeechEvent represents a speech detection event during streaming.
//
// For EventSpeechStart: only Segment.SpeechStartAt is populated, Prob is the triggering probability.
// For EventSpeechEnd:   both SpeechStartAt and SpeechEndAt are populated, FrameProbs and Samples are filled.
type SpeechEvent struct {
	Type       EventType
	Segment    Segment
	Prob       float32
	FrameProbs []float32
	Samples    []float32
}

// circularBuffer is an internal ring buffer for audio sample retention.
type circularBuffer struct {
	data []float32
	head int // logical index of the oldest sample
	size int
	cap  int
}

func newCircularBuffer(capacity int) *circularBuffer {
	return &circularBuffer{
		data: make([]float32, capacity),
		cap:  capacity,
	}
}

func (b *circularBuffer) Push(samples []float32) {
	for _, s := range samples {
		idx := (b.head + b.size) % b.cap
		if b.size == b.cap {
			b.data[idx] = s
			b.head = (b.head + 1) % b.cap
		} else {
			b.data[idx] = s
			b.size++
		}
	}
}

func (b *circularBuffer) Get(logicalStart, length int) []float32 {
	offset := logicalStart - b.Head()
	if offset < 0 {
		length += offset
		offset = 0
	}
	if offset >= b.size || length <= 0 {
		return nil
	}
	if offset+length > b.size {
		length = b.size - offset
	}
	out := make([]float32, length)
	start := (b.head + offset) % b.cap
	for i := 0; i < length; i++ {
		out[i] = b.data[(start+i)%b.cap]
	}
	return out
}

func (b *circularBuffer) Head() int {
	return b.head
}

func (b *circularBuffer) Tail() int {
	return b.head + b.size
}

func (b *circularBuffer) Size() int {
	return b.size
}

func (b *circularBuffer) Pop(n int) {
	if n > b.size {
		n = b.size
	}
	b.head = (b.head + n) % b.cap
	b.size -= n
}

// streamState holds additional state for enhanced streaming detection.
type streamState struct {
	buffer     *circularBuffer
	frameProbs []float32 // probs collected since speech start
	tempStart  int       // first above-threshold sample for min_speech pre-validation
	confirmed  bool      // true after min_speech pre-validation passes
	// logical sample counter for buffer indexing (total samples pushed)
	totalPushed       int
	silenceFrameCount int // consecutive frames below threshold (for state reset)
	framesSinceReset  int // total frames since last time-based periodic reset
}

// initStreamState initializes streaming state if not already done.
func (sd *Detector) initStreamState() {
	if sd.stream == nil {
		bufCap := sd.cfg.SampleRate * 120 // 120 seconds of audio buffer
		sd.stream = &streamState{
			buffer: newCircularBuffer(bufCap),
		}
	}
}

// ProcessChunk feeds a chunk of audio samples to the detector and returns
// any speech events detected so far. The chunk can be any size; partial
// windows are buffered internally until enough samples accumulate.
//
// This method maintains state across calls, so it must be used with a
// single audio stream. Call Flush when the stream ends, then Reset before
// reusing the detector for a new stream.
func (sd *Detector) ProcessChunk(pcm []float32) ([]SpeechEvent, error) {
	if sd == nil {
		return nil, fmt.Errorf("invalid nil detector")
	}

	sd.initStreamState()
	sd.stream.buffer.Push(pcm)
	sd.stream.totalPushed += len(pcm)

	windowSize := 512
	if sd.cfg.SampleRate == 8000 {
		windowSize = 256
	}

	var data []float32
	if len(sd.residual) > 0 {
		data = make([]float32, 0, len(sd.residual)+len(pcm))
		data = append(data, sd.residual...)
		data = append(data, pcm...)
		sd.residual = nil
	} else {
		data = pcm
	}

	if len(data) < windowSize {
		sd.residual = append(sd.residual, data...)
		return nil, nil
	}

	srPerMs := sd.cfg.SampleRate / 1000
	minSilenceSamples := sd.cfg.MinSilenceDurationMs * srPerMs
	lookback := sd.lookbackSamples()
	lookahead := sd.lookaheadSamples()
	minSpeechSamples := sd.cfg.MinSpeechDurationMs * srPerMs
	minSilenceSamplesAtMaxSpeech := 98 * srPerMs
	negThreshold := sd.negThresholdValue()

	maxSpeechSamples := math.Inf(1)
	if sd.cfg.MaxSpeechDurationS > 0 {
		maxSpeechSamples = float64(sd.cfg.SampleRate)*sd.cfg.MaxSpeechDurationS -
			float64(windowSize) - float64(lookback+lookahead)
	}

	// Dynamic threshold parameters for max_speech approach
	dynamicThreshold := float32(0.90)
	dynamicMinSilenceMs := 100
	dynamicTriggerRatio := 0.9

	// State reset: how many consecutive silence frames trigger a full RNN reset
	stateResetFrames := 0
	if sd.cfg.StateResetSilenceMs > 0 {
		stateResetFrames = (sd.cfg.StateResetSilenceMs * srPerMs) / windowSize
		if stateResetFrames < 1 {
			stateResetFrames = 1
		}
	}

	// Time-based periodic reset: unconditional reset every N ms
	stateResetIntervalFrames := 0
	if sd.cfg.StateResetIntervalMs > 0 {
		stateResetIntervalFrames = (sd.cfg.StateResetIntervalMs * srPerMs) / windowSize
		if stateResetIntervalFrames < 1 {
			stateResetIntervalFrames = 1
		}
	}

	var events []SpeechEvent
	i := 0

	for ; i+windowSize <= len(data); i += windowSize {
		speechProb, err := sd.infer(data[i : i+windowSize])
		if err != nil {
			return nil, fmt.Errorf("infer failed: %w", err)
		}

		sd.currSample += windowSize
		curSample := sd.currSample - windowSize

		// Time-based periodic reset: reset when interval elapsed and not in speech
		sd.stream.framesSinceReset++
		if stateResetIntervalFrames > 0 && sd.stream.framesSinceReset >= stateResetIntervalFrames {
			if !sd.triggered {
				sd.resetRNNState()
				sd.stream.framesSinceReset = 0
			}
		}

		// Dynamic threshold: when speech approaches max duration, tighten params
		effectiveThreshold := sd.cfg.Threshold
		effectiveNegThreshold := negThreshold
		effectiveMinSilence := minSilenceSamples
		if sd.triggered && sd.stream.confirmed && sd.cfg.MaxSpeechDurationS > 0 {
			speechDur := float64(curSample - sd.speechStart)
			if speechDur > maxSpeechSamples*dynamicTriggerRatio {
				effectiveThreshold = dynamicThreshold
				effectiveNegThreshold = dynamicThreshold - 0.15
				effectiveMinSilence = dynamicMinSilenceMs * srPerMs
			}
		}

		// Speech resumes after temp_end: record possible split point
		if speechProb >= effectiveThreshold && sd.tempEnd != 0 {
			silDur := curSample - sd.tempEnd
			if silDur > minSilenceSamplesAtMaxSpeech {
				sd.possibleEnds = append(sd.possibleEnds, possibleEnd{sd.tempEnd, silDur})
			}
			sd.tempEnd = 0
			if sd.nextStart < sd.prevEnd {
				sd.nextStart = curSample
			}
		}

		// min_speech pre-validation: track potential start
		if speechProb >= effectiveThreshold && !sd.triggered {
			sd.stream.silenceFrameCount = 0 // speech detected, reset silence counter
			if sd.stream.tempStart == 0 {
				sd.stream.tempStart = curSample
			}
			// Check if consecutive speech exceeds min_speech_duration
			if curSample-sd.stream.tempStart+windowSize >= minSpeechSamples {
				sd.triggered = true
				sd.stream.confirmed = true
				sd.speechStart = sd.stream.tempStart
				sd.stream.tempStart = 0
				sd.stream.frameProbs = nil // reset for new segment

				startAt := sd.sampleToMs(sd.speechStart - lookback)
				slog.Debug("stream: speech start (confirmed)", slog.Int("startAtMs", startAt))
				events = append(events, SpeechEvent{
					Type:    EventSpeechStart,
					Segment: Segment{SpeechStartAt: startAt},
					Prob:    speechProb,
				})
			}
			if sd.triggered {
				sd.stream.frameProbs = append(sd.stream.frameProbs, speechProb)
			}
			continue
		}

		// Reset tempStart if below threshold and not yet triggered
		if speechProb < effectiveThreshold && !sd.triggered {
			sd.stream.tempStart = 0
			// Track consecutive silence frames for state reset
			sd.stream.silenceFrameCount++
			if stateResetFrames > 0 && sd.stream.silenceFrameCount >= stateResetFrames {
				sd.resetRNNState()
				sd.stream.silenceFrameCount = 0
			}
			continue
		}

		// Collect frame probs while triggered
		if sd.triggered {
			sd.stream.frameProbs = append(sd.stream.frameProbs, speechProb)
		}

		// Max duration check
		if sd.triggered && float64(curSample-sd.speechStart) > maxSpeechSamples {
			ev := sd.handleMaxDuration(curSample, lookback, lookahead)
			events = append(events, ev...)
			continue
		}

		// Hysteresis zone
		if speechProb >= effectiveNegThreshold {
			continue
		}

		// Silence handling
		if sd.triggered {
			if sd.tempEnd == 0 {
				sd.tempEnd = curSample
			}

			if sd.currSample-sd.tempEnd > minSilenceSamplesAtMaxSpeech {
				sd.prevEnd = sd.tempEnd
			}

			if sd.currSample-sd.tempEnd >= effectiveMinSilence {
				speechEnd := sd.tempEnd
				speechDuration := speechEnd - sd.speechStart

				if speechDuration > minSpeechSamples {
					startAt := sd.sampleToMs(sd.speechStart - lookback)
					endAt := sd.sampleToMs(speechEnd + lookahead)
					slog.Debug("stream: speech end",
						slog.Int("startAtMs", startAt), slog.Int("endAtMs", endAt))

					samples := sd.extractPaddedSamples(sd.speechStart-lookback, speechEnd+lookahead)
					if sd.passesEnergyFilter(samples, sd.stream.frameProbs) {
						events = append(events, SpeechEvent{
							Type:       EventSpeechEnd,
							Segment:    Segment{SpeechStartAt: startAt, SpeechEndAt: endAt},
							FrameProbs: append([]float32{}, sd.stream.frameProbs...),
							Samples:    samples,
						})
					}
				}

				sd.prevEnd = 0
				sd.nextStart = 0
				sd.tempEnd = 0
				sd.triggered = false
				sd.stream.confirmed = false
				sd.stream.frameProbs = nil
				sd.possibleEnds = nil
				sd.applyStateDecay()
			}
		}
	}

	if i < len(data) {
		sd.residual = append([]float32{}, data[i:]...)
	}

	return events, nil
}

// passesEnergyFilter checks whether a segment passes the energy validation.
// It examines frames with prob >= threshold and checks if at least minEnergyRatio
// of them have RMS energy >= minEnergyDb. Returns true if the filter is disabled
// or if the segment passes.
// The samples slice may include lookback padding before the actual speech.
// lookbackSamples indicates how many samples of padding precede the speech data.
func (sd *Detector) passesEnergyFilter(samples []float32, frameProbs []float32) bool {
	if !sd.cfg.EnergyFilterEnabled {
		return true
	}

	minDb := sd.cfg.MinEnergyDb
	if minDb == 0 {
		minDb = -60
	}
	minRatio := sd.cfg.MinEnergyRatio
	if minRatio <= 0 {
		minRatio = 0.5
	}
	threshold := sd.cfg.Threshold
	if threshold <= 0 {
		threshold = 0.5
	}

	windowSize := 512
	if sd.cfg.SampleRate == 8000 {
		windowSize = 256
	}

	// Calculate lookback offset: samples includes lookback padding before speech,
	// but frameProbs starts from speech start. Offset aligns them.
	lookbackOffset := sd.lookbackSamples()

	probCount := 0
	energyCount := 0
	for f, prob := range frameProbs {
		if prob >= threshold {
			probCount++
			frameStart := lookbackOffset + f*windowSize
			frameEnd := frameStart + windowSize
			if frameEnd > len(samples) {
				frameEnd = len(samples)
			}
			if frameStart >= len(samples) {
				continue
			}
			var sumSq float64
			n := frameEnd - frameStart
			if n > 0 {
				for s := frameStart; s < frameEnd; s++ {
					sumSq += float64(samples[s]) * float64(samples[s])
				}
				rms := math.Sqrt(sumSq / float64(n))
				db := float64(-100)
				if rms > 0 {
					db = 20 * math.Log10(rms)
				}
				if db >= minDb {
					energyCount++
				}
			}
		}
	}

	if probCount == 0 {
		return true
	}
	return float64(energyCount)/float64(probCount) >= minRatio
}

// handleMaxDuration handles the case when speech exceeds max duration.
func (sd *Detector) handleMaxDuration(curSample, lookback, lookahead int) []SpeechEvent {
	var events []SpeechEvent

	if len(sd.possibleEnds) > 0 {
		bestIdx := 0
		for j := 1; j < len(sd.possibleEnds); j++ {
			if sd.possibleEnds[j].silDur > sd.possibleEnds[bestIdx].silDur {
				bestIdx = j
			}
		}
		bestEnd := sd.possibleEnds[bestIdx].pos
		startAt := sd.sampleToMs(sd.speechStart - lookback)
		endAt := sd.sampleToMs(bestEnd + lookahead)
		samples := sd.extractPaddedSamples(sd.speechStart-lookback, bestEnd+lookahead)
		if sd.passesEnergyFilter(samples, sd.stream.frameProbs) {
			events = append(events, SpeechEvent{
				Type:       EventSpeechEnd,
				Segment:    Segment{SpeechStartAt: startAt, SpeechEndAt: endAt},
				FrameProbs: append([]float32{}, sd.stream.frameProbs...),
				Samples:    samples,
			})
		}
		if sd.nextStart < bestEnd {
			sd.triggered = false
			sd.stream.confirmed = false
			sd.stream.frameProbs = nil
			sd.applyStateDecay()
		} else {
			sd.speechStart = sd.nextStart
			sd.stream.frameProbs = nil
			events = append(events, SpeechEvent{
				Type:    EventSpeechStart,
				Segment: Segment{SpeechStartAt: sd.sampleToMs(sd.speechStart - lookback)},
			})
		}
	} else if sd.prevEnd > 0 {
		startAt := sd.sampleToMs(sd.speechStart - lookback)
		endAt := sd.sampleToMs(sd.prevEnd + lookahead)
		samples := sd.extractPaddedSamples(sd.speechStart-lookback, sd.prevEnd+lookahead)
		if sd.passesEnergyFilter(samples, sd.stream.frameProbs) {
			events = append(events, SpeechEvent{
				Type:       EventSpeechEnd,
				Segment:    Segment{SpeechStartAt: startAt, SpeechEndAt: endAt},
				FrameProbs: append([]float32{}, sd.stream.frameProbs...),
				Samples:    samples,
			})
		}
		if sd.nextStart < sd.prevEnd {
			sd.triggered = false
			sd.stream.confirmed = false
			sd.stream.frameProbs = nil
			sd.applyStateDecay()
		} else {
			sd.speechStart = sd.nextStart
			sd.stream.frameProbs = nil
			events = append(events, SpeechEvent{
				Type:    EventSpeechStart,
				Segment: Segment{SpeechStartAt: sd.sampleToMs(sd.speechStart - lookback)},
			})
		}
	} else {
		startAt := sd.sampleToMs(sd.speechStart - lookback)
		endAt := sd.sampleToMs(sd.currSample)
		samples := sd.extractPaddedSamples(sd.speechStart-lookback, sd.currSample)
		if sd.passesEnergyFilter(samples, sd.stream.frameProbs) {
			events = append(events, SpeechEvent{
				Type:       EventSpeechEnd,
				Segment:    Segment{SpeechStartAt: startAt, SpeechEndAt: endAt},
				FrameProbs: append([]float32{}, sd.stream.frameProbs...),
				Samples:    samples,
			})
		}
		sd.triggered = false
		sd.stream.confirmed = false
		sd.stream.frameProbs = nil
		sd.applyStateDecay()
	}

	sd.prevEnd = 0
	sd.nextStart = 0
	sd.tempEnd = 0
	sd.possibleEnds = nil
	return events
}

// extractPaddedSamples retrieves audio from the circular buffer by sample position.
func (sd *Detector) extractPaddedSamples(startSample, endSample int) []float32 {
	if sd.stream == nil || sd.stream.buffer == nil {
		return nil
	}
	if startSample < 0 {
		startSample = 0
	}
	length := endSample - startSample
	if length <= 0 {
		return nil
	}

	// The buffer uses relative indexing: totalPushed - buffer.Size() is the logical head
	bufferHead := sd.stream.totalPushed - sd.stream.buffer.Size()
	relStart := startSample - bufferHead
	if relStart < 0 {
		length += relStart
		relStart = 0
	}
	if relStart >= sd.stream.buffer.Size() || length <= 0 {
		return nil
	}
	if relStart+length > sd.stream.buffer.Size() {
		length = sd.stream.buffer.Size() - relStart
	}

	out := make([]float32, length)
	bufStart := (sd.stream.buffer.head + relStart) % sd.stream.buffer.cap
	for i := 0; i < length; i++ {
		out[i] = sd.stream.buffer.data[(bufStart+i)%sd.stream.buffer.cap]
	}
	return out
}

// Flush finalises the current stream. If speech is still in progress at the
// end of the audio, it emits a final SpeechEnd event. Call Reset after Flush
// before starting a new stream.
func (sd *Detector) Flush() ([]SpeechEvent, error) {
	if sd == nil {
		return nil, fmt.Errorf("invalid nil detector")
	}

	sd.initStreamState()

	var events []SpeechEvent

	if sd.triggered {
		srPerMs := sd.cfg.SampleRate / 1000
		lookback := sd.lookbackSamples()
		minSpeechSamples := sd.cfg.MinSpeechDurationMs * srPerMs

		speechDuration := sd.currSample - sd.speechStart
		if speechDuration > minSpeechSamples {
			startAt := sd.sampleToMs(sd.speechStart - lookback)
			endAt := sd.sampleToMs(sd.currSample)
			slog.Debug("stream: speech end (flush)",
				slog.Int("startAtMs", startAt), slog.Int("endAtMs", endAt))

			samples := sd.extractPaddedSamples(sd.speechStart-lookback, sd.currSample)
			if sd.passesEnergyFilter(samples, sd.stream.frameProbs) {
				events = append(events, SpeechEvent{
					Type:       EventSpeechEnd,
					Segment:    Segment{SpeechStartAt: startAt, SpeechEndAt: endAt},
					FrameProbs: append([]float32{}, sd.stream.frameProbs...),
					Samples:    samples,
				})
			}
		}

		sd.triggered = false
		sd.tempEnd = 0
		sd.prevEnd = 0
		sd.nextStart = 0
		sd.possibleEnds = nil
		sd.stream.confirmed = false
		sd.stream.frameProbs = nil
	}

	sd.residual = nil

	return events, nil
}

// FrameProb holds a single frame's probability and its start time in milliseconds.
type FrameProb struct {
	TimeMs int
	Prob   float32
}

// ProcessChunkWithProbs is like ProcessChunk but additionally returns per-frame
// probabilities for ALL frames processed in this call (not just triggered ones).
func (sd *Detector) ProcessChunkWithProbs(pcm []float32) ([]SpeechEvent, []FrameProb, error) {
	if sd == nil {
		return nil, nil, fmt.Errorf("invalid nil detector")
	}

	sd.initStreamState()
	sd.stream.buffer.Push(pcm)
	sd.stream.totalPushed += len(pcm)

	windowSize := 512
	if sd.cfg.SampleRate == 8000 {
		windowSize = 256
	}

	var data []float32
	if len(sd.residual) > 0 {
		data = make([]float32, 0, len(sd.residual)+len(pcm))
		data = append(data, sd.residual...)
		data = append(data, pcm...)
		sd.residual = nil
	} else {
		data = pcm
	}

	if len(data) < windowSize {
		sd.residual = append(sd.residual, data...)
		return nil, nil, nil
	}

	srPerMs := sd.cfg.SampleRate / 1000
	minSilenceSamples := sd.cfg.MinSilenceDurationMs * srPerMs
	lookback := sd.lookbackSamples()
	lookahead := sd.lookaheadSamples()
	minSpeechSamples := sd.cfg.MinSpeechDurationMs * srPerMs
	minSilenceSamplesAtMaxSpeech := 98 * srPerMs
	negThreshold := sd.negThresholdValue()

	maxSpeechSamples := math.Inf(1)
	if sd.cfg.MaxSpeechDurationS > 0 {
		maxSpeechSamples = float64(sd.cfg.SampleRate)*sd.cfg.MaxSpeechDurationS -
			float64(windowSize) - float64(lookback+lookahead)
	}

	dynamicThreshold := float32(0.90)
	dynamicMinSilenceMs := 100
	dynamicTriggerRatio := 0.9

	stateResetFrames := 0
	if sd.cfg.StateResetSilenceMs > 0 {
		stateResetFrames = (sd.cfg.StateResetSilenceMs * srPerMs) / windowSize
		if stateResetFrames < 1 {
			stateResetFrames = 1
		}
	}

	// Time-based periodic reset: unconditional reset every N ms
	stateResetIntervalFrames := 0
	if sd.cfg.StateResetIntervalMs > 0 {
		stateResetIntervalFrames = (sd.cfg.StateResetIntervalMs * srPerMs) / windowSize
		if stateResetIntervalFrames < 1 {
			stateResetIntervalFrames = 1
		}
	}

	var events []SpeechEvent
	var frameProbs []FrameProb
	i := 0

	for ; i+windowSize <= len(data); i += windowSize {
		speechProb, err := sd.infer(data[i : i+windowSize])
		if err != nil {
			return nil, nil, fmt.Errorf("infer failed: %w", err)
		}

		sd.currSample += windowSize
		curSample := sd.currSample - windowSize

		// Time-based periodic reset: reset when interval elapsed and not in speech
		sd.stream.framesSinceReset++
		if stateResetIntervalFrames > 0 && sd.stream.framesSinceReset >= stateResetIntervalFrames {
			if !sd.triggered {
				sd.resetRNNState()
				sd.stream.framesSinceReset = 0
			}
		}

		frameProbs = append(frameProbs, FrameProb{
			TimeMs: sd.sampleToMs(curSample),
			Prob:   speechProb,
		})

		effectiveThreshold := sd.cfg.Threshold
		effectiveNegThreshold := negThreshold
		effectiveMinSilence := minSilenceSamples
		if sd.triggered && sd.stream.confirmed && sd.cfg.MaxSpeechDurationS > 0 {
			speechDur := float64(curSample - sd.speechStart)
			if speechDur > maxSpeechSamples*dynamicTriggerRatio {
				effectiveThreshold = dynamicThreshold
				effectiveNegThreshold = dynamicThreshold - 0.15
				effectiveMinSilence = dynamicMinSilenceMs * srPerMs
			}
		}

		if speechProb >= effectiveThreshold && sd.tempEnd != 0 {
			silDur := curSample - sd.tempEnd
			if silDur > minSilenceSamplesAtMaxSpeech {
				sd.possibleEnds = append(sd.possibleEnds, possibleEnd{sd.tempEnd, silDur})
			}
			sd.tempEnd = 0
			if sd.nextStart < sd.prevEnd {
				sd.nextStart = curSample
			}
		}

		if speechProb >= effectiveThreshold && !sd.triggered {
			sd.stream.silenceFrameCount = 0
			if sd.stream.tempStart == 0 {
				sd.stream.tempStart = curSample
			}
			if curSample-sd.stream.tempStart+windowSize >= minSpeechSamples {
				sd.triggered = true
				sd.stream.confirmed = true
				sd.speechStart = sd.stream.tempStart
				sd.stream.tempStart = 0
				sd.stream.frameProbs = nil

				startAt := sd.sampleToMs(sd.speechStart - lookback)
				events = append(events, SpeechEvent{
					Type:    EventSpeechStart,
					Segment: Segment{SpeechStartAt: startAt},
					Prob:    speechProb,
				})
			}
			if sd.triggered {
				sd.stream.frameProbs = append(sd.stream.frameProbs, speechProb)
			}
			continue
		}

		if speechProb < effectiveThreshold && !sd.triggered {
			sd.stream.tempStart = 0
			sd.stream.silenceFrameCount++
			if stateResetFrames > 0 && sd.stream.silenceFrameCount >= stateResetFrames {
				sd.resetRNNState()
				sd.stream.silenceFrameCount = 0
			}
			continue
		}

		if sd.triggered {
			sd.stream.frameProbs = append(sd.stream.frameProbs, speechProb)
		}

		if sd.triggered && float64(curSample-sd.speechStart) > maxSpeechSamples {
			ev := sd.handleMaxDuration(curSample, lookback, lookahead)
			events = append(events, ev...)
			continue
		}

		if speechProb >= effectiveNegThreshold {
			continue
		}

		if sd.triggered {
			if sd.tempEnd == 0 {
				sd.tempEnd = curSample
			}

			if sd.currSample-sd.tempEnd > minSilenceSamplesAtMaxSpeech {
				sd.prevEnd = sd.tempEnd
			}

			if sd.currSample-sd.tempEnd >= effectiveMinSilence {
				speechEnd := sd.tempEnd
				speechDuration := speechEnd - sd.speechStart

				if speechDuration > minSpeechSamples {
					startAt := sd.sampleToMs(sd.speechStart - lookback)
					endAt := sd.sampleToMs(speechEnd + lookahead)
					samples := sd.extractPaddedSamples(sd.speechStart-lookback, speechEnd+lookahead)
					if sd.passesEnergyFilter(samples, sd.stream.frameProbs) {
						events = append(events, SpeechEvent{
							Type:       EventSpeechEnd,
							Segment:    Segment{SpeechStartAt: startAt, SpeechEndAt: endAt},
							FrameProbs: append([]float32{}, sd.stream.frameProbs...),
							Samples:    samples,
						})
					}
				}

				sd.triggered = false
				sd.tempEnd = 0
				sd.prevEnd = 0
				sd.nextStart = 0
				sd.possibleEnds = nil
				sd.stream.confirmed = false
				sd.stream.frameProbs = nil
				sd.applyStateDecay()
			}
		}
	}

	if i < len(data) {
		sd.residual = append(sd.residual, data[i:]...)
	}

	return events, frameProbs, nil
}

// resetRNNState fully resets the RNN hidden state and context to zeros.
func (sd *Detector) resetRNNState() {
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
}

// decayRNNState multiplies all RNN state values by the given factor.
func (sd *Detector) decayRNNState(factor float32) {
	if factor <= 0 {
		sd.resetRNNState()
		return
	}
	if factor >= 1 {
		return
	}
	for i := 0; i < stateV5Len; i++ {
		sd.stateV5[i] *= factor
	}
	for i := 0; i < stateV3Len; i++ {
		sd.stateH[i] *= factor
		sd.stateC[i] *= factor
	}
}

// applyStateDecay applies the configured StateDecayOnEnd to the RNN state.
func (sd *Detector) applyStateDecay() {
	if sd.cfg.StateDecayOnEnd > 0 && sd.cfg.StateDecayOnEnd < 1 {
		sd.decayRNNState(sd.cfg.StateDecayOnEnd)
	}
}
