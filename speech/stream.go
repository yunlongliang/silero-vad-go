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
// For EventSpeechStart: only Segment.SpeechStartAt is populated.
// For EventSpeechEnd:   both SpeechStartAt and SpeechEndAt are populated.
type SpeechEvent struct {
	Type    EventType
	Segment Segment
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
	speechPadSamples := sd.cfg.SpeechPadMs * srPerMs
	minSpeechSamples := sd.cfg.MinSpeechDurationMs * srPerMs
	minSilenceSamplesAtMaxSpeech := 98 * srPerMs

	maxSpeechSamples := math.Inf(1)
	if sd.cfg.MaxSpeechDurationS > 0 {
		maxSpeechSamples = float64(sd.cfg.SampleRate)*sd.cfg.MaxSpeechDurationS -
			float64(windowSize) - float64(2*speechPadSamples)
	}

	var events []SpeechEvent
	i := 0

	for ; i+windowSize <= len(data); i += windowSize {
		speechProb, err := sd.infer(data[i : i+windowSize])
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
				startAt := sd.sampleToMs(sd.speechStart - speechPadSamples)
				slog.Debug("stream: speech start", slog.Int("startAtMs", startAt))
				events = append(events, SpeechEvent{
					Type:    EventSpeechStart,
					Segment: Segment{SpeechStartAt: startAt},
				})
			}
			continue
		}

		if sd.triggered && float64(sd.currSample-sd.speechStart) > maxSpeechSamples {
			if sd.prevEnd > 0 {
				endAt := sd.sampleToMs(sd.prevEnd + speechPadSamples)
				startAt := sd.sampleToMs(sd.speechStart - speechPadSamples)
				events = append(events, SpeechEvent{
					Type:    EventSpeechEnd,
					Segment: Segment{SpeechStartAt: startAt, SpeechEndAt: endAt},
				})

				if sd.nextStart < sd.prevEnd {
					sd.triggered = false
				} else {
					sd.speechStart = sd.nextStart
					events = append(events, SpeechEvent{
						Type:    EventSpeechStart,
						Segment: Segment{SpeechStartAt: sd.sampleToMs(sd.speechStart - speechPadSamples)},
					})
				}
				sd.prevEnd = 0
				sd.nextStart = 0
				sd.tempEnd = 0
			} else {
				endAt := sd.sampleToMs(sd.currSample)
				startAt := sd.sampleToMs(sd.speechStart - speechPadSamples)
				events = append(events, SpeechEvent{
					Type:    EventSpeechEnd,
					Segment: Segment{SpeechStartAt: startAt, SpeechEndAt: endAt},
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
					slog.Debug("stream: speech end",
						slog.Int("startAtMs", startAt), slog.Int("endAtMs", endAt))
					events = append(events, SpeechEvent{
						Type:    EventSpeechEnd,
						Segment: Segment{SpeechStartAt: startAt, SpeechEndAt: endAt},
					})
				}

				sd.prevEnd = 0
				sd.nextStart = 0
				sd.tempEnd = 0
				sd.triggered = false
			}
		}
	}

	if i < len(data) {
		sd.residual = append([]float32{}, data[i:]...)
	}

	return events, nil
}

// Flush finalises the current stream. If speech is still in progress at the
// end of the audio, it emits a final SpeechEnd event. Call Reset after Flush
// before starting a new stream.
func (sd *Detector) Flush() ([]SpeechEvent, error) {
	if sd == nil {
		return nil, fmt.Errorf("invalid nil detector")
	}

	var events []SpeechEvent

	if sd.triggered {
		srPerMs := sd.cfg.SampleRate / 1000
		speechPadSamples := sd.cfg.SpeechPadMs * srPerMs
		minSpeechSamples := sd.cfg.MinSpeechDurationMs * srPerMs

		speechDuration := sd.currSample - sd.speechStart
		if speechDuration > minSpeechSamples {
			startAt := sd.sampleToMs(sd.speechStart - speechPadSamples)
			endAt := sd.sampleToMs(sd.currSample)
			slog.Debug("stream: speech end (flush)",
				slog.Int("startAtMs", startAt), slog.Int("endAtMs", endAt))
			events = append(events, SpeechEvent{
				Type:    EventSpeechEnd,
				Segment: Segment{SpeechStartAt: startAt, SpeechEndAt: endAt},
			})
		}

		sd.triggered = false
		sd.tempEnd = 0
		sd.prevEnd = 0
		sd.nextStart = 0
	}

	sd.residual = nil

	return events, nil
}
