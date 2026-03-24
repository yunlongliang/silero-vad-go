package speech

import (
	"encoding/binary"
	"log/slog"
	"math"
	"os"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestDetectorConfigIsValid(t *testing.T) {
	tcs := []struct {
		name string
		cfg  DetectorConfig
		err  string
	}{
		{
			name: "missing ModelPath",
			cfg: DetectorConfig{
				ModelPath: "",
			},
			err: "invalid ModelPath: should not be empty",
		},
		{
			name: "invalid SampleRate",
			cfg: DetectorConfig{
				ModelPath:  "../testfiles/silero_vad.onnx",
				SampleRate: 48000,
			},
			err: "invalid SampleRate: valid values are 8000 and 16000",
		},
		{
			name: "invalid Threshold",
			cfg: DetectorConfig{
				ModelPath:  "../testfiles/silero_vad.onnx",
				SampleRate: 16000,
				Threshold:  0,
			},
			err: "invalid Threshold: should be in range (0, 1)",
		},
		{
			name: "invalid MinSilenceDurationMs",
			cfg: DetectorConfig{
				ModelPath:            "../testfiles/silero_vad.onnx",
				SampleRate:           16000,
				Threshold:            0.5,
				MinSilenceDurationMs: -1,
			},
			err: "invalid MinSilenceDurationMs: should be a non-negative number",
		},
		{
			name: "invalid SpeechPadMs",
			cfg: DetectorConfig{
				ModelPath:   "../testfiles/silero_vad.onnx",
				SampleRate:  16000,
				Threshold:   0.5,
				SpeechPadMs: -1,
			},
			err: "invalid SpeechPadMs: should be a non-negative number",
		},
		{
			name: "invalid MinSpeechDurationMs",
			cfg: DetectorConfig{
				ModelPath:           "../testfiles/silero_vad.onnx",
				SampleRate:          16000,
				Threshold:           0.5,
				MinSpeechDurationMs: -1,
			},
			err: "invalid MinSpeechDurationMs: should be a non-negative number",
		},
		{
			name: "invalid MaxSpeechDurationS",
			cfg: DetectorConfig{
				ModelPath:          "../testfiles/silero_vad.onnx",
				SampleRate:         16000,
				Threshold:          0.5,
				MaxSpeechDurationS: -1,
			},
			err: "invalid MaxSpeechDurationS: should be a non-negative number",
		},
		{
			name: "valid",
			cfg: DetectorConfig{
				ModelPath:  "../testfiles/silero_vad.onnx",
				SampleRate: 16000,
				Threshold:  0.5,
			},
		},
	}

	for _, tc := range tcs {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.cfg.IsValid()
			if tc.err != "" {
				require.EqualError(t, err, tc.err)
			} else {
				require.NoError(t, err)
			}
		})
	}
}

func TestNewDetector(t *testing.T) {
	cfg := DetectorConfig{
		ModelPath:  "../testfiles/silero_vad.onnx",
		SampleRate: 16000,
		Threshold:  0.5,
	}

	sd, err := NewDetector(cfg)
	require.NoError(t, err)
	require.NotNil(t, sd)

	err = sd.Destroy()
	require.NoError(t, err)
}

func TestSpeechDetection(t *testing.T) {
	cfg := DetectorConfig{
		ModelPath:  "../testfiles/silero_vad.onnx",
		SampleRate: 16000,
		Threshold:  0.5,
	}

	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{
		AddSource: true,
		Level:     slog.LevelDebug,
	}))
	slog.SetDefault(logger)

	sd, err := NewDetector(cfg)
	require.NoError(t, err)
	require.NotNil(t, sd)
	defer func() {
		require.NoError(t, sd.Destroy())
	}()

	readSamplesFromFile := func(path string) []float32 {
		data, err := os.ReadFile(path)
		require.NoError(t, err)

		samples := make([]float32, 0, len(data)/4)
		for i := 0; i < len(data); i += 4 {
			samples = append(samples, math.Float32frombits(binary.LittleEndian.Uint32(data[i:i+4])))
		}
		return samples
	}

	samples := readSamplesFromFile("../testfiles/samples.pcm")
	samples2 := readSamplesFromFile("../testfiles/samples2.pcm")

	t.Run("detect", func(t *testing.T) {
		segments, err := sd.Detect(samples)
		require.NoError(t, err)
		require.NotEmpty(t, segments)
		t.Logf("segments: %+v", segments)

		err = sd.Reset()
		require.NoError(t, err)

		segments, err = sd.Detect(samples2)
		require.NoError(t, err)
		require.NotEmpty(t, segments)
		t.Logf("segments2: %+v", segments)
	})

	t.Run("reset", func(t *testing.T) {
		err = sd.Reset()
		require.NoError(t, err)

		segments, err := sd.Detect(samples)
		require.NoError(t, err)
		require.NotEmpty(t, segments)
		t.Logf("segments after reset: %+v", segments)
	})

	t.Run("speech padding", func(t *testing.T) {
		cfg.SpeechPadMs = 30
		sd2, err := NewDetector(cfg)
		require.NoError(t, err)
		require.NotNil(t, sd2)
		defer func() {
			require.NoError(t, sd2.Destroy())
		}()

		segments, err := sd2.Detect(samples)
		require.NoError(t, err)
		require.NotEmpty(t, segments)
		t.Logf("segments with padding: %+v", segments)
	})

	t.Run("stream", func(t *testing.T) {
		cfg.SpeechPadMs = 30
		cfg.MinSpeechDurationMs = 250
		cfg.MaxSpeechDurationS = 30
		sd3, err := NewDetector(cfg)
		require.NoError(t, err)
		require.NotNil(t, sd3)
		defer func() {
			require.NoError(t, sd3.Destroy())
		}()

		chunkSize := 320
		var allEvents []SpeechEvent
		for i := 0; i < len(samples); i += chunkSize {
			end := i + chunkSize
			if end > len(samples) {
				end = len(samples)
			}
			events, err := sd3.ProcessChunk(samples[i:end])
			require.NoError(t, err)
			allEvents = append(allEvents, events...)
		}
		flushEvents, err := sd3.Flush()
		require.NoError(t, err)
		allEvents = append(allEvents, flushEvents...)
		require.NotEmpty(t, allEvents)
		t.Logf("stream events: %+v", allEvents)
	})
}

func TestDetectorPool(t *testing.T) {
	pool, err := NewDetectorPool(PoolConfig{
		ModelPath:  "../testfiles/silero_vad.onnx",
		SampleRate: 16000,
		PoolSize:   2,
		LogLevel:   LogLevelError,
	})
	require.NoError(t, err)
	defer pool.Destroy()

	readSamplesFromFile := func(path string) []float32 {
		data, err := os.ReadFile(path)
		require.NoError(t, err)

		samples := make([]float32, 0, len(data)/4)
		for i := 0; i < len(data); i += 4 {
			samples = append(samples, math.Float32frombits(binary.LittleEndian.Uint32(data[i:i+4])))
		}
		return samples
	}

	samples := readSamplesFromFile("../testfiles/samples.pcm")

	t.Run("batch detect via pool", func(t *testing.T) {
		segments, err := pool.Detect(samples, DetectOptions{
			Threshold:            0.5,
			MinSilenceDurationMs: 100,
			SpeechPadMs:          30,
			MinSpeechDurationMs:  250,
		})
		require.NoError(t, err)
		require.NotEmpty(t, segments)
		t.Logf("pool segments: %+v", segments)
	})

	t.Run("streaming via pool", func(t *testing.T) {
		sd, err := pool.Acquire(DetectOptions{
			Threshold:            0.5,
			MinSilenceDurationMs: 100,
			SpeechPadMs:          30,
			MinSpeechDurationMs:  250,
			MaxSpeechDurationS:   30,
		})
		require.NoError(t, err)
		require.NotNil(t, sd)

		chunkSize := 320
		var allEvents []SpeechEvent
		for i := 0; i < len(samples); i += chunkSize {
			end := i + chunkSize
			if end > len(samples) {
				end = len(samples)
			}
			events, err := sd.ProcessChunk(samples[i:end])
			require.NoError(t, err)
			allEvents = append(allEvents, events...)
		}
		flushEvents, err := sd.Flush()
		require.NoError(t, err)
		allEvents = append(allEvents, flushEvents...)

		pool.Release(sd)
		require.NotEmpty(t, allEvents)
		t.Logf("pool stream events: %+v", allEvents)
	})
}
