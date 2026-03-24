package main

import (
	"flag"
	"log"
	"os"

	"github.com/go-audio/wav"
	"github.com/yunlongliang/silero-vad-go/speech"
)

func main() {
	modelPath := flag.String("model", "../testfiles/silero_vad.onnx", "path to silero_vad.onnx model file")
	wavPath := flag.String("wav", "/home/yunlong/go/src/aicc-nlulab/module_onnx/recorder.wav", "path to input WAV file (mono, 8kHz or 16kHz)")
	sampleRate := flag.Int("rate", 8000, "sample rate (8000 or 16000)")
	poolSize := flag.Int("pool", 2, "detector pool size")
	threshold := flag.Float64("threshold", 0.5, "speech probability threshold")
	minSilenceMs := flag.Int("min-silence", 100, "minimum silence duration (ms) to split segments")
	speechPadMs := flag.Int("pad", 30, "speech padding (ms)")
	minSpeechMs := flag.Int("min-speech", 250, "minimum speech duration (ms)")
	maxSpeechS := flag.Float64("max-speech", 30.0, "maximum speech duration (s)")
	flag.Parse()

	if *modelPath == "" || *wavPath == "" {
		flag.Usage()
		log.Fatal("both -model and -wav are required")
	}

	pool, err := speech.NewDetectorPool(speech.PoolConfig{
		ModelPath:  *modelPath,
		SampleRate: *sampleRate,
		PoolSize:   *poolSize,
		LogLevel:   speech.LogLevelError,
	})
	if err != nil {
		log.Fatalf("failed to create detector pool: %s", err)
	}
	defer pool.Destroy()

	pcmData := readWAV(*wavPath)

	opts := speech.DetectOptions{
		Threshold:            float32(*threshold),
		MinSilenceDurationMs: *minSilenceMs,
		SpeechPadMs:          *speechPadMs,
		MinSpeechDurationMs:  *minSpeechMs,
		MaxSpeechDurationS:   *maxSpeechS,
	}

	log.Println("=== Batch mode ===")
	segments, err := pool.Detect(pcmData, opts)
	if err != nil {
		log.Fatalf("Detect failed: %s", err)
	}
	for _, s := range segments {
		log.Printf("  [%dms - %dms]", s.SpeechStartAt, s.SpeechEndAt)
	}

	log.Println("=== Stream mode ===")
	sd, err := pool.Acquire(opts)
	if err != nil {
		log.Fatalf("failed to acquire detector: %s", err)
	}

	chunkSize := 320
	for i := 0; i < len(pcmData); i += chunkSize {
		end := i + chunkSize
		if end > len(pcmData) {
			end = len(pcmData)
		}

		events, err := sd.ProcessChunk(pcmData[i:end])
		if err != nil {
			log.Fatalf("ProcessChunk failed: %s", err)
		}

		for _, e := range events {
			switch e.Type {
			case speech.EventSpeechStart:
				log.Printf("  >> speech started at %dms", e.Segment.SpeechStartAt)
			case speech.EventSpeechEnd:
				log.Printf("  << speech ended   [%dms - %dms]", e.Segment.SpeechStartAt, e.Segment.SpeechEndAt)
			}
		}
	}

	events, err := sd.Flush()
	if err != nil {
		log.Fatalf("Flush failed: %s", err)
	}
	for _, e := range events {
		log.Printf("  << speech ended (flush) [%dms - %dms]", e.Segment.SpeechStartAt, e.Segment.SpeechEndAt)
	}

	pool.Release(sd)
}

func readWAV(path string) []float32 {
	f, err := os.Open(path)
	if err != nil {
		log.Fatalf("failed to open WAV file: %s", err)
	}
	defer f.Close()

	dec := wav.NewDecoder(f)
	if ok := dec.IsValidFile(); !ok {
		log.Fatalf("invalid WAV file: %s", path)
	}

	buf, err := dec.FullPCMBuffer()
	if err != nil {
		log.Fatalf("failed to read PCM buffer: %s", err)
	}

	return buf.AsFloat32Buffer().Data
}
