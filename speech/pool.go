package speech

import (
	"fmt"
	"sync"
	"sync/atomic"
)

// PoolConfig defines configuration for creating a detector pool.
// These are model-level settings that are shared across all detectors
// and cannot be changed per-call.
type PoolConfig struct {
	ModelPath  string
	SampleRate int
	PoolSize   int
	LogLevel   LogLevel
}

// DetectOptions defines per-call detection parameters.
// These can differ for every call to DetectorPool.Detect.
type DetectOptions struct {
	Threshold            float32
	MinSilenceDurationMs int
	SpeechPadMs          int
	MinSpeechDurationMs  int
	MaxSpeechDurationS   float64
}

func (o DetectOptions) validate() error {
	if o.Threshold <= 0 || o.Threshold >= 1 {
		return fmt.Errorf("invalid Threshold: should be in range (0, 1)")
	}
	if o.MinSilenceDurationMs < 0 {
		return fmt.Errorf("invalid MinSilenceDurationMs: should be >= 0")
	}
	if o.SpeechPadMs < 0 {
		return fmt.Errorf("invalid SpeechPadMs: should be >= 0")
	}
	if o.MinSpeechDurationMs < 0 {
		return fmt.Errorf("invalid MinSpeechDurationMs: should be >= 0")
	}
	if o.MaxSpeechDurationS < 0 {
		return fmt.Errorf("invalid MaxSpeechDurationS: should be >= 0")
	}
	return nil
}

// DetectorPool manages a fixed-size pool of Detector instances.
// It is safe for concurrent use from multiple goroutines.
type DetectorPool struct {
	pool        chan *Detector
	closed      atomic.Bool
	destroyOnce sync.Once
}

// NewDetectorPool creates a pool of pre-initialized detectors.
// Each detector loads the ONNX model once; the pool amortises this cost.
func NewDetectorPool(cfg PoolConfig) (*DetectorPool, error) {
	if cfg.PoolSize <= 0 {
		return nil, fmt.Errorf("invalid PoolSize: must be > 0")
	}

	detCfg := DetectorConfig{
		ModelPath:            cfg.ModelPath,
		SampleRate:           cfg.SampleRate,
		LogLevel:             cfg.LogLevel,
		Threshold:            0.5,
		MinSilenceDurationMs: 100,
		SpeechPadMs:          30,
	}

	p := &DetectorPool{
		pool: make(chan *Detector, cfg.PoolSize),
	}

	for i := 0; i < cfg.PoolSize; i++ {
		sd, err := NewDetector(detCfg)
		if err != nil {
			p.Destroy()
			return nil, fmt.Errorf("failed to create detector %d: %w", i, err)
		}
		p.pool <- sd
	}

	return p, nil
}

func (p *DetectorPool) applyOpts(sd *Detector, opts DetectOptions) {
	sd.cfg.Threshold = opts.Threshold
	sd.cfg.MinSilenceDurationMs = opts.MinSilenceDurationMs
	sd.cfg.SpeechPadMs = opts.SpeechPadMs
	sd.cfg.MinSpeechDurationMs = opts.MinSpeechDurationMs
	sd.cfg.MaxSpeechDurationS = opts.MaxSpeechDurationS
}

// Detect borrows a detector from the pool, applies the given options,
// runs detection, resets the detector, and returns it to the pool.
// If the pool is empty it blocks until a detector becomes available.
func (p *DetectorPool) Detect(pcm []float32, opts DetectOptions) ([]Segment, error) {
	if err := opts.validate(); err != nil {
		return nil, fmt.Errorf("invalid options: %w", err)
	}
	if p.closed.Load() {
		return nil, fmt.Errorf("pool is destroyed")
	}

	sd := <-p.pool
	defer func() {
		_ = sd.Reset()
		p.pool <- sd
	}()

	p.applyOpts(sd, opts)
	return sd.Detect(pcm)
}

// Acquire borrows a detector from the pool for streaming use.
// It applies the given options and returns the detector ready for
// ProcessChunk calls. Blocks if all detectors are in use.
// The caller MUST call Release when the stream is done.
func (p *DetectorPool) Acquire(opts DetectOptions) (*Detector, error) {
	if err := opts.validate(); err != nil {
		return nil, fmt.Errorf("invalid options: %w", err)
	}
	if p.closed.Load() {
		return nil, fmt.Errorf("pool is destroyed")
	}

	sd := <-p.pool
	p.applyOpts(sd, opts)
	return sd, nil
}

// Release resets the detector and returns it to the pool.
// Must be called after Flush when the audio stream is complete.
func (p *DetectorPool) Release(sd *Detector) {
	if sd == nil {
		return
	}
	_ = sd.Reset()
	if p.closed.Load() {
		_ = sd.Destroy()
		return
	}
	p.pool <- sd
}

// Destroy releases all detectors and their ONNX resources.
// Safe to call multiple times. Detectors that are still acquired will
// be destroyed when they are released.
func (p *DetectorPool) Destroy() {
	p.destroyOnce.Do(func() {
		p.closed.Store(true)
		close(p.pool)
		for sd := range p.pool {
			_ = sd.Destroy()
		}
	})
}
