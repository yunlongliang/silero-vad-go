<h1 align="center">
  <br>
  silero-vad-go
  <br>
</h1>
<h4 align="center">A Golang (CGO + ONNX Runtime) speech detector powered by <a href="https://github.com/snakers4/silero-vad">Silero VAD</a></h4>
<p align="center">
  <a href="https://pkg.go.dev/github.com/yunlongliang/silero-vad-go/speech"><img src="https://pkg.go.dev/badge/github.com/yunlongliang/silero-vad-go/speech.svg" alt="Go Reference"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
</p>
<br>

## Features

- **Batch detection** — feed a complete audio buffer and get all speech segments at once
- **Streaming detection** — feed arbitrary-sized chunks in real-time and receive speech start/end events
- **Detector pool** — goroutine-safe pool of pre-loaded detectors for concurrent workloads
- **Min/Max speech duration** — automatically discard short segments and split long ones
- **Speech padding** — configurable padding to avoid aggressive cutting

## Requirements

- [Golang](https://go.dev/doc/install) >= v1.21
- A C compiler (e.g. GCC)
- [ONNX Runtime](https://github.com/microsoft/onnxruntime/releases) (v1.18.x recommended)
- A [Silero VAD](https://github.com/snakers4/silero-vad) ONNX model (v5)

## Installation

```sh
go get github.com/yunlongliang/silero-vad-go/speech
```

### ONNX Runtime Setup

You need to download and install ONNX Runtime, then set the following environment variables.

#### Linux

```sh
export LD_RUN_PATH="/usr/local/lib/onnxruntime-linux-x64-1.18.1/lib"
export LIBRARY_PATH="/usr/local/lib/onnxruntime-linux-x64-1.18.1/lib"
export C_INCLUDE_PATH="/usr/local/include/onnxruntime-linux-x64-1.18.1/include"
```

#### Darwin (macOS)

```sh
export LIBRARY_PATH="/usr/local/lib/onnxruntime-osx-arm64-1.18.1/lib"
export C_INCLUDE_PATH="/usr/local/include/onnxruntime-osx-arm64-1.18.1/include"
sudo update_dyld_shared_cache
```

## Usage

### Batch Mode

Process a complete audio buffer and get all speech segments:

```go
package main

import (
    "fmt"
    "log"

    "github.com/yunlongliang/silero-vad-go/speech"
)

func main() {
    sd, err := speech.NewDetector(speech.DetectorConfig{
        ModelPath:            "/path/to/silero_vad.onnx",
        SampleRate:           16000,
        Threshold:            0.5,
        MinSilenceDurationMs: 100,
        SpeechPadMs:          30,
        MinSpeechDurationMs:  250,
    })
    if err != nil {
        log.Fatal(err)
    }
    defer sd.Destroy()

    // pcm: []float32 mono audio at 16kHz
    segments, err := sd.Detect(pcm)
    if err != nil {
        log.Fatal(err)
    }

    for _, s := range segments {
        fmt.Printf("Speech: %dms - %dms\n", s.SpeechStartAt, s.SpeechEndAt)
    }
}
```

### Streaming Mode

Feed audio chunks in real-time and receive events:

```go
sd, _ := speech.NewDetector(speech.DetectorConfig{
    ModelPath:            "/path/to/silero_vad.onnx",
    SampleRate:           16000,
    Threshold:            0.5,
    MinSilenceDurationMs: 100,
    SpeechPadMs:          30,
    MinSpeechDurationMs:  250,
    MaxSpeechDurationS:   30,
})
defer sd.Destroy()

// Feed chunks as they arrive (any size works)
for chunk := range audioChunks {
    events, err := sd.ProcessChunk(chunk)
    if err != nil {
        log.Fatal(err)
    }
    for _, e := range events {
        switch e.Type {
        case speech.EventSpeechStart:
            fmt.Printf("Speech started at %dms\n", e.Segment.SpeechStartAt)
        case speech.EventSpeechEnd:
            fmt.Printf("Speech ended [%dms - %dms]\n", e.Segment.SpeechStartAt, e.Segment.SpeechEndAt)
        }
    }
}

// Flush remaining speech at end of stream
events, _ := sd.Flush()
for _, e := range events {
    fmt.Printf("Speech ended (flush) [%dms - %dms]\n", e.Segment.SpeechStartAt, e.Segment.SpeechEndAt)
}
```

### Detector Pool (Concurrent)

Use a pool for safe concurrent detection from multiple goroutines:

```go
pool, err := speech.NewDetectorPool(speech.PoolConfig{
    ModelPath:  "/path/to/silero_vad.onnx",
    SampleRate: 16000,
    PoolSize:   4,
    LogLevel:   speech.LogLevelError,
})
if err != nil {
    log.Fatal(err)
}
defer pool.Destroy()

// Batch mode via pool
segments, err := pool.Detect(pcm, speech.DetectOptions{
    Threshold:            0.5,
    MinSilenceDurationMs: 100,
    SpeechPadMs:          30,
    MinSpeechDurationMs:  250,
})

// Streaming mode via pool
sd, _ := pool.Acquire(speech.DetectOptions{
    Threshold:            0.5,
    MinSilenceDurationMs: 100,
    SpeechPadMs:          30,
    MinSpeechDurationMs:  250,
    MaxSpeechDurationS:   30,
})
// ... use sd.ProcessChunk / sd.Flush ...
pool.Release(sd)
```

## Running the Example

```sh
cd examples
go run main.go -model /path/to/silero_vad.onnx -wav /path/to/audio.wav -rate 16000
```

See `examples/main.go` for the full example demonstrating both batch and streaming modes.

## API Reference

| Type | Description |
|------|-------------|
| `DetectorConfig` | Configuration for creating a `Detector` |
| `Detector` | Core VAD detector (batch + streaming) |
| `Segment` | Speech segment with start/end timestamps (ms) |
| `SpeechEvent` | Streaming event with type and segment |
| `EventType` | `EventSpeechStart` or `EventSpeechEnd` |
| `DetectOptions` | Per-call options for `DetectorPool` |
| `DetectorPool` | Goroutine-safe pool of detectors |
| `PoolConfig` | Configuration for creating a `DetectorPool` |

## License

MIT License — see [LICENSE](LICENSE) for full text.

## Credits

Based on [streamer45/silero-vad-go](https://github.com/streamer45/silero-vad-go) and the Go example from [snakers4/silero-vad](https://github.com/snakers4/silero-vad).
