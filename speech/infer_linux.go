//go:build !darwin

package speech

// #cgo CFLAGS: -Wall -Werror -std=c99
// #cgo LDFLAGS: -lonnxruntime
// #include "ort_bridge.h"
import "C"

import (
	"fmt"
	"unsafe"
)

func (sd *Detector) infer(samples []float32) (float32, error) {
	pcm := samples
	if sd.currSample > 0 {
		pcm = append(sd.ctx[:], samples...)
	}
	copy(sd.ctx[:], samples[len(samples)-contextLen:])

	var pcmValue *C.OrtValue
	pcmInputDims := []C.long{
		1,
		C.long(len(pcm)),
	}
	status := C.OrtApiCreateTensorWithDataAsOrtValue(sd.api, sd.memoryInfo, unsafe.Pointer(&pcm[0]), C.size_t(len(pcm)*4), &pcmInputDims[0], C.size_t(len(pcmInputDims)), C.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &pcmValue)
	defer C.OrtApiReleaseStatus(sd.api, status)
	if status != nil {
		return 0, fmt.Errorf("failed to create value: %s", C.GoString(C.OrtApiGetErrorMessage(sd.api, status)))
	}
	defer C.OrtApiReleaseValue(sd.api, pcmValue)

	if sd.separateHC {
		return sd.inferV3(pcmValue)
	}
	return sd.inferV5(pcmValue)
}

// inferV3 handles the sherpa/v3 model format: inputs=[x, h, c], outputs=[output, hn, cn]
func (sd *Detector) inferV3(pcmValue *C.OrtValue) (float32, error) {
	var hValue *C.OrtValue
	hDims := []C.long{2, 1, 64}
	status := C.OrtApiCreateTensorWithDataAsOrtValue(sd.api, sd.memoryInfo, unsafe.Pointer(&sd.stateH[0]), C.size_t(stateV3Len*4), &hDims[0], C.size_t(len(hDims)), C.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &hValue)
	defer C.OrtApiReleaseStatus(sd.api, status)
	if status != nil {
		return 0, fmt.Errorf("failed to create h value: %s", C.GoString(C.OrtApiGetErrorMessage(sd.api, status)))
	}
	defer C.OrtApiReleaseValue(sd.api, hValue)

	var cValue *C.OrtValue
	cDims := []C.long{2, 1, 64}
	status = C.OrtApiCreateTensorWithDataAsOrtValue(sd.api, sd.memoryInfo, unsafe.Pointer(&sd.stateC[0]), C.size_t(stateV3Len*4), &cDims[0], C.size_t(len(cDims)), C.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &cValue)
	defer C.OrtApiReleaseStatus(sd.api, status)
	if status != nil {
		return 0, fmt.Errorf("failed to create c value: %s", C.GoString(C.OrtApiGetErrorMessage(sd.api, status)))
	}
	defer C.OrtApiReleaseValue(sd.api, cValue)

	inputs := []*C.OrtValue{pcmValue, hValue, cValue}
	outputs := make([]*C.OrtValue, len(sd.outputNames))

	status = C.OrtApiRun(sd.api, sd.session, nil, &sd.inputNames[0], &inputs[0], C.size_t(len(sd.inputNames)), &sd.outputNames[0], C.size_t(len(sd.outputNames)), &outputs[0])
	defer C.OrtApiReleaseStatus(sd.api, status)
	if status != nil {
		return 0, fmt.Errorf("failed to run: %s", C.GoString(C.OrtApiGetErrorMessage(sd.api, status)))
	}

	var prob unsafe.Pointer
	status = C.OrtApiGetTensorMutableData(sd.api, outputs[0], &prob)
	defer C.OrtApiReleaseStatus(sd.api, status)
	if status != nil {
		return 0, fmt.Errorf("failed to get tensor data: %s", C.GoString(C.OrtApiGetErrorMessage(sd.api, status)))
	}

	var hn unsafe.Pointer
	status = C.OrtApiGetTensorMutableData(sd.api, outputs[1], &hn)
	defer C.OrtApiReleaseStatus(sd.api, status)
	if status != nil {
		return 0, fmt.Errorf("failed to get tensor data: %s", C.GoString(C.OrtApiGetErrorMessage(sd.api, status)))
	}
	C.memcpy(unsafe.Pointer(&sd.stateH[0]), hn, stateV3Len*4)

	var cn unsafe.Pointer
	status = C.OrtApiGetTensorMutableData(sd.api, outputs[2], &cn)
	defer C.OrtApiReleaseStatus(sd.api, status)
	if status != nil {
		return 0, fmt.Errorf("failed to get tensor data: %s", C.GoString(C.OrtApiGetErrorMessage(sd.api, status)))
	}
	C.memcpy(unsafe.Pointer(&sd.stateC[0]), cn, stateV3Len*4)

	for i := range outputs {
		C.OrtApiReleaseValue(sd.api, outputs[i])
	}

	return *(*float32)(prob), nil
}

// inferV5 handles the original v5 model format: inputs=[input, state, sr], outputs=[output, stateN]
func (sd *Detector) inferV5(pcmValue *C.OrtValue) (float32, error) {
	var stateValue *C.OrtValue
	stateNodeInputDims := []C.long{2, 1, 128}
	status := C.OrtApiCreateTensorWithDataAsOrtValue(sd.api, sd.memoryInfo, unsafe.Pointer(&sd.stateV5[0]), C.size_t(stateV5Len*4), &stateNodeInputDims[0], C.size_t(len(stateNodeInputDims)), C.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &stateValue)
	defer C.OrtApiReleaseStatus(sd.api, status)
	if status != nil {
		return 0, fmt.Errorf("failed to create value: %s", C.GoString(C.OrtApiGetErrorMessage(sd.api, status)))
	}
	defer C.OrtApiReleaseValue(sd.api, stateValue)

	var rateValue *C.OrtValue
	rateInputDims := []C.long{1}
	rate := []C.int64_t{C.int64_t(sd.cfg.SampleRate)}
	status = C.OrtApiCreateTensorWithDataAsOrtValue(sd.api, sd.memoryInfo, unsafe.Pointer(&rate[0]), C.size_t(8), &rateInputDims[0], C.size_t(len(rateInputDims)), C.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &rateValue)
	defer C.OrtApiReleaseStatus(sd.api, status)
	if status != nil {
		return 0, fmt.Errorf("failed to create value: %s", C.GoString(C.OrtApiGetErrorMessage(sd.api, status)))
	}
	defer C.OrtApiReleaseValue(sd.api, rateValue)

	inputs := []*C.OrtValue{pcmValue, stateValue, rateValue}
	outputs := []*C.OrtValue{nil, nil}

	status = C.OrtApiRun(sd.api, sd.session, nil, &sd.inputNames[0], &inputs[0], C.size_t(len(sd.inputNames)), &sd.outputNames[0], C.size_t(len(sd.outputNames)), &outputs[0])
	defer C.OrtApiReleaseStatus(sd.api, status)
	if status != nil {
		return 0, fmt.Errorf("failed to run: %s", C.GoString(C.OrtApiGetErrorMessage(sd.api, status)))
	}

	var prob unsafe.Pointer
	var stateN unsafe.Pointer

	status = C.OrtApiGetTensorMutableData(sd.api, outputs[0], &prob)
	defer C.OrtApiReleaseStatus(sd.api, status)
	if status != nil {
		return 0, fmt.Errorf("failed to get tensor data: %s", C.GoString(C.OrtApiGetErrorMessage(sd.api, status)))
	}

	status = C.OrtApiGetTensorMutableData(sd.api, outputs[1], &stateN)
	defer C.OrtApiReleaseStatus(sd.api, status)
	if status != nil {
		return 0, fmt.Errorf("failed to get tensor data: %s", C.GoString(C.OrtApiGetErrorMessage(sd.api, status)))
	}

	C.memcpy(unsafe.Pointer(&sd.stateV5[0]), stateN, stateV5Len*4)

	C.OrtApiReleaseValue(sd.api, outputs[0])
	C.OrtApiReleaseValue(sd.api, outputs[1])

	return *(*float32)(prob), nil
}
