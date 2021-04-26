//
// Created by redra on 14.04.21.
//

#include "stream.hpp"
#include "graph.hpp"
#include "event.hpp"

namespace cuda {

Stream::Stream() {
  throwIfCudaError(cudaStreamCreate(&stream_));
}

Stream::Stream(unsigned int flags) {
  throwIfCudaError(cudaStreamCreateWithFlags(&stream_, flags));
}

Stream::Stream(unsigned int flags, int priority) {
  throwIfCudaError(cudaStreamCreateWithPriority(&stream_, flags, priority));
}

Stream::~Stream() {
  cudaStreamDestroy(stream_);
}

void Stream::beginCapture(StreamCaptureMode mode) {
  throwIfCudaError(cudaStreamBeginCapture(stream_, static_cast<cudaStreamCaptureMode>(mode)));
}

void Stream::endCapture(Graph& graph) {
  throwIfCudaError(cudaStreamEndCapture(stream_, &graph.handle()));
}

void Stream::waitEvent(Event& event, unsigned int flags) {
  throwIfCudaError(cudaStreamWaitEvent(stream_, event.handle(), flags));
}

//void Stream::addCallback(StreamCallback callback, void *userData, unsigned int flags) {
//  throwIfCudaError(cudaStreamAddCallback(stream_, event.handle(), userData, flags));
//}

void Stream::synchronize() {
  throwIfCudaError(cudaStreamSynchronize(stream_));
}

std::pair<StreamCaptureStatus, Stream::CaptureSequenceId>
Stream::getCaptureInfo() {
  cudaStreamCaptureStatus captureStatus;
  unsigned long long id;
  throwIfCudaError(cudaStreamGetCaptureInfo(stream_, &captureStatus, &id));
  return {static_cast<StreamCaptureStatus>(captureStatus), id};
}

Stream::CaptureFlags
Stream::getFlags() {
  cudaStreamCaptureStatus captureStatus;
  unsigned int flags;
  throwIfCudaError(cudaStreamGetFlags(stream_, &flags));
  return flags;
}

}
