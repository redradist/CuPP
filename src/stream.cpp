//
// Created by redra on 14.04.21.
//

#include "stream.hpp"
#include "graph.hpp"
#include "event.hpp"

namespace cuda {

Stream::Stream() {
  THROW_IF_CUDA_ERROR(cudaStreamCreate(&stream_));
}

Stream::Stream(unsigned int flags) {
  THROW_IF_CUDA_ERROR(cudaStreamCreateWithFlags(&stream_, flags));
}

Stream::Stream(unsigned int flags, int priority) {
  THROW_IF_CUDA_ERROR(cudaStreamCreateWithPriority(&stream_, flags, priority));
}

Stream::~Stream() {
  cudaStreamDestroy(stream_);
}

void Stream::beginCapture(StreamCaptureMode mode) {
  THROW_IF_CUDA_ERROR(cudaStreamBeginCapture(stream_, static_cast<cudaStreamCaptureMode>(mode)));
}

void Stream::endCapture(Graph& graph) {
  THROW_IF_CUDA_ERROR(cudaStreamEndCapture(stream_, &graph.handle()));
}

void Stream::waitEvent(Event& event, unsigned int flags) {
  THROW_IF_CUDA_ERROR(cudaStreamWaitEvent(stream_, event.handle(), flags));
}

//void Stream::addCallback(StreamCallback callback, void *userData, unsigned int flags) {
//  THROW_IF_CUDA_ERROR(cudaStreamAddCallback(stream_, event.handle(), userData, flags));
//}

void Stream::synchronize() {
  THROW_IF_CUDA_ERROR(cudaStreamSynchronize(stream_));
}

std::pair<StreamCaptureStatus, Stream::CaptureSequenceId>
Stream::getCaptureInfo() {
  cudaStreamCaptureStatus captureStatus;
  unsigned long long id;
  THROW_IF_CUDA_ERROR(cudaStreamGetCaptureInfo(stream_, &captureStatus, &id));
  return {static_cast<StreamCaptureStatus>(captureStatus), id};
}

Stream::CaptureFlags
Stream::getFlags() {
  cudaStreamCaptureStatus captureStatus;
  unsigned int flags;
  THROW_IF_CUDA_ERROR(cudaStreamGetFlags(stream_, &flags));
  return flags;
}

}
