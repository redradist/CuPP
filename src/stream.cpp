//
// Created by redra on 14.04.21.
//

#include "stream.hpp"
#include "graph.hpp"
#include "event.hpp"

namespace cuda {

Stream::Stream() {
  throwIfCudaError(cudaStreamCreate(&handle_));
}

Stream::Stream(unsigned int flags) {
  throwIfCudaError(cudaStreamCreateWithFlags(&handle_, flags));
}

Stream::Stream(unsigned int flags, int priority) {
  throwIfCudaError(cudaStreamCreateWithPriority(&handle_, flags, priority));
}

Stream::~Stream() {
  cudaStreamDestroy(handle_);
}

void Stream::beginCapture(StreamCaptureMode mode) {
  throwIfCudaError(cudaStreamBeginCapture(handle_, static_cast<cudaStreamCaptureMode>(mode)));
}

void Stream::endCapture(Graph& graph) {
  throwIfCudaError(cudaStreamEndCapture(handle_, &handleFrom(graph)));
}

void Stream::waitEvent(Event& event, unsigned int flags) {
  throwIfCudaError(cudaStreamWaitEvent(handle_, handleFrom(event), flags));
}

void Stream::onStreamEvent(cudaStream_t stream, cudaError_t status, void *userData) {
  auto streamUserData = reinterpret_cast<StreamUserData*>(userData);
  streamUserData->callback_(*streamUserData->stream_, status, streamUserData->user_data_);
}

void Stream::addCallback(const StreamCallback& callback, void *userData, unsigned int flags) {
  auto streamUserData = std::make_unique<StreamUserData>(this, callback, userData);
  throwIfCudaError(cudaStreamAddCallback(handle_, &Stream::onStreamEvent, streamUserData.get(), flags));
  users_data_.push_back(std::move(streamUserData));
}

void Stream::query() {
  throwIfCudaError(cudaStreamQuery(handle_));
}

void Stream::synchronize() {
  throwIfCudaError(cudaStreamSynchronize(handle_));
}

StreamCaptureStatus
Stream::isCapturing() {
  cudaStreamCaptureStatus captureStatus;
  throwIfCudaError(cudaStreamIsCapturing(handle_, &captureStatus));
  return static_cast<StreamCaptureStatus>(captureStatus);
}

std::pair<StreamCaptureStatus, Stream::CaptureSequenceId>
Stream::getCaptureInfo() {
  cudaStreamCaptureStatus captureStatus;
  unsigned long long id;
  throwIfCudaError(cudaStreamGetCaptureInfo(handle_, &captureStatus, &id));
  return {static_cast<StreamCaptureStatus>(captureStatus), id};
}

Stream::CaptureFlags
Stream::getFlags() {
  cudaStreamCaptureStatus captureStatus;
  unsigned int flags;
  throwIfCudaError(cudaStreamGetFlags(handle_, &flags));
  return flags;
}

}
