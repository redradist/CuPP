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

void Stream::onStreamEvent(cudaStream_t stream, cudaError_t status, void *userData) {
  auto streamUserData = reinterpret_cast<StreamUserData*>(userData);
  streamUserData->callback_(*streamUserData->stream_, status, streamUserData->user_data_);
}

void Stream::addCallback(const StreamCallback& callback, void *userData, unsigned int flags) {
  auto streamUserData = std::make_unique<StreamUserData>(this, callback, userData);
  throwIfCudaError(cudaStreamAddCallback(stream_, &Stream::onStreamEvent, streamUserData.get(), flags));
  users_data_.push_back(std::move(streamUserData));
}

void Stream::query() {
  throwIfCudaError(cudaStreamQuery(stream_));
}

void Stream::synchronize() {
  throwIfCudaError(cudaStreamSynchronize(stream_));
}

StreamCaptureStatus
Stream::isCapturing() {
  cudaStreamCaptureStatus captureStatus;
  throwIfCudaError(cudaStreamIsCapturing(stream_, &captureStatus));
  return static_cast<StreamCaptureStatus>(captureStatus);
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
