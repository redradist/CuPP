//
// Created by redra on 21.04.21.
//

#include "event.hpp"
#include "stream.hpp"
#include "exceptions/cuda_exception.hpp"

namespace cuda {

Event::Event() {
  throwIfCudaError(cudaEventCreate(&handle_));
}

Event::Event(unsigned int flags) {
  throwIfCudaError(cudaEventCreateWithFlags(&handle_, flags));
}

Event::~Event() {
  cudaEventDestroy(handle_);
}

void Event::query() {
  throwIfCudaError(cudaEventQuery(handle_));
}

void Event::record() {
  throwIfCudaError(cudaEventRecord(handle_));
}

void Event::record(Stream& stream) {
  throwIfCudaError(cudaEventRecord(handle_, handleFrom(stream)));
}

float Event::elapsedTime(Event& start, Event& end) {
  float ms;
  throwIfCudaError(cudaEventElapsedTime(&ms, start.handle_, end.handle_));
  return ms;
}

void Event::synchronize() {
  throwIfCudaError(cudaEventSynchronize(handle_));
}

}
