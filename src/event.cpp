//
// Created by redra on 21.04.21.
//

#include "event.hpp"
#include "stream.hpp"
#include "exceptions/cuda_exception.hpp"

namespace cuda {

Event::Event() {
  throwIfCudaError(cudaEventCreate(&event_));
}

Event::Event(unsigned int flags) {
  throwIfCudaError(cudaEventCreateWithFlags(&event_, flags));
}

Event::~Event() {
  cudaEventDestroy(event_);
}

void Event::query() {
  throwIfCudaError(cudaEventQuery(event_));
}

void Event::record() {
  throwIfCudaError(cudaEventRecord(event_));
}

void Event::record(Stream& stream) {
  throwIfCudaError(cudaEventRecord(event_, stream.stream_));
}

float Event::elapsedTime(Event& start, Event& end) {
  float ms;
  throwIfCudaError(cudaEventElapsedTime(&ms, start.event_, end.event_));
  return ms;
}

void Event::synchronize() {
  throwIfCudaError(cudaEventSynchronize(event_));
}

}
