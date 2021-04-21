//
// Created by redra on 21.04.21.
//

#include "event.hpp"
#include "stream.cuh"

namespace cuda {

Event::Event() {
  cudaEventCreate(&start_event_);
  cudaEventCreate(&end_event_);
}

void Event::record(Stream& stream) {
  cudaEventRecord(start_event_, stream);
}

void Event::stop() {
  cudaEventRecord(end_event_);
  cudaEventSynchronize(end_event_);
}

Event::~Event() {
  cudaEventDestroy(start_event_);
  cudaEventDestroy(end_event_);
}

}
