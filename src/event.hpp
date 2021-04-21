//
// Created by redra on 21.04.21.
//

#ifndef CUDAPP_EVENT_HPP
#define CUDAPP_EVENT_HPP

#include <cuda_runtime.h>
#include "stream.cuh"

namespace cuda {

class Event {
 public:
  Event();
  ~Event();

  void record(Stream& stream);
  void stop();

 private:
  cudaEvent_t start_event_;
  cudaEvent_t end_event_;
};

}

#endif //CUDAPP_EVENT_HPP
