//
// Created by redra on 21.04.21.
//

#ifndef CUDAPP_EVENT_HPP
#define CUDAPP_EVENT_HPP

#include <cuda_runtime.h>
#include "stream.hpp"

namespace cuda {

class Event {
 public:
  Event();

  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;

  Event(Event&&) = default;
  Event& operator=(Event&&) = default;

  ~Event();

  operator cudaEvent_t() const;

 private:
  cudaEvent_t event_;
};

inline
Event::operator cudaEvent_t() const {
  return event_;
}

}

#endif //CUDAPP_EVENT_HPP
