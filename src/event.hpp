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

 private:
  friend class Stream;

  cudaEvent_t& handle();
  const cudaEvent_t& handle() const;

  cudaEvent_t event_;
};

inline
cudaEvent_t& Event::handle() {
  return event_;
}

inline
const cudaEvent_t& Event::handle() const {
  return event_;
}

}

#endif //CUDAPP_EVENT_HPP
