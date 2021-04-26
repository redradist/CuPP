//
// Created by redra on 21.04.21.
//

#ifndef CUDAPP_EVENT_HPP
#define CUDAPP_EVENT_HPP

#include <cuda_runtime.h>
#include "stream.hpp"

namespace cuda {

class Stream;

class Event final {
 public:
  Event();
  explicit Event(unsigned int flags);

  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;

  Event(Event&&) = default;
  Event& operator=(Event&&) = default;

  ~Event();

  void query();
  void record();
  void record(Stream& stream);
  void synchronize();
  static float elapsedTime(Event& start, Event& end);

 private:
  friend class Stream;

  cudaEvent_t event_;
};

}

#endif //CUDAPP_EVENT_HPP
