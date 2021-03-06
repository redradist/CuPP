//
// Created by redra on 21.04.21.
//

#ifndef CUPP_EVENT_HPP
#define CUPP_EVENT_HPP

#include <cuda_runtime.h>
#include "stream.hpp"

namespace cuda {

class Stream;

class Event final : public Handle<cudaEvent_t> {
 public:
  Event();
  explicit Event(unsigned int flags);

  ~Event();

  static float elapsedTime(Event& start, Event& end);

  void query();
  void record();
  void record(Stream& stream);
  void synchronize();
};

}

#endif //CUPP_EVENT_HPP
