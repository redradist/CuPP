//
// Created by redra on 21.04.21.
//

#include "event.hpp"
#include "stream.hpp"

namespace cuda {

Event::Event() {
  cudaEventCreate(&event_);
}

Event::~Event() {
  cudaEventDestroy(event_);
}

}
