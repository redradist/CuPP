//
// Created by redra on 21.04.21.
//

#ifndef CUDAPP_PINNED_MEMORY_HPP
#define CUDAPP_PINNED_MEMORY_HPP

#include <cuda_runtime.h>

namespace cuda {

class PinnedMemory {
 public:
  PinnedMemory() {
  }

  PinnedMemory(const PinnedMemory&) = delete;
  PinnedMemory& operator=(const PinnedMemory&) = delete;

  PinnedMemory(PinnedMemory&&) = default;
  PinnedMemory& operator=(PinnedMemory&&) = default;
};

}

#endif //CUDAPP_PINNED_MEMORY_HPP
