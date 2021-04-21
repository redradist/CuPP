//
// Created by redra on 21.04.21.
//

#ifndef CUDAPP_PINNED_MEMORY_HPP
#define CUDAPP_PINNED_MEMORY_HPP

namespace cuda {

class PinnedMemory {
 public:
  PinnedMemory() {
    cudaHostAlloc();
  }
};

}

#endif //CUDAPP_PINNED_MEMORY_HPP
