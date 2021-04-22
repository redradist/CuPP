//
// Created by redra on 22.04.21.
//

#ifndef CUDAPP_MEMORY_TYPES_HPP
#define CUDAPP_MEMORY_TYPES_HPP

namespace cuda {

enum class PinnedMemoryFlag : unsigned {
  AllocDefault = 0,
  AllocPortable = 1,
  AllocMapped = 2,
  AllocWriteCombined = 4,
};

}

#endif //CUDAPP_MEMORY_TYPES_HPP
