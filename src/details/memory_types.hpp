//
// Created by redra on 22.04.21.
//

#ifndef CUPP_MEMORY_TYPES_HPP
#define CUPP_MEMORY_TYPES_HPP

#include <memory>
#include <type_traits>

namespace cuda {

enum class PinnedMemoryFlag : unsigned {
  AllocDefault = 0,
  AllocPortable = 1,
  AllocMapped = 2,
  AllocWriteCombined = 4,
};

template<typename T>
struct CudaDeleter {
  constexpr CudaDeleter() noexcept = default;

  template<typename U,
      typename = typename std::enable_if<std::is_convertible<U*, T*>::value>::type>
  CudaDeleter(const CudaDeleter<U>&) noexcept { }

  void operator()(T* ptr) const {
    cudaFreeHost(ptr);
  }
};

}

#endif //CUPP_MEMORY_TYPES_HPP
