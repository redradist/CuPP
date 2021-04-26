//
// Created by redra on 15.03.21.
//

#ifndef CUDA_UNIQUE_PTR_CUH
#define CUDA_UNIQUE_PTR_CUH

#include <memory>
#include <atomic>
#include <functional>
#include <type_traits>

#include <gsl/pointers>
#include <cuda_runtime.h>

#include "memory_types.hpp"

namespace cuda {

template <typename T>
using UniquePtr = std::unique_ptr<T, CudaDeleter<T>>;

template<typename T,
         typename = typename std::enable_if<not std::is_array<T>::value>::type,
         typename = typename std::enable_if<std::is_trivially_constructible<T>::value>>
UniquePtr<T> makeUnique() {
  T *t;
  cudaMalloc(&t, sizeof(T));
  return UniquePtr<T>(t);
}

template<typename T,
         typename = typename std::enable_if<std::is_array<T>::value>::type,
         typename = typename std::enable_if<std::is_trivially_constructible<T>::value>>
UniquePtr<typename std::remove_extent<T>::type>
makeUnique(const size_t size) {
  using ArrayType = typename std::remove_extent<T>::type;
  ArrayType *t;
  cudaMalloc(&t, size * sizeof(ArrayType));
  return UniquePtr<ArrayType>(t);
}

template<typename T,
         typename = typename std::enable_if<not std::is_array<T>::value>::type,
         typename = typename std::enable_if<std::is_trivially_constructible<T>::value>>
UniquePtr<T> makeUniquePinned(const PinnedMemoryFlag flags) {
  T *t;
  cudaHostAlloc(&t, sizeof(T), static_cast<unsigned>(flags));
  return UniquePtr<T>(t);
}

template<typename T,
         typename = typename std::enable_if<std::is_array<T>::value>::type,
         typename = typename std::enable_if<std::is_trivially_constructible<T>::value>>
UniquePtr<typename std::remove_extent<T>::type>
makeUniquePinned(const size_t size, const PinnedMemoryFlag flags) {
  using ArrayType = typename std::remove_extent<T>::type;
  ArrayType *t;
  cudaHostAlloc(&t, size * sizeof(ArrayType), static_cast<unsigned>(flags));
  return UniquePtr<ArrayType>(t);
}

}

#endif //CUDA_UNIQUE_PTR_CUH
