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
#include "memory_types.hpp"

namespace cuda {

template<typename T>
using DefaultUniquePtrDeleter = std::function<void(T*)>;

template<typename T, typename Deleter = DefaultUniquePtrDeleter<T>>
class UniquePtr {
 public:
  UniquePtr(T* ptr, Deleter deleter)
      : ptr_{ptr}
      , deleter_{std::move(deleter)} {
  }

  UniquePtr(const UniquePtr&) = delete;

  UniquePtr(UniquePtr&& unq_ptr)
      : ptr_{unq_ptr.ptr_} {
    unq_ptr.ptr_ = nullptr;
  }

  ~UniquePtr() {
    if (ptr_) {
      deleter_(ptr_);
    }
  }

  UniquePtr& operator=(const UniquePtr&) = delete;

  UniquePtr& operator=(UniquePtr&& unq_ptr) {
    ptr_ = unq_ptr.ptr_;
    unq_ptr.ptr_ = nullptr;
  }

  explicit operator bool() const {
    return ptr_ != nullptr;
  }

  T& operator*() {
    return *ptr_;
  }

  T* operator->() {
    return ptr_;
  }

  T& operator[](ptrdiff_t idx) const {
    return ptr_[idx];
  }

  T* get() const {
    return ptr_;
  }

 private:
  gsl::owner<T*> ptr_ = nullptr;
  Deleter deleter_;
};

template<typename T,
         typename = typename std::enable_if<not std::is_array<T>::value>::type,
         typename = typename std::enable_if<std::is_trivially_constructible<T>::value>>
UniquePtr<T> makeUnique() {
  T *t;
  cudaMalloc(&t, sizeof(T));
  return UniquePtr<T>(t, [](T* ptr) {
    cudaFree(ptr);
  });
}

template<typename T,
         typename = typename std::enable_if<std::is_array<T>::value>::type,
         typename = typename std::enable_if<std::is_trivially_constructible<T>::value>>
UniquePtr<typename std::remove_extent<T>::type>
makeUnique(const size_t size) {
  using ArrayType = typename std::remove_extent<T>::type;
  ArrayType *t;
  cudaMalloc(&t, size * sizeof(ArrayType));
  return UniquePtr<ArrayType>(t, [](ArrayType* ptr) {
    cudaFree(ptr);
  });
}

template<typename T,
         typename = typename std::enable_if<not std::is_array<T>::value>::type,
         typename = typename std::enable_if<std::is_trivially_constructible<T>::value>>
UniquePtr<T> makeUniquePinned(const PinnedMemoryFlag flags) {
  T *t;
  cudaHostAlloc(&t, sizeof(T), static_cast<unsigned>(flags));
  return UniquePtr<T>(t, [](T* ptr) {
    cudaFreeHost(ptr);
  });
}

template<typename T,
         typename = typename std::enable_if<std::is_array<T>::value>::type,
         typename = typename std::enable_if<std::is_trivially_constructible<T>::value>>
UniquePtr<typename std::remove_extent<T>::type>
makeUniquePinned(const size_t size, const PinnedMemoryFlag flags) {
  using ArrayType = typename std::remove_extent<T>::type;
  ArrayType *t;
  cudaHostAlloc(&t, size * sizeof(ArrayType), static_cast<unsigned>(flags));
  return UniquePtr<ArrayType>(t, [](ArrayType* ptr) {
    cudaFreeHost(ptr);
  });
}

}

#endif //CUDA_UNIQUE_PTR_CUH
