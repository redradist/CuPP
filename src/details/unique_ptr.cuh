//
// Created by redra on 15.03.21.
//

#ifndef CUDA_UNIQUE_PTR_CUH
#define CUDA_UNIQUE_PTR_CUH

#include <memory>
#include <atomic>
#include <type_traits>

namespace cuda {

template<typename T>
class UniquePtr {
 public:
  UniquePtr(T* ptr)
  : ptr_{ptr} {
  }

  UniquePtr(UniquePtr&& unq_ptr)
      : ptr_{unq_ptr.ptr_} {
    unq_ptr.ptr_ = nullptr;
  }

  ~UniquePtr() {
    if (ptr_) {
      cudaFree(ptr_);
    }
  }

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
  T* ptr_ = nullptr;
};

template<typename T, typename ... TArgs,
    typename = typename std::enable_if<not std::is_array<T>::value>::type,
    typename = typename std::enable_if<std::is_trivially_constructible<T>::value>>
UniquePtr<T> makeUnique() {
  T *t;
  cudaMalloc(&t, sizeof(T));
  return UniquePtr<T>(t);
}

template<typename T, typename ... TArgs,
    typename = typename std::enable_if<std::is_array<T>::value>::type,
    typename = typename std::enable_if<std::is_trivially_constructible<T>::value>>
UniquePtr<typename std::remove_extent<T>::type>
makeUnique(size_t size) {
  using array_type = typename std::remove_extent<T>::type;
  array_type *t;
  cudaMalloc(&t, size * sizeof(array_type));
  return UniquePtr<array_type>(t);
}

}

#endif //CUDA_UNIQUE_PTR_CUH
