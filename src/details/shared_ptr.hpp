//
// Created by redra on 15.03.21.
//

#ifndef CUDA_SHARED_PTR_HPP
#define CUDA_SHARED_PTR_HPP

#include <memory>
#include <atomic>
#include "memory_types.hpp"

namespace cuda {

template<typename T>
 class SharedPtr : protected std::shared_ptr<T> {
 public:
  explicit SharedPtr(T* ptr)
    : std::shared_ptr<T>{ptr, CudaDeleter<T>{}} {
  }

   using std::shared_ptr<T>::operator=;
   using std::shared_ptr<T>::reset;
   using std::shared_ptr<T>::swap;
   using std::shared_ptr<T>::get;
   using std::shared_ptr<T>::operator*;
   using std::shared_ptr<T>::operator->;
#if __cplusplus >= 201703L
   using std::shared_ptr<T>::operator[];
#endif
   using std::shared_ptr<T>::use_count;
#if __cplusplus > 201703L
   using std::shared_ptr<T>::unique;
#endif
   using std::shared_ptr<T>::operator bool;
   using std::shared_ptr<T>::owner_before;
};

template<typename T,
         typename = typename std::enable_if<not std::is_array<T>::value>::type,
         typename = typename std::enable_if<std::is_trivially_constructible<T>::value>>
SharedPtr<T> makeShared() {
  T *t;
  cudaMalloc(&t, sizeof(T));
  return SharedPtr<T>(t);
}

template<typename T,
         typename = typename std::enable_if<std::is_array<T>::value>::type,
         typename = typename std::enable_if<std::is_trivially_constructible<T>::value>>
SharedPtr<typename std::remove_extent<T>::type>
makeShared(size_t size) {
  using array_type = typename std::remove_extent<T>::type;
  array_type *t;
  cudaMalloc(&t, size * sizeof(array_type));
  return SharedPtr<array_type>(t);
}

}

#endif //CUDA_SHARED_PTR_HPP
