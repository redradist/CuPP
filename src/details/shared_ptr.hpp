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
class SharedPtr {
 public:
  SharedPtr(T* ptr)
  : ptr_{ptr}
  , sync_block_{std::make_shared<SyncBlock>()} {
    sync_block_->strong_.fetch_add(1, std::memory_order_acq_rel);
  }

  SharedPtr(const SharedPtr& shr_ptr)
      : ptr_{shr_ptr.ptr_}
      , sync_block_{shr_ptr.sync_block_} {
    sync_block_->strong_.fetch_add(1, std::memory_order_acq_rel);
  }

  SharedPtr(SharedPtr&& shr_ptr)
      : ptr_{std::move(shr_ptr.ptr_)}
      , sync_block_{std::move(shr_ptr.sync_block_)} {
  }

  ~SharedPtr() {
    sync_block_->strong_.fetch_sub(1, std::memory_order_acq_rel);
    if (0 == sync_block_->strong_.load(std::memory_order_acquire)) {
      if (ptr_) {
        cudaFree(ptr_);
      }
    }
  }

  SharedPtr& operator=(const SharedPtr& shr_ptr) {
    ptr_ = shr_ptr.ptr_;
    sync_block_ = shr_ptr.sync_block_;
    sync_block_->strong_.fetch_add(1, std::memory_order_acq_rel);
  }

  SharedPtr& operator=(SharedPtr&& shr_ptr) {
    ptr_ = std::move(shr_ptr.ptr_);
    sync_block_ = std::move(shr_ptr.sync_block_);
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
  struct SyncBlock {
    std::atomic<ptrdiff_t> strong_{0};
    std::atomic<ptrdiff_t> weak_{0};
  };

  T* ptr_ = nullptr;
  std::shared_ptr<SyncBlock> sync_block_;
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
