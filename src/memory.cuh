//
// Created by redra on 25.03.21.
//

#ifndef TEST0_MEMORY_HPP
#define TEST0_MEMORY_HPP

#include <memory>
#if __has_include(<span>)
#include <span>
#endif
#include "span.cuh"
#include "details/shared_ptr.cuh"
#include "details/unique_ptr.cuh"
#include "exceptions/cuda_exception.hpp"

namespace cuda {

template <typename TDes, typename TSrc>
void memcpy(UniquePtr<TDes>& dest, const std::unique_ptr<TSrc>& src, std::size_t n) {
  const cudaError_t err = cudaMemcpy(dest.get(), src.get(), n, cudaMemcpyHostToDevice);
  if (cudaSuccess != err) {
    throw CudaException(cudaGetErrorString(err));
  }
}

template <typename TDes, typename TSrc>
void memcpy(SharedPtr<TDes>& dest, const std::shared_ptr<TSrc>& src, std::size_t n) {
  const cudaError_t err = cudaMemcpy(dest.get(), src.get(), n, cudaMemcpyHostToDevice);
  if (cudaSuccess != err) {
    throw CudaException(cudaGetErrorString(err));
  }
}

#if __has_include(<span>)
template <typename TDes, typename TSrc>
void memcpy(Span<TDes> dest, std::span<const TSrc> src) {
  const cudaError_t err = cudaMemcpy(dest, src, src.size_bytes(), cudaMemcpyHostToDevice);
  if (cudaSuccess != err) {
    throw CudaException(cudaGetErrorString(err));
  }
}
#endif

template <typename TDes, typename TSrc>
void memcpyRawToDevice(TDes* dest, const TSrc* src, std::size_t n) {
  const cudaError_t err = cudaMemcpy(dest, src, n, cudaMemcpyHostToDevice);
  if (cudaSuccess != err) {
    throw CudaException(cudaGetErrorString(err));
  }
}

template <typename TDes, typename TSrc>
void memcpy(std::unique_ptr<TDes>& dest, const UniquePtr<TSrc>& src, std::size_t n) {
  const cudaError_t err = cudaMemcpy(dest.get(), src.get(), n, cudaMemcpyDeviceToHost);
  if (cudaSuccess != err) {
    throw CudaException(cudaGetErrorString(err));
  }
}

template <typename TDes, typename TSrc>
void memcpy(std::shared_ptr<TDes>& dest, const SharedPtr<TSrc>& src, std::size_t n) {
  const cudaError_t err = cudaMemcpy(dest.get(), src.get(), n, cudaMemcpyDeviceToHost);
  if (cudaSuccess != err) {
    throw CudaException(cudaGetErrorString(err));
  }
}

#if __has_include(<span>)
template <typename TDes, typename TSrc>
void memcpy(std::span<TDes>& dest, const Span<TSrc>& src) {
  const cudaError_t err = cudaMemcpy(dest.get(), src.get(), src.size_bytes(), cudaMemcpyDeviceToHost);
  if (cudaSuccess != err) {
    throw CudaException(cudaGetErrorString(err));
  }
}
#endif

template <typename TDes, typename TSrc>
void memcpyRawToHost(TDes* dest, const TSrc* src, std::size_t n) {
  const cudaError_t err = cudaMemcpy(dest, src, n, cudaMemcpyDeviceToHost);
  if (cudaSuccess != err) {
    throw CudaException(cudaGetErrorString(err));
  }
}

}

#endif //TEST0_MEMORY_HPP
