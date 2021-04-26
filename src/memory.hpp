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
#include "details/shared_ptr.hpp"
#include "details/unique_ptr.hpp"
#include "exceptions/cuda_exception.hpp"
#include "stream.hpp"

namespace cuda {

template <typename TDes, typename TSrc>
void memcpy(UniquePtr<TDes>& dest, const std::unique_ptr<TSrc>& src, std::size_t n) {
  throwIfCudaError(cudaMemcpy(dest.get(), src.get(), n, cudaMemcpyHostToDevice));
}

template <typename TDes, typename TSrc>
void memcpy(SharedPtr<TDes>& dest, const std::shared_ptr<TSrc>& src, std::size_t n) {
  throwIfCudaError(cudaMemcpy(dest.get(), src.get(), n, cudaMemcpyHostToDevice));
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
  throwIfCudaError(cudaMemcpy(dest, src, n, cudaMemcpyHostToDevice));
}

template <typename TDes, typename TSrc>
void memcpy(std::unique_ptr<TDes>& dest, const UniquePtr<TSrc>& src, std::size_t n) {
  throwIfCudaError(cudaMemcpy(dest.get(), src.get(), n, cudaMemcpyDeviceToHost));
}

template <typename TDes, typename TSrc>
void memcpy(std::shared_ptr<TDes>& dest, const SharedPtr<TSrc>& src, std::size_t n) {
  throwIfCudaError(cudaMemcpy(dest.get(), src.get(), n, cudaMemcpyDeviceToHost));
}

#if __has_include(<span>)
template <typename TDes, typename TSrc>
void memcpy(std::span<TDes>& dest, const Span<TSrc>& src) {
  throwIfCudaError(cudaMemcpy(dest.get(), src.get(), src.size_bytes(), cudaMemcpyDeviceToHost));
}
#endif

template <typename TDes, typename TSrc>
void memcpyRawToHost(TDes* dest, const TSrc* src, std::size_t n) {
  throwIfCudaError(cudaMemcpy(dest, src, n, cudaMemcpyDeviceToHost));
}

template <typename TDes, typename TSrc>
void memcpyAsync(UniquePtr<TDes>& dest, const std::unique_ptr<TSrc>& src, std::size_t n, const Stream& stream) {
  stream.memcpyAsync(dest.get(), src.get(), n, cudaMemcpyHostToDevice);
}

template <typename TDes, typename TSrc>
void memcpyAsync(std::unique_ptr<TDes>& dest, const UniquePtr<TSrc>& src, std::size_t n, const Stream& stream) {
  stream.memcpyAsync(dest.get(), src.get(), n, cudaMemcpyDeviceToHost);
}

}

#endif //TEST0_MEMORY_HPP
