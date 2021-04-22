//
// Created by redra on 14.04.21.
//

#ifndef CUDAPP_STREAM_HPP
#define CUDAPP_STREAM_HPP

#include <memory>
#include <cuda_runtime.h>
#include "exceptions/cuda_exception.hpp"
#include "details/unique_ptr.hpp"

namespace cuda {

class Graph;
class Event;

enum class StreamCaptureMode : int {
  CaptureModeGlobal      = 0,
  CaptureModeThreadLocal = 1,
  CaptureModeRelaxed     = 2
};

enum class MemAttachMode : unsigned {
  MemAttachGlobal = 0,
  MemAttachHost   = 1,
  MemAttachSingle = 4,
};


class Stream final {
 public:
  using StreamCallback = void(Stream& stream, cudaError_t status, void *userData);

  Stream();

  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;

  Stream(Stream&&) = default;
  Stream& operator=(Stream&&) = default;

  ~Stream();

  template <typename TDes, typename TSrc>
  void memcpyAsync(UniquePtr<TDes>& dest, const std::unique_ptr<TSrc>& src, std::size_t n);

  template <typename TDes, typename TSrc>
  void memcpyAsync(std::unique_ptr<TDes>& dest, const UniquePtr<TSrc>& src, std::size_t n);

  template <typename T>
  void attachMemAsync(T *devPtr, size_t length = 0, MemAttachMode flags = MemAttachMode::MemAttachSingle);

  void beginCapture(StreamCaptureMode mode);
  void endCapture(Graph& graph);
  void waitEvent(Event& event, unsigned int flags);
  void synchronize();

 private:
  cudaStream_t stream_;
};

template <typename TDes, typename TSrc>
void Stream::memcpyAsync(UniquePtr<TDes>& dest, const std::unique_ptr<TSrc>& src, std::size_t n) {
  THROW_IF_CUDA_ERROR(cudaMemcpyAsync(dest.get(), src.get(), n, cudaMemcpyHostToDevice, stream_));
}

template <typename TDes, typename TSrc>
void Stream::memcpyAsync(std::unique_ptr<TDes>& dest, const UniquePtr<TSrc>& src, std::size_t n) {
  THROW_IF_CUDA_ERROR(cudaMemcpyAsync(dest.get(), src.get(), n, cudaMemcpyDeviceToHost, stream_));
}

template <typename T>
void Stream::attachMemAsync(T *devPtr, size_t length, MemAttachMode flags) {
  THROW_IF_CUDA_ERROR(cudaStreamAttachMemAsync(stream_, devPtr, length, static_cast<unsigned>(flags)));
}

}

#endif //CUDAPP_STREAM_HPP
