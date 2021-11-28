//
// Created by redra on 14.04.21.
//

#ifndef CUPP_STREAM_HPP
#define CUPP_STREAM_HPP

#include <memory>
#include <cuda_runtime.h>
#include <utility>
#include <vector>

#include "exceptions/cuda_exception.hpp"
#include "details/resource.hpp"
#include "details/unique_ptr.hpp"
#include "capture_graph.hpp"

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

enum class StreamCaptureStatus : int {
  StatusNone        = 0, /**< Stream is not capturing */
  StatusActive      = 1, /**< Stream is actively capturing */
  StatusInvalidated = 2, /**< Stream is part of a capture sequence that */
};

class Stream final : public Resource<cudaStream_t> {
 public:
  using StreamCallback = std::function<void(Stream& stream, cudaError_t status, void *userData)>;
  using CaptureSequenceId = unsigned long long;
  using CaptureFlags = unsigned int;

  Stream();
  explicit Stream(unsigned int flags);
  Stream(unsigned int flags, int priority);

  ~Stream();

  template <typename TDes, typename TSrc>
  void memcpyAsync(UniquePtr<TDes>& dest, const std::unique_ptr<TSrc>& src, std::size_t n);
  template <typename TDes, typename TSrc>
  void memcpyAsync(std::unique_ptr<TDes>& dest, const UniquePtr<TSrc>& src, std::size_t n);
  template <typename T>
  void attachMemAsync(T *devPtr, size_t length = 0, MemAttachMode flags = MemAttachMode::MemAttachSingle);

  void beginCapture(StreamCaptureMode mode);
  void endCapture(Graph& graph);
  CaptureGraph endCapture();
  void waitEvent(Event& event, unsigned int flags);
  void addCallback(const StreamCallback& callback, void *userData, unsigned int flags);
  void query();
  void synchronize();
  [[nodiscard]]StreamCaptureStatus isCapturing();
  [[nodiscard]]std::pair<StreamCaptureStatus, CaptureSequenceId>
  getCaptureInfo();
  [[nodiscard]]CaptureFlags getFlags();

 private:
  struct StreamUserData {
    Stream* stream_;
    StreamCallback callback_;
    void* user_data_;

    StreamUserData(Stream* stream, StreamCallback callback, void* userData)
      : stream_{stream}
      , callback_{std::move(callback)}
      , user_data_{userData} {
    }
  };

  static void onStreamEvent(cudaStream_t stream, cudaError_t status, void *userData);

  std::vector<std::unique_ptr<StreamUserData>> users_data_;
};

template <typename TDes, typename TSrc>
void Stream::memcpyAsync(UniquePtr<TDes>& dest, const std::unique_ptr<TSrc>& src, std::size_t n) {
  throwIfCudaError(cudaMemcpyAsync(dest.get(), src.get(), n, cudaMemcpyHostToDevice, handle_));
}

template <typename TDes, typename TSrc>
void Stream::memcpyAsync(std::unique_ptr<TDes>& dest, const UniquePtr<TSrc>& src, std::size_t n) {
  throwIfCudaError(cudaMemcpyAsync(dest.get(), src.get(), n, cudaMemcpyDeviceToHost, handle_));
}

template <typename T>
void Stream::attachMemAsync(T *devPtr, size_t length, MemAttachMode flags) {
  throwIfCudaError(cudaStreamAttachMemAsync(handle_, devPtr, length, static_cast<unsigned>(flags)));
}

}

#endif //CUPP_STREAM_HPP
