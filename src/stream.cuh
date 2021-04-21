//
// Created by redra on 14.04.21.
//

#ifndef CUDAPP_STREAM_CUH
#define CUDAPP_STREAM_CUH

#include <memory>
#include <cuda_runtime.h>
#include "details/unique_ptr.cuh"

namespace cuda {

class Stream final {
 public:
  Stream();
  ~Stream();

  operator cudaStream_t() const;

  template <typename TDes, typename TSrc>
  void memcpyAsync(UniquePtr<TDes>& dest, const std::unique_ptr<TSrc>& src, std::size_t n) {
    cudaMemcpyAsync(dest.get(), src.get(), n, cudaMemcpyHostToDevice, stream_);
  }

  template <typename TDes, typename TSrc>
  void memcpy(std::unique_ptr<TDes>& dest, const UniquePtr<TSrc>& src, std::size_t n) {
    cudaMemcpyAsync(dest.get(), src.get(), n, cudaMemcpyDeviceToHost, stream_);
  }

  void synchronize();

 private:
  cudaStream_t stream_;
};

inline
Stream::operator cudaStream_t() const {
  return stream_;
}

}

#endif //CUDAPP_STREAM_CUH
