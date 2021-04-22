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

class Stream final {
 public:
  Stream();

  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;

  Stream(Stream&&) = default;
  Stream& operator=(Stream&&) = default;

  ~Stream();

  operator cudaStream_t() const;

  void synchronize();

 private:
  cudaStream_t stream_;
};

inline
Stream::operator cudaStream_t() const {
  return stream_;
}

}

#endif //CUDAPP_STREAM_HPP
