//
// Created by redra on 22.04.21.
//

#ifndef CUDAPP_CUDA_EXCEPTION_HPP
#define CUDAPP_CUDA_EXCEPTION_HPP

#include <stdexcept>
#include <string>
#include <utility>

#include <cuda_runtime.h>

namespace cuda {

class CudaException : public std::exception {
 public:
  explicit CudaException(std::string msg)
    : msg_{std::move(msg)} {
  }

  const char* what() const noexcept override {
    return msg_.c_str();
  }

 private:
  std::string msg_;
};

inline constexpr void throwIfCudaError(cudaError_t err) {
  if (cudaSuccess != err) {
    throw CudaException(cudaGetErrorString(err));
  }
}

}

#endif //CUDAPP_CUDA_EXCEPTION_HPP
