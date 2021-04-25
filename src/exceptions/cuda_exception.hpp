//
// Created by redra on 22.04.21.
//

#ifndef CUDAPP_CUDA_EXCEPTION_HPP
#define CUDAPP_CUDA_EXCEPTION_HPP

#include <stdexcept>
#include <string>
#include <utility>

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

#define THROW_IF_CUDA_ERROR(err) \
  if (cudaSuccess != err) { \
    throw CudaException(cudaGetErrorString(err)); \
  }
}

#endif //CUDAPP_CUDA_EXCEPTION_HPP
