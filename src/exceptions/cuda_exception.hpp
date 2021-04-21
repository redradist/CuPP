//
// Created by redra on 22.04.21.
//

#ifndef CUDAPP_CUDA_EXCEPTION_HPP
#define CUDAPP_CUDA_EXCEPTION_HPP

#include <stdexcept>
#include <string>

namespace cuda {

class CudaException : public std::exception {
 public:
  explicit CudaException(std::string msg)
    : msg_{msg} {
  }

  const char* what() const noexcept override {
    return msg_.c_str();
  }

 private:
  std::string msg_;
};

}

#endif //CUDAPP_CUDA_EXCEPTION_HPP
