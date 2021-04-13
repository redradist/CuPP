//
// Created by redra on 27.03.21.
//

#ifndef TEST0_SPAN_HPP
#define TEST0_SPAN_HPP

#include <limits>
#include <gsl/>

namespace cuda {

template<typename T, std::size_t Extent = std::numeric_limits<std::size_t>::max()>
class Span {
 public:
  Span(T* t, std::size_t n)
    : ptr_{t}
    , size_{n} {
  }

  T& operator[](ptrdiff_t idx) const {
    return ptr_[idx];
  }

  T* data() const {
    return ptr_;
  }

  std::size_t size() const {
    return size_;
  }

  std::size_t size_bytes() const {
    return sizeof(T) * size_;
  }

 private:
  T* ptr_;
  std::size_t size_;
};

}

#endif //TEST0_SPAN_HPP
