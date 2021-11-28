//
// Created by redra on 12.04.21.
//

#ifndef CUPP_CAPTUREGRAPH_HPP
#define CUPP_CAPTUREGRAPH_HPP

#include <utility>
#include "details/resource.hpp"

namespace cuda {

class Stream;

class CaptureGraph final : public Resource<cudaGraph_t> {
 public:
  explicit CaptureGraph(Stream& stream);
  ~CaptureGraph();

  void launch();

};

}

#endif //CUPP_CAPTUREGRAPH_HPP
