//
// Created by redra on 26.03.21.
//

#ifndef TEST0_GRAPH_HPP
#define TEST0_GRAPH_HPP

#include <cuda_runtime.h>

namespace cuda {

class Graph {
 public:
  Graph(const Graph&) = delete;
  Graph& operator=(const Graph&) = delete;

  Graph(Graph&&) = default;
  Graph& operator=(Graph&&) = default;

  void run() {

  }

 private:
  cudaGraph_t graph_;
};

}

#endif //TEST0_GRAPH_HPP
