//
// Created by redra on 26.03.21.
//

#ifndef TEST0_GRAPH_HPP
#define TEST0_GRAPH_HPP

#include <cuda_runtime.h>

namespace cuda {

class Graph final {
 public:
  Graph(unsigned int flags);
  ~Graph();

  Graph(const Graph&) = delete;
  Graph& operator=(const Graph&) = delete;

  Graph(Graph&&) = default;
  Graph& operator=(Graph&&) = default;

 private:
  friend class Stream;

  cudaGraph_t& handle();
  const cudaGraph_t& handle() const;

  cudaGraph_t graph_;
};

inline
cudaGraph_t& Graph::handle() {
  return graph_;
}

inline
const cudaGraph_t& Graph::handle() const {
  return graph_;
}

}

#endif //TEST0_GRAPH_HPP
