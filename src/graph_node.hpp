//
// Created by redra on 26.04.21.
//

#ifndef CUDAPP_SRC_GRAPH_NODE_HPP_
#define CUDAPP_SRC_GRAPH_NODE_HPP_

#include <cuda_runtime.h>

#include "graph.hpp"

namespace cuda {

class Graph::Node {
 public:
  virtual ~Node() = 0;

 protected:
  friend class Graph;

  cudaGraphNode_t& node();

  cudaGraphNode_t graph_node_;
};

inline
Graph::Node::~Node() {
}

inline
cudaGraphNode_t&
Graph::Node::node() {
  return graph_node_;
}

}

#endif //CUDAPP_SRC_GRAPH_NODE_HPP_
