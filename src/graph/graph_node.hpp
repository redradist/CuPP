//
// Created by redra on 26.04.21.
//

#ifndef CUPP_SRC_GRAPH_NODE_HPP_
#define CUPP_SRC_GRAPH_NODE_HPP_

#include <cuda_runtime.h>

#include <graph.hpp>
#include <details/resource.hpp>

namespace cuda {

class Graph::Node : public Resource<cudaGraphNode_t> {
 public:
  virtual ~Node() = 0;
};

inline
Graph::Node::~Node() {
}

}

#endif //CUPP_SRC_GRAPH_NODE_HPP_
