//
// Created by redra on 22.04.21.
//

#include "graph.hpp"
#include "exceptions/cuda_exception.hpp"

namespace cuda {

Graph::Graph(unsigned int flags) {
  THROW_IF_CUDA_ERROR(cudaGraphCreate(&graph_, flags));
}

Graph::~Graph() {
  THROW_IF_CUDA_ERROR(cudaGraphDestroy(graph_));
}

}
