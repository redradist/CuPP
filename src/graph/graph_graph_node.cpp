//
// Created by redra on 01.05.21.
//

#include "graph_graph_node.hpp"

#include "graph_host_node.hpp"
#include "exceptions/cuda_exception.hpp"

namespace cuda {

Graph::GraphNode::GraphNode(const this_is_private &,
                            cudaGraph_t graph,
                            Graph& graph_) {
  throwIfCudaError(cudaGraphAddChildGraphNode(&handle_, graph, nullptr, 0, handleFrom(graph_)));
}

Graph::GraphNode::~GraphNode() {
  cudaGraphDestroyNode(handle_);
}

}
