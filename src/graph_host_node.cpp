//
// Created by redra on 25.04.21.
//

#include "graph_host_node.hpp"
#include "exceptions/cuda_exception.hpp"

namespace cuda {

Graph::HostNode::HostNode(const this_is_private &,
                          cudaGraph_t graph,
                          cudaHostNodeParams& nodeParams) {
  THROW_IF_CUDA_ERROR(cudaGraphAddHostNode(&graph_node_, graph, nullptr, 0, &nodeParams));
}

Graph::HostNode::~HostNode() {
  cudaGraphDestroyNode(graph_node_);
}

}