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
  throwIfCudaError(cudaGraphAddChildGraphNode(&graph_node_, graph, nullptr, 0, graph_.graph_));
}

Graph::GraphNode::~GraphNode() {
  cudaGraphDestroyNode(graph_node_);
}

}
