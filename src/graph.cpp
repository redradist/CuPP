//
// Created by redra on 22.04.21.
//

#include "graph.hpp"
#include "graph/graph_host_node.hpp"
#include "graph/graph_kernel_node.hpp"
#include "graph/graph_graph_node.hpp"
#include "exceptions/cuda_exception.hpp"

namespace cuda {

Graph::Graph(unsigned int flags) {
  throwIfCudaError(cudaGraphCreate(&handle_, flags));
}

Graph::~Graph() {
  cudaGraphDestroy(handle_);
}

std::shared_ptr<Graph::HostNode>
Graph::createHostNode(cudaHostNodeParams& nodeParams) {
  return std::make_shared<HostNode>(
      HostNode::this_is_private{},
      handle_, nodeParams
  );
}

std::shared_ptr<Graph::KernelNode>
Graph::createKernelNode(cudaKernelNodeParams& nodeParams) {
  return std::make_shared<KernelNode>(
      KernelNode::this_is_private{},
      handle_, nodeParams
  );
}

std::shared_ptr<Graph::GraphNode>
Graph::createGraphNode(Graph& graph) {
  return std::make_shared<GraphNode>(
      GraphNode::this_is_private{},
      handle_, graph
  );
}

void Graph::addDependency(std::shared_ptr<Graph::Node>& leftNode,
                          std::shared_ptr<Graph::Node>& rightNode) {
  throwIfCudaError(cudaGraphAddDependencies(handle_, &handleFrom(*leftNode), &handleFrom(*rightNode), 1));
}

}
