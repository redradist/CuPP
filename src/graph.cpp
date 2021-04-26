//
// Created by redra on 22.04.21.
//

#include "graph.hpp"
#include "graph_host_node.hpp"
#include "graph_kernel_node.hpp"
#include "exceptions/cuda_exception.hpp"

namespace cuda {

Graph::Graph(unsigned int flags) {
  throwIfCudaError(cudaGraphCreate(&graph_, flags));
}

Graph::~Graph() {
  cudaGraphDestroy(graph_);
}

std::shared_ptr<Graph::HostNode>
Graph::createHostNode() {
  cudaHostNodeParams nodeParams;
  return std::make_shared<HostNode>(
      HostNode::this_is_private{},
      graph_, nodeParams
  );
}

std::shared_ptr<Graph::KernelNode>
Graph::createKernelNode() {
  cudaKernelNodeParams nodeParams;
  return std::make_shared<KernelNode>(
      KernelNode::this_is_private{},
      graph_, nodeParams
  );
}

void Graph::addDependency(std::shared_ptr<Graph::Node>& leftNode,
                          std::shared_ptr<Graph::Node>& rightNode) {
  throwIfCudaError(cudaGraphAddDependencies(graph_, &leftNode->node(), &rightNode->node(), 1));
}

}
