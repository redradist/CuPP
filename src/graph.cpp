//
// Created by redra on 22.04.21.
//

#include "graph.hpp"
#include "graph_host_node.hpp"
#include "graph_kernel_node.hpp"
#include "exceptions/cuda_exception.hpp"

namespace cuda {

Graph::Graph(unsigned int flags) {
  THROW_IF_CUDA_ERROR(cudaGraphCreate(&graph_, flags));
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

Graph::~Graph() {
  cudaGraphDestroy(graph_);
}

}
