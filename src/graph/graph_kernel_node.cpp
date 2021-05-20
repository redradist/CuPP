//
// Created by redra on 25.04.21.
//

#include "graph_host_node.hpp"
#include "graph_kernel_node.hpp"
#include "exceptions/cuda_exception.hpp"

namespace cuda {

Graph::KernelNode::KernelNode(const this_is_private &,
                              cudaGraph_t graph,
                              cudaKernelNodeParams& nodeParams) {
  throwIfCudaError(cudaGraphAddKernelNode(&handle_, graph, nullptr, 0, &nodeParams));
}

Graph::KernelNode::~KernelNode() {
  cudaGraphDestroyNode(handle_);
}

}
