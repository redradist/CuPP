//
// Created by redra on 25.04.21.
//

#ifndef CUDAPP_SRC_GRAPH_KERNEL_NODE_HPP_
#define CUDAPP_SRC_GRAPH_KERNEL_NODE_HPP_

#include "graph.hpp"
#include "graph_node.hpp"

namespace cuda {

class Graph::KernelNode : public Graph::Node {
 protected:
  struct this_is_private;

 public:
  explicit KernelNode(const this_is_private &,
                      cudaGraph_t graph,
                      cudaKernelNodeParams& nodeParams);
  ~KernelNode() override;

  KernelNode(const KernelNode&) = delete;
  KernelNode& operator=(const KernelNode&) = delete;

  KernelNode(KernelNode&&) = default;
  KernelNode& operator=(KernelNode&&) = default;

 protected:
  friend class Graph;

  struct this_is_private {
    explicit this_is_private() = default;
  };

  cudaGraphNode_t graph_node_;
};

}

#endif //CUDAPP_SRC_GRAPH_KERNEL_NODE_HPP_
