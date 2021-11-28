//
// Created by redra on 25.04.21.
//

#ifndef CUPP_SRC_GRAPH_KERNEL_NODE_HPP_
#define CUPP_SRC_GRAPH_KERNEL_NODE_HPP_

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

 protected:
  friend class Graph;

  struct this_is_private {
    explicit this_is_private() = default;
  };
};

}

#endif //CUPP_SRC_GRAPH_KERNEL_NODE_HPP_
