//
// Created by redra on 25.04.21.
//

#ifndef CUDAPP_SRC_GRAPH_HOST_NODE_HPP_
#define CUDAPP_SRC_GRAPH_HOST_NODE_HPP_

#include "graph.hpp"
#include "graph_node.hpp"

namespace cuda {

class Graph::HostNode : public Graph::Node {
 protected:
  struct this_is_private;

 public:
  explicit HostNode(const this_is_private &,
                    cudaGraph_t graph,
                    cudaHostNodeParams& nodeParams);
  ~HostNode() override;

  HostNode(const HostNode&) = delete;
  HostNode& operator=(const HostNode&) = delete;

  HostNode(HostNode&&) = default;
  HostNode& operator=(HostNode&&) = default;

 protected:
  friend class Graph;

  struct this_is_private {
    explicit this_is_private() = default;
  };

  cudaGraphNode_t graph_node_;
};

}

#endif //CUDAPP_SRC_GRAPH_HOST_NODE_HPP_
