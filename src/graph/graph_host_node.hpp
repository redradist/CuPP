//
// Created by redra on 25.04.21.
//

#ifndef CUPP_SRC_GRAPH_HOST_NODE_HPP_
#define CUPP_SRC_GRAPH_HOST_NODE_HPP_

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

 protected:
  friend class Graph;

  struct this_is_private {
    explicit this_is_private() = default;
  };
};

}

#endif //CUPP_SRC_GRAPH_HOST_NODE_HPP_
