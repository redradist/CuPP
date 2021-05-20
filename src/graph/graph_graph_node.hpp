//
// Created by redra on 01.05.21.
//

#ifndef CUPP_SRC_GRAPH_GRAPH_NODE_HPP_
#define CUPP_SRC_GRAPH_GRAPH_NODE_HPP_

#include "graph.hpp"
#include "graph_node.hpp"

namespace cuda {

class Graph::GraphNode : public Graph::Node {
 protected:
  struct this_is_private;

 public:
  explicit GraphNode(const this_is_private &,
                     cudaGraph_t graph,
                     Graph& graph_);
  ~GraphNode() override;

 protected:
  friend class Graph;

  struct this_is_private {
    explicit this_is_private() = default;
  };
};

}

#endif //CUPP_SRC_GRAPH_GRAPH_NODE_HPP_
