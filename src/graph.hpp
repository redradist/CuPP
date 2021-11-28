//
// Created by redra on 26.03.21.
//

#ifndef TEST0_GRAPH_HPP
#define TEST0_GRAPH_HPP

#include <memory>

#include <cuda_runtime.h>

#include "details/resource.hpp"

namespace cuda {

class Graph final : public Handle<cudaGraph_t> {
 public:
  class Node;
  class HostNode;
  class KernelNode;
  class GraphNode;

  explicit Graph(unsigned int flags = 0);
  ~Graph();

  [[nodiscard]]std::shared_ptr<HostNode> createHostNode(cudaHostNodeParams& nodeParams);
  [[nodiscard]]std::shared_ptr<KernelNode> createKernelNode(cudaKernelNodeParams& nodeParams);
  [[nodiscard]]std::shared_ptr<GraphNode> createGraphNode(Graph& graph);

  void addDependency(std::shared_ptr<Graph::Node>& leftNode,
                     std::shared_ptr<Graph::Node>& rightNode);
};

}

#endif //TEST0_GRAPH_HPP
