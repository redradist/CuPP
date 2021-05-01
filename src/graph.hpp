//
// Created by redra on 26.03.21.
//

#ifndef TEST0_GRAPH_HPP
#define TEST0_GRAPH_HPP

#include <memory>

#include <cuda_runtime.h>

namespace cuda {

class Graph final {
 public:
  class Node;
  class HostNode;
  class KernelNode;
  class GraphNode;

  explicit Graph(unsigned int flags = 0);
  ~Graph();

  Graph(const Graph&) = delete;
  Graph& operator=(const Graph&) = delete;

  Graph(Graph&&) = default;
  Graph& operator=(Graph&&) = default;

  std::shared_ptr<HostNode> createHostNode();
  std::shared_ptr<KernelNode> createKernelNode();
  std::shared_ptr<GraphNode> createGraphNode(Graph& graph);

  void addDependency(std::shared_ptr<Graph::Node>& leftNode,
                     std::shared_ptr<Graph::Node>& rightNode);

 private:
  friend class Stream;

  cudaGraph_t& handle();
  const cudaGraph_t& handle() const;

  cudaGraph_t graph_;
};

inline
cudaGraph_t& Graph::handle() {
  return graph_;
}

inline
const cudaGraph_t& Graph::handle() const {
  return graph_;
}

}

#endif //TEST0_GRAPH_HPP
