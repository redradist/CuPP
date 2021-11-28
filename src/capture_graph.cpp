//
// Created by redra on 7/3/21.
//

#include "capture_graph.hpp"
#include "stream.hpp"

namespace cuda {

CaptureGraph::CaptureGraph(Stream& stream) {
  throwIfCudaError(cudaStreamEndCapture(handleFrom(stream), &handle_));
}

CaptureGraph::~CaptureGraph() {
  cudaGraphDestroy(handle_);
}

}
