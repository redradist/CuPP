//
// Created by redra on 14.04.21.
//

#include "stream.cuh"

namespace cuda {

Stream::Stream() {
  cudaStreamCreate(&stream_);
}

Stream::~Stream() {
  cudaStreamDestroy(stream_);
}

void Stream::synchronize() {
  cudaStreamSynchronize(stream_);
}

}