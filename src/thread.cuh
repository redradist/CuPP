//
// Created by redra on 25.03.21.
//

#ifndef TEST0_THREADS_HPP
#define TEST0_THREADS_HPP

namespace cuda {

__device__ void barrier() {
  __syncthreads();
}

}

#endif //TEST0_THREADS_HPP
