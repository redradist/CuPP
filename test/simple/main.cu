#include <cstdio>
#include <cmath>
#include <memory.hpp>
#include <thread.cuh>

__global__
void saxpy(int n, float a, float *x, float *y)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = a*x[i] + y[i];
  }

  cuda::barrier();
}

class SaxpyKernel {
 public:
  SaxpyKernel() {
  }

  void compute(int N, float *d_x, float *d_y) {
    saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);
  }
};

int main() {
  int N = 1 << 20;
  auto x = std::make_unique<float[]>(N);
  auto y = std::make_unique<float[]>(N);

  auto d_x = cuda::makeUnique<float[]>(N);
  auto d_y = cuda::makeUnique<float[]>(N);

  cudaStream_t streamForGraph;
  cudaError_t err = cudaStreamCreateWithFlags(&streamForGraph, cudaStreamNonBlocking);

  cudaGraph_t graph2;
  err = cudaGraphCreate(&graph2, 0);
  int count;
  err = cudaGetDeviceCount(&count);

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  cuda::memcpy(d_x, x, N*sizeof(float));
  cuda::memcpy(d_y, y, N*sizeof(float));

  // Perform SAXPY on 1M elements
  SaxpyKernel kr;
  kr.compute(N, d_x.get(), d_y.get());

  cuda::memcpy(y, d_y, N*sizeof(float));

  float maxError = 0.0f;
  for (int i = 0; i < N; i++) {
    maxError = max(maxError, abs(y[i]-4.0f));
  }

  printf("Max error: %f\n", maxError);
}
