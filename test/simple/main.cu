#include <cstdio>
#include <cmath>
#include <memory.hpp>
#include <thread.cuh>
#include <graph.hpp>
#include <gsl/pointers>
#include <type_traits>

__global__
void saxpy(int n, float a, float *x, float *y)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = a*x[i] + y[i];
  }

  cuda::kernel::barrier();
}

class SaxpyKernel {
 public:
  SaxpyKernel() {
  }

  void compute(int N, float *d_x, float *d_y) {
    saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);
  }
};

template <typename T>
class DevicePointer : gsl::not_null<T> {
 public:
  static_assert(std::is_pointer<T>::value, "T should be a pointer type");
  using gsl::not_null<T>::not_null;
  template<typename U>
  DevicePointer<U*> cast() const noexcept {
    static_assert(!std::is_pointer<U>::value, "U should not be a pointer type");
    return DevicePointer<U*>{static_cast<U*>(gsl::not_null<T>::get())};
  }
  T get() const noexcept {
    return gsl::not_null<T>::get();
  }
};

int main() {
  auto devTensorA = DevicePointer<void*>{(void*)24};
  auto devTypedTensorA = devTensorA.cast<int>();
  auto rawTensorA = devTypedTensorA.get();

  int N = 1 << 20;
  auto x = std::make_unique<float[]>(N);
  auto y = std::make_unique<float[]>(N);

  auto d_x = cuda::makeUnique<float[]>(N);
  auto d_y = cuda::makeUnique<float[]>(N);
  auto d_z = cuda::makeShared<float[]>(N);

  cuda::Stream stream{cudaStreamNonBlocking};

  cuda::Graph graph{};
  cudaKernelNodeParams kernelNodeParams;
  int n = 1;
  float a;
  float *xx = new float{0};
  float *yy = new float{1};
  int blocks = 10;
  int threads = 10;
  kernelNodeParams.func = reinterpret_cast<void *>(&saxpy);
  kernelNodeParams.gridDim = dim3(blocks, 1, 1);
  kernelNodeParams.blockDim = dim3(threads, 1, 1);
  kernelNodeParams.sharedMemBytes = 0;
  void* kernelArgs[4] = { &n, &a, &xx, &yy };
  kernelNodeParams.kernelParams = kernelArgs;
  kernelNodeParams.extra = nullptr;
  auto kernelNode = graph.createKernelNode(kernelNodeParams);
  int count;
  cudaGetDeviceCount(&count);

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
