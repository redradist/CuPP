#include <cstdio>
#include <cmath>
#include <memory.hpp>
#include <thread.cuh>
#include <cudnn.h>
#include <vector>
#include <iostream>

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

void convolutionRun() {
  cudnnHandle_t handle_;
  // Create a cuDNN handle:
  (cudnnCreate(&handle_));

  // Create your tensor descriptors:
  cudnnTensorDescriptor_t cudnnIdesc;
  cudnnFilterDescriptor_t cudnnFdesc;
  cudnnTensorDescriptor_t cudnnOdesc;
  cudnnConvolutionDescriptor_t cudnnConvDesc;
  ( cudnnCreateTensorDescriptor( &cudnnIdesc ));
  ( cudnnCreateFilterDescriptor( &cudnnFdesc ));
  ( cudnnCreateTensorDescriptor( &cudnnOdesc ));
  ( cudnnCreateConvolutionDescriptor( &cudnnConvDesc ));

  // Set NCHW tensor dimensions, not necessarily as multiples of eight (only the input tensor is shown here):
  int dimA[] = {1, 7, 32, 32};
  int strideA[] = {7168, 1024, 32, 1};

  size_t convDim = 2;
  int padA[] = { 1, 1 };
  int convstrideA[] = { 1, 1 };
  int dilationA[] = { 1, 1 };

  ( cudnnSetTensorNdDescriptor(cudnnIdesc, CUDNN_DATA_FLOAT,
                                            convDim+2, dimA, strideA) );

  // Allocate and initialize tensors (again, only the input tensor is shown):
  float *alpha;
  float *devPtrI;
  float *devPtrF;
  float *devPtrO;
  float *beta;
  size_t insize = 10000;
  ( cudaMalloc((void**)&(alpha), (insize) * sizeof(alpha[0]) ));
  ( cudaMalloc((void**)&(devPtrI), (insize) * sizeof(devPtrI[0]) ));
  ( cudaMalloc((void**)&(devPtrF), (insize) * sizeof(devPtrF[0]) ));
  ( cudaMalloc((void**)&(beta), (insize) * sizeof(beta[0]) ));
  ( cudaMalloc((void**)&(devPtrO), (insize) * sizeof(devPtrO[0]) ));

  // No host memory prepared !

  // Set the compute data type (below as CUDNN_DATA_FLOAT):
  ( cudnnSetConvolutionNdDescriptor(cudnnConvDesc, convDim, padA, convstrideA, dilationA, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT) );

  // Set the math type to allow cuDNN to use Tensor Cores:
  ( cudnnSetConvolutionMathType(cudnnConvDesc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION) );

  // Choose a supported algorithm:
  float *workSpace;
  size_t workSpaceSize;
  cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

  // Allocate your workspace:
  ( cudnnGetConvolutionForwardWorkspaceSize(handle_, cudnnIdesc,
                                                         cudnnFdesc, cudnnConvDesc,
                                                         cudnnOdesc, algo, &workSpaceSize) );

  if (workSpaceSize > 0) {
    cudaMalloc(&workSpace, workSpaceSize);
  }

  // Invoke the convolution:
  auto err = ( cudnnConvolutionForward(handle_, (void*)(&alpha), cudnnIdesc, devPtrI,
                                         cudnnFdesc, devPtrF, cudnnConvDesc, algo,
                                         workSpace, workSpaceSize, (void*)(&beta),
                                         cudnnOdesc, devPtrO) );
}

int main() {
  cudaGraph_t graph;
  cudaGraphExec_t instance;
  cudaStream_t stream;
  cudaError_t err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  cudnnHandle_t cudnn;
  cudnnCreate(&cudnn);

  convolutionRun();

  int N = 1 << 20;
  auto x = std::make_unique<float[]>(N);
  auto y = std::make_unique<float[]>(N);

  auto d_x = cuda::makeUnique<float[]>(N);
  auto d_y = cuda::makeUnique<float[]>(N);

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  cuda::memcpy(d_x, x, N*sizeof(float));
  cuda::memcpy(d_y, y, N*sizeof(float));

  cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

  SaxpyKernel().compute(N, d_x.get(), d_y.get());

  cudaStreamEndCapture(stream, &graph);
  cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);
  cudaGraphLaunch(instance, stream);
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);
}
