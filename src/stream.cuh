//
// Created by redra on 14.04.21.
//

#ifndef CUDAPP_STREAM_CUH
#define CUDAPP_STREAM_CUH

namespace cuda {

class Stream {
 public:
  class Context;

  Stream();
  ~Stream();

  Context createContext();

 protected:
  cudaStream_t stream;
};

}

#endif //CUDAPP_STREAM_CUH
