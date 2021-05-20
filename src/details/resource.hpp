//
// Created by redra on 22.04.21.
//

#ifndef CUDAPP_RESOURCE_HPP
#define CUDAPP_RESOURCE_HPP

namespace cuda {

template <typename TResource>
class Resource {
 public:
  template <typename TExternalResource>
  friend class Resource;

  Resource() = default;
  Resource(const Resource&) = delete;
  Resource& operator=(const Resource&) = delete;
  Resource(Resource&&) = delete;
  Resource& operator=(Resource&&) = delete;

  template<typename ... TArgs>
  void call(cudaError_t(*cudaFunction)(TResource, TArgs...), TArgs&&... args) {
    throwIfCudaError(cudaFunction(this->handle_, std::forward<TArgs>(args)...));
  }

 protected:
  template <typename TExternalResource>
  static TExternalResource& handleFrom(Resource<TExternalResource>& res);
  template <typename TExternalResource>
  static const TExternalResource& handleFrom(const Resource<TExternalResource>& res);

  TResource handle_;
};

template <typename TResource>
template <typename TExternalResource>
TExternalResource& Resource<TResource>::handleFrom(Resource<TExternalResource>& res) {
  return res.handle_;
}

template <typename TResource>
template <typename TExternalResource>
const TExternalResource& Resource<TResource>::handleFrom(const Resource<TExternalResource>& res) {
  return res.handle_;
}

}

#endif //CUDAPP_RESOURCE_HPP
