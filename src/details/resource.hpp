//
// Created by redra on 22.04.21.
//

#ifndef CUPP_RESOURCE_HPP
#define CUPP_RESOURCE_HPP

#include <driver_types.h>

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
  void call(cudaError_t(*cudaFunction)(TResource, TArgs...), TArgs&&... args);

 protected:
  template <typename TExternalResource>
  static TExternalResource& handleFrom(Resource<TExternalResource>& res);
  template <typename TExternalResource>
  static const TExternalResource& handleFrom(const Resource<TExternalResource>& res);

  TResource handle_;
};

template<typename Arg>
constexpr Arg&& tryHandleFrom(typename std::remove_reference<Arg>::type& arg) noexcept {
  return static_cast<Arg&&>(arg);
}

template<typename Arg>
constexpr Arg&& tryHandleFrom(typename std::remove_reference<Arg>::type&& arg) noexcept {
  static_assert(
    !std::is_lvalue_reference<Arg>::value,
    "template argument substituting _Tp is an lvalue reference type");
  return static_cast<Arg&&>(arg);
}

template <typename TArg>
inline TArg& tryHandleFrom(TArg&& res) {

}

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

template<typename TResource>
template<typename ... TArgs>
void Resource<TResource>::call(cudaError_t(*cudaFunction)(TResource, TArgs...), TArgs&&... args) {
  throwIfCudaError(cudaFunction(this->handle_, tryHandleFrom<TArgs>(args)...));
}

}

#endif //CUPP_RESOURCE_HPP
