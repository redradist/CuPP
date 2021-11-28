//
// Created by redra on 22.04.21.
//

#ifndef CUPP_RESOURCE_HPP
#define CUPP_RESOURCE_HPP

#include <driver_types.h>

namespace cuda {

template <typename THandle>
class Handle {
 public:
  template <typename TExternalHandle>
  friend class Handle;

  Handle() = default;
  Handle(const Handle&) = delete;
  Handle& operator=(const Handle&) = delete;
  Handle(Handle&&) = delete;
  Handle& operator=(Handle&&) = delete;

  template<typename ... TArgs>
  void call(cudaError_t(*cudaFunction)(THandle, TArgs...), TArgs&&... args);

 protected:
  template <typename TExternalHandle>
  static TExternalHandle& handleFrom(Handle<TExternalHandle>& res);
  template <typename TExternalHandle>
  static const TExternalHandle& handleFrom(const Handle<TExternalHandle>& res);

  THandle handle_;
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

template <typename THandle>
template <typename TExternalHandle>
TExternalHandle& Handle<THandle>::handleFrom(Handle<TExternalHandle>& res) {
  return res.handle_;
}

template <typename THandle>
template <typename TExternalHandle>
const TExternalHandle& Handle<THandle>::handleFrom(const Handle<TExternalHandle>& res) {
  return res.handle_;
}

template<typename THandle>
template<typename ... TArgs>
void Handle<THandle>::call(cudaError_t(*cudaFunction)(THandle, TArgs...), TArgs&&... args) {
  throwIfCudaError(cudaFunction(this->handle_, tryHandleFrom<TArgs>(args)...));
}

}

#endif //CUPP_RESOURCE_HPP
