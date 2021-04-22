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

 protected:
  TResource& handle();
  const TResource& handle() const;

  template <typename TExternalResource>
  static TExternalResource& handleFrom(Resource<TExternalResource>& res);
  template <typename TExternalResource>
  static const TExternalResource& handleFrom(const Resource<TExternalResource>& res);

  TResource resource_;
};

template <typename TResource>
TResource& Resource<TResource>::handle() {
  return resource_;
}

template <typename TResource>
const TResource& Resource<TResource>::handle() const {
  return resource_;
}

template <typename TResource>
template <typename TExternalResource>
TExternalResource& Resource<TResource>::handleFrom(Resource<TExternalResource>& res) {
  return res.resource_;
}

template <typename TResource>
template <typename TExternalResource>
const TExternalResource& Resource<TResource>::handleFrom(const Resource<TExternalResource>& res) {
  return res.resource_;
}

}

#endif //CUDAPP_RESOURCE_HPP
