#pragma once

#include <concepts>
#include <memory>
#include <type_traits>
#include <utility>

#include "agents_framework/core/result.hpp"
#include "agents_framework/graph/channel_map.hpp"
#include "agents_framework/graph/state.hpp"

namespace agents_framework::graph {

class Node {
 public:
  virtual ~Node() = default;

  virtual core::Result<StateUpdate> run(const ChannelMap& state) = 0;
};

template <class F, class S>
concept NodeFn =
    std::invocable<F&, StateView<S>> &&
    (std::same_as<std::invoke_result_t<F&, StateView<S>>, Update<S>> ||
     std::same_as<std::invoke_result_t<F&, StateView<S>>, core::Result<Update<S>>>);

template <class S, NodeFn<S> F>
class FnNode final : public Node {
 public:
  explicit FnNode(F fn) : fn_(std::move(fn)) {}

  core::Result<StateUpdate> run(const ChannelMap& state) override {
    if constexpr (std::same_as<std::invoke_result_t<F&, StateView<S>>, Update<S>>) {
      return fn_(StateView<S>(state)).take();
    } else {
      AF_TRY(auto update, fn_(StateView<S>(state)));
      return std::move(update).take();
    }
  }

 private:
  F fn_;
};

template <class S, NodeFn<S> F>
[[nodiscard]] std::unique_ptr<Node> make_fn_node(F fn) {
  return std::make_unique<FnNode<S, F>>(std::move(fn));
}

}
