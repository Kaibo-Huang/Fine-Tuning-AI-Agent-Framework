#pragma once

#include <concepts>
#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "agents_framework/core/result.hpp"
#include "agents_framework/graph/graph.hpp"
#include "agents_framework/graph/node.hpp"
#include "agents_framework/graph/state.hpp"

namespace agents_framework::graph {

template <class F, class S>
concept RouterFn =
    std::invocable<F&, StateView<S>> &&
    (std::same_as<std::invoke_result_t<F&, StateView<S>>, std::string> ||
     std::same_as<std::invoke_result_t<F&, StateView<S>>, std::vector<std::string>> ||
     std::same_as<std::invoke_result_t<F&, StateView<S>>, core::Result<std::string>> ||
     std::same_as<std::invoke_result_t<F&, StateView<S>>, core::Result<std::vector<std::string>>>);

template <class S>
class GraphBuilder {
 public:
  using schema_type = S;

  GraphBuilder& add_node(std::string name, std::unique_ptr<Node> node) {
    spec_.add_node(std::move(name), std::move(node));
    return *this;
  }

  template <NodeFn<S> F>
  GraphBuilder& add_node(std::string name, F fn) {
    return add_node(std::move(name), make_fn_node<S>(std::move(fn)));
  }

  GraphBuilder& add_edge(std::string from, std::string to) {
    spec_.add_edge(std::move(from), std::move(to));
    return *this;
  }

  GraphBuilder& set_entry(std::string node) {
    return add_edge(std::string{kStart}, std::move(node));
  }

  GraphBuilder& set_finish(std::string node) {
    return add_edge(std::move(node), std::string{kEnd});
  }

  template <RouterFn<S> F>
  GraphBuilder& add_conditional_edge(std::string from, F router,
                                     std::vector<std::string> targets = {}) {
    spec_.add_router(
        std::move(from),
        [router = std::move(router)](
            const ChannelMap& state) mutable -> core::Result<std::vector<std::string>> {
          using R = std::invoke_result_t<F&, StateView<S>>;
          if constexpr (std::same_as<R, std::string>) {
            return std::vector<std::string>{router(StateView<S>(state))};
          } else if constexpr (std::same_as<R, std::vector<std::string>>) {
            return router(StateView<S>(state));
          } else if constexpr (std::same_as<R, core::Result<std::string>>) {
            AF_TRY(auto name, router(StateView<S>(state)));
            return std::vector<std::string>{std::move(name)};
          } else {
            return router(StateView<S>(state));
          }
        },
        std::move(targets));
    return *this;
  }

  [[nodiscard]] core::Result<CompiledGraph> compile() && { return std::move(spec_).compile(); }

  [[nodiscard]] std::size_t node_count() const noexcept { return spec_.node_count(); }
  [[nodiscard]] bool has_node(std::string_view name) const noexcept {
    return spec_.has_node(name);
  }
  [[nodiscard]] std::span<const Edge> edges() const noexcept { return spec_.edges(); }

 private:
  GraphSpec spec_;
};

}
