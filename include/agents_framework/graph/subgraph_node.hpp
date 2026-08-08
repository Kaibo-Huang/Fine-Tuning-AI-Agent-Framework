#pragma once

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "agents_framework/core/result.hpp"
#include "agents_framework/graph/executor.hpp"
#include "agents_framework/graph/graph.hpp"
#include "agents_framework/graph/node.hpp"
#include "agents_framework/graph/state.hpp"

namespace agents_framework::graph {

template <class ParentS, class ChildS>
class SubgraphNode final : public Node {
 public:
  using ToChild = std::function<core::Result<State<ChildS>>(StateView<ParentS>)>;
  using FromChild = std::function<core::Result<Update<ParentS>>(const State<ChildS>&)>;

  SubgraphNode(CompiledGraph graph, ToChild to_child, FromChild from_child,
               ExecutorOptions executor_options = {}, RunOptions run_options = {})
      : graph_(std::move(graph)), to_child_(std::move(to_child)),
        from_child_(std::move(from_child)), executor_(executor_options),
        run_options_(std::move(run_options)) {
    if (!to_child_ || !from_child_) {
      throw std::invalid_argument("SubgraphNode requires both state mappers");
    }
  }

  core::Result<StateUpdate> run(const ChannelMap& state) override {
    AF_TRY(auto child, to_child_(StateView<ParentS>(state)));
    AF_TRY(const auto stats, executor_.run(graph_, child, run_options_));
    if (stats.status != RunStatus::Completed) {
      return core::fail(core::ErrorCode::Invalid,
                        "subgraph was interrupted; interrupts must be handled by the parent run");
    }
    AF_TRY(auto update, from_child_(child));
    return std::move(update).take();
  }

 private:
  CompiledGraph graph_;
  ToChild to_child_;
  FromChild from_child_;
  Executor executor_;
  RunOptions run_options_;
};

template <class ParentS, class ChildS>
[[nodiscard]] std::unique_ptr<Node> make_subgraph_node(
    CompiledGraph graph, typename SubgraphNode<ParentS, ChildS>::ToChild to_child,
    typename SubgraphNode<ParentS, ChildS>::FromChild from_child,
    ExecutorOptions executor_options = {}, RunOptions run_options = {}) {
  return std::make_unique<SubgraphNode<ParentS, ChildS>>(
      std::move(graph), std::move(to_child), std::move(from_child), executor_options,
      std::move(run_options));
}

}
