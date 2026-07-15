#include "agents_framework/graph/executor.hpp"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace agents_framework::graph {

namespace {

core::Error annotate(core::Error error, std::string_view node, std::size_t step) {
  std::string where = "node '" + std::string{node} + "' (step " + std::to_string(step) + ")";
  if (error.context.empty()) {
    error.context = std::move(where);
  } else {
    error.context += "; " + where;
  }
  return error;
}

}

core::Result<RunStats> Executor::run(CompiledGraph& graph, ChannelMap& state,
                                     RunOptions options) {
  RunStats stats;
  std::vector<std::size_t> active(graph.entry_nodes().begin(), graph.entry_nodes().end());

  while (!active.empty()) {
    if (stats.steps >= options.max_steps) {
      return core::fail(core::ErrorCode::Cancelled, "step budget exhausted",
                        "after " + std::to_string(stats.steps) + " super-steps");
    }
    ++stats.steps;

    std::vector<core::Result<StateUpdate>> results;
    results.reserve(active.size());
    for (const std::size_t index : active) {
      results.push_back(graph.node(index).run(state));
    }
    stats.node_runs += active.size();

    for (std::size_t i = 0; i < active.size(); ++i) {
      if (!results[i]) {
        return std::unexpected(
            annotate(std::move(results[i]).error(), graph.node_name(active[i]), stats.steps));
      }
    }
    for (auto& result : results) {
      state.apply(std::move(*result));
    }

    std::vector<std::size_t> next;
    for (const std::size_t index : active) {
      const auto targets = graph.static_targets(index);
      next.insert(next.end(), targets.begin(), targets.end());
      if (graph.has_router(index)) {
        auto routed = graph.route(index, state);
        if (!routed) {
          return std::unexpected(
              annotate(std::move(routed).error(), graph.node_name(index), stats.steps));
        }
        next.insert(next.end(), routed->begin(), routed->end());
      }
    }
    std::sort(next.begin(), next.end());
    next.erase(std::unique(next.begin(), next.end()), next.end());
    active = std::move(next);
  }

  return stats;
}

}
