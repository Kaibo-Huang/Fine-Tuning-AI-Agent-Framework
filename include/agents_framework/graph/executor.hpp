#pragma once

#include <cstddef>
#include <memory>

#include "agents_framework/core/result.hpp"
#include "agents_framework/core/thread_pool.hpp"
#include "agents_framework/graph/channel_map.hpp"
#include "agents_framework/graph/graph.hpp"
#include "agents_framework/graph/state.hpp"

namespace agents_framework::graph {

struct ExecutorOptions {
  std::size_t workers{1};
};

struct RunOptions {
  std::size_t max_steps{100};
};

struct RunStats {
  std::size_t steps{0};
  std::size_t node_runs{0};
};

class Executor {
 public:
  Executor() = default;
  explicit Executor(ExecutorOptions options);

  core::Result<RunStats> run(CompiledGraph& graph, ChannelMap& state, RunOptions options = {});

  template <class S>
  core::Result<RunStats> run(CompiledGraph& graph, State<S>& state, RunOptions options = {}) {
    return run(graph, state.map(), options);
  }

  [[nodiscard]] const ExecutorOptions& options() const noexcept { return options_; }

 private:
  ExecutorOptions options_;
  std::unique_ptr<core::ThreadPool> pool_;
};

}
