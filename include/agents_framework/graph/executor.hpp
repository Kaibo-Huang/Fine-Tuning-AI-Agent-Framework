#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "agents_framework/core/result.hpp"
#include "agents_framework/core/thread_pool.hpp"
#include "agents_framework/graph/channel_map.hpp"
#include "agents_framework/graph/checkpoint.hpp"
#include "agents_framework/graph/graph.hpp"
#include "agents_framework/graph/state.hpp"

namespace agents_framework::graph {

struct ExecutorOptions {
  std::size_t workers{1};
};

struct RunOptions {
  std::size_t max_steps{100};
  std::string run_id;
  Checkpointer* checkpointer{nullptr};
  std::vector<std::string> interrupt_before;
};

enum class RunStatus { Completed, Interrupted };

struct RunStats {
  std::size_t steps{0};
  std::size_t node_runs{0};
  RunStatus status{RunStatus::Completed};
  std::vector<std::string> pending_nodes;
};

class Executor {
 public:
  Executor() = default;
  explicit Executor(ExecutorOptions options);

  core::Result<RunStats> run(CompiledGraph& graph, ChannelMap& state, RunOptions options = {});

  core::Result<RunStats> resume(CompiledGraph& graph, ChannelMap& state,
                                const Checkpoint& checkpoint, RunOptions options = {});

  template <class S>
  core::Result<RunStats> run(CompiledGraph& graph, State<S>& state, RunOptions options = {}) {
    return run(graph, state.map(), std::move(options));
  }

  template <class S>
  core::Result<RunStats> resume(CompiledGraph& graph, State<S>& state,
                                const Checkpoint& checkpoint, RunOptions options = {}) {
    return resume(graph, state.map(), checkpoint, std::move(options));
  }

  [[nodiscard]] const ExecutorOptions& options() const noexcept { return options_; }

 private:
  core::Result<RunStats> run_loop(CompiledGraph& graph, ChannelMap& state,
                                  const RunOptions& options, std::vector<std::size_t> active,
                                  std::uint64_t start_step, bool resuming);

  ExecutorOptions options_;
  std::unique_ptr<core::ThreadPool> pool_;
};

}
