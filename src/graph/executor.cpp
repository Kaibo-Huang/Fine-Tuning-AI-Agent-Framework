#include "agents_framework/graph/executor.hpp"

#include <algorithm>
#include <exception>
#include <latch>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "agents_framework/core/clock.hpp"
#include "agents_framework/core/id.hpp"
#include "agents_framework/core/rng.hpp"

namespace agents_framework::graph {

namespace {

core::Error annotate(core::Error error, std::string_view node, std::uint64_t step) {
  std::string where = "node '" + std::string{node} + "' (step " + std::to_string(step) + ")";
  if (error.context.empty()) {
    error.context = std::move(where);
  } else {
    error.context += "; " + where;
  }
  return error;
}

std::string generate_run_id() {
  core::SystemClock clock;
  std::random_device device;
  core::Rng rng{(static_cast<std::uint64_t>(device()) << 32) ^ device()};
  return core::Ulid::generate(clock, rng).to_string();
}

}

Executor::Executor(ExecutorOptions options) : options_(options) {
  if (options_.workers > 1) {
    pool_ = std::make_unique<core::ThreadPool>(options_.workers);
  }
}

core::Result<RunStats> Executor::run(CompiledGraph& graph, ChannelMap& state,
                                     RunOptions options) {
  if (options.run_id.empty() && (options.checkpointer != nullptr || options.events)) {
    options.run_id = generate_run_id();
  }
  std::vector<std::size_t> active(graph.entry_nodes().begin(), graph.entry_nodes().end());
  return run_loop(graph, state, options, std::move(active), 0, false);
}

core::Result<RunStats> Executor::resume(CompiledGraph& graph, ChannelMap& state,
                                        const Checkpoint& checkpoint, RunOptions options) {
  if (checkpoint.next_nodes.empty()) {
    return core::fail(core::ErrorCode::Invalid, "checkpoint has no pending nodes to resume",
                      "run '" + checkpoint.run_id + "' completed at step " +
                          std::to_string(checkpoint.step));
  }
  std::vector<std::size_t> active;
  active.reserve(checkpoint.next_nodes.size());
  for (const std::string& name : checkpoint.next_nodes) {
    const auto index = graph.find_node(name);
    if (!index) {
      return core::fail(core::ErrorCode::NotFound, "checkpoint references an unknown node", name);
    }
    active.push_back(*index);
  }
  std::sort(active.begin(), active.end());
  active.erase(std::unique(active.begin(), active.end()), active.end());

  options.run_id = checkpoint.run_id;
  return run_loop(graph, state, options, std::move(active), checkpoint.step, true);
}

core::Result<RunStats> Executor::run_loop(CompiledGraph& graph, ChannelMap& state,
                                          const RunOptions& options,
                                          std::vector<std::size_t> active,
                                          std::uint64_t start_step, bool resuming) {
  RunStats stats;

  const auto publish = [&options](ExecEvent event) {
    if (options.events) options.events->publish(event);
  };
  const auto names_of = [&graph](const std::vector<std::size_t>& indices) {
    std::vector<std::string> names;
    names.reserve(indices.size());
    for (const std::size_t index : indices) {
      names.emplace_back(graph.node_name(index));
    }
    return names;
  };
  const auto save_checkpoint = [&](std::uint64_t at_step, const std::vector<std::size_t>& next,
                                   CheckpointStatus status) -> core::Result<void> {
    if (options.checkpointer == nullptr) return {};
    const Checkpoint checkpoint{options.run_id, at_step, state.serialize(), names_of(next),
                                status};
    auto saved = options.checkpointer->save(checkpoint);
    if (!saved) {
      core::Error error = std::move(saved).error();
      std::string where = "saving checkpoint (step " + std::to_string(at_step) + ")";
      error.context = error.context.empty() ? std::move(where) : error.context + "; " + where;
      publish(RunFinished{options.run_id, at_step, false, error.to_string()});
      return std::unexpected(std::move(error));
    }
    return {};
  };

  std::uint64_t step = start_step;
  publish(RunStarted{options.run_id, step});
  if (!resuming) {
    AF_TRY_VOID(save_checkpoint(step, active,
                                active.empty() ? CheckpointStatus::Completed
                                               : CheckpointStatus::Running));
  }

  bool approved = resuming;
  while (!active.empty()) {
    if (!approved && !options.interrupt_before.empty()) {
      const bool interrupted =
          std::any_of(active.begin(), active.end(), [&](std::size_t index) {
            return std::find(options.interrupt_before.begin(), options.interrupt_before.end(),
                             graph.node_name(index)) != options.interrupt_before.end();
          });
      if (interrupted) {
        stats.status = RunStatus::Interrupted;
        stats.pending_nodes = names_of(active);
        AF_TRY_VOID(save_checkpoint(step, active, CheckpointStatus::Interrupted));
        publish(RunInterrupted{options.run_id, step, stats.pending_nodes});
        return stats;
      }
    }
    approved = false;

    if (stats.steps >= options.max_steps) {
      core::Error error{core::ErrorCode::Cancelled, "step budget exhausted",
                        "after " + std::to_string(stats.steps) + " super-steps"};
      publish(RunFinished{options.run_id, step, false, error.to_string()});
      return std::unexpected(std::move(error));
    }
    ++stats.steps;
    ++step;

    publish(StepStarted{options.run_id, step, names_of(active)});
    for (const std::size_t index : active) {
      publish(NodeStarted{options.run_id, step, std::string{graph.node_name(index)}});
    }

    std::vector<core::Result<StateUpdate>> results;
    if (pool_ && active.size() > 1) {
      results.resize(active.size());
      std::vector<std::exception_ptr> exceptions(active.size());
      std::latch done{static_cast<std::ptrdiff_t>(active.size())};
      for (std::size_t i = 0; i < active.size(); ++i) {
        pool_->submit([&graph, &state, &results, &exceptions, &done, &active, i] {
          try {
            results[i] = graph.node(active[i]).run(state);
          } catch (...) {
            exceptions[i] = std::current_exception();
          }
          done.count_down();
        });
      }
      done.wait();
      for (const std::exception_ptr& thrown : exceptions) {
        if (thrown) std::rethrow_exception(thrown);
      }
    } else {
      results.reserve(active.size());
      for (const std::size_t index : active) {
        results.push_back(graph.node(index).run(state));
      }
    }
    stats.node_runs += active.size();

    for (std::size_t i = 0; i < active.size(); ++i) {
      publish(NodeFinished{options.run_id, step, std::string{graph.node_name(active[i])},
                           results[i].has_value(),
                           results[i] ? std::string{} : results[i].error().to_string()});
    }
    for (std::size_t i = 0; i < active.size(); ++i) {
      if (!results[i]) {
        core::Error error =
            annotate(std::move(results[i]).error(), graph.node_name(active[i]), step);
        publish(RunFinished{options.run_id, step, false, error.to_string()});
        return std::unexpected(std::move(error));
      }
    }
    for (auto& result : results) {
      state.apply(std::move(*result));
    }
    if (options.events && options.events->has_subscribers()) {
      publish(StateUpdated{options.run_id, step, state.to_json()});
    }

    std::vector<std::size_t> next;
    for (const std::size_t index : active) {
      const auto targets = graph.static_targets(index);
      next.insert(next.end(), targets.begin(), targets.end());
      if (graph.has_router(index)) {
        auto routed = graph.route(index, state);
        if (!routed) {
          core::Error error = annotate(std::move(routed).error(), graph.node_name(index), step);
          publish(RunFinished{options.run_id, step, false, error.to_string()});
          return std::unexpected(std::move(error));
        }
        next.insert(next.end(), routed->begin(), routed->end());
      }
    }
    std::sort(next.begin(), next.end());
    next.erase(std::unique(next.begin(), next.end()), next.end());
    active = std::move(next);

    AF_TRY_VOID(save_checkpoint(step, active,
                                active.empty() ? CheckpointStatus::Completed
                                               : CheckpointStatus::Running));
  }

  publish(RunFinished{options.run_id, step, true, {}});
  return stats;
}

}
