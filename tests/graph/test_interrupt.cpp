#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "agents_framework/core/result.hpp"
#include "agents_framework/graph/builder.hpp"
#include "agents_framework/graph/checkpoint.hpp"
#include "agents_framework/graph/executor.hpp"
#include "agents_framework/store/checkpoint_store.hpp"
#include "agents_framework/store/db.hpp"

namespace graph = agents_framework::graph;
namespace store = agents_framework::store;

namespace {

using Steps = graph::Channel<"steps", int>;
using Log = graph::Channel<"log", std::vector<std::string>, graph::Append>;
using DemoSchema = graph::Schema<Steps, Log>;

auto log_node(std::string label) {
  return [label = std::move(label)](graph::StateView<DemoSchema> view) {
    return graph::Update<DemoSchema>{}
        .write<"steps">(view.get<"steps">() + 1)
        .write<"log">({label});
  };
}

graph::CompiledGraph make_linear_graph() {
  graph::GraphBuilder<DemoSchema> builder;
  builder.add_node("plan", log_node("plan"))
      .add_node("act", log_node("act"))
      .add_node("summarize", log_node("summarize"))
      .set_entry("plan")
      .add_edge("plan", "act")
      .add_edge("act", "summarize")
      .set_finish("summarize");
  auto compiled = std::move(builder).compile();
  REQUIRE(compiled);
  return std::move(*compiled);
}

store::CheckpointStore make_store() {
  auto opened = store::Db::open_memory();
  REQUIRE(opened);
  auto checkpoints =
      store::CheckpointStore::open(std::make_shared<store::Db>(std::move(*opened)));
  REQUIRE(checkpoints);
  return std::move(*checkpoints);
}

}

TEST_CASE("a run pauses before a gated node and reports the pending work",
          "[graph][interrupt]") {
  auto compiled = make_linear_graph();
  graph::State<DemoSchema> state;
  graph::Executor executor;

  const auto stats =
      executor.run(compiled, state, graph::RunOptions{.interrupt_before = {"act"}});
  REQUIRE(stats);
  REQUIRE(stats->status == graph::RunStatus::Interrupted);
  REQUIRE(stats->pending_nodes == std::vector<std::string>{"act"});
  REQUIRE(stats->steps == 1);
  REQUIRE(state.get<"log">() == std::vector<std::string>{"plan"});
}

TEST_CASE("an interrupted run persists a paused checkpoint and resumes on approval",
          "[graph][interrupt]") {
  auto checkpoints = make_store();
  auto compiled = make_linear_graph();
  graph::State<DemoSchema> state;
  graph::Executor executor;

  const auto paused = executor.run(compiled, state,
                                   graph::RunOptions{.run_id = "run-1",
                                                     .checkpointer = &checkpoints,
                                                     .interrupt_before = {"act"}});
  REQUIRE(paused);
  REQUIRE(paused->status == graph::RunStatus::Interrupted);

  const auto pending = checkpoints.latest("run-1");
  REQUIRE(pending);
  REQUIRE(pending->status == graph::CheckpointStatus::Interrupted);
  REQUIRE(pending->next_nodes == std::vector<std::string>{"act"});

  auto restored = graph::State<DemoSchema>::deserialize(pending->state);
  REQUIRE(restored);
  const auto resumed = executor.resume(compiled, *restored, *pending,
                                       graph::RunOptions{.checkpointer = &checkpoints,
                                                         .interrupt_before = {"act"}});
  REQUIRE(resumed);
  REQUIRE(resumed->status == graph::RunStatus::Completed);
  REQUIRE(restored->get<"log">() == std::vector<std::string>{"plan", "act", "summarize"});
}

TEST_CASE("approval only covers the interrupted step, not later gated nodes",
          "[graph][interrupt]") {
  auto checkpoints = make_store();
  auto compiled = make_linear_graph();
  graph::State<DemoSchema> state;
  graph::Executor executor;

  const graph::RunOptions options{.run_id = "run-2",
                                  .checkpointer = &checkpoints,
                                  .interrupt_before = {"act", "summarize"}};
  const auto first = executor.run(compiled, state, options);
  REQUIRE(first);
  REQUIRE(first->pending_nodes == std::vector<std::string>{"act"});

  auto pending = checkpoints.latest("run-2");
  REQUIRE(pending);
  auto restored = graph::State<DemoSchema>::deserialize(pending->state);
  REQUIRE(restored);
  const auto second = executor.resume(compiled, *restored, *pending, options);
  REQUIRE(second);
  REQUIRE(second->status == graph::RunStatus::Interrupted);
  REQUIRE(second->pending_nodes == std::vector<std::string>{"summarize"});
  REQUIRE(restored->get<"log">() == std::vector<std::string>{"plan", "act"});

  pending = checkpoints.latest("run-2");
  REQUIRE(pending);
  auto final_state = graph::State<DemoSchema>::deserialize(pending->state);
  REQUIRE(final_state);
  const auto third = executor.resume(compiled, *final_state, *pending, options);
  REQUIRE(third);
  REQUIRE(third->status == graph::RunStatus::Completed);
  REQUIRE(final_state->get<"log">() ==
          std::vector<std::string>{"plan", "act", "summarize"});
}

TEST_CASE("interrupts on nodes not in the graph never fire", "[graph][interrupt]") {
  auto compiled = make_linear_graph();
  graph::State<DemoSchema> state;
  graph::Executor executor;

  const auto stats =
      executor.run(compiled, state, graph::RunOptions{.interrupt_before = {"missing"}});
  REQUIRE(stats);
  REQUIRE(stats->status == graph::RunStatus::Completed);
  REQUIRE(state.get<"log">() == std::vector<std::string>{"plan", "act", "summarize"});
}
