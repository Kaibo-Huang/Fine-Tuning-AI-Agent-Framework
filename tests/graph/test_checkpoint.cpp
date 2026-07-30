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

namespace core = agents_framework::core;
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

std::shared_ptr<store::Db> open_db() {
  auto opened = store::Db::open_memory();
  REQUIRE(opened);
  return std::make_shared<store::Db>(std::move(*opened));
}

store::CheckpointStore make_store(std::shared_ptr<store::Db> db) {
  auto checkpoints = store::CheckpointStore::open(std::move(db));
  REQUIRE(checkpoints);
  return std::move(*checkpoints);
}

}

TEST_CASE("a run saves a checkpoint per super-step plus the initial state",
          "[graph][checkpoint]") {
  auto checkpoints = make_store(open_db());
  auto compiled = make_linear_graph();

  graph::State<DemoSchema> state;
  graph::Executor executor;
  const auto stats = executor.run(compiled, state,
                                  graph::RunOptions{.run_id = "run-1", .checkpointer = &checkpoints});
  REQUIRE(stats);
  REQUIRE(stats->status == graph::RunStatus::Completed);

  const auto listed = checkpoints.list("run-1");
  REQUIRE(listed);
  REQUIRE(listed->size() == 4);
  REQUIRE(listed->front().step == 0);
  REQUIRE(listed->back().step == 3);
  REQUIRE(listed->back().status == graph::CheckpointStatus::Completed);

  const auto last = checkpoints.latest("run-1");
  REQUIRE(last);
  REQUIRE(last->next_nodes.empty());
  const auto restored = graph::State<DemoSchema>::deserialize(last->state);
  REQUIRE(restored);
  REQUIRE(restored->get<"log">() == std::vector<std::string>{"plan", "act", "summarize"});
}

TEST_CASE("a killed run resumes from its last checkpoint and finishes identically",
          "[graph][checkpoint]") {
  auto checkpoints = make_store(open_db());
  auto compiled = make_linear_graph();

  graph::State<DemoSchema> state;
  graph::Executor executor;
  const auto killed = executor.run(
      compiled, state,
      graph::RunOptions{.max_steps = 2, .run_id = "run-2", .checkpointer = &checkpoints});
  REQUIRE(!killed);
  REQUIRE(killed.error().code == core::ErrorCode::Cancelled);

  const auto last = checkpoints.latest("run-2");
  REQUIRE(last);
  REQUIRE(last->step == 2);
  REQUIRE(last->next_nodes == std::vector<std::string>{"summarize"});

  auto restored = graph::State<DemoSchema>::deserialize(last->state);
  REQUIRE(restored);
  const auto resumed = executor.resume(compiled, *restored, *last,
                                       graph::RunOptions{.checkpointer = &checkpoints});
  REQUIRE(resumed);
  REQUIRE(resumed->status == graph::RunStatus::Completed);
  REQUIRE(resumed->steps == 1);
  REQUIRE(restored->get<"steps">() == 3);
  REQUIRE(restored->get<"log">() == std::vector<std::string>{"plan", "act", "summarize"});

  graph::State<DemoSchema> uninterrupted;
  REQUIRE(executor.run(compiled, uninterrupted));
  REQUIRE(restored->serialize() == uninterrupted.serialize());
}

TEST_CASE("resuming a completed checkpoint is rejected", "[graph][checkpoint]") {
  auto compiled = make_linear_graph();
  graph::State<DemoSchema> state;
  graph::Executor executor;

  const graph::Checkpoint done{"run-3", 3, state.serialize(), {},
                               graph::CheckpointStatus::Completed};
  const auto resumed = executor.resume(compiled, state, done);
  REQUIRE(!resumed);
  REQUIRE(resumed.error().code == core::ErrorCode::Invalid);
}

TEST_CASE("resuming a checkpoint naming an unknown node is rejected", "[graph][checkpoint]") {
  auto compiled = make_linear_graph();
  graph::State<DemoSchema> state;
  graph::Executor executor;

  const graph::Checkpoint bogus{"run-4", 1, state.serialize(), {"missing"},
                                graph::CheckpointStatus::Running};
  const auto resumed = executor.resume(compiled, state, bogus);
  REQUIRE(!resumed);
  REQUIRE(resumed.error().code == core::ErrorCode::NotFound);
}

TEST_CASE("a run without a run id gets a generated one when checkpointing",
          "[graph][checkpoint]") {
  auto checkpoints = make_store(open_db());
  auto compiled = make_linear_graph();

  graph::State<DemoSchema> state;
  graph::Executor executor;
  REQUIRE(executor.run(compiled, state, graph::RunOptions{.checkpointer = &checkpoints}));

  const auto runs = checkpoints.runs();
  REQUIRE(runs);
  REQUIRE(runs->size() == 1);
  REQUIRE(runs->front().run_id.size() == 26);
  REQUIRE(runs->front().status == graph::CheckpointStatus::Completed);
}
