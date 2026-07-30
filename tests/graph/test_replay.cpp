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

TEST_CASE("a completed run can be replayed from any earlier step via fork",
          "[graph][replay]") {
  auto checkpoints = make_store();
  auto compiled = make_linear_graph();

  graph::State<DemoSchema> state;
  graph::Executor executor;
  REQUIRE(executor.run(compiled, state,
                       graph::RunOptions{.run_id = "run-1", .checkpointer = &checkpoints}));

  const auto forked = checkpoints.fork("run-1", 1, "replay-1");
  REQUIRE(forked);

  auto replayed = graph::State<DemoSchema>::deserialize(forked->state);
  REQUIRE(replayed);
  REQUIRE(replayed->get<"log">() == std::vector<std::string>{"plan"});

  const auto stats = executor.resume(compiled, *replayed, *forked,
                                     graph::RunOptions{.checkpointer = &checkpoints});
  REQUIRE(stats);
  REQUIRE(stats->steps == 2);
  REQUIRE(replayed->serialize() == state.serialize());

  const auto replay_checkpoints = checkpoints.list("replay-1");
  REQUIRE(replay_checkpoints);
  REQUIRE(replay_checkpoints->size() == 3);

  const auto original = checkpoints.list("run-1");
  REQUIRE(original);
  REQUIRE(original->size() == 4);
}

TEST_CASE("a forked run can diverge from the original without touching it",
          "[graph][replay]") {
  auto checkpoints = make_store();
  auto compiled = make_linear_graph();

  graph::State<DemoSchema> state;
  graph::Executor executor;
  REQUIRE(executor.run(compiled, state,
                       graph::RunOptions{.run_id = "run-2", .checkpointer = &checkpoints}));

  const auto forked = checkpoints.fork("run-2", 2, "replay-2");
  REQUIRE(forked);

  auto replayed = graph::State<DemoSchema>::deserialize(forked->state);
  REQUIRE(replayed);
  replayed->set<"log">({"edited-history"});

  REQUIRE(executor.resume(compiled, *replayed, *forked,
                          graph::RunOptions{.checkpointer = &checkpoints}));
  REQUIRE(replayed->get<"log">() == std::vector<std::string>{"edited-history", "summarize"});

  const auto original_latest = checkpoints.latest("run-2");
  REQUIRE(original_latest);
  const auto original_state = graph::State<DemoSchema>::deserialize(original_latest->state);
  REQUIRE(original_state);
  REQUIRE(original_state->get<"log">() ==
          std::vector<std::string>{"plan", "act", "summarize"});
}
