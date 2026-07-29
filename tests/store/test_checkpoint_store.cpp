#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "agents_framework/core/result.hpp"
#include "agents_framework/graph/checkpoint.hpp"
#include "agents_framework/store/checkpoint_store.hpp"
#include "agents_framework/store/db.hpp"

namespace core = agents_framework::core;
namespace graph = agents_framework::graph;
namespace store = agents_framework::store;

namespace {

std::shared_ptr<store::Db> open_db() {
  auto opened = store::Db::open_memory();
  REQUIRE(opened);
  return std::make_shared<store::Db>(std::move(*opened));
}

store::CheckpointStore make_store() {
  auto checkpoints = store::CheckpointStore::open(open_db());
  REQUIRE(checkpoints);
  return std::move(*checkpoints);
}

graph::Checkpoint make_checkpoint(std::string run_id, std::uint64_t step,
                                  std::vector<std::string> next) {
  const auto status = next.empty() ? graph::CheckpointStatus::Completed
                                   : graph::CheckpointStatus::Running;
  return graph::Checkpoint{std::move(run_id), step, R"({"steps":)" + std::to_string(step) + "}",
                           std::move(next), status};
}

}

TEST_CASE("checkpoints round-trip through the store", "[store][checkpoint]") {
  auto checkpoints = make_store();
  const graph::Checkpoint saved{"run-1", 2, R"({"steps":2})", {"act", "plan"},
                                graph::CheckpointStatus::Running};
  REQUIRE(checkpoints.save(saved));

  const auto loaded = checkpoints.load("run-1", 2);
  REQUIRE(loaded);
  REQUIRE(*loaded == saved);
}

TEST_CASE("saving the same step twice overwrites it", "[store][checkpoint]") {
  auto checkpoints = make_store();
  REQUIRE(checkpoints.save(make_checkpoint("run-1", 1, {"a"})));
  REQUIRE(checkpoints.save(graph::Checkpoint{"run-1", 1, "{}", {"b"},
                                             graph::CheckpointStatus::Interrupted}));

  const auto loaded = checkpoints.load("run-1", 1);
  REQUIRE(loaded);
  REQUIRE(loaded->next_nodes == std::vector<std::string>{"b"});
  REQUIRE(loaded->status == graph::CheckpointStatus::Interrupted);

  const auto listed = checkpoints.list("run-1");
  REQUIRE(listed);
  REQUIRE(listed->size() == 1);
}

TEST_CASE("list and latest order checkpoints by step", "[store][checkpoint]") {
  auto checkpoints = make_store();
  REQUIRE(checkpoints.save(make_checkpoint("run-1", 0, {"plan"})));
  REQUIRE(checkpoints.save(make_checkpoint("run-1", 1, {"act"})));
  REQUIRE(checkpoints.save(make_checkpoint("run-1", 2, {})));

  const auto listed = checkpoints.list("run-1");
  REQUIRE(listed);
  REQUIRE(listed->size() == 3);
  REQUIRE(listed->at(0).step == 0);
  REQUIRE(listed->at(2).step == 2);
  REQUIRE(!listed->at(0).created_at.empty());

  const auto last = checkpoints.latest("run-1");
  REQUIRE(last);
  REQUIRE(last->step == 2);
  REQUIRE(last->status == graph::CheckpointStatus::Completed);
}

TEST_CASE("missing runs and steps report NotFound", "[store][checkpoint]") {
  auto checkpoints = make_store();
  REQUIRE(checkpoints.save(make_checkpoint("run-1", 0, {"plan"})));

  const auto missing_run = checkpoints.latest("ghost");
  REQUIRE(!missing_run);
  REQUIRE(missing_run.error().code == core::ErrorCode::NotFound);

  const auto missing_step = checkpoints.load("run-1", 7);
  REQUIRE(!missing_step);
  REQUIRE(missing_step.error().code == core::ErrorCode::NotFound);
}

TEST_CASE("an empty run id is rejected on save", "[store][checkpoint]") {
  auto checkpoints = make_store();
  const auto result = checkpoints.save(make_checkpoint("", 0, {"plan"}));
  REQUIRE(!result);
  REQUIRE(result.error().code == core::ErrorCode::Invalid);
}

TEST_CASE("fork copies a checkpoint under a new run id with lineage", "[store][checkpoint]") {
  auto checkpoints = make_store();
  REQUIRE(checkpoints.save(make_checkpoint("run-1", 0, {"plan"})));
  REQUIRE(checkpoints.save(make_checkpoint("run-1", 1, {"act"})));
  REQUIRE(checkpoints.save(make_checkpoint("run-1", 2, {})));

  const auto forked = checkpoints.fork("run-1", 1, "run-1b");
  REQUIRE(forked);
  REQUIRE(forked->run_id == "run-1b");
  REQUIRE(forked->step == 1);
  REQUIRE(forked->next_nodes == std::vector<std::string>{"act"});

  const auto original = checkpoints.list("run-1");
  REQUIRE(original);
  REQUIRE(original->size() == 3);

  const auto runs = checkpoints.runs();
  REQUIRE(runs);
  REQUIRE(runs->size() == 2);
  REQUIRE(runs->at(1).run_id == "run-1b");
  REQUIRE(runs->at(1).parent_run_id == "run-1");
}

TEST_CASE("forking onto an existing run id is rejected", "[store][checkpoint]") {
  auto checkpoints = make_store();
  REQUIRE(checkpoints.save(make_checkpoint("run-1", 0, {"plan"})));
  REQUIRE(checkpoints.save(make_checkpoint("run-2", 0, {"plan"})));

  const auto result = checkpoints.fork("run-1", 0, "run-2");
  REQUIRE(!result);
  REQUIRE(result.error().code == core::ErrorCode::Invalid);
}
