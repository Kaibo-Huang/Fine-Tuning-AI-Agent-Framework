#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <string>
#include <vector>

#include "agents_framework/graph/executor.hpp"
#include "agents_framework/graph/graph.hpp"
#include "agents_framework/graph/node.hpp"
#include "agents_framework/graph/state.hpp"
#include "agents_framework/trace/dataset.hpp"
#include "agents_framework/trace/training_channel.hpp"

namespace core = agents_framework::core;
namespace graph = agents_framework::graph;
namespace trace = agents_framework::trace;

namespace {

using Training = trace::TrainingDataChannel<>;
using S = graph::Schema<Training>;

}

TEST_CASE("nodes accumulate training data through the reducer channel",
          "[trace][training_channel]") {
  graph::GraphSpec spec;
  spec.add_node("solver", graph::make_fn_node<S>([](graph::StateView<S>) {
    trace::TrainingData data;
    data.samples.push_back({"case-1", "How many customers?", "SELECT COUNT(*) FROM customers",
                            true, 1.0});
    graph::Update<S> update;
    update.write<"training_data">(std::move(data));
    return update;
  }));
  spec.add_node("retry", graph::make_fn_node<S>([](graph::StateView<S>) {
    trace::TrainingData data;
    data.corrections.push_back({"case-2", "Total revenue?", "SELECT SUM(total)",
                                "no such column: total_", "SELECT SUM(total) FROM orders"});
    graph::Update<S> update;
    update.write<"training_data">(std::move(data));
    return update;
  }));
  spec.add_edge(std::string{graph::kStart}, "solver");
  spec.add_edge("solver", "retry");
  spec.add_edge("retry", std::string{graph::kEnd});
  auto compiled = std::move(spec).compile();
  REQUIRE(compiled);

  graph::State<S> state;
  graph::Executor executor;
  REQUIRE(executor.run(*compiled, state));

  const trace::TrainingData& collected = state.get<"training_data">();
  REQUIRE(collected.size() == 2);
  REQUIRE(collected.samples.size() == 1);
  REQUIRE(collected.corrections.size() == 1);
  REQUIRE(collected.samples[0].verified);
  REQUIRE(collected.corrections[0].error == "no such column: total_");
}

TEST_CASE("training data survives state serialization", "[trace][training_channel]") {
  graph::State<S> state;
  trace::TrainingData data;
  data.samples.push_back({"case-1", "q", "a", true, 1.0});
  data.corrections.push_back({"case-2", "q2", "bad", "err", "good"});
  state.reduce<"training_data">(std::move(data));

  const auto restored = graph::State<S>::deserialize(state.serialize());
  REQUIRE(restored);
  REQUIRE(restored->get<"training_data">() == state.get<"training_data">());
}

TEST_CASE("channel contents convert straight into training artifacts",
          "[trace][training_channel]") {
  trace::TrainingData data;
  data.samples.push_back({"case-1", "How many customers?", "SELECT COUNT(*) FROM customers",
                          true, 1.0});
  data.samples.push_back({"case-2", "unverified", "guess", false, 0.0});
  data.corrections.push_back({"case-3", "Total revenue?", "SELECT SUM(total)",
                              "no such column", "SELECT SUM(total) FROM orders"});

  const auto examples = trace::samples_to_examples(data.samples);
  REQUIRE(examples);
  REQUIRE(examples->size() == 1);
  REQUIRE(examples->front().turns.size() == 2);
  REQUIRE(examples->front().turns[1].train);

  const auto pairs = trace::corrections_to_pairs(data.corrections);
  REQUIRE(pairs.size() == 1);
  REQUIRE(pairs.front().rejected == "SELECT SUM(total)");
  REQUIRE(pairs.front().chosen == "SELECT SUM(total) FROM orders");
  REQUIRE(pairs.front().metadata.at("error") == "no such column");
}
