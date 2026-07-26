#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <chrono>
#include <latch>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "agents_framework/core/result.hpp"
#include "agents_framework/core/thread_pool.hpp"
#include "agents_framework/graph/builder.hpp"
#include "agents_framework/graph/executor.hpp"

namespace core = agents_framework::core;
namespace graph = agents_framework::graph;

namespace {

using Log = graph::Channel<"log", std::vector<std::string>, graph::Append>;
using DemoSchema = graph::Schema<Log>;

auto log_node(std::string label, std::chrono::milliseconds delay = {}) {
  return [label = std::move(label), delay](graph::StateView<DemoSchema>) {
    if (delay.count() > 0) std::this_thread::sleep_for(delay);
    return graph::Update<DemoSchema>{}.write<"log">({label});
  };
}

graph::CompiledGraph fan_out_graph(std::vector<std::unique_ptr<graph::Node>> branches) {
  graph::GraphBuilder<DemoSchema> builder;
  builder.add_node("split", log_node("split")).set_entry("split");
  for (std::size_t i = 0; i < branches.size(); ++i) {
    const std::string name = "branch" + std::to_string(i);
    builder.add_node(name, std::move(branches[i]));
    builder.add_edge("split", name);
    builder.set_finish(name);
  }
  auto compiled = std::move(builder).compile();
  REQUIRE(compiled);
  return std::move(*compiled);
}

}

TEST_CASE("the thread pool runs submitted tasks on its workers", "[core][thread_pool]") {
  core::ThreadPool pool{4};
  REQUIRE(pool.worker_count() == 4);

  std::atomic<int> counter{0};
  std::latch done{32};
  for (int i = 0; i < 32; ++i) {
    pool.submit([&] {
      counter.fetch_add(1);
      done.count_down();
    });
  }
  done.wait();
  REQUIRE(counter.load() == 32);
}

TEST_CASE("a zero-worker pool still makes progress", "[core][thread_pool]") {
  core::ThreadPool pool{0};
  REQUIRE(pool.worker_count() == 1);

  std::atomic<bool> ran{false};
  std::latch done{1};
  pool.submit([&] {
    ran = true;
    done.count_down();
  });
  done.wait();
  REQUIRE(ran.load());
}

TEST_CASE("branches in one super-step really run concurrently", "[graph][concurrency]") {
  std::atomic<int> in_flight{0};
  std::atomic<int> peak{0};

  const auto observing_branch = [&](std::chrono::milliseconds deadline) {
    return [&, deadline](graph::StateView<DemoSchema>) {
      const int now = in_flight.fetch_add(1) + 1;
      int expected = peak.load();
      while (expected < now && !peak.compare_exchange_weak(expected, now)) {
      }
      const auto give_up = std::chrono::steady_clock::now() + deadline;
      while (in_flight.load() < 2 && std::chrono::steady_clock::now() < give_up) {
        std::this_thread::yield();
      }
      in_flight.fetch_sub(1);
      return graph::Update<DemoSchema>{}.write<"log">({"observed"});
    };
  };

  SECTION("a pooled executor overlaps the branches") {
    std::vector<std::unique_ptr<graph::Node>> branches;
    branches.push_back(graph::make_fn_node<DemoSchema>(observing_branch(std::chrono::seconds{2})));
    branches.push_back(graph::make_fn_node<DemoSchema>(observing_branch(std::chrono::seconds{2})));
    auto compiled = fan_out_graph(std::move(branches));

    graph::State<DemoSchema> state;
    graph::Executor executor{graph::ExecutorOptions{.workers = 4}};
    REQUIRE(executor.run(compiled, state));
    REQUIRE(peak.load() == 2);
  }

  SECTION("the sequential executor never overlaps them") {
    std::vector<std::unique_ptr<graph::Node>> branches;
    branches.push_back(
        graph::make_fn_node<DemoSchema>(observing_branch(std::chrono::milliseconds{50})));
    branches.push_back(
        graph::make_fn_node<DemoSchema>(observing_branch(std::chrono::milliseconds{50})));
    auto compiled = fan_out_graph(std::move(branches));

    graph::State<DemoSchema> state;
    graph::Executor executor;
    REQUIRE(executor.run(compiled, state));
    REQUIRE(peak.load() == 1);
  }
}

TEST_CASE("parallel execution keeps the deterministic barrier order", "[graph][concurrency]") {
  const auto build = [] {
    std::vector<std::unique_ptr<graph::Node>> branches;
    for (int i = 0; i < 4; ++i) {
      branches.push_back(graph::make_fn_node<DemoSchema>(log_node(
          "branch" + std::to_string(i), std::chrono::milliseconds{(4 - i) * 15})));
    }
    return fan_out_graph(std::move(branches));
  };
  const std::vector<std::string> expected{"split",   "branch0", "branch1",
                                          "branch2", "branch3"};

  auto sequential_graph = build();
  graph::State<DemoSchema> sequential_state;
  graph::Executor sequential;
  REQUIRE(sequential.run(sequential_graph, sequential_state));
  REQUIRE(sequential_state.get<"log">() == expected);

  auto parallel_graph = build();
  graph::Executor parallel{graph::ExecutorOptions{.workers = 4}};
  for (int run = 0; run < 5; ++run) {
    graph::State<DemoSchema> state;
    REQUIRE(parallel.run(parallel_graph, state));
    REQUIRE(state.get<"log">() == expected);
    REQUIRE(state.serialize() == sequential_state.serialize());
  }
}

TEST_CASE("errors from parallel branches are reported deterministically",
          "[graph][concurrency]") {
  const auto failing = [](std::string label) {
    return [label = std::move(label)](
               graph::StateView<DemoSchema>) -> core::Result<graph::Update<DemoSchema>> {
      return core::fail(core::ErrorCode::Tool, label);
    };
  };

  std::vector<std::unique_ptr<graph::Node>> branches;
  branches.push_back(graph::make_fn_node<DemoSchema>(log_node("ok")));
  branches.push_back(graph::make_fn_node<DemoSchema>(failing("first failure")));
  branches.push_back(graph::make_fn_node<DemoSchema>(failing("second failure")));
  auto compiled = fan_out_graph(std::move(branches));

  graph::State<DemoSchema> state;
  graph::Executor executor{graph::ExecutorOptions{.workers = 4}};
  const auto stats = executor.run(compiled, state);
  REQUIRE(!stats);
  REQUIRE(stats.error().message == "first failure");
  REQUIRE(stats.error().context == "node 'branch1' (step 2)");
  REQUIRE(state.get<"log">() == std::vector<std::string>{"split"});
}

TEST_CASE("exceptions from worker threads resurface on the caller", "[graph][concurrency]") {
  std::vector<std::unique_ptr<graph::Node>> branches;
  branches.push_back(graph::make_fn_node<DemoSchema>(log_node("fine")));
  branches.push_back(
      graph::make_fn_node<DemoSchema>([](graph::StateView<DemoSchema>) -> graph::Update<DemoSchema> {
        throw std::runtime_error("broken invariant");
      }));
  auto compiled = fan_out_graph(std::move(branches));

  graph::State<DemoSchema> state;
  graph::Executor executor{graph::ExecutorOptions{.workers = 4}};
  REQUIRE_THROWS_AS(executor.run(compiled, state), std::runtime_error);
}
