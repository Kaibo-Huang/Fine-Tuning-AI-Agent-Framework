#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <string>
#include <vector>

#include "agents_framework/graph/channel.hpp"
#include "agents_framework/graph/events.hpp"
#include "agents_framework/graph/executor.hpp"
#include "agents_framework/graph/graph.hpp"
#include "agents_framework/graph/node.hpp"
#include "agents_framework/graph/state.hpp"
#include "agents_framework/llm/message.hpp"
#include "agents_framework/trace/capture.hpp"

namespace core = agents_framework::core;
namespace graph = agents_framework::graph;
namespace llm = agents_framework::llm;
namespace trace = agents_framework::trace;

namespace {

using Messages = graph::Channel<"messages", std::vector<llm::Message>, graph::Append>;
using S = graph::Schema<Messages>;

graph::CompiledGraph make_agent_graph(bool fail_second_node) {
  graph::GraphSpec spec;
  spec.add_node("think", graph::make_fn_node<S>([](graph::StateView<S>) {
    graph::Update<S> update;
    update.write<"messages">({llm::Message::assistant_text("thinking...")});
    return update;
  }));
  spec.add_node("answer", graph::make_fn_node<S>(
                              [fail_second_node](graph::StateView<S>)
                                  -> core::Result<graph::Update<S>> {
                                if (fail_second_node) {
                                  return core::fail(core::ErrorCode::Tool,
                                                    "database exploded mid-query");
                                }
                                graph::Update<S> update;
                                update.write<"messages">(
                                    {llm::Message::assistant_text("SELECT 42")});
                                return update;
                              }));
  spec.add_edge(std::string{graph::kStart}, "think");
  spec.add_edge("think", "answer");
  spec.add_edge("answer", std::string{graph::kEnd});
  auto compiled = std::move(spec).compile();
  REQUIRE(compiled);
  return std::move(*compiled);
}

}

TEST_CASE("a capture assembles transcript, node runs, and final output", "[trace][capture]") {
  auto graph_def = make_agent_graph(false);
  auto events = std::make_shared<graph::EventBus>();
  trace::TraceCapture capture{events};

  graph::State<S> state;
  state.set<"messages">({llm::Message::user_text("What is the answer?")});

  graph::Executor executor;
  graph::RunOptions options;
  options.events = events;
  REQUIRE(executor.run(graph_def, state, options));

  const auto captured = capture.take();
  REQUIRE(captured);
  REQUIRE(!captured->trace_id.empty());
  REQUIRE(!captured->run_id.empty());
  REQUIRE(captured->transcript.size() == 3);
  REQUIRE(captured->transcript.front().role == llm::Role::User);
  REQUIRE(captured->final_output == "SELECT 42");
  REQUIRE(captured->node_runs.size() == 2);
  REQUIRE(captured->node_runs[0].node == "think");
  REQUIRE(captured->node_runs[0].ok);
  REQUIRE(captured->node_runs[1].node == "answer");
}

TEST_CASE("a failed node's raw error text is captured", "[trace][capture]") {
  auto graph_def = make_agent_graph(true);
  auto events = std::make_shared<graph::EventBus>();
  trace::TraceCapture capture{events};

  graph::State<S> state;
  state.set<"messages">({llm::Message::user_text("What is the answer?")});

  graph::Executor executor;
  graph::RunOptions options;
  options.events = events;
  const auto run = executor.run(graph_def, state, options);
  REQUIRE(!run);

  const auto captured = capture.take();
  REQUIRE(captured);
  REQUIRE(captured->node_runs.size() == 2);
  REQUIRE(captured->node_runs[1].node == "answer");
  REQUIRE(!captured->node_runs[1].ok);
  REQUIRE(captured->node_runs[1].error.find("database exploded mid-query") != std::string::npos);
  REQUIRE(captured->transcript.size() == 2);
}

TEST_CASE("two distinct trace ids come from two takes", "[trace][capture]") {
  auto events = std::make_shared<graph::EventBus>();
  trace::TraceCapture capture{events};
  const auto first = capture.take();
  const auto second = capture.take();
  REQUIRE(first);
  REQUIRE(second);
  REQUIRE(first->trace_id != second->trace_id);
}

TEST_CASE("a run_id filter ignores other runs", "[trace][capture]") {
  auto events = std::make_shared<graph::EventBus>();
  trace::CaptureOptions options;
  options.run_id = "wanted";
  trace::TraceCapture capture{events, options};

  events->publish(graph::RunStarted{"other", 0});
  events->publish(graph::NodeFinished{"other", 1, "noise", true, ""});
  events->publish(graph::RunStarted{"wanted", 0});
  events->publish(graph::NodeFinished{"wanted", 1, "signal", true, ""});

  const auto captured = capture.take();
  REQUIRE(captured);
  REQUIRE(captured->run_id == "wanted");
  REQUIRE(captured->node_runs.size() == 1);
  REQUIRE(captured->node_runs[0].node == "signal");
}
