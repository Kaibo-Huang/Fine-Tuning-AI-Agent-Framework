#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "agents_framework/core/result.hpp"
#include "agents_framework/graph/builder.hpp"
#include "agents_framework/graph/executor.hpp"
#include "agents_framework/graph/llm_node.hpp"
#include "agents_framework/graph/subgraph_node.hpp"
#include "agents_framework/llm/mock_backend.hpp"

namespace core = agents_framework::core;
namespace graph = agents_framework::graph;
namespace llm = agents_framework::llm;

namespace {

using Task = graph::Channel<"task", std::string>;
using Results = graph::Channel<"results", std::vector<std::string>, graph::Append>;
using SupervisorSchema = graph::Schema<Task, Results>;

using WorkerMessages = graph::Channel<"messages", std::vector<llm::Message>, graph::Append>;
using WorkerSchema = graph::Schema<WorkerMessages>;

std::shared_ptr<llm::MockBackend> worker_backend(std::string prefix) {
  auto backend = std::make_shared<llm::MockBackend>();
  backend->set_handler(
      [prefix = std::move(prefix)](const llm::ChatRequest& request)
          -> core::Result<llm::ChatResponse> {
        std::string prompt;
        for (const auto& message : request.messages) {
          for (const auto& block : message.content) {
            if (const auto* text = std::get_if<llm::TextBlock>(&block)) prompt += text->text;
          }
        }
        llm::ChatResponse response;
        response.content.push_back(llm::TextBlock{prefix + ": " + prompt});
        return response;
      });
  return backend;
}

graph::CompiledGraph worker_graph(std::shared_ptr<llm::MockBackend> backend) {
  graph::GraphBuilder<WorkerSchema> builder;
  builder.add_node("llm", graph::make_llm_node<WorkerSchema>(std::move(backend)))
      .set_entry("llm")
      .set_finish("llm");
  auto compiled = std::move(builder).compile();
  REQUIRE(compiled);
  return std::move(*compiled);
}

std::unique_ptr<graph::Node> worker_node(std::string prefix) {
  return graph::make_subgraph_node<SupervisorSchema, WorkerSchema>(
      worker_graph(worker_backend(std::move(prefix))),
      [](graph::StateView<SupervisorSchema> parent)
          -> core::Result<graph::State<WorkerSchema>> {
        graph::State<WorkerSchema> child;
        child.set<"messages">({llm::Message::user_text(parent.get<"task">())});
        return child;
      },
      [](const graph::State<WorkerSchema>& child)
          -> core::Result<graph::Update<SupervisorSchema>> {
        const auto& messages = child.get<"messages">();
        std::string answer;
        for (const auto& block : messages.back().content) {
          if (const auto* text = std::get_if<llm::TextBlock>(&block)) answer += text->text;
        }
        return graph::Update<SupervisorSchema>{}.write<"results">({std::move(answer)});
      });
}

}

TEST_CASE("a supervisor delegates to a worker subgraph and collects its result",
          "[graph][multiagent]") {
  graph::GraphBuilder<SupervisorSchema> builder;
  builder.add_node("worker", worker_node("RESEARCHER"))
      .set_entry("worker")
      .set_finish("worker");
  auto compiled = std::move(builder).compile();
  REQUIRE(compiled);

  graph::State<SupervisorSchema> state;
  state.set<"task">("find the answer");
  graph::Executor executor;
  REQUIRE(executor.run(*compiled, state));

  REQUIRE(state.get<"results">() ==
          std::vector<std::string>{"RESEARCHER: find the answer"});
}

TEST_CASE("a supervisor routes work to the right specialist agent", "[graph][multiagent]") {
  graph::GraphBuilder<SupervisorSchema> builder;
  builder
      .add_node("supervisor",
                [](graph::StateView<SupervisorSchema>) {
                  return graph::Update<SupervisorSchema>{};
                })
      .add_node("math_agent", worker_node("MATH"))
      .add_node("writing_agent", worker_node("WRITER"))
      .set_entry("supervisor")
      .add_conditional_edge("supervisor",
                            [](graph::StateView<SupervisorSchema> view) -> std::string {
                              const auto& task = view.get<"task">();
                              return task.find("essay") != std::string::npos ? "writing_agent"
                                                                             : "math_agent";
                            })
      .set_finish("math_agent")
      .set_finish("writing_agent");
  auto compiled = std::move(builder).compile();
  REQUIRE(compiled);

  graph::Executor executor;

  graph::State<SupervisorSchema> math_state;
  math_state.set<"task">("integrate x squared");
  REQUIRE(executor.run(*compiled, math_state));
  REQUIRE(math_state.get<"results">() ==
          std::vector<std::string>{"MATH: integrate x squared"});

  graph::State<SupervisorSchema> essay_state;
  essay_state.set<"task">("write an essay on agents");
  REQUIRE(executor.run(*compiled, essay_state));
  REQUIRE(essay_state.get<"results">() ==
          std::vector<std::string>{"WRITER: write an essay on agents"});
}

TEST_CASE("parallel worker agents fan out and merge deterministically",
          "[graph][multiagent]") {
  const auto build = [] {
    graph::GraphBuilder<SupervisorSchema> builder;
    builder
        .add_node("plan",
                  [](graph::StateView<SupervisorSchema>) {
                    return graph::Update<SupervisorSchema>{};
                  })
        .add_node("alpha_agent", worker_node("ALPHA"))
        .add_node("beta_agent", worker_node("BETA"))
        .add_node("join",
                  [](graph::StateView<SupervisorSchema> view) {
                    return graph::Update<SupervisorSchema>{}.write<"results">(
                        {"combined " + std::to_string(view.get<"results">().size())});
                  })
        .set_entry("plan")
        .add_edge("plan", "alpha_agent")
        .add_edge("plan", "beta_agent")
        .add_edge("alpha_agent", "join")
        .add_edge("beta_agent", "join")
        .set_finish("join");
    auto compiled = std::move(builder).compile();
    REQUIRE(compiled);
    return std::move(*compiled);
  };

  auto sequential_graph = build();
  graph::State<SupervisorSchema> sequential;
  sequential.set<"task">("survey");
  graph::Executor single;
  REQUIRE(single.run(sequential_graph, sequential));

  auto pooled_graph = build();
  graph::State<SupervisorSchema> pooled;
  pooled.set<"task">("survey");
  graph::Executor parallel{graph::ExecutorOptions{.workers = 4}};
  REQUIRE(parallel.run(pooled_graph, pooled));

  REQUIRE(sequential.serialize() == pooled.serialize());
  REQUIRE(pooled.get<"results">() ==
          std::vector<std::string>{"ALPHA: survey", "BETA: survey", "combined 2"});
}

TEST_CASE("a failing worker surfaces its error through the parent", "[graph][multiagent]") {
  auto backend = std::make_shared<llm::MockBackend>();
  backend->push_error(core::Error{core::ErrorCode::Network, "provider unreachable"});

  graph::GraphBuilder<SupervisorSchema> builder;
  builder
      .add_node("worker",
                graph::make_subgraph_node<SupervisorSchema, WorkerSchema>(
                    worker_graph(backend),
                    [](graph::StateView<SupervisorSchema> parent)
                        -> core::Result<graph::State<WorkerSchema>> {
                      graph::State<WorkerSchema> child;
                      child.set<"messages">({llm::Message::user_text(parent.get<"task">())});
                      return child;
                    },
                    [](const graph::State<WorkerSchema>&)
                        -> core::Result<graph::Update<SupervisorSchema>> {
                      return graph::Update<SupervisorSchema>{};
                    }))
      .set_entry("worker")
      .set_finish("worker");
  auto compiled = std::move(builder).compile();
  REQUIRE(compiled);

  graph::State<SupervisorSchema> state;
  state.set<"task">("doomed");
  graph::Executor executor;
  const auto stats = executor.run(*compiled, state);
  REQUIRE(!stats);
  REQUIRE(stats.error().code == core::ErrorCode::Network);
  REQUIRE(stats.error().context.find("node 'llm'") != std::string::npos);
  REQUIRE(stats.error().context.find("node 'worker'") != std::string::npos);
}
