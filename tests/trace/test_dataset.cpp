#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <string>
#include <vector>

#include "agents_framework/core/result.hpp"
#include "agents_framework/llm/message.hpp"
#include "agents_framework/trace/dataset.hpp"

namespace core = agents_framework::core;
namespace llm = agents_framework::llm;
namespace trace = agents_framework::trace;

namespace {

trace::Trace tool_use_trace() {
  trace::Trace t;
  t.trace_id = "t1";
  t.suite = "ref-a";
  t.task_id = "count-customers";
  t.score = 1.0;
  t.verified = true;
  t.transcript = {
      llm::Message::user_text("How many customers are there?"),
      llm::Message{llm::Role::Assistant,
                   {llm::TextBlock{"Let me check."},
                    llm::ToolUseBlock{"call_1", "sql", {{"query", "SELECT COUNT(*)"}}}}},
      llm::Message{llm::Role::User, {llm::ToolResultBlock{"call_1", "[[\"4\"]]", false}}},
      llm::Message::assistant_text("SELECT COUNT(*) FROM customers"),
  };
  return t;
}

void require_masks_valid(const std::vector<trace::TrainingExample>& examples) {
  for (const trace::TrainingExample& example : examples) {
    REQUIRE(trace::validate_example(example));
    for (const trace::ExampleTurn& turn : example.turns) {
      if (turn.train) {
        REQUIRE(turn.message.role == llm::Role::Assistant);
        for (const llm::ContentBlock& block : turn.message.content) {
          REQUIRE(!std::holds_alternative<llm::ToolResultBlock>(block));
        }
      }
    }
  }
}

}

TEST_CASE("whole-trajectory trains every assistant turn and only those", "[trace][dataset]") {
  const auto examples =
      trace::trace_to_examples(tool_use_trace(), trace::TraceToExample::WholeTrajectory);
  REQUIRE(examples);
  REQUIRE(examples->size() == 1);
  const auto& turns = examples->front().turns;
  REQUIRE(turns.size() == 4);
  REQUIRE(!turns[0].train);
  REQUIRE(turns[1].train);
  REQUIRE(!turns[2].train);
  REQUIRE(turns[3].train);
  require_masks_valid(*examples);
  REQUIRE(examples->front().metadata.at("strategy") == "whole_trajectory");
  REQUIRE(examples->front().metadata.at("task_id") == "count-customers");
}

TEST_CASE("per-turn yields one example per assistant turn", "[trace][dataset]") {
  const auto examples =
      trace::trace_to_examples(tool_use_trace(), trace::TraceToExample::PerTurn);
  REQUIRE(examples);
  REQUIRE(examples->size() == 2);
  REQUIRE(examples->at(0).turns.size() == 2);
  REQUIRE(examples->at(0).turns[1].train);
  REQUIRE(examples->at(1).turns.size() == 4);
  REQUIRE(!examples->at(1).turns[1].train);
  REQUIRE(examples->at(1).turns[3].train);
  require_masks_valid(*examples);
}

TEST_CASE("final-answer-only trains just the last assistant turn", "[trace][dataset]") {
  const auto examples =
      trace::trace_to_examples(tool_use_trace(), trace::TraceToExample::FinalAnswerOnly);
  REQUIRE(examples);
  REQUIRE(examples->size() == 1);
  const auto& turns = examples->front().turns;
  REQUIRE(turns.size() == 4);
  REQUIRE(!turns[1].train);
  REQUIRE(turns[3].train);
  require_masks_valid(*examples);
}

TEST_CASE("tool-calls-only trains just tool-calling turns", "[trace][dataset]") {
  const auto examples =
      trace::trace_to_examples(tool_use_trace(), trace::TraceToExample::ToolCallsOnly);
  REQUIRE(examples);
  REQUIRE(examples->size() == 1);
  const auto& turns = examples->front().turns;
  REQUIRE(turns.size() == 2);
  REQUIRE(turns[1].train);
  require_masks_valid(*examples);
}

TEST_CASE("a trace without assistant turns yields no examples", "[trace][dataset]") {
  trace::Trace t;
  t.transcript = {llm::Message::user_text("hello?")};
  for (const auto strategy :
       {trace::TraceToExample::WholeTrajectory, trace::TraceToExample::PerTurn,
        trace::TraceToExample::FinalAnswerOnly, trace::TraceToExample::ToolCallsOnly}) {
    const auto examples = trace::trace_to_examples(t, strategy);
    REQUIRE(examples);
    REQUIRE(examples->empty());
  }
}

TEST_CASE("a malformed transcript is an error, not a silent mask", "[trace][dataset]") {
  trace::Trace t = tool_use_trace();
  t.transcript.push_back(llm::Message{
      llm::Role::Assistant, {llm::ToolResultBlock{"call_9", "smuggled tokens", false}}});
  const auto examples = trace::trace_to_examples(t, trace::TraceToExample::WholeTrajectory);
  REQUIRE(!examples);
  REQUIRE(examples.error().code == core::ErrorCode::Invalid);
}

TEST_CASE("validate_example rejects loss on tokens the model cannot produce",
          "[trace][dataset]") {
  trace::TrainingExample bad_role;
  bad_role.turns = {{llm::Message::user_text("q"), true}};
  REQUIRE(!trace::validate_example(bad_role));

  trace::TrainingExample no_loss;
  no_loss.turns = {{llm::Message::user_text("q"), false},
                   {llm::Message::assistant_text("a"), false}};
  REQUIRE(!trace::validate_example(no_loss));

  trace::TrainingExample good;
  good.turns = {{llm::Message::user_text("q"), false},
                {llm::Message::assistant_text("a"), true}};
  REQUIRE(trace::validate_example(good));
}

TEST_CASE("build_dataset filters on verification, score, and limit", "[trace][dataset]") {
  trace::Trace verified = tool_use_trace();
  trace::Trace failed = tool_use_trace();
  failed.trace_id = "t2";
  failed.score = 0.0;
  failed.verified = false;
  trace::Trace partial = tool_use_trace();
  partial.trace_id = "t3";
  partial.score = 0.6;
  partial.verified = false;

  const std::vector<trace::Trace> traces{verified, failed, partial};

  trace::DatasetOptions defaults;
  auto dataset = trace::build_dataset(traces, defaults);
  REQUIRE(dataset);
  REQUIRE(dataset->size() == 1);

  trace::DatasetOptions by_score;
  by_score.verified_only = false;
  by_score.min_score = 0.5;
  dataset = trace::build_dataset(traces, by_score);
  REQUIRE(dataset);
  REQUIRE(dataset->size() == 2);

  trace::DatasetOptions capped;
  capped.verified_only = false;
  capped.strategy = trace::TraceToExample::PerTurn;
  capped.limit = 3;
  dataset = trace::build_dataset(traces, capped);
  REQUIRE(dataset);
  REQUIRE(dataset->size() == 3);
}

TEST_CASE("datasets round-trip through JSONL", "[trace][dataset]") {
  const auto examples =
      trace::trace_to_examples(tool_use_trace(), trace::TraceToExample::WholeTrajectory);
  REQUIRE(examples);

  const auto path = std::filesystem::temp_directory_path() / "af_test_dataset.jsonl";
  REQUIRE(trace::write_jsonl(*examples, path));
  const auto restored = trace::read_jsonl(path);
  std::filesystem::remove(path);
  REQUIRE(restored);
  REQUIRE(*restored == *examples);
}

TEST_CASE("strategy names round-trip", "[trace][dataset]") {
  for (const auto strategy :
       {trace::TraceToExample::WholeTrajectory, trace::TraceToExample::PerTurn,
        trace::TraceToExample::FinalAnswerOnly, trace::TraceToExample::ToolCallsOnly}) {
    const auto parsed = trace::parse_trace_to_example(trace::trace_to_example_name(strategy));
    REQUIRE(parsed);
    REQUIRE(*parsed == strategy);
  }
  REQUIRE(!trace::parse_trace_to_example("nonsense"));
}
