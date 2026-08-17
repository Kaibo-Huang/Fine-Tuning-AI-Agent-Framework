#include <catch2/catch_test_macros.hpp>

#include <string>
#include <utility>
#include <vector>

#include "agents_framework/core/result.hpp"
#include "agents_framework/eval/evaluator.hpp"
#include "agents_framework/eval/verifiers.hpp"

namespace core = agents_framework::core;
namespace eval = agents_framework::eval;
namespace trace = agents_framework::trace;

namespace {

class FakeScorer final : public eval::SequenceScorer {
 public:
  explicit FakeScorer(std::vector<eval::ContinuationScore> scores)
      : scores_(std::move(scores)) {}

  core::Result<std::vector<eval::ContinuationScore>> score(
      std::string_view, std::span<const std::string> continuations) override {
    REQUIRE(continuations.size() == scores_.size());
    return scores_;
  }

 private:
  std::vector<eval::ContinuationScore> scores_;
};

eval::TaskInstance mc_instance(int answer_index) {
  eval::TaskInstance instance;
  instance.id = "mc-1";
  instance.input = {{"prompt", "The capital of France is"}};
  instance.expected["choices"] = {" Paris", " London", " a very long wrong answer"};
  instance.expected["answer_index"] = answer_index;
  return instance;
}

}

TEST_CASE("the task-metric evaluator applies the verifier to the final output",
          "[eval][evaluator]") {
  auto verifier = eval::make_exact_match_verifier();
  eval::TaskMetricEvaluator evaluator{*verifier};
  REQUIRE(evaluator.needs_agent());

  eval::TaskInstance instance;
  instance.id = "case-1";
  instance.expected["answer"] = "42";

  trace::Trace right;
  right.final_output = "42";
  auto evaluation = evaluator.evaluate(instance, right);
  REQUIRE(evaluation);
  REQUIRE(evaluation->score == 1.0);
  REQUIRE(evaluation->verified);

  trace::Trace wrong;
  wrong.final_output = "41";
  evaluation = evaluator.evaluate(instance, wrong);
  REQUIRE(evaluation);
  REQUIRE(evaluation->score == 0.0);
}

TEST_CASE("acc_norm needs no agent and length-normalizes", "[eval][evaluator]") {
  FakeScorer scorer{{
      {-4.0, 2},
      {-9.0, 3},
      {-13.0, 6},
  }};
  eval::AccNormEvaluator evaluator{scorer};
  REQUIRE(!evaluator.needs_agent());

  auto evaluation = evaluator.evaluate(mc_instance(0), trace::Trace{});
  REQUIRE(evaluation);
  REQUIRE(evaluation->score == 1.0);
  REQUIRE(evaluation->verified);
  REQUIRE(evaluation->detail == "picked choice 0");

  evaluation = evaluator.evaluate(mc_instance(2), trace::Trace{});
  REQUIRE(evaluation);
  REQUIRE(evaluation->score == 0.0);
}

TEST_CASE("acc_norm_pick validates its inputs", "[eval][evaluator]") {
  REQUIRE(!eval::acc_norm_pick({}));

  const std::vector<eval::ContinuationScore> zero_tokens{{-1.0, 0}};
  const auto broken = eval::acc_norm_pick(zero_tokens);
  REQUIRE(!broken);
  REQUIRE(broken.error().code == core::ErrorCode::Invalid);

  const std::vector<eval::ContinuationScore> tie{{-2.0, 2}, {-1.0, 1}};
  const auto pick = eval::acc_norm_pick(tie);
  REQUIRE(pick);
  REQUIRE(*pick == 0);
}

TEST_CASE("acc_norm rejects malformed instances", "[eval][evaluator]") {
  FakeScorer scorer{{{-1.0, 1}}};
  eval::AccNormEvaluator evaluator{scorer};

  eval::TaskInstance no_prompt;
  no_prompt.id = "bad-1";
  no_prompt.expected["choices"] = {"a"};
  no_prompt.expected["answer_index"] = 0;
  REQUIRE(!evaluator.evaluate(no_prompt, trace::Trace{}));

  eval::TaskInstance no_choices = mc_instance(0);
  no_choices.expected.erase("choices");
  REQUIRE(!evaluator.evaluate(no_choices, trace::Trace{}));

  eval::TaskInstance out_of_range = mc_instance(7);
  REQUIRE(!evaluator.evaluate(out_of_range, trace::Trace{}));
}
