#include "agents_framework/eval/evaluator.hpp"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace agents_framework::eval {

core::Result<std::size_t> acc_norm_pick(std::span<const ContinuationScore> scores) {
  if (scores.empty()) {
    return core::fail(core::ErrorCode::Invalid, "acc_norm needs at least one continuation");
  }
  std::size_t best = 0;
  double best_value = 0.0;
  for (std::size_t i = 0; i < scores.size(); ++i) {
    if (scores[i].tokens == 0) {
      return core::fail(core::ErrorCode::Invalid,
                        "continuation has zero tokens — cannot length-normalize",
                        "continuation " + std::to_string(i));
    }
    const double value = scores[i].logprob / static_cast<double>(scores[i].tokens);
    if (i == 0 || value > best_value) {
      best = i;
      best_value = value;
    }
  }
  return best;
}

core::Result<Evaluation> TaskMetricEvaluator::evaluate(const TaskInstance& instance,
                                                       const trace::Trace& trace) {
  AF_TRY(const double score, verifier_->score(instance, trace.final_output));
  return Evaluation{std::clamp(score, 0.0, 1.0), verifier_->trusted(), {}};
}

core::Result<Evaluation> AccNormEvaluator::evaluate(const TaskInstance& instance,
                                                    const trace::Trace&) {
  std::string prompt;
  if (instance.input.is_string()) {
    prompt = instance.input.get<std::string>();
  } else if (instance.input.is_object()) {
    prompt = instance.input.value("prompt", std::string{});
  }
  if (prompt.empty()) {
    return core::fail(core::ErrorCode::Invalid, "acc_norm instance needs input.prompt",
                      "'" + instance.id + "'");
  }

  const auto choices_it = instance.expected.find("choices");
  if (choices_it == instance.expected.end() || !choices_it->is_array() || choices_it->empty()) {
    return core::fail(core::ErrorCode::Invalid,
                      "acc_norm instance needs expected.choices (a non-empty array)",
                      "'" + instance.id + "'");
  }
  std::vector<std::string> choices;
  choices.reserve(choices_it->size());
  for (const auto& choice : *choices_it) {
    if (!choice.is_string()) {
      return core::fail(core::ErrorCode::Invalid, "expected.choices must contain strings",
                        "'" + instance.id + "'");
    }
    choices.push_back(choice.get<std::string>());
  }

  const auto answer_it = instance.expected.find("answer_index");
  if (answer_it == instance.expected.end() || !answer_it->is_number_integer()) {
    return core::fail(core::ErrorCode::Invalid,
                      "acc_norm instance needs expected.answer_index (an integer)",
                      "'" + instance.id + "'");
  }
  const auto answer = answer_it->get<std::int64_t>();
  if (answer < 0 || static_cast<std::size_t>(answer) >= choices.size()) {
    return core::fail(core::ErrorCode::Invalid, "expected.answer_index is out of range",
                      "'" + instance.id + "'");
  }

  AF_TRY(const auto scores, scorer_->score(prompt, choices));
  if (scores.size() != choices.size()) {
    return core::fail(core::ErrorCode::Invalid,
                      "scorer returned a different number of scores than continuations",
                      "'" + instance.id + "'");
  }
  AF_TRY(const std::size_t pick, acc_norm_pick(scores));
  Evaluation evaluation;
  evaluation.score = pick == static_cast<std::size_t>(answer) ? 1.0 : 0.0;
  evaluation.verified = true;
  evaluation.detail = "picked choice " + std::to_string(pick);
  return evaluation;
}

}
