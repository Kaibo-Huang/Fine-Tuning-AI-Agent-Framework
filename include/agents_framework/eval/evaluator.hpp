#pragma once

#include <string>

#include "agents_framework/core/result.hpp"
#include "agents_framework/eval/scorer.hpp"
#include "agents_framework/eval/task_suite.hpp"
#include "agents_framework/trace/trace.hpp"

namespace agents_framework::eval {

struct Evaluation {
  double score{0.0};
  bool verified{false};
  std::string detail;
};

class Evaluator {
 public:
  virtual ~Evaluator() = default;

  virtual core::Result<Evaluation> evaluate(const TaskInstance& instance,
                                            const trace::Trace& trace) = 0;

  [[nodiscard]] virtual bool needs_agent() const { return true; }
};

class TaskMetricEvaluator final : public Evaluator {
 public:
  explicit TaskMetricEvaluator(Verifier& verifier) : verifier_(&verifier) {}

  core::Result<Evaluation> evaluate(const TaskInstance& instance,
                                    const trace::Trace& trace) override;

 private:
  Verifier* verifier_;
};

class AccNormEvaluator final : public Evaluator {
 public:
  explicit AccNormEvaluator(SequenceScorer& scorer) : scorer_(&scorer) {}

  core::Result<Evaluation> evaluate(const TaskInstance& instance,
                                    const trace::Trace& trace) override;

  [[nodiscard]] bool needs_agent() const override { return false; }

 private:
  SequenceScorer* scorer_;
};

}
