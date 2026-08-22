#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "agents_framework/core/result.hpp"
#include "agents_framework/eval/evaluator.hpp"
#include "agents_framework/eval/task_suite.hpp"
#include "agents_framework/trace/trace.hpp"
#include "agents_framework/trace/trace_store.hpp"

namespace agents_framework::eval {

class EvalStore;

struct AgentOutput {
  std::string output;
  std::optional<trace::Trace> trace;
};

using AgentFn = std::function<core::Result<AgentOutput>(const TaskInstance&)>;

struct EvalOptions {
  std::optional<Split> split;
  std::size_t limit{0};
  std::size_t workers{1};
  std::uint64_t seed{0};
  std::string model;
  std::string adapter;
  std::string label;
  trace::TraceStore* traces{nullptr};
  EvalStore* store{nullptr};
};

struct InstanceResult {
  std::string task_id;
  Split split{Split::Train};
  double score{0.0};
  bool verified{false};
  bool ok{true};
  std::string error;
  std::string detail;
  std::string trace_id;

  bool operator==(const InstanceResult&) const = default;
};

struct Interval {
  double low{0.0};
  double high{0.0};

  bool operator==(const Interval&) const = default;
};

struct EvalReport {
  std::string eval_id;
  std::string suite;
  std::string model;
  std::string adapter;
  std::string label;
  std::string split;
  std::uint64_t seed{0};
  std::string framework_version;
  std::string framework_commit;
  std::size_t total{0};
  std::size_t failures{0};
  double mean_score{0.0};
  Interval ci;
  std::vector<InstanceResult> results;
};

[[nodiscard]] core::Result<EvalReport> run_eval(TaskSuite& suite, const AgentFn& agent,
                                                Evaluator& evaluator,
                                                const EvalOptions& options = {});

[[nodiscard]] core::Result<EvalReport> run_eval(TaskSuite& suite, const AgentFn& agent,
                                                const EvalOptions& options = {});

[[nodiscard]] core::Result<EvalReport> run_eval(TaskSuite& suite, Evaluator& evaluator,
                                                const EvalOptions& options = {});

struct EvalComparison {
  double baseline_mean{0.0};
  double candidate_mean{0.0};
  double delta{0.0};
  Interval ci;
  bool significant{false};
  std::size_t common{0};
  std::size_t improved{0};
  std::size_t regressed{0};
  std::size_t unchanged{0};
};

[[nodiscard]] core::Result<EvalComparison> compare(const EvalReport& baseline,
                                                   const EvalReport& candidate);

struct PassAtK {
  std::size_t k{0};
  std::size_t instances{0};
  double pass_at_1{0.0};
  double pass_at_k{0.0};
  Interval ci;
};

[[nodiscard]] core::Result<PassAtK> pass_at_k(std::span<const EvalReport> runs,
                                              double threshold = 1.0);

[[nodiscard]] Interval wilson_interval(double mean, std::size_t n);
[[nodiscard]] Interval bootstrap_interval(std::span<const double> values, std::uint64_t seed);
[[nodiscard]] Interval confidence_interval(std::span<const double> scores, std::uint64_t seed);

}
