#include "agents_framework/eval/harness.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <latch>
#include <numeric>
#include <random>
#include <unordered_map>
#include <utility>

#include "agents_framework/core/clock.hpp"
#include "agents_framework/core/id.hpp"
#include "agents_framework/core/rng.hpp"
#include "agents_framework/core/thread_pool.hpp"
#include "agents_framework/eval/eval_store.hpp"
#include "agents_framework/version.hpp"

namespace agents_framework::eval {

namespace {

constexpr double kZ95 = 1.959963984540054;
constexpr std::size_t kBootstrapResamples = 2000;
constexpr std::uint64_t kCompareSeed = 0x5eedC0DEULL;

std::string generate_ulid() {
  core::SystemClock clock;
  std::random_device device;
  core::Rng rng{(static_cast<std::uint64_t>(device()) << 32) ^ device()};
  return core::Ulid::generate(clock, rng).to_string();
}

[[nodiscard]] bool is_binary(std::span<const double> scores) {
  constexpr double kEpsilon = 1e-12;
  return std::all_of(scores.begin(), scores.end(), [](double s) {
    return std::abs(s) < kEpsilon || std::abs(s - 1.0) < kEpsilon;
  });
}

struct InstanceOutcome {
  InstanceResult result;
  std::optional<trace::Trace> trace;
};

InstanceOutcome run_instance(const TaskInstance& instance, const AgentFn* agent,
                             Evaluator& evaluator, const EvalOptions& options) {
  InstanceOutcome outcome;
  outcome.result.task_id = instance.id;
  outcome.result.split = instance.split;

  trace::Trace trace;
  bool keep_trace = false;
  if (agent != nullptr) {
    auto produced = (*agent)(instance);
    if (!produced) {
      outcome.result.ok = false;
      outcome.result.error = produced.error().to_string();
      return outcome;
    }
    keep_trace = produced->trace.has_value();
    if (keep_trace) trace = *std::move(produced->trace);
    trace.final_output = std::move(produced->output);
  }

  auto evaluated = evaluator.evaluate(instance, trace);
  if (!evaluated) {
    outcome.result.ok = false;
    outcome.result.error = evaluated.error().to_string();
    return outcome;
  }
  outcome.result.score = std::clamp(evaluated->score, 0.0, 1.0);
  outcome.result.verified = evaluated->verified;
  outcome.result.detail = std::move(evaluated->detail);

  if (keep_trace) {
    trace.task_id = instance.id;
    trace.model = options.model;
    trace.score = outcome.result.score;
    trace.verified = outcome.result.verified && outcome.result.score >= 1.0;
    outcome.trace = std::move(trace);
  }
  return outcome;
}

core::Result<EvalReport> run_eval_impl(TaskSuite& suite, const AgentFn* agent,
                                       Evaluator& evaluator, const EvalOptions& options) {
  if (evaluator.needs_agent() && agent == nullptr) {
    return core::fail(core::ErrorCode::Invalid,
                      "this evaluator needs an agent — use the run_eval overload that takes one",
                      std::string{suite.name()});
  }

  const std::span<const TaskInstance> all = suite.instances();
  std::vector<std::size_t> selected;
  selected.reserve(all.size());
  for (std::size_t i = 0; i < all.size(); ++i) {
    if (!options.split || all[i].split == *options.split) selected.push_back(i);
  }
  if (selected.empty()) {
    return core::fail(core::ErrorCode::Invalid, "no instances selected",
                      "suite '" + std::string{suite.name()} + "'" +
                          (options.split ? std::string{", split "} +
                                               std::string{split_name(*options.split)}
                                         : std::string{}));
  }
  if (options.limit > 0 && options.limit < selected.size()) {
    std::mt19937_64 engine{options.seed};
    std::shuffle(selected.begin(), selected.end(), engine);
    selected.resize(options.limit);
    std::sort(selected.begin(), selected.end());
  }

  std::vector<InstanceOutcome> outcomes(selected.size());
  const auto work = [&](std::size_t i) {
    outcomes[i] = run_instance(all[selected[i]], agent, evaluator, options);
  };
  if (options.workers > 1 && selected.size() > 1) {
    core::ThreadPool pool{options.workers};
    std::vector<std::exception_ptr> exceptions(selected.size());
    std::latch done{static_cast<std::ptrdiff_t>(selected.size())};
    for (std::size_t i = 0; i < selected.size(); ++i) {
      pool.submit([&work, &exceptions, &done, i] {
        try {
          work(i);
        } catch (...) {
          exceptions[i] = std::current_exception();
        }
        done.count_down();
      });
    }
    done.wait();
    for (const std::exception_ptr& thrown : exceptions) {
      if (thrown) std::rethrow_exception(thrown);
    }
  } else {
    for (std::size_t i = 0; i < selected.size(); ++i) work(i);
  }

  EvalReport report;
  report.eval_id = generate_ulid();
  report.suite = std::string{suite.name()};
  report.model = options.model;
  report.adapter = options.adapter;
  report.label = options.label;
  report.split = options.split ? std::string{split_name(*options.split)} : std::string{};
  report.seed = options.seed;
  report.framework_version = std::string{version()};
  report.framework_commit = std::string{commit()};
  report.total = outcomes.size();

  std::vector<double> scores;
  scores.reserve(outcomes.size());
  for (InstanceOutcome& outcome : outcomes) {
    if (outcome.trace && options.traces != nullptr) {
      if (outcome.trace->trace_id.empty()) outcome.trace->trace_id = generate_ulid();
      outcome.trace->suite = report.suite;
      AF_TRY_VOID(options.traces->save(*outcome.trace));
      outcome.result.trace_id = outcome.trace->trace_id;
    }
    if (!outcome.result.ok) ++report.failures;
    scores.push_back(outcome.result.score);
    report.results.push_back(std::move(outcome.result));
  }
  report.mean_score =
      std::accumulate(scores.begin(), scores.end(), 0.0) / static_cast<double>(scores.size());
  report.ci = confidence_interval(scores, options.seed);

  if (options.store != nullptr) {
    AF_TRY_VOID(options.store->save(report));
  }
  return report;
}

}

Interval wilson_interval(double mean, std::size_t n) {
  if (n == 0) return {0.0, 1.0};
  const double p = std::clamp(mean, 0.0, 1.0);
  const double nd = static_cast<double>(n);
  const double z2 = kZ95 * kZ95;
  const double denominator = 1.0 + z2 / nd;
  const double center = (p + z2 / (2.0 * nd)) / denominator;
  const double half =
      (kZ95 / denominator) * std::sqrt(p * (1.0 - p) / nd + z2 / (4.0 * nd * nd));
  return {std::max(0.0, center - half), std::min(1.0, center + half)};
}

Interval bootstrap_interval(std::span<const double> values, std::uint64_t seed) {
  if (values.empty()) return {0.0, 0.0};
  if (values.size() == 1) return {values[0], values[0]};

  std::mt19937_64 engine{seed ^ 0x9e3779b97f4a7c15ULL};
  std::uniform_int_distribution<std::size_t> pick{0, values.size() - 1};
  std::vector<double> means;
  means.reserve(kBootstrapResamples);
  for (std::size_t b = 0; b < kBootstrapResamples; ++b) {
    double sum = 0.0;
    for (std::size_t i = 0; i < values.size(); ++i) sum += values[pick(engine)];
    means.push_back(sum / static_cast<double>(values.size()));
  }
  std::sort(means.begin(), means.end());
  const auto at = [&means](double quantile) {
    const double position = quantile * static_cast<double>(means.size() - 1);
    return means[static_cast<std::size_t>(std::llround(position))];
  };
  return {at(0.025), at(0.975)};
}

Interval confidence_interval(std::span<const double> scores, std::uint64_t seed) {
  if (scores.empty()) return {0.0, 0.0};
  const double mean =
      std::accumulate(scores.begin(), scores.end(), 0.0) / static_cast<double>(scores.size());
  if (is_binary(scores)) return wilson_interval(mean, scores.size());
  return bootstrap_interval(scores, seed);
}

core::Result<EvalReport> run_eval(TaskSuite& suite, const AgentFn& agent, Evaluator& evaluator,
                                  const EvalOptions& options) {
  if (!agent) {
    return core::fail(core::ErrorCode::Invalid, "run_eval was given an empty agent function");
  }
  return run_eval_impl(suite, &agent, evaluator, options);
}

core::Result<EvalReport> run_eval(TaskSuite& suite, const AgentFn& agent,
                                  const EvalOptions& options) {
  TaskMetricEvaluator evaluator{suite.verifier()};
  return run_eval(suite, agent, evaluator, options);
}

core::Result<EvalReport> run_eval(TaskSuite& suite, Evaluator& evaluator,
                                  const EvalOptions& options) {
  return run_eval_impl(suite, nullptr, evaluator, options);
}

core::Result<PassAtK> pass_at_k(std::span<const EvalReport> runs, double threshold) {
  if (runs.empty()) {
    return core::fail(core::ErrorCode::Invalid, "pass@k needs at least one run");
  }

  std::unordered_map<std::string_view, std::size_t> passes;
  for (const InstanceResult& result : runs.front().results) {
    passes.emplace(result.task_id, 0);
  }
  double accuracy_sum = 0.0;
  for (const EvalReport& run : runs) {
    if (run.results.size() != passes.size()) {
      return core::fail(core::ErrorCode::Invalid,
                        "pass@k runs must cover the same instances",
                        "run '" + run.eval_id + "' has a different instance count");
    }
    std::size_t solved = 0;
    for (const InstanceResult& result : run.results) {
      const auto it = passes.find(result.task_id);
      if (it == passes.end()) {
        return core::fail(core::ErrorCode::Invalid,
                          "pass@k runs must cover the same instances",
                          "'" + result.task_id + "' is missing from the first run");
      }
      if (result.score >= threshold) {
        ++it->second;
        ++solved;
      }
    }
    accuracy_sum += static_cast<double>(solved) / static_cast<double>(passes.size());
  }

  PassAtK out;
  out.k = runs.size();
  out.instances = passes.size();
  out.pass_at_1 = accuracy_sum / static_cast<double>(runs.size());
  std::size_t solved_by_any = 0;
  for (const auto& [id, count] : passes) {
    if (count > 0) ++solved_by_any;
  }
  out.pass_at_k = static_cast<double>(solved_by_any) / static_cast<double>(passes.size());
  out.ci = wilson_interval(out.pass_at_k, passes.size());
  return out;
}

core::Result<EvalComparison> compare(const EvalReport& baseline, const EvalReport& candidate) {
  std::unordered_map<std::string_view, double> baseline_scores;
  baseline_scores.reserve(baseline.results.size());
  for (const InstanceResult& result : baseline.results) {
    baseline_scores.emplace(result.task_id, result.score);
  }

  EvalComparison comparison;
  std::vector<double> diffs;
  double baseline_sum = 0.0;
  double candidate_sum = 0.0;
  for (const InstanceResult& result : candidate.results) {
    const auto it = baseline_scores.find(result.task_id);
    if (it == baseline_scores.end()) continue;
    const double diff = result.score - it->second;
    diffs.push_back(diff);
    baseline_sum += it->second;
    candidate_sum += result.score;
    if (diff > 0.0) {
      ++comparison.improved;
    } else if (diff < 0.0) {
      ++comparison.regressed;
    } else {
      ++comparison.unchanged;
    }
  }
  if (diffs.empty()) {
    return core::fail(core::ErrorCode::Invalid,
                      "the two reports share no task ids — nothing to compare",
                      baseline.suite + " vs " + candidate.suite);
  }

  comparison.common = diffs.size();
  comparison.baseline_mean = baseline_sum / static_cast<double>(diffs.size());
  comparison.candidate_mean = candidate_sum / static_cast<double>(diffs.size());
  comparison.delta = comparison.candidate_mean - comparison.baseline_mean;
  comparison.ci = bootstrap_interval(diffs, kCompareSeed);
  comparison.significant = comparison.ci.low > 0.0 || comparison.ci.high < 0.0;
  return comparison;
}

}
