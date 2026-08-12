#pragma once

#include <cstddef>
#include <filesystem>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "agents_framework/core/result.hpp"
#include "agents_framework/trace/example.hpp"
#include "agents_framework/trace/trace.hpp"

namespace agents_framework::trace {

enum class TraceToExample {
  WholeTrajectory,
  PerTurn,
  FinalAnswerOnly,
  ToolCallsOnly,
};

[[nodiscard]] std::string_view trace_to_example_name(TraceToExample strategy) noexcept;
[[nodiscard]] std::optional<TraceToExample> parse_trace_to_example(std::string_view text) noexcept;

[[nodiscard]] core::Result<std::vector<TrainingExample>> trace_to_examples(
    const Trace& trace, TraceToExample strategy = TraceToExample::WholeTrajectory);

struct DatasetOptions {
  TraceToExample strategy{TraceToExample::WholeTrajectory};
  bool verified_only{true};
  std::optional<double> min_score;
  std::size_t limit{0};
};

[[nodiscard]] core::Result<std::vector<TrainingExample>> build_dataset(
    std::span<const Trace> traces, const DatasetOptions& options = {});

[[nodiscard]] core::Result<std::vector<TrainingExample>> samples_to_examples(
    std::span<const TrainingSample> samples, bool verified_only = true);

[[nodiscard]] std::vector<PreferencePair> corrections_to_pairs(
    std::span<const CorrectionSample> corrections);

[[nodiscard]] core::Result<void> write_jsonl(std::span<const TrainingExample> examples,
                                             const std::filesystem::path& path);
[[nodiscard]] core::Result<std::vector<TrainingExample>> read_jsonl(
    const std::filesystem::path& path);

}
