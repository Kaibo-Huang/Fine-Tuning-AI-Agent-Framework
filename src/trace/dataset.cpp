#include "agents_framework/trace/dataset.hpp"

#include <exception>
#include <fstream>
#include <string>
#include <utility>
#include <variant>

namespace agents_framework::trace {

namespace {

[[nodiscard]] bool has_tool_use(const llm::Message& message) {
  for (const llm::ContentBlock& block : message.content) {
    if (std::holds_alternative<llm::ToolUseBlock>(block)) return true;
  }
  return false;
}

[[nodiscard]] bool has_tool_result(const llm::Message& message) {
  for (const llm::ContentBlock& block : message.content) {
    if (std::holds_alternative<llm::ToolResultBlock>(block)) return true;
  }
  return false;
}

[[nodiscard]] nlohmann::json example_metadata(const Trace& trace, TraceToExample strategy) {
  nlohmann::json meta = nlohmann::json::object();
  meta["trace_id"] = trace.trace_id;
  meta["suite"] = trace.suite;
  meta["task_id"] = trace.task_id;
  meta["strategy"] = std::string{trace_to_example_name(strategy)};
  if (trace.score) meta["score"] = *trace.score;
  meta["verified"] = trace.verified;
  return meta;
}

[[nodiscard]] TrainingExample slice_example(const Trace& trace, TraceToExample strategy,
                                            std::size_t last,
                                            const std::vector<std::size_t>& train_turns) {
  TrainingExample example;
  example.turns.reserve(last + 1);
  for (std::size_t i = 0; i <= last; ++i) {
    example.turns.push_back(ExampleTurn{trace.transcript[i], false});
  }
  for (const std::size_t index : train_turns) {
    example.turns[index].train = true;
  }
  example.metadata = example_metadata(trace, strategy);
  return example;
}

}

std::string_view trace_to_example_name(TraceToExample strategy) noexcept {
  switch (strategy) {
    case TraceToExample::WholeTrajectory: return "whole_trajectory";
    case TraceToExample::PerTurn:         return "per_turn";
    case TraceToExample::FinalAnswerOnly: return "final_answer_only";
    case TraceToExample::ToolCallsOnly:   return "tool_calls_only";
  }
  return "whole_trajectory";
}

std::optional<TraceToExample> parse_trace_to_example(std::string_view text) noexcept {
  if (text == "whole_trajectory") return TraceToExample::WholeTrajectory;
  if (text == "per_turn") return TraceToExample::PerTurn;
  if (text == "final_answer_only") return TraceToExample::FinalAnswerOnly;
  if (text == "tool_calls_only") return TraceToExample::ToolCallsOnly;
  return std::nullopt;
}

core::Result<void> validate_example(const TrainingExample& example) {
  bool any_train = false;
  for (std::size_t i = 0; i < example.turns.size(); ++i) {
    const ExampleTurn& turn = example.turns[i];
    if (!turn.train) continue;
    any_train = true;
    if (turn.message.role != llm::Role::Assistant) {
      return core::fail(core::ErrorCode::Invalid,
                        "loss on a non-assistant turn — the model cannot produce those tokens",
                        "turn " + std::to_string(i));
    }
    if (has_tool_result(turn.message)) {
      return core::fail(core::ErrorCode::Invalid,
                        "loss on a turn carrying tool results — the model cannot produce those "
                        "tokens",
                        "turn " + std::to_string(i));
    }
  }
  if (!any_train) {
    return core::fail(core::ErrorCode::Invalid, "example has no turn to train on");
  }
  return {};
}

core::Result<std::vector<TrainingExample>> trace_to_examples(const Trace& trace,
                                                             TraceToExample strategy) {
  std::vector<std::size_t> assistant_turns;
  for (std::size_t i = 0; i < trace.transcript.size(); ++i) {
    const llm::Message& message = trace.transcript[i];
    if (message.role != llm::Role::Assistant) continue;
    if (has_tool_result(message)) {
      return core::fail(core::ErrorCode::Invalid,
                        "malformed transcript: an assistant turn carries tool results",
                        "trace '" + trace.trace_id + "', turn " + std::to_string(i));
    }
    assistant_turns.push_back(i);
  }

  std::vector<TrainingExample> examples;
  if (assistant_turns.empty()) return examples;

  switch (strategy) {
    case TraceToExample::WholeTrajectory:
      examples.push_back(
          slice_example(trace, strategy, trace.transcript.size() - 1, assistant_turns));
      break;
    case TraceToExample::PerTurn:
      for (const std::size_t index : assistant_turns) {
        examples.push_back(slice_example(trace, strategy, index, {index}));
      }
      break;
    case TraceToExample::FinalAnswerOnly: {
      const std::size_t last = assistant_turns.back();
      examples.push_back(slice_example(trace, strategy, last, {last}));
      break;
    }
    case TraceToExample::ToolCallsOnly:
      for (const std::size_t index : assistant_turns) {
        if (!has_tool_use(trace.transcript[index])) continue;
        examples.push_back(slice_example(trace, strategy, index, {index}));
      }
      break;
  }

  for (const TrainingExample& example : examples) {
    AF_TRY_VOID(validate_example(example));
  }
  return examples;
}

core::Result<std::vector<TrainingExample>> build_dataset(std::span<const Trace> traces,
                                                         const DatasetOptions& options) {
  std::vector<TrainingExample> dataset;
  for (const Trace& trace : traces) {
    if (options.verified_only && !trace.verified) continue;
    if (options.min_score && (!trace.score || *trace.score < *options.min_score)) continue;

    AF_TRY(auto examples, trace_to_examples(trace, options.strategy));
    for (TrainingExample& example : examples) {
      if (options.limit > 0 && dataset.size() >= options.limit) return dataset;
      dataset.push_back(std::move(example));
    }
  }
  return dataset;
}

core::Result<std::vector<TrainingExample>> samples_to_examples(
    std::span<const TrainingSample> samples, bool verified_only) {
  std::vector<TrainingExample> examples;
  for (const TrainingSample& sample : samples) {
    if (verified_only && !sample.verified) continue;
    TrainingExample example;
    example.turns.push_back(ExampleTurn{llm::Message::user_text(sample.input), false});
    example.turns.push_back(ExampleTurn{llm::Message::assistant_text(sample.output), true});
    example.metadata["task_id"] = sample.task_id;
    example.metadata["score"] = sample.score;
    example.metadata["verified"] = sample.verified;
    AF_TRY_VOID(validate_example(example));
    examples.push_back(std::move(example));
  }
  return examples;
}

std::vector<PreferencePair> corrections_to_pairs(std::span<const CorrectionSample> corrections) {
  std::vector<PreferencePair> pairs;
  pairs.reserve(corrections.size());
  for (const CorrectionSample& correction : corrections) {
    PreferencePair pair;
    pair.context.push_back(llm::Message::user_text(correction.input));
    pair.rejected = correction.failed_attempt;
    pair.chosen = correction.correction;
    pair.metadata["task_id"] = correction.task_id;
    pair.metadata["error"] = correction.error;
    pairs.push_back(std::move(pair));
  }
  return pairs;
}

core::Result<void> write_jsonl(std::span<const TrainingExample> examples,
                               const std::filesystem::path& path) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    return core::fail(core::ErrorCode::Io, "cannot open dataset file for writing",
                      path.string());
  }
  for (const TrainingExample& example : examples) {
    out << nlohmann::json(example).dump() << '\n';
  }
  out.flush();
  if (!out) {
    return core::fail(core::ErrorCode::Io, "failed writing dataset file", path.string());
  }
  return {};
}

core::Result<std::vector<TrainingExample>> read_jsonl(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return core::fail(core::ErrorCode::Io, "cannot open dataset file for reading",
                      path.string());
  }
  std::vector<TrainingExample> examples;
  std::string line;
  std::size_t line_number = 0;
  while (std::getline(in, line)) {
    ++line_number;
    if (line.empty()) continue;
    const nlohmann::json j = nlohmann::json::parse(line, nullptr, false);
    if (j.is_discarded()) {
      return core::fail(core::ErrorCode::Parse, "dataset line is not valid JSON",
                        path.string() + ":" + std::to_string(line_number));
    }
    try {
      examples.push_back(j.get<TrainingExample>());
    } catch (const std::exception& error) {
      return core::fail(core::ErrorCode::Parse, error.what(),
                        path.string() + ":" + std::to_string(line_number));
    }
  }
  return examples;
}

}
