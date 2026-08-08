#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "agents_framework/llm/message.hpp"

namespace agents_framework::trace {

struct NodeRun {
  std::uint64_t step{0};
  std::string node;
  bool ok{true};
  std::string error;

  bool operator==(const NodeRun&) const = default;
};

struct Trace {
  std::string trace_id;
  std::string run_id;
  std::string suite;
  std::string task_id;
  std::string model;
  std::vector<llm::Message> transcript;
  std::vector<NodeRun> node_runs;
  std::string final_output;
  std::optional<double> score;
  bool verified{false};
  nlohmann::json metadata = nlohmann::json::object();

  bool operator==(const Trace&) const = default;
};

[[nodiscard]] inline std::string final_assistant_text(const std::vector<llm::Message>& transcript) {
  for (auto it = transcript.rbegin(); it != transcript.rend(); ++it) {
    if (it->role != llm::Role::Assistant) continue;
    std::string out;
    for (const llm::ContentBlock& block : it->content) {
      if (const auto* text = std::get_if<llm::TextBlock>(&block)) out += text->text;
    }
    return out;
  }
  return {};
}

inline void to_json(nlohmann::json& j, const NodeRun& run) {
  j = nlohmann::json::object();
  j["step"] = run.step;
  j["node"] = run.node;
  j["ok"] = run.ok;
  j["error"] = run.error;
}

inline void from_json(const nlohmann::json& j, NodeRun& run) {
  run.step = j.value("step", std::uint64_t{0});
  run.node = j.value("node", std::string{});
  run.ok = j.value("ok", true);
  run.error = j.value("error", std::string{});
}

inline void to_json(nlohmann::json& j, const Trace& trace) {
  j = nlohmann::json::object();
  j["trace_id"] = trace.trace_id;
  j["run_id"] = trace.run_id;
  j["suite"] = trace.suite;
  j["task_id"] = trace.task_id;
  j["model"] = trace.model;
  j["transcript"] = trace.transcript;
  j["node_runs"] = trace.node_runs;
  j["final_output"] = trace.final_output;
  if (trace.score) {
    j["score"] = *trace.score;
  } else {
    j["score"] = nullptr;
  }
  j["verified"] = trace.verified;
  j["metadata"] = trace.metadata;
}

inline void from_json(const nlohmann::json& j, Trace& trace) {
  trace.trace_id = j.value("trace_id", std::string{});
  trace.run_id = j.value("run_id", std::string{});
  trace.suite = j.value("suite", std::string{});
  trace.task_id = j.value("task_id", std::string{});
  trace.model = j.value("model", std::string{});
  trace.transcript = j.value("transcript", std::vector<llm::Message>{});
  trace.node_runs = j.value("node_runs", std::vector<NodeRun>{});
  trace.final_output = j.value("final_output", std::string{});
  const auto score = j.find("score");
  trace.score = (score == j.end() || score->is_null()) ? std::nullopt
                                                       : std::optional{score->get<double>()};
  trace.verified = j.value("verified", false);
  trace.metadata = j.value("metadata", nlohmann::json::object());
}

}
