#pragma once

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "agents_framework/core/result.hpp"
#include "agents_framework/llm/message.hpp"

namespace agents_framework::trace {

struct ExampleTurn {
  llm::Message message;
  bool train{false};

  bool operator==(const ExampleTurn&) const = default;
};

struct TrainingExample {
  std::vector<ExampleTurn> turns;
  nlohmann::json metadata = nlohmann::json::object();

  bool operator==(const TrainingExample&) const = default;
};

[[nodiscard]] core::Result<void> validate_example(const TrainingExample& example);

struct PreferencePair {
  std::vector<llm::Message> context;
  std::string rejected;
  std::string chosen;
  nlohmann::json metadata = nlohmann::json::object();

  bool operator==(const PreferencePair&) const = default;
};

struct TrainingSample {
  std::string task_id;
  std::string input;
  std::string output;
  bool verified{false};
  double score{0.0};

  bool operator==(const TrainingSample&) const = default;
};

struct CorrectionSample {
  std::string task_id;
  std::string input;
  std::string failed_attempt;
  std::string error;
  std::string correction;

  bool operator==(const CorrectionSample&) const = default;
};

inline void to_json(nlohmann::json& j, const ExampleTurn& turn) {
  to_json(j, turn.message);
  j["train"] = turn.train;
}

inline void from_json(const nlohmann::json& j, ExampleTurn& turn) {
  from_json(j, turn.message);
  turn.train = j.value("train", false);
}

inline void to_json(nlohmann::json& j, const TrainingExample& example) {
  j = nlohmann::json::object();
  j["turns"] = example.turns;
  j["metadata"] = example.metadata;
}

inline void from_json(const nlohmann::json& j, TrainingExample& example) {
  example.turns = j.at("turns").get<std::vector<ExampleTurn>>();
  example.metadata = j.value("metadata", nlohmann::json::object());
}

inline void to_json(nlohmann::json& j, const PreferencePair& pair) {
  j = nlohmann::json::object();
  j["context"] = pair.context;
  j["rejected"] = pair.rejected;
  j["chosen"] = pair.chosen;
  j["metadata"] = pair.metadata;
}

inline void from_json(const nlohmann::json& j, PreferencePair& pair) {
  pair.context = j.value("context", std::vector<llm::Message>{});
  pair.rejected = j.value("rejected", std::string{});
  pair.chosen = j.value("chosen", std::string{});
  pair.metadata = j.value("metadata", nlohmann::json::object());
}

inline void to_json(nlohmann::json& j, const TrainingSample& sample) {
  j = nlohmann::json::object();
  j["task_id"] = sample.task_id;
  j["input"] = sample.input;
  j["output"] = sample.output;
  j["verified"] = sample.verified;
  j["score"] = sample.score;
}

inline void from_json(const nlohmann::json& j, TrainingSample& sample) {
  sample.task_id = j.value("task_id", std::string{});
  sample.input = j.value("input", std::string{});
  sample.output = j.value("output", std::string{});
  sample.verified = j.value("verified", false);
  sample.score = j.value("score", 0.0);
}

inline void to_json(nlohmann::json& j, const CorrectionSample& sample) {
  j = nlohmann::json::object();
  j["task_id"] = sample.task_id;
  j["input"] = sample.input;
  j["failed_attempt"] = sample.failed_attempt;
  j["error"] = sample.error;
  j["correction"] = sample.correction;
}

inline void from_json(const nlohmann::json& j, CorrectionSample& sample) {
  sample.task_id = j.value("task_id", std::string{});
  sample.input = j.value("input", std::string{});
  sample.failed_attempt = j.value("failed_attempt", std::string{});
  sample.error = j.value("error", std::string{});
  sample.correction = j.value("correction", std::string{});
}

}
