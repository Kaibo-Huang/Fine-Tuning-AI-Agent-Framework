#pragma once

#include <iterator>
#include <vector>

#include <nlohmann/json.hpp>

#include "agents_framework/core/fixed_string.hpp"
#include "agents_framework/graph/channel.hpp"
#include "agents_framework/trace/example.hpp"

namespace agents_framework::trace {

struct TrainingData {
  std::vector<TrainingSample> samples;
  std::vector<CorrectionSample> corrections;

  [[nodiscard]] bool empty() const noexcept { return samples.empty() && corrections.empty(); }
  [[nodiscard]] std::size_t size() const noexcept { return samples.size() + corrections.size(); }

  bool operator==(const TrainingData&) const = default;
};

struct MergeTrainingData {
  static void apply(TrainingData& current, TrainingData update) {
    current.samples.insert(current.samples.end(),
                           std::make_move_iterator(update.samples.begin()),
                           std::make_move_iterator(update.samples.end()));
    current.corrections.insert(current.corrections.end(),
                               std::make_move_iterator(update.corrections.begin()),
                               std::make_move_iterator(update.corrections.end()));
  }
};

inline void to_json(nlohmann::json& j, const TrainingData& data) {
  j = nlohmann::json::object();
  j["samples"] = data.samples;
  j["corrections"] = data.corrections;
}

inline void from_json(const nlohmann::json& j, TrainingData& data) {
  data.samples = j.value("samples", std::vector<TrainingSample>{});
  data.corrections = j.value("corrections", std::vector<CorrectionSample>{});
}

template <core::FixedString Name = "training_data">
using TrainingDataChannel = graph::Channel<Name, TrainingData, MergeTrainingData>;

}
