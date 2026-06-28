#pragma once

#include <filesystem>

#include <nlohmann/json.hpp>

#include "agents_framework/core/result.hpp"

namespace agents_framework::core {

class Config {
 public:
  static Result<Config> from_file(const std::filesystem::path& path);
  static Result<Config> from_json(nlohmann::json data);

  [[nodiscard]] const nlohmann::json& data() const noexcept { return data_; }

 private:
  explicit Config(nlohmann::json data) : data_(std::move(data)) {}

  nlohmann::json data_;
};

}  // namespace agents_framework::core
