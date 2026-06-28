#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

#include "agents_framework/core/result.hpp"

namespace agents_framework::core {

class Config {
 public:
  static Result<Config> from_file(const std::filesystem::path& path);
  static Result<Config> from_json(nlohmann::json data);

  [[nodiscard]] Result<std::string> get_string(std::string_view key) const;
  [[nodiscard]] Result<std::int64_t> get_int(std::string_view key) const;
  [[nodiscard]] Result<double> get_double(std::string_view key) const;
  [[nodiscard]] Result<bool> get_bool(std::string_view key) const;

  [[nodiscard]] std::string get_string_or(std::string_view key, std::string fallback) const;
  [[nodiscard]] std::int64_t get_int_or(std::string_view key, std::int64_t fallback) const;
  [[nodiscard]] double get_double_or(std::string_view key, double fallback) const;
  [[nodiscard]] bool get_bool_or(std::string_view key, bool fallback) const;

  [[nodiscard]] const nlohmann::json& data() const noexcept { return data_; }

 private:
  explicit Config(nlohmann::json data) : data_(std::move(data)) {}

  nlohmann::json data_;
};

}  
