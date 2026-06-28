#include "agents_framework/core/config.hpp"

#include <fstream>

namespace agents_framework::core {
namespace {

const nlohmann::json* find_key(const nlohmann::json& root, std::string_view key) {
  const nlohmann::json* node = &root;
  std::size_t pos = 0;
  while (true) {
    const std::size_t dot = key.find('.', pos);
    const std::string_view segment =
        (dot == std::string_view::npos) ? key.substr(pos) : key.substr(pos, dot - pos);
    if (!node->is_object()) {
      return nullptr;
    }
    const auto it = node->find(std::string{segment});
    if (it == node->end()) {
      return nullptr;
    }
    node = &*it;
    if (dot == std::string_view::npos) {
      return node;
    }
    pos = dot + 1;
  }
}

}  // namespace

Result<Config> Config::from_file(const std::filesystem::path& path) {
  std::ifstream in{path};
  if (!in) {
    return fail(ErrorCode::Io, "cannot open config file", path.string());
  }
  nlohmann::json data;
  try {
    in >> data;
  } catch (const nlohmann::json::exception& e) {
    return fail(ErrorCode::Parse, "invalid JSON in config file", e.what());
  }
  return from_json(std::move(data));
}

Result<Config> Config::from_json(nlohmann::json data) {
  if (!data.is_object()) {
    return fail(ErrorCode::Config, "config root must be a JSON object");
  }
  return Config{std::move(data)};
}

Result<std::string> Config::get_string(std::string_view key) const {
  const nlohmann::json* node = find_key(data_, key);
  if (!node) {
    return fail(ErrorCode::Config, "missing config key", std::string{key});
  }
  if (!node->is_string()) {
    return fail(ErrorCode::Config, "config value is not a string", std::string{key});
  }
  return node->get<std::string>();
}

Result<std::int64_t> Config::get_int(std::string_view key) const {
  const nlohmann::json* node = find_key(data_, key);
  if (!node) {
    return fail(ErrorCode::Config, "missing config key", std::string{key});
  }
  if (!node->is_number_integer()) {
    return fail(ErrorCode::Config, "config value is not an integer", std::string{key});
  }
  return node->get<std::int64_t>();
}

Result<double> Config::get_double(std::string_view key) const {
  const nlohmann::json* node = find_key(data_, key);
  if (!node) {
    return fail(ErrorCode::Config, "missing config key", std::string{key});
  }
  if (!node->is_number()) {
    return fail(ErrorCode::Config, "config value is not a number", std::string{key});
  }
  return node->get<double>();
}

Result<bool> Config::get_bool(std::string_view key) const {
  const nlohmann::json* node = find_key(data_, key);
  if (!node) {
    return fail(ErrorCode::Config, "missing config key", std::string{key});
  }
  if (!node->is_boolean()) {
    return fail(ErrorCode::Config, "config value is not a boolean", std::string{key});
  }
  return node->get<bool>();
}

std::string Config::get_string_or(std::string_view key, std::string fallback) const {
  auto value = get_string(key);
  return value ? std::move(*value) : std::move(fallback);
}

std::int64_t Config::get_int_or(std::string_view key, std::int64_t fallback) const {
  auto value = get_int(key);
  return value ? *value : fallback;
}

double Config::get_double_or(std::string_view key, double fallback) const {
  auto value = get_double(key);
  return value ? *value : fallback;
}

bool Config::get_bool_or(std::string_view key, bool fallback) const {
  auto value = get_bool(key);
  return value ? *value : fallback;
}

}  // namespace agents_framework::core
