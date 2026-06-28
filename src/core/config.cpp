#include "agents_framework/core/config.hpp"

#include <fstream>

namespace agents_framework::core {

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

}  // namespace agents_framework::core
