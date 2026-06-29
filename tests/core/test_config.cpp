#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include "agents_framework/core/config.hpp"

using namespace agents_framework::core;

namespace {

void set_env(const char* name, const char* value) {
#ifdef _WIN32
  _putenv_s(name, value);
#else
  setenv(name, value, 1);
#endif
}

void unset_env(const char* name) {
#ifdef _WIN32
  _putenv_s(name, "");
#else
  unsetenv(name);
#endif
}

}  

TEST_CASE("Config reads typed values via dotted keys", "[config]") {
  auto config = Config::from_json(nlohmann::json::parse(R"({
    "anthropic": { "model": "claude-haiku-4-5", "max_tokens": 1024 },
    "stream": true,
    "temperature": 0.7
  })"));
  REQUIRE(config.has_value());

  REQUIRE(*config->get_string("anthropic.model") == "claude-haiku-4-5");
  REQUIRE(*config->get_int("anthropic.max_tokens") == 1024);
  REQUIRE(*config->get_bool("stream") == true);

  const double temperature = *config->get_double("temperature");
  REQUIRE(temperature > 0.69);
  REQUIRE(temperature < 0.71);
}

TEST_CASE("Config loads from a JSON file", "[config]") {
  const auto path = std::filesystem::temp_directory_path() / "agents_framework_config_test.json";
  {
    std::ofstream out{path};
    out << R"({ "name": "from-file", "retries": 3 })";
  }

  auto config = Config::from_file(path);
  REQUIRE(config.has_value());
  REQUIRE(*config->get_string("name") == "from-file");
  REQUIRE(*config->get_int("retries") == 3);

  std::filesystem::remove(path);
}

TEST_CASE("Config reports a missing file", "[config]") {
  auto config = Config::from_file("this-config-file-does-not-exist.json");
  REQUIRE_FALSE(config.has_value());
  REQUIRE(config.error().code == ErrorCode::Io);
}

TEST_CASE("Config rejects a non-object root", "[config]") {
  auto config = Config::from_json(nlohmann::json::array());
  REQUIRE_FALSE(config.has_value());
  REQUIRE(config.error().code == ErrorCode::Config);
}

TEST_CASE("Config reports missing keys", "[config]") {
  auto config = Config::from_json(nlohmann::json::object());
  REQUIRE(config.has_value());

  const auto value = config->get_string("does.not.exist");
  REQUIRE_FALSE(value.has_value());
  REQUIRE(value.error().code == ErrorCode::Config);
}

TEST_CASE("Config reports type mismatches", "[config]") {
  auto config = Config::from_json(nlohmann::json::parse(R"({ "port": "not-a-number" })"));
  REQUIRE(config.has_value());

  const auto value = config->get_int("port");
  REQUIRE_FALSE(value.has_value());
  REQUIRE(value.error().code == ErrorCode::Config);
}

TEST_CASE("Config *_or getters fall back on error", "[config]") {
  auto config = Config::from_json(nlohmann::json::object());
  REQUIRE(config->get_string_or("missing", "default") == "default");
  REQUIRE(config->get_int_or("missing", 42) == 42);
  REQUIRE(config->get_bool_or("missing", true) == true);
}

TEST_CASE("Environment variable overrides config value", "[config]") {
  set_env("AGENTS_ANTHROPIC_MODEL", "env-model");
  auto config = Config::from_json(nlohmann::json::parse(R"({
    "anthropic": { "model": "file-model" }
  })"));
  REQUIRE(*config->get_string("anthropic.model") == "env-model");
  unset_env("AGENTS_ANTHROPIC_MODEL");
}
