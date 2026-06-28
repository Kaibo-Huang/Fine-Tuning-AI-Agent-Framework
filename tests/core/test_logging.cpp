#include <catch2/catch_test_macros.hpp>

#include <string>

#include "agents_framework/core/logging.hpp"

using namespace agents_framework::core;

TEST_CASE("redact masks registered secrets", "[logging]") {
  add_redaction("super-secret-token-xyz");
  const std::string masked = redact("authorization: super-secret-token-xyz end");
  REQUIRE(masked.find("super-secret-token-xyz") == std::string::npos);
  REQUIRE(masked.find("***") != std::string::npos);
}

TEST_CASE("redact leaves unrelated text untouched", "[logging]") {
  REQUIRE(redact("nothing to hide here") == "nothing to hide here");
}

TEST_CASE("logging API runs without throwing", "[logging]") {
  init_logging(LogLevel::Warn);
  REQUIRE_NOTHROW(log(LogLevel::Info, "ignored_below_threshold"));
  REQUIRE_NOTHROW(log(LogLevel::Warn, "request", {{"model", "claude-haiku-4-5"}}));
}
