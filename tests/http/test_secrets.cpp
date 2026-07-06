#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <sstream>
#include <string>

#include "agents_framework/core/logging.hpp"
#include "agents_framework/http/secrets.hpp"

using namespace agents_framework;

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

TEST_CASE("SecretStore::from_env reads a set variable", "[secret]") {
  set_env("AGENTS_TEST_SECRET", "sk-test-123");
  const auto secret = http::SecretStore::from_env("AGENTS_TEST_SECRET");
  REQUIRE(secret.has_value());
  REQUIRE(secret->reveal() == "sk-test-123");
  unset_env("AGENTS_TEST_SECRET");
}

TEST_CASE("SecretStore::from_env fails when the variable is unset", "[secret]") {
  unset_env("AGENTS_TEST_SECRET_MISSING");
  const auto secret = http::SecretStore::from_env("AGENTS_TEST_SECRET_MISSING");
  REQUIRE_FALSE(secret.has_value());
  REQUIRE(secret.error().code == core::ErrorCode::Auth);
}

TEST_CASE("Secret prints as *** instead of its value", "[secret]") {
  const http::Secret secret{"top-secret-value"};
  std::ostringstream out;
  out << secret;
  REQUIRE(out.str() == "***");
}

TEST_CASE("Constructing a Secret registers it for log redaction", "[secret]") {
  const http::Secret secret{"redact-me-9f8a7b"};
  const std::string masked = core::redact("Authorization: Bearer redact-me-9f8a7b done");
  REQUIRE(masked.find("redact-me-9f8a7b") == std::string::npos);
  REQUIRE(masked.find("***") != std::string::npos);
}
