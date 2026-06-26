#include <catch2/catch_test_macros.hpp>

#include <string>

#include "agents_framework/core/result.hpp"

using namespace agents_framework::core;

namespace {

Result<int> parse_positive(int x) {
  if (x < 0) {
    return fail(ErrorCode::Invalid, "value must be positive", "x");
  }
  return x;
}

Result<int> doubled_if_positive(int x) {
  AF_TRY(auto value, parse_positive(x));
  return value * 2;
}

Result<void> require_positive(int x) {
  AF_TRY_VOID(parse_positive(x));
  return {};
}

}  // namespace

TEST_CASE("Result carries either a value or an error", "[result]") {
  const auto ok = parse_positive(5);
  REQUIRE(ok.has_value());
  REQUIRE(*ok == 5);

  const auto err = parse_positive(-1);
  REQUIRE_FALSE(err.has_value());
  REQUIRE(err.error().code == ErrorCode::Invalid);
}

TEST_CASE("AF_TRY unwraps values and propagates errors", "[result]") {
  REQUIRE(*doubled_if_positive(4) == 8);

  const auto err = doubled_if_positive(-2);
  REQUIRE_FALSE(err.has_value());
  REQUIRE(err.error().code == ErrorCode::Invalid);
}

TEST_CASE("AF_TRY_VOID propagates errors from void results", "[result]") {
  REQUIRE(require_positive(1).has_value());

  const auto err = require_positive(-1);
  REQUIRE_FALSE(err.has_value());
  REQUIRE(err.error().code == ErrorCode::Invalid);
}

TEST_CASE("Error::to_string includes code, message, and context", "[result]") {
  const Error error{ErrorCode::NotFound, "missing thing", "key=foo"};
  const std::string text = error.to_string();
  REQUIRE(text.find("NotFound") != std::string::npos);
  REQUIRE(text.find("missing thing") != std::string::npos);
  REQUIRE(text.find("key=foo") != std::string::npos);
}
