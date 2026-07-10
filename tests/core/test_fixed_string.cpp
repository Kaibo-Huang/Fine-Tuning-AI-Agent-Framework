#include <catch2/catch_test_macros.hpp>

#include <string_view>

#include "agents_framework/core/fixed_string.hpp"

using agents_framework::core::FixedString;

namespace {

template <FixedString Name>
constexpr std::string_view name_of() {
  return Name.view();
}

}

TEST_CASE("FixedString exposes its text and length", "[fixed_string]") {
  constexpr FixedString name{"messages"};
  STATIC_REQUIRE(name.view() == "messages");
  STATIC_REQUIRE(name.size() == 8);
}

TEST_CASE("FixedString compares by content across lengths", "[fixed_string]") {
  STATIC_REQUIRE(FixedString{"abc"} == FixedString{"abc"});
  STATIC_REQUIRE(FixedString{"abc"} != FixedString{"abcd"});
  STATIC_REQUIRE(FixedString{"abc"} != FixedString{"ab"});
}

TEST_CASE("FixedString works as a non-type template parameter", "[fixed_string]") {
  STATIC_REQUIRE(name_of<"steps">() == "steps");
}
