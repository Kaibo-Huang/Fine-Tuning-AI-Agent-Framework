#include <catch2/catch_test_macros.hpp>

#include <chrono>

#include "agents_framework/core/clock.hpp"

using namespace agents_framework::core;
using namespace std::chrono_literals;

TEST_CASE("ManualClock advances by the given delta", "[clock]") {
  ManualClock clock;
  const auto start = clock.now();
  clock.advance(1500ms);
  REQUIRE(clock.now() - start == 1500ms);
}

TEST_CASE("SystemClock does not move backwards", "[clock]") {
  SystemClock clock;
  const auto first = clock.now();
  const auto second = clock.now();
  REQUIRE(second >= first);
}
