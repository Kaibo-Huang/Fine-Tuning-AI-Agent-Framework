#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>

#include "agents_framework/core/rng.hpp"

using namespace agents_framework::core;

TEST_CASE("Rng is reproducible for a fixed seed", "[rng]") {
  Rng a{12345};
  Rng b{12345};
  for (int i = 0; i < 100; ++i) {
    REQUIRE(a.next_u64() == b.next_u64());
  }
}

TEST_CASE("Rng diverges for different seeds", "[rng]") {
  Rng a{1};
  Rng b{2};
  bool any_different = false;
  for (int i = 0; i < 10; ++i) {
    if (a.next_u64() != b.next_u64()) {
      any_different = true;
    }
  }
  REQUIRE(any_different);
}

TEST_CASE("Rng::next_double stays within [0, 1)", "[rng]") {
  Rng rng{7};
  for (int i = 0; i < 1000; ++i) {
    const double value = rng.next_double();
    REQUIRE(value >= 0.0);
    REQUIRE(value < 1.0);
  }
}

TEST_CASE("Rng::fill is deterministic for a fixed seed", "[rng]") {
  Rng a{99};
  Rng b{99};
  std::array<std::byte, 20> first{};
  std::array<std::byte, 20> second{};
  a.fill(first);
  b.fill(second);
  REQUIRE(first == second);
}
