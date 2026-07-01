#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <set>
#include <string>

#include "agents_framework/core/clock.hpp"
#include "agents_framework/core/id.hpp"
#include "agents_framework/core/rng.hpp"

using namespace agents_framework::core;
using namespace std::chrono_literals;

TEST_CASE("Ulid round-trips through its string form", "[ulid]") {
  SystemClock clock;
  Rng rng{99};

  const Ulid id = Ulid::generate(clock, rng);
  const std::string text = id.to_string();
  REQUIRE(text.size() == 26);

  const auto parsed = Ulid::parse(text);
  REQUIRE(parsed.has_value());
  REQUIRE(*parsed == id);
  REQUIRE(parsed->to_string() == text);
}

TEST_CASE("Ulid::parse rejects malformed input", "[ulid]") {
  REQUIRE_FALSE(Ulid::parse("too-short").has_value());
  REQUIRE_FALSE(Ulid::parse(std::string(26, '!')).has_value());
  REQUIRE(Ulid::parse(std::string(26, '!')).error().code == ErrorCode::Parse);
}

TEST_CASE("Ulid is lexicographically sortable by time", "[ulid]") {
  ManualClock clock;
  Rng rng{5};

  const Ulid earlier = Ulid::generate(clock, rng);
  clock.advance(1ms);
  const Ulid later = Ulid::generate(clock, rng);

  REQUIRE(earlier < later);
  REQUIRE(earlier.to_string() < later.to_string());
}

TEST_CASE("Ulid::generate produces unique values", "[ulid]") {
  SystemClock clock;
  Rng rng{2024};

  std::set<std::string> seen;
  for (int i = 0; i < 10000; ++i) {
    REQUIRE(seen.insert(Ulid::generate(clock, rng).to_string()).second);
  }
}
