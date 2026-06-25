#include <catch2/catch_test_macros.hpp>
#include "agents_framework/version.hpp"

TEST_CASE("version string is non-empty", "[version]") {
  REQUIRE_FALSE(agents_framework::version().empty());
}
