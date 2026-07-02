#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <string>

#include "agents_framework/core/dotenv.hpp"

using namespace agents_framework;

namespace {

std::filesystem::path write_temp_env(const std::string& name, std::string_view contents) {
  const auto dir = std::filesystem::temp_directory_path() / "af_dotenv_tests";
  std::filesystem::create_directories(dir);
  const auto path = dir / name;
  std::ofstream file{path, std::ios::binary | std::ios::trunc};
  file << contents;
  return path;
}

}

TEST_CASE("parse_dotenv reads key/value pairs", "[core][dotenv]") {
  const auto pairs = core::parse_dotenv(
      "# a comment\n"
      "\n"
      "PLAIN=value\n"
      "  SPACED   =   padded  \n"
      "export EXPORTED=shell-style\n"
      "QUOTED=\"quoted value\"\n"
      "SINGLE='single value'\n"
      "TRAILING=value # trailing comment\n"
      "HASH_IN_QUOTES=\"keeps # inside\"\n");

  REQUIRE(pairs.size() == 7);
  CHECK(pairs[0] == std::pair<std::string, std::string>{"PLAIN", "value"});
  CHECK(pairs[1] == std::pair<std::string, std::string>{"SPACED", "padded"});
  CHECK(pairs[2] == std::pair<std::string, std::string>{"EXPORTED", "shell-style"});
  CHECK(pairs[3] == std::pair<std::string, std::string>{"QUOTED", "quoted value"});
  CHECK(pairs[4] == std::pair<std::string, std::string>{"SINGLE", "single value"});
  CHECK(pairs[5] == std::pair<std::string, std::string>{"TRAILING", "value"});
  CHECK(pairs[6] == std::pair<std::string, std::string>{"HASH_IN_QUOTES", "keeps # inside"});
}

TEST_CASE("parse_dotenv skips malformed lines", "[core][dotenv]") {
  const auto pairs = core::parse_dotenv(
      "no equals sign here\n"
      "1INVALID=leading digit\n"
      "has-dash=not a valid identifier\n"
      "GOOD=kept\n");
  REQUIRE(pairs.size() == 1);
  CHECK(pairs[0].first == "GOOD");
}

TEST_CASE("load_dotenv_file does not clobber the real environment", "[core][dotenv]") {
  core::set_env("AF_TEST_PRESET", "from-shell");
  core::unset_env("AF_TEST_FRESH");

  const auto path = write_temp_env("preset.env",
                                   "AF_TEST_PRESET=from-file\n"
                                   "AF_TEST_FRESH=from-file\n");
  const auto loaded = core::load_dotenv_file(path);
  REQUIRE(loaded.has_value());

  CHECK(core::get_env("AF_TEST_PRESET") == "from-shell");
  CHECK(core::get_env("AF_TEST_FRESH") == "from-file");
  CHECK(loaded->applied == std::vector<std::string>{"AF_TEST_FRESH"});
  CHECK(loaded->skipped == std::vector<std::string>{"AF_TEST_PRESET"});

  core::unset_env("AF_TEST_PRESET");
  core::unset_env("AF_TEST_FRESH");
}

TEST_CASE("load_dotenv_file reports a missing file", "[core][dotenv]") {
  const auto missing = core::load_dotenv_file("definitely-not-here.env");
  REQUIRE_FALSE(missing.has_value());
  CHECK(missing.error().code == core::ErrorCode::Io);
}

TEST_CASE("find_dotenv walks up toward the repository root", "[core][dotenv]") {
  const auto dir = std::filesystem::temp_directory_path() / "af_dotenv_tests" / "nested" / "deep";
  std::filesystem::create_directories(dir);
  const auto root = write_temp_env("walk.env", "AF_WALK=1\n");

  const auto found = core::find_dotenv("walk.env", dir);
  REQUIRE(found.has_value());
  CHECK(std::filesystem::equivalent(*found, root));
}

TEST_CASE("get_env treats an empty variable as unset", "[core][dotenv]") {
  core::set_env("AF_TEST_EMPTY", "");
  CHECK_FALSE(core::get_env("AF_TEST_EMPTY").has_value());
  core::unset_env("AF_TEST_EMPTY");
}
