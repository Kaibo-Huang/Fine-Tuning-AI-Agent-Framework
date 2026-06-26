#pragma once

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "agents_framework/core/result.hpp"

namespace agents_framework::core {

struct DotenvResult {
  std::filesystem::path path;
  std::vector<std::string> applied;
  std::vector<std::string> skipped;
};

[[nodiscard]] std::vector<std::pair<std::string, std::string>> parse_dotenv(std::string_view text);

[[nodiscard]] std::optional<std::filesystem::path> find_dotenv(
    std::string_view filename = ".env", const std::filesystem::path& start = {},
    std::size_t max_depth = 6);

[[nodiscard]] Result<DotenvResult> load_dotenv_file(const std::filesystem::path& path);

[[nodiscard]] Result<DotenvResult> load_dotenv(std::string_view filename = ".env",
                                               const std::filesystem::path& start = {});

[[nodiscard]] std::optional<std::string> get_env(std::string_view name);

void set_env(std::string_view name, std::string_view value);

void unset_env(std::string_view name);

}
