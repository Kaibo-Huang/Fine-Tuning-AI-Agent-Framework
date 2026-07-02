#include "agents_framework/core/dotenv.hpp"

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <system_error>
#include <utility>

namespace agents_framework::core {
namespace {

constexpr std::string_view kWhitespace = " \t\r\n";

std::string_view trim(std::string_view text) {
  const auto begin = text.find_first_not_of(kWhitespace);
  if (begin == std::string_view::npos) return {};
  const auto end = text.find_last_not_of(kWhitespace);
  return text.substr(begin, end - begin + 1);
}

std::string_view strip_quotes(std::string_view value) {
  if (value.size() >= 2 && (value.front() == '"' || value.front() == '\'') &&
      value.back() == value.front()) {
    return value.substr(1, value.size() - 2);
  }
  return value;
}

bool valid_key(std::string_view key) {
  if (key.empty()) return false;
  if (std::isdigit(static_cast<unsigned char>(key.front())) != 0) return false;
  for (const char c : key) {
    const bool ok = std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_';
    if (!ok) return false;
  }
  return true;
}

}

std::vector<std::pair<std::string, std::string>> parse_dotenv(std::string_view text) {
  std::vector<std::pair<std::string, std::string>> out;
  std::istringstream stream{std::string{text}};
  std::string line;

  while (std::getline(stream, line)) {
    std::string_view view = trim(line);
    if (view.empty() || view.front() == '#') continue;

    if (view.starts_with("export ")) view = trim(view.substr(7));

    const auto equals = view.find('=');
    if (equals == std::string_view::npos) continue;

    const std::string_view key = trim(view.substr(0, equals));
    if (!valid_key(key)) continue;

    std::string_view value = trim(view.substr(equals + 1));
    const bool quoted = value.size() >= 2 && (value.front() == '"' || value.front() == '\'');
    if (!quoted) {
      const auto comment = value.find(" #");
      if (comment != std::string_view::npos) value = trim(value.substr(0, comment));
    }
    out.emplace_back(std::string{key}, std::string{strip_quotes(value)});
  }
  return out;
}

std::optional<std::filesystem::path> find_dotenv(std::string_view filename,
                                                 const std::filesystem::path& start,
                                                 std::size_t max_depth) {
  std::error_code ec;
  std::filesystem::path dir = start.empty() ? std::filesystem::current_path(ec) : start;
  if (ec) return std::nullopt;

  for (std::size_t depth = 0; depth <= max_depth; ++depth) {
    const std::filesystem::path candidate = dir / filename;
    if (std::filesystem::exists(candidate, ec) && !ec) return candidate;
    if (!dir.has_parent_path() || dir.parent_path() == dir) break;
    dir = dir.parent_path();
  }
  return std::nullopt;
}

Result<DotenvResult> load_dotenv_file(const std::filesystem::path& path) {
  std::ifstream file{path, std::ios::binary};
  if (!file) {
    return fail(ErrorCode::Io, "could not open dotenv file", path.string());
  }
  std::ostringstream buffer;
  buffer << file.rdbuf();

  DotenvResult result;
  result.path = path;
  for (auto& [key, value] : parse_dotenv(buffer.str())) {
    if (get_env(key).has_value()) {
      result.skipped.push_back(key);
      continue;
    }
    set_env(key, value);
    result.applied.push_back(key);
  }
  return result;
}

Result<DotenvResult> load_dotenv(std::string_view filename, const std::filesystem::path& start) {
  const auto path = find_dotenv(filename, start);
  if (!path) {
    return fail(ErrorCode::NotFound, "no dotenv file found", std::string{filename});
  }
  return load_dotenv_file(*path);
}

std::optional<std::string> get_env(std::string_view name) {
  const std::string key{name};
#ifdef _MSC_VER
  char* buffer = nullptr;
  std::size_t size = 0;
  if (_dupenv_s(&buffer, &size, key.c_str()) != 0 || buffer == nullptr) {
    return std::nullopt;
  }
  std::string value{buffer};
  std::free(buffer);
  if (value.empty()) return std::nullopt;
  return value;
#else
  const char* value = std::getenv(key.c_str());
  if (value == nullptr || *value == '\0') return std::nullopt;
  return std::string{value};
#endif
}

void set_env(std::string_view name, std::string_view value) {
  const std::string key{name};
  const std::string val{value};
#ifdef _WIN32
  _putenv_s(key.c_str(), val.c_str());
#else
  setenv(key.c_str(), val.c_str(), 1);
#endif
}

void unset_env(std::string_view name) {
  const std::string key{name};
#ifdef _WIN32
  _putenv_s(key.c_str(), "");
#else
  unsetenv(key.c_str());
#endif
}

}
