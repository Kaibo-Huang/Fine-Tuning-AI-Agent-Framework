#include "agents_framework/tools/text_protocol.hpp"

#include <array>
#include <cctype>
#include <utility>

namespace agents_framework::tools {

namespace {

std::string_view trim(std::string_view text) {
  while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front()))) {
    text.remove_prefix(1);
  }
  while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back()))) {
    text.remove_suffix(1);
  }
  return text;
}

std::string_view strip_wrapping(std::string_view text) {
  text = trim(text);
  while (text.size() >= 2 &&
         ((text.front() == '"' && text.back() == '"') ||
          (text.front() == '\'' && text.back() == '\'') ||
          (text.front() == '`' && text.back() == '`'))) {
    text = trim(text.substr(1, text.size() - 2));
  }
  return text;
}

bool starts_with_insensitive(std::string_view text, std::string_view prefix) {
  if (text.size() < prefix.size()) return false;
  for (std::size_t i = 0; i < prefix.size(); ++i) {
    if (std::tolower(static_cast<unsigned char>(text[i])) !=
        std::tolower(static_cast<unsigned char>(prefix[i]))) {
      return false;
    }
  }
  return true;
}

std::optional<nlohmann::json> first_json_object(std::string_view text) {
  for (std::size_t start = text.find('{'); start != std::string_view::npos;
       start = text.find('{', start + 1)) {
    int depth = 0;
    bool in_string = false;
    bool escaped = false;
    for (std::size_t i = start; i < text.size(); ++i) {
      const char ch = text[i];
      if (in_string) {
        if (escaped) {
          escaped = false;
        } else if (ch == '\\') {
          escaped = true;
        } else if (ch == '"') {
          in_string = false;
        }
        continue;
      }
      if (ch == '"') {
        in_string = true;
      } else if (ch == '{') {
        ++depth;
      } else if (ch == '}') {
        if (--depth == 0) {
          auto candidate =
              nlohmann::json::parse(text.substr(start, i - start + 1), nullptr, false);
          if (candidate.is_object()) {
            return std::optional<nlohmann::json>{std::move(candidate)};
          }
          break;
        }
      }
    }
  }
  return std::nullopt;
}

nlohmann::json coerce_input(const nlohmann::json& value) {
  if (value.is_object()) return value;
  if (value.is_string()) {
    const auto nested = nlohmann::json::parse(value.get<std::string>(), nullptr, false);
    if (nested.is_object()) return nested;
    return nlohmann::json{{"input", value.get<std::string>()}};
  }
  return nlohmann::json::object();
}

std::optional<ParsedToolCall> extract_call(const nlohmann::json& j) {
  if (!j.is_object()) return std::nullopt;

  if (const auto function = j.find("function");
      function != j.end() && function->is_object()) {
    if (auto nested = extract_call(*function)) return nested;
  }

  static constexpr std::array<std::string_view, 3> kToolKeys{"tool", "tool_name", "action"};
  static constexpr std::array<std::string_view, 6> kInputKeys{
      "input", "args", "arguments", "parameters", "action_input", "tool_input"};

  std::string name;
  bool explicit_tool_key = false;
  for (const std::string_view key : kToolKeys) {
    if (const auto found = j.find(key); found != j.end() && found->is_string()) {
      name = found->get<std::string>();
      explicit_tool_key = true;
      break;
    }
  }
  if (name.empty()) {
    if (const auto found = j.find("name"); found != j.end() && found->is_string()) {
      name = found->get<std::string>();
    }
  }
  if (name.empty()) return std::nullopt;

  const nlohmann::json* input = nullptr;
  for (const std::string_view key : kInputKeys) {
    if (const auto found = j.find(key); found != j.end()) {
      input = &*found;
      break;
    }
  }
  if (!explicit_tool_key && input == nullptr) return std::nullopt;

  return ParsedToolCall{std::move(name),
                        input != nullptr ? coerce_input(*input) : nlohmann::json::object()};
}

std::optional<ParsedToolCall> parse_fenced_blocks(std::string_view text) {
  for (std::size_t fence = text.find("```"); fence != std::string_view::npos;
       fence = text.find("```", fence + 3)) {
    std::size_t body = fence + 3;
    const std::size_t newline = text.find('\n', body);
    const std::size_t closing = text.find("```", body);
    if (closing == std::string_view::npos) break;
    if (newline != std::string_view::npos && newline < closing) {
      const std::string_view tag = trim(text.substr(body, newline - body));
      if (tag.empty() || tag == "json") body = newline + 1;
    }
    const std::string_view content = trim(text.substr(body, closing - body));
    if (const auto object = first_json_object(content)) {
      if (auto call = extract_call(*object)) return call;
    }
    fence = closing;
  }
  return std::nullopt;
}

std::optional<ParsedToolCall> parse_action_lines(std::string_view text) {
  std::size_t offset = 0;
  std::string name;
  while (offset < text.size()) {
    std::size_t end = text.find('\n', offset);
    if (end == std::string_view::npos) end = text.size();
    const std::string_view line = trim(text.substr(offset, end - offset));
    if (starts_with_insensitive(line, "action input:") ||
        starts_with_insensitive(line, "action_input:")) {
      if (name.empty()) return std::nullopt;
      std::string_view rest = text.substr(offset);
      rest.remove_prefix(rest.find(':') + 1);
      std::size_t stop = rest.size();
      std::size_t line_start = 0;
      while (line_start < rest.size()) {
        std::size_t line_end = rest.find('\n', line_start);
        if (line_end == std::string_view::npos) line_end = rest.size();
        if (starts_with_insensitive(trim(rest.substr(line_start, line_end - line_start)),
                                    "observation")) {
          stop = line_start;
          break;
        }
        line_start = line_end + 1;
      }
      const std::string_view raw = rest.substr(0, stop);
      if (const auto object = first_json_object(raw)) {
        return ParsedToolCall{std::move(name), *object};
      }
      const std::string_view value = strip_wrapping(raw);
      nlohmann::json input = nlohmann::json::object();
      if (!value.empty()) input = nlohmann::json{{"input", std::string{value}}};
      return ParsedToolCall{std::move(name), std::move(input)};
    }
    if (starts_with_insensitive(line, "action:")) {
      name = std::string{strip_wrapping(line.substr(line.find(':') + 1))};
    }
    offset = end + 1;
  }
  return std::nullopt;
}

}

std::optional<ParsedToolCall> parse_text_tool_call(std::string_view text) {
  if (auto call = parse_fenced_blocks(text)) return call;
  if (auto call = parse_action_lines(text)) return call;
  if (const auto object = first_json_object(text)) {
    if (auto call = extract_call(*object)) return call;
  }
  return std::nullopt;
}

}
