#pragma once

#include <optional>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

namespace agents_framework::tools {

struct ParsedToolCall {
  std::string name;
  nlohmann::json input;
  bool operator==(const ParsedToolCall&) const = default;
};

[[nodiscard]] std::optional<ParsedToolCall> parse_text_tool_call(std::string_view text);

}
