#pragma once

#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "agents_framework/llm/message.hpp"
#include "agents_framework/llm/tool.hpp"

namespace agents_framework::tools {

struct ParsedToolCall {
  std::string name;
  nlohmann::json input;
  bool operator==(const ParsedToolCall&) const = default;
};

[[nodiscard]] std::optional<ParsedToolCall> parse_text_tool_call(std::string_view text);

[[nodiscard]] std::string render_tool_instructions(std::span<const llm::ToolDef> tools);

[[nodiscard]] std::vector<llm::Message> render_text_messages(
    const std::vector<llm::Message>& messages);

}
