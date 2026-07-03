#pragma once

#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "agents_framework/llm/message.hpp"

namespace agents_framework::llm {

// Why llm calls stop
enum class StopReason { EndTurn, MaxTokens, StopSequence, ToolUse, Error };

[[nodiscard]] inline std::string_view stop_reason_name(StopReason reason) noexcept {
  switch (reason) {
    case StopReason::EndTurn:      return "end_turn";
    case StopReason::MaxTokens:    return "max_tokens";
    case StopReason::StopSequence: return "stop_sequence";
    case StopReason::ToolUse:      return "tool_use";
    case StopReason::Error:        return "error";
  }
  return "end_turn";
}

[[nodiscard]] inline StopReason stop_reason_from_string(std::string_view text) noexcept {
  if (text == "max_tokens") return StopReason::MaxTokens;
  if (text == "stop_sequence") return StopReason::StopSequence;
  if (text == "tool_use") return StopReason::ToolUse;
  if (text == "error") return StopReason::Error;
  return StopReason::EndTurn;
}

struct Usage {
  int input_tokens{0};
  int output_tokens{0};
  bool operator==(const Usage&) const = default;
};

struct ChatResponse {
  std::string id;
  Role role{Role::Assistant};
  std::vector<ContentBlock> content;
  StopReason stop_reason{StopReason::EndTurn};
  Usage usage;

  [[nodiscard]] std::string text() const {
    std::string out;
    for (const auto& block : content) {
      if (const auto* text = std::get_if<TextBlock>(&block)) {
        out += text->text;
      }
    }
    return out;
  }

  [[nodiscard]] std::vector<ToolUseBlock> tool_uses() const {
    std::vector<ToolUseBlock> uses;
    for (const auto& block : content) {
      if (const auto* use = std::get_if<ToolUseBlock>(&block)) {
        uses.push_back(*use);
      }
    }
    return uses;
  }

  bool operator==(const ChatResponse&) const = default;
};

inline void to_json(nlohmann::json& j, const Usage& usage) {
  j = nlohmann::json::object();
  j["input_tokens"] = usage.input_tokens;
  j["output_tokens"] = usage.output_tokens;
}

inline void from_json(const nlohmann::json& j, Usage& usage) {
  usage.input_tokens = j.value("input_tokens", 0);
  usage.output_tokens = j.value("output_tokens", 0);
}

inline void to_json(nlohmann::json& j, const ChatResponse& response) {
  j = nlohmann::json::object();
  j["id"] = response.id;
  j["role"] = std::string{role_name(response.role)};
  j["content"] = response.content;
  j["stop_reason"] = std::string{stop_reason_name(response.stop_reason)};
  j["usage"] = response.usage;
}

inline void from_json(const nlohmann::json& j, ChatResponse& response) {
  response.id = j.value("id", std::string{});
  response.role = role_from_string(j.value("role", std::string{"assistant"}));
  response.content = j.at("content").get<std::vector<ContentBlock>>();
  response.stop_reason = stop_reason_from_string(j.value("stop_reason", std::string{"end_turn"}));
  response.usage = j.value("usage", Usage{});
}

}  // namespace agents_framework::llm
