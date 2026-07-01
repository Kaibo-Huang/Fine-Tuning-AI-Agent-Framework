#pragma once

#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

namespace agents_framework::llm {
enum class Role { System, User, Assistant, Tool };

[[nodiscard]] inline std::string_view role_name(Role role) noexcept {
  switch (role) {
    case Role::System:    return "system";
    case Role::User:      return "user";
    case Role::Assistant: return "assistant";
    case Role::Tool:      return "tool";
  }
  return "user";
}

[[nodiscard]] inline Role role_from_string(std::string_view text) noexcept {
  if (text == "system") return Role::System;
  if (text == "assistant") return Role::Assistant;
  if (text == "tool") return Role::Tool;
  return Role::User;
}

struct TextBlock {
  std::string text;
  bool operator==(const TextBlock&) const = default;
};

struct ToolUseBlock {
  std::string id;
  std::string name;
  nlohmann::json input;
  bool operator==(const ToolUseBlock&) const = default;
};

struct ToolResultBlock {
  std::string tool_use_id;
  std::string content;
  bool is_error{false};
  bool operator==(const ToolResultBlock&) const = default;
};

using ContentBlock = std::variant<TextBlock, ToolUseBlock, ToolResultBlock>;

struct Message {
  Role role{Role::User};
  std::vector<ContentBlock> content;

  static Message system_text(std::string text) {
    return Message{Role::System, {TextBlock{std::move(text)}}};
  }
  static Message user_text(std::string text) {
    return Message{Role::User, {TextBlock{std::move(text)}}};
  }
  static Message assistant_text(std::string text) {
    return Message{Role::Assistant, {TextBlock{std::move(text)}}};
  }

  bool operator==(const Message&) const = default;
};

inline void to_json(nlohmann::json& j, const TextBlock& block) {
  j = nlohmann::json::object();
  j["type"] = "text";
  j["text"] = block.text;
}

inline void from_json(const nlohmann::json& j, TextBlock& block) {
  block.text = j.at("text").get<std::string>();
}

inline void to_json(nlohmann::json& j, const ToolUseBlock& block) {
  j = nlohmann::json::object();
  j["type"] = "tool_use";
  j["id"] = block.id;
  j["name"] = block.name;
  j["input"] = block.input;
}

inline void from_json(const nlohmann::json& j, ToolUseBlock& block) {
  block.id = j.at("id").get<std::string>();
  block.name = j.at("name").get<std::string>();
  block.input = j.value("input", nlohmann::json::object());
}

inline void to_json(nlohmann::json& j, const ToolResultBlock& block) {
  j = nlohmann::json::object();
  j["type"] = "tool_result";
  j["tool_use_id"] = block.tool_use_id;
  j["content"] = block.content;
  j["is_error"] = block.is_error;
}

inline void from_json(const nlohmann::json& j, ToolResultBlock& block) {
  block.tool_use_id = j.at("tool_use_id").get<std::string>();
  block.content = j.value("content", std::string{});
  block.is_error = j.value("is_error", false);
}

inline void to_json(nlohmann::json& j, const ContentBlock& block) {
  std::visit([&j](const auto& value) { to_json(j, value); }, block);
}

inline void from_json(const nlohmann::json& j, ContentBlock& block) {
  const auto type = j.at("type").get<std::string>();
  if (type == "text") {
    block = j.get<TextBlock>();
  } else if (type == "tool_use") {
    block = j.get<ToolUseBlock>();
  } else if (type == "tool_result") {
    block = j.get<ToolResultBlock>();
  } else {
    throw std::invalid_argument("unknown content block type: " + type);
  }
}

inline void to_json(nlohmann::json& j, const Message& message) {
  j = nlohmann::json::object();
  j["role"] = std::string{role_name(message.role)};
  j["content"] = message.content;
}

inline void from_json(const nlohmann::json& j, Message& message) {
  message.role = role_from_string(j.at("role").get<std::string>());
  message.content = j.at("content").get<std::vector<ContentBlock>>();
}

}  
