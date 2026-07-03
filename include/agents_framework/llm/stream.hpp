#pragma once

#include <functional>
#include <string>
#include <variant>

#include "agents_framework/core/result.hpp"
#include "agents_framework/llm/response.hpp"

namespace agents_framework::llm {

// sends llm response in small pieces as it's generating them

struct MessageStart {
  std::string id;
};

struct TextDelta {
  std::string text;
};

struct ToolUseStart {
  std::string id;
  std::string name;
};

struct ToolUseInputDelta {
  std::string partial_json;
};

struct MessageStop {
  StopReason reason{StopReason::EndTurn};
  Usage usage;
};

struct StreamError {
  core::Error error;
};

using StreamEvent = std::variant<MessageStart, TextDelta, ToolUseStart, ToolUseInputDelta,
                                 MessageStop, StreamError>;

using StreamCallback = std::function<void(const StreamEvent&)>;

}  
