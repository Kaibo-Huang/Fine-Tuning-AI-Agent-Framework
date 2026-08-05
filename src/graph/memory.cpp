#include "agents_framework/graph/memory.hpp"

#include <algorithm>
#include <variant>

namespace agents_framework::graph {

std::vector<llm::Message> window_messages(const std::vector<llm::Message>& messages,
                                          std::size_t max_messages) {
  if (messages.size() <= max_messages) return messages;

  std::size_t start = messages.size() - max_messages;
  const auto has_tool_result = [](const llm::Message& message) {
    return std::any_of(message.content.begin(), message.content.end(),
                       [](const llm::ContentBlock& block) {
                         return std::holds_alternative<llm::ToolResultBlock>(block);
                       });
  };
  while (start < messages.size() && has_tool_result(messages[start])) {
    ++start;
  }
  return {messages.begin() + static_cast<std::ptrdiff_t>(start), messages.end()};
}

}
