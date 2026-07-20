#pragma once

#include <concepts>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "agents_framework/core/fixed_string.hpp"
#include "agents_framework/core/result.hpp"
#include "agents_framework/graph/node.hpp"
#include "agents_framework/graph/state.hpp"
#include "agents_framework/llm/message.hpp"
#include "agents_framework/tools/registry.hpp"

namespace agents_framework::graph {

template <class S, core::FixedString MessagesChannel = "messages">
class ToolNode final : public Node {
  static_assert(std::same_as<typename S::template value_of<MessagesChannel>,
                             std::vector<llm::Message>>,
                "ToolNode requires a std::vector<llm::Message> channel");

 public:
  explicit ToolNode(std::shared_ptr<tools::ToolRegistry> registry)
      : registry_(std::move(registry)) {
    if (!registry_) throw std::invalid_argument("ToolNode requires a non-null registry");
  }

  core::Result<StateUpdate> run(const ChannelMap& state) override {
    StateView<S> view(state);
    const auto& messages = view.template get<MessagesChannel>();
    if (messages.empty()) {
      return core::fail(core::ErrorCode::Invalid, "tool node ran with no messages");
    }

    std::vector<llm::ContentBlock> results;
    for (const llm::ContentBlock& block : messages.back().content) {
      const auto* use = std::get_if<llm::ToolUseBlock>(&block);
      if (!use) continue;
      auto outcome = registry_->invoke(use->name, use->input);
      if (outcome) {
        results.push_back(llm::ToolResultBlock{use->id, std::move(*outcome), false});
      } else {
        results.push_back(llm::ToolResultBlock{use->id, outcome.error().to_string(), true});
      }
    }
    if (results.empty()) {
      return core::fail(core::ErrorCode::Invalid,
                        "tool node ran but the last message has no tool calls");
    }

    Update<S> update;
    update.template write<MessagesChannel>(
        {llm::Message{llm::Role::User, std::move(results)}});
    return std::move(update).take();
  }

 private:
  std::shared_ptr<tools::ToolRegistry> registry_;
};

template <class S, core::FixedString MessagesChannel = "messages">
[[nodiscard]] std::unique_ptr<Node> make_tool_node(std::shared_ptr<tools::ToolRegistry> registry) {
  return std::make_unique<ToolNode<S, MessagesChannel>>(std::move(registry));
}

}
