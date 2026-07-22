#pragma once

#include <concepts>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "agents_framework/core/fixed_string.hpp"
#include "agents_framework/core/result.hpp"
#include "agents_framework/graph/node.hpp"
#include "agents_framework/graph/state.hpp"
#include "agents_framework/llm/backend.hpp"
#include "agents_framework/tools/text_protocol.hpp"

namespace agents_framework::graph {

enum class ToolCallFormat { Native, TextProtocol };

struct LlmNodeOptions {
  std::string model;
  std::optional<std::string> system;
  std::vector<llm::ToolDef> tools;
  llm::SamplingParams sampling;
  ToolCallFormat tool_format{ToolCallFormat::Native};
};

template <class S, core::FixedString MessagesChannel = "messages">
class LlmNode final : public Node {
  static_assert(std::same_as<typename S::template value_of<MessagesChannel>,
                             std::vector<llm::Message>>,
                "LlmNode requires a std::vector<llm::Message> channel");

 public:
  LlmNode(std::shared_ptr<llm::LLMBackend> backend, LlmNodeOptions options)
      : backend_(std::move(backend)), options_(std::move(options)) {
    if (!backend_) throw std::invalid_argument("LlmNode requires a non-null backend");
  }

  core::Result<StateUpdate> run(const ChannelMap& state) override {
    StateView<S> view(state);
    const auto& messages = view.template get<MessagesChannel>();

    llm::ChatRequest request;
    request.model = options_.model;
    request.sampling = options_.sampling;
    if (options_.tool_format == ToolCallFormat::Native) {
      request.system = options_.system;
      request.messages = messages;
      request.tools = options_.tools;
    } else {
      std::string system = options_.system.value_or("");
      if (!options_.tools.empty()) {
        if (!system.empty()) system += "\n\n";
        system += tools::render_tool_instructions(options_.tools);
      }
      if (!system.empty()) request.system = std::move(system);
      request.messages = tools::render_text_messages(messages);
    }

    AF_TRY(auto response, backend_->generate(request));

    llm::Message message{llm::Role::Assistant, std::move(response.content)};
    if (options_.tool_format == ToolCallFormat::TextProtocol) {
      bool has_native = false;
      std::string text;
      for (const llm::ContentBlock& block : message.content) {
        if (std::holds_alternative<llm::ToolUseBlock>(block)) has_native = true;
        if (const auto* piece = std::get_if<llm::TextBlock>(&block)) text += piece->text;
      }
      if (!has_native) {
        if (const auto call = tools::parse_text_tool_call(text)) {
          message.content.push_back(
              llm::ToolUseBlock{next_call_id(), call->name, call->input});
        }
      }
    }

    Update<S> update;
    update.template write<MessagesChannel>({std::move(message)});
    return std::move(update).take();
  }

 private:
  std::string next_call_id() { return "call_" + std::to_string(++text_call_counter_); }

  std::shared_ptr<llm::LLMBackend> backend_;
  LlmNodeOptions options_;
  std::size_t text_call_counter_{0};
};

template <class S, core::FixedString MessagesChannel = "messages">
[[nodiscard]] std::unique_ptr<Node> make_llm_node(std::shared_ptr<llm::LLMBackend> backend,
                                                  LlmNodeOptions options = {}) {
  return std::make_unique<LlmNode<S, MessagesChannel>>(std::move(backend), std::move(options));
}

}
