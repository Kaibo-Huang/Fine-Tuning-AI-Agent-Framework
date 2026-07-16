#pragma once

#include <concepts>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "agents_framework/core/fixed_string.hpp"
#include "agents_framework/core/result.hpp"
#include "agents_framework/graph/node.hpp"
#include "agents_framework/graph/state.hpp"
#include "agents_framework/llm/backend.hpp"

namespace agents_framework::graph {

struct LlmNodeOptions {
  std::string model;
  std::optional<std::string> system;
  std::vector<llm::ToolDef> tools;
  llm::SamplingParams sampling;
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
    request.system = options_.system;
    request.messages = messages;
    request.tools = options_.tools;

    AF_TRY(auto response, backend_->generate(request));

    Update<S> update;
    update.template write<MessagesChannel>(
        {llm::Message{llm::Role::Assistant, std::move(response.content)}});
    return std::move(update).take();
  }

 private:
  std::shared_ptr<llm::LLMBackend> backend_;
  LlmNodeOptions options_;
};

template <class S, core::FixedString MessagesChannel = "messages">
[[nodiscard]] std::unique_ptr<Node> make_llm_node(std::shared_ptr<llm::LLMBackend> backend,
                                                  LlmNodeOptions options = {}) {
  return std::make_unique<LlmNode<S, MessagesChannel>>(std::move(backend), std::move(options));
}

}
