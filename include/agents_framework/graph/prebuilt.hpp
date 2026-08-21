#pragma once

#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "agents_framework/core/result.hpp"
#include "agents_framework/graph/builder.hpp"
#include "agents_framework/graph/llm_node.hpp"
#include "agents_framework/graph/tool_node.hpp"
#include "agents_framework/llm/message.hpp"
#include "agents_framework/tools/registry.hpp"

namespace agents_framework::graph {

// The conversation channel and schema every chat-shaped agent shares. LlmNode
// and ToolNode already read a "messages" channel by convention; this names it
// once so callers stop redeclaring it. Custom graphs add their own channels
// alongside Messages; a plain chat agent needs nothing beyond ChatSchema.
using Messages = Channel<"messages", std::vector<llm::Message>, Append>;
using ChatSchema = Schema<Messages>;

// A ChatSchema state seeded with a single user question.
[[nodiscard]] inline State<ChatSchema> chat_state(std::string question) {
  State<ChatSchema> state;
  state.set<"messages">({llm::Message::user_text(std::move(question))});
  return state;
}

// The plain text of the most recent assistant message in a transcript, with
// multiple text blocks joined; empty when no assistant message exists yet.
[[nodiscard]] inline std::string last_assistant_text(
    const std::vector<llm::Message>& messages) {
  for (auto it = messages.rbegin(); it != messages.rend(); ++it) {
    if (it->role != llm::Role::Assistant) continue;
    std::string text;
    for (const llm::ContentBlock& block : it->content) {
      if (const auto* piece = std::get_if<llm::TextBlock>(&block)) text += piece->text;
    }
    return text;
  }
  return {};
}

// The standard ReAct loop as one call: an LLM node and a tool node over
// ChatSchema, routed until the model answers in plain text. The registry's
// tool definitions are advertised to the model automatically unless the
// options already name a tool list. Run the result with an Executor and a
// max_steps budget, exactly like any hand-built graph.
[[nodiscard]] inline core::Result<CompiledGraph> make_react_agent(
    std::shared_ptr<llm::LLMBackend> backend, std::shared_ptr<tools::ToolRegistry> registry,
    LlmNodeOptions options = {}) {
  if (options.tools.empty() && registry) options.tools = registry->defs();

  GraphBuilder<ChatSchema> builder;
  builder
      .add_node("agent",
                make_llm_node<ChatSchema>(std::move(backend), std::move(options)))
      .add_node("tools", make_tool_node<ChatSchema>(std::move(registry)))
      .set_entry("agent")
      .add_conditional_edge("agent", tools_router<ChatSchema>("tools"))
      .add_edge("tools", "agent");
  return std::move(builder).compile();
}

}
