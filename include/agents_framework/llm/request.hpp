#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "agents_framework/llm/message.hpp"
#include "agents_framework/llm/tool.hpp"

namespace agents_framework::llm {

struct SamplingParams {
  double temperature{1.0};
  std::optional<double> top_p;
  std::optional<int> top_k;
  int max_tokens{1024};
  std::vector<std::string> stop;
  std::optional<std::uint64_t> seed;
  bool operator==(const SamplingParams&) const = default;
};

struct ChatRequest {
  std::string model;
  std::optional<std::string> system;
  std::vector<Message> messages;
  std::vector<ToolDef> tools;
  SamplingParams sampling;
  bool stream{false};
  bool operator==(const ChatRequest&) const = default;
};

}  
