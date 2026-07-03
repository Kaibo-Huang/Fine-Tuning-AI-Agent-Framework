#pragma once

#include <string>

#include "agents_framework/core/result.hpp"
#include "agents_framework/llm/request.hpp"
#include "agents_framework/llm/response.hpp"
#include "agents_framework/llm/stream.hpp"

namespace agents_framework::llm {

struct BackendCapabilities {
  bool native_tools{false};
  bool streaming{false};
};

// shared interface for all llms
class LLMBackend {
 public:
  virtual ~LLMBackend() = default;

  virtual core::Result<ChatResponse> generate(const ChatRequest& request) = 0;

  virtual core::Result<ChatResponse> generate_stream(const ChatRequest& request,
                                                     const StreamCallback& on_event) = 0;

  [[nodiscard]] virtual BackendCapabilities capabilities() const = 0;
  [[nodiscard]] virtual std::string name() const = 0;
};

}  
