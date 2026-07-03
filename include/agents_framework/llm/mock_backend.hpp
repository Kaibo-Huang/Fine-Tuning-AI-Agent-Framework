#pragma once

#include <deque>
#include <functional>
#include <string>
#include <vector>

#include "agents_framework/core/result.hpp"
#include "agents_framework/llm/backend.hpp"

namespace agents_framework::llm {

// offline LLMBackend for tests
class MockBackend : public LLMBackend {
 public:
  using Handler = std::function<core::Result<ChatResponse>(const ChatRequest&)>;

  void push_response(ChatResponse response);
  void push_error(core::Error error);
  void set_handler(Handler handler);

  void set_capabilities(BackendCapabilities capabilities) { capabilities_ = capabilities; }
  void set_name(std::string name) { name_ = std::move(name); }

  core::Result<ChatResponse> generate(const ChatRequest& request) override;
  core::Result<ChatResponse> generate_stream(const ChatRequest& request,
                                             const StreamCallback& on_event) override;
  [[nodiscard]] BackendCapabilities capabilities() const override { return capabilities_; }
  [[nodiscard]] std::string name() const override { return name_; }

  [[nodiscard]] const std::vector<ChatRequest>& calls() const noexcept { return calls_; }

 private:
  core::Result<ChatResponse> take_response(const ChatRequest& request);

  Handler handler_;
  std::deque<core::Result<ChatResponse>> queue_;
  std::vector<ChatRequest> calls_;
  BackendCapabilities capabilities_{true, true};
  std::string name_{"mock"};
};

}  
