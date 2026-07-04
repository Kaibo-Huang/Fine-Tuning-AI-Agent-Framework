#include "agents_framework/llm/mock_backend.hpp"

#include <utility>
#include <variant>

namespace agents_framework::llm {
namespace {

constexpr std::size_t kTextChunk = 5;

// mock llm replies
void emit_stream(const ChatResponse& response, const StreamCallback& on_event) {
  on_event(MessageStart{response.id});
  for (const auto& block : response.content) {
    if (const auto* text = std::get_if<TextBlock>(&block)) {
      for (std::size_t i = 0; i < text->text.size(); i += kTextChunk) {
        on_event(TextDelta{text->text.substr(i, kTextChunk)});
      }
    } else if (const auto* use = std::get_if<ToolUseBlock>(&block)) {
      on_event(ToolUseStart{use->id, use->name});
      on_event(ToolUseInputDelta{use->input.dump()});
    }
  }
  on_event(MessageStop{response.stop_reason, response.usage});
}

} 

void MockBackend::push_response(ChatResponse response) {
  queue_.push_back(std::move(response));
}

void MockBackend::push_error(core::Error error) {
  queue_.push_back(std::unexpected(std::move(error)));
}

void MockBackend::set_handler(Handler handler) { handler_ = std::move(handler); }

core::Result<ChatResponse> MockBackend::take_response(const ChatRequest& request) {
  if (handler_) {
    return handler_(request);
  }
  if (queue_.empty()) {
    return core::fail(core::ErrorCode::Invalid, "MockBackend has no scripted response");
  }
  core::Result<ChatResponse> response = std::move(queue_.front());
  queue_.pop_front();
  return response;
}

core::Result<ChatResponse> MockBackend::generate(const ChatRequest& request) {
  calls_.push_back(request);
  return take_response(request);
}

core::Result<ChatResponse> MockBackend::generate_stream(const ChatRequest& request,
                                                        const StreamCallback& on_event) {
  calls_.push_back(request);
  auto response = take_response(request);
  if (!response) {
    if (on_event) {
      on_event(StreamError{response.error()});
    }
    return response;
  }
  if (on_event) {
    emit_stream(*response, on_event);
  }
  return response;
}

}  
