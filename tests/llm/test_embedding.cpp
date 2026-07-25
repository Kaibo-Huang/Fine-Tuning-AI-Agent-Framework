#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "../live_env.hpp"
#include "agents_framework/http/http_client.hpp"
#include "agents_framework/http/secrets.hpp"
#include "agents_framework/llm/mock_embedding.hpp"
#include "agents_framework/llm/openai_embeddings.hpp"

namespace core = agents_framework::core;
namespace http = agents_framework::http;
namespace llm = agents_framework::llm;

namespace {

class ScriptedTransport : public http::Transport {
 public:
  core::Result<http::Response> next = http::Response{200, {}, {}};
  std::vector<http::Request> calls;

  core::Result<http::Response> send(const http::Request& request,
                                    std::chrono::milliseconds) override {
    calls.push_back(request);
    return next;
  }

  core::Result<http::Response> send_stream(const http::Request& request,
                                           std::chrono::milliseconds,
                                           const http::StreamChunkCallback&) override {
    calls.push_back(request);
    return next;
  }
};

float cosine(const std::vector<float>& a, const std::vector<float>& b) {
  float dot = 0.0F;
  float na = 0.0F;
  float nb = 0.0F;
  for (std::size_t i = 0; i < a.size(); ++i) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (std::sqrt(na) * std::sqrt(nb));
}

}

TEST_CASE("the mock embedding backend is deterministic and unit-length", "[llm][embedding]") {
  llm::MockEmbeddingBackend backend{32};
  const auto first = backend.embed({"", {"the capital of France"}});
  const auto second = backend.embed({"", {"the capital of France"}});
  REQUIRE(first);
  REQUIRE(second);
  REQUIRE(first->embeddings == second->embeddings);
  REQUIRE(first->embeddings.front().size() == 32);

  float norm = 0.0F;
  for (const float value : first->embeddings.front()) norm += value * value;
  REQUIRE(std::abs(norm - 1.0F) < 1e-5F);
}

TEST_CASE("texts sharing words score higher than disjoint texts", "[llm][embedding]") {
  llm::MockEmbeddingBackend backend{64};
  const auto response = backend.embed(
      {"", {"the capital of France is Paris", "what is the capital of France", "quantum flux"}});
  REQUIRE(response);
  const auto& doc = response->embeddings[0];
  const auto& query = response->embeddings[1];
  const auto& noise = response->embeddings[2];
  REQUIRE(cosine(doc, query) > cosine(doc, noise));
}

TEST_CASE("an empty embedding request is rejected", "[llm][embedding]") {
  llm::MockEmbeddingBackend backend;
  const auto response = backend.embed({"", {}});
  REQUIRE(!response);
  REQUIRE(response.error().code == core::ErrorCode::Invalid);
}

TEST_CASE("the OpenAI embeddings adapter builds the request and parses the response",
          "[llm][embedding]") {
  auto transport = std::make_shared<ScriptedTransport>();
  transport->next = http::Response{200, {}, R"({
    "data": [
      {"index": 1, "embedding": [0.5, 0.6]},
      {"index": 0, "embedding": [0.1, 0.2]}
    ],
    "usage": {"prompt_tokens": 7, "total_tokens": 7}
  })"};
  auto client = std::make_shared<http::HttpClient>(http::ClientOptions{}, transport);

  llm::OpenAiEmbeddingBackend backend{http::Secret{"sk-test"}, {}, client};
  const auto response = backend.embed({"", {"alpha", "beta"}});
  REQUIRE(response);
  REQUIRE(response->embeddings.size() == 2);
  REQUIRE(response->embeddings[0] == std::vector<float>{0.1F, 0.2F});
  REQUIRE(response->embeddings[1] == std::vector<float>{0.5F, 0.6F});
  REQUIRE(response->usage.input_tokens == 7);

  REQUIRE(transport->calls.size() == 1);
  const auto& request = transport->calls.front();
  REQUIRE(request.url == "https://api.openai.com/v1/embeddings");
  const auto body = nlohmann::json::parse(request.body);
  REQUIRE(body["model"] == "text-embedding-3-small");
  REQUIRE(body["input"] == nlohmann::json::array({"alpha", "beta"}));
  bool authorized = false;
  for (const auto& [key, value] : request.headers) {
    if (key == "authorization") authorized = value == "Bearer sk-test";
  }
  REQUIRE(authorized);
}

TEST_CASE("the OpenAI embeddings adapter surfaces API errors", "[llm][embedding]") {
  auto transport = std::make_shared<ScriptedTransport>();
  transport->next = http::Response{401, {}, R"({"error":{"message":"Incorrect API key"}})"};
  auto client = std::make_shared<http::HttpClient>(http::ClientOptions{}, transport);

  llm::OpenAiEmbeddingBackend backend{http::Secret{"bad"}, {}, client};
  const auto response = backend.embed({"", {"alpha"}});
  REQUIRE(!response);
  REQUIRE(response.error().code == core::ErrorCode::Auth);
  REQUIRE(response.error().message == "Incorrect API key");
}

TEST_CASE("a count mismatch between input and response is a protocol error",
          "[llm][embedding]") {
  auto transport = std::make_shared<ScriptedTransport>();
  transport->next =
      http::Response{200, {}, R"({"data": [{"index": 0, "embedding": [0.1]}]})"};
  auto client = std::make_shared<http::HttpClient>(http::ClientOptions{}, transport);

  llm::OpenAiEmbeddingBackend backend{http::Secret{"sk-test"}, {}, client};
  const auto response = backend.embed({"", {"alpha", "beta"}});
  REQUIRE(!response);
  REQUIRE(response.error().code == core::ErrorCode::Protocol);
}

TEST_CASE("OpenAiEmbeddingBackend live round-trip", "[.live][embedding]") {
  test_support::load_env_once();
  auto key = http::SecretStore::from_env("OPENAI_API_KEY");
  if (!key) {
    SKIP("set OPENAI_API_KEY to run live OpenAI tests");
  }
  llm::OpenAiEmbeddingBackend backend{std::move(*key)};
  const auto response = backend.embed({"", {"hello world", "goodbye world"}});
  REQUIRE(response.has_value());
  REQUIRE(response->embeddings.size() == 2);
  REQUIRE(response->embeddings.front().size() == 1536);
  CHECK(response->embeddings.back().size() == 1536);
  CHECK(response->usage.input_tokens > 0);

  double norm = 0.0;
  for (const float value : response->embeddings.front()) {
    norm += static_cast<double>(value) * static_cast<double>(value);
  }
  CHECK(std::abs(std::sqrt(norm) - 1.0) < 0.01);
}
