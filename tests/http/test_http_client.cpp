#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "agents_framework/http/http_client.hpp"

using namespace agents_framework;
using namespace agents_framework::http;

namespace {

Response make_response(long status, std::string body = {}, Headers headers = {}) {
  return Response{status, std::move(headers), std::move(body)};
}

class MockTransport : public Transport {
 public:
  std::vector<core::Result<Response>> script;
  std::vector<Request> calls;

  core::Result<Response> send(const Request& request,
                              std::chrono::milliseconds) override {
    calls.push_back(request);
    return current();
  }

  core::Result<Response> send_stream(const Request& request, std::chrono::milliseconds,
                                     const StreamChunkCallback& on_chunk) override {
    calls.push_back(request);
    auto result = current();
    if (result) {
      const std::string& body = result->body;
      for (std::size_t i = 0; i < body.size(); i += 2) {
        if (!on_chunk(std::string_view{body}.substr(i, 2))) break;
      }
    }
    return result;
  }

 private:
  core::Result<Response> current() {
    const std::size_t index = std::min(calls.size() - 1, script.size() - 1);
    return script[index];
  }
};

ClientOptions fast_retry(int max_retries = 3) {
  ClientOptions options;
  options.retry.max_retries = max_retries;
  options.retry.base_delay = std::chrono::milliseconds{0};
  options.jitter_seed = 1;
  return options;
}

}

TEST_CASE("HttpClient retries on 429 and then succeeds", "[http]") {
  auto mock = std::make_shared<MockTransport>();
  mock->script = {make_response(429), make_response(429), make_response(200, "ok")};
  HttpClient client{fast_retry(), mock};

  const auto response = client.send(Request{"GET", "https://example.test", {}, {}});
  REQUIRE(response.has_value());
  REQUIRE(response->status == 200);
  REQUIRE(response->body == "ok");
  REQUIRE(mock->calls.size() == 3);
}

TEST_CASE("HttpClient gives up after max retries on a persistent 5xx", "[http]") {
  auto mock = std::make_shared<MockTransport>();
  mock->script = {make_response(500)};
  HttpClient client{fast_retry(3), mock};

  const auto response = client.send(Request{"GET", "https://example.test", {}, {}});
  REQUIRE_FALSE(response.has_value());
  REQUIRE(response.error().code == core::ErrorCode::Network);
  REQUIRE(mock->calls.size() == 4);
}

TEST_CASE("HttpClient retries transient transport errors", "[http]") {
  auto mock = std::make_shared<MockTransport>();
  mock->script = {core::fail(core::ErrorCode::Network, "connection reset"),
                  make_response(200, "recovered")};
  HttpClient client{fast_retry(), mock};

  const auto response = client.send(Request{"GET", "https://example.test", {}, {}});
  REQUIRE(response.has_value());
  REQUIRE(response->body == "recovered");
  REQUIRE(mock->calls.size() == 2);
}

TEST_CASE("HttpClient returns 4xx responses without retrying", "[http]") {
  auto mock = std::make_shared<MockTransport>();
  mock->script = {make_response(404, "nope")};
  HttpClient client{fast_retry(), mock};

  const auto response = client.send(Request{"GET", "https://example.test", {}, {}});
  REQUIRE(response.has_value());
  REQUIRE(response->status == 404);
  REQUIRE(mock->calls.size() == 1);
}

TEST_CASE("HttpClient honors Retry-After before retrying", "[http]") {
  auto mock = std::make_shared<MockTransport>();
  mock->script = {make_response(429, "", {{"Retry-After", "0"}}), make_response(200, "done")};
  HttpClient client{fast_retry(), mock};

  const auto response = client.send(Request{"GET", "https://example.test", {}, {}});
  REQUIRE(response.has_value());
  REQUIRE(response->status == 200);
  REQUIRE(mock->calls.size() == 2);
}

TEST_CASE("post_json sets Content-Type, serializes the body, and parses the reply", "[http]") {
  auto mock = std::make_shared<MockTransport>();
  mock->script = {make_response(200, R"({"reply":"pong"})")};
  HttpClient client{fast_retry(), mock};

  const nlohmann::json body = {{"ping", 1}};
  const auto result = client.post_json("https://example.test/v1", body,
                                       {{"x-api-key", "abc"}});
  REQUIRE(result.has_value());
  REQUIRE((*result)["reply"] == "pong");

  REQUIRE(mock->calls.size() == 1);
  const Request& sent = mock->calls.front();
  REQUIRE(sent.method == "POST");
  REQUIRE(sent.body == body.dump());

  const bool has_content_type =
      std::any_of(sent.headers.begin(), sent.headers.end(), [](const auto& h) {
        return h.first == "Content-Type" && h.second == "application/json";
      });
  const bool has_api_key =
      std::any_of(sent.headers.begin(), sent.headers.end(),
                  [](const auto& h) { return h.first == "x-api-key"; });
  REQUIRE(has_content_type);
  REQUIRE(has_api_key);
}

TEST_CASE("post_json maps a non-2xx status to an error", "[http]") {
  auto mock = std::make_shared<MockTransport>();
  mock->script = {make_response(401, R"({"error":"bad key"})")};
  HttpClient client{fast_retry(), mock};

  const auto result = client.post_json("https://example.test", nlohmann::json::object());
  REQUIRE_FALSE(result.has_value());
  REQUIRE(result.error().code == core::ErrorCode::Auth);
}

TEST_CASE("send_stream delivers body chunks to the callback", "[http]") {
  auto mock = std::make_shared<MockTransport>();
  mock->script = {make_response(200, "streamed")};
  HttpClient client{fast_retry(), mock};

  std::string assembled;
  const auto response = client.send_stream(
      Request{"GET", "https://example.test", {}, {}},
      [&](std::string_view chunk) {
        assembled.append(chunk);
        return true;
      });
  REQUIRE(response.has_value());
  REQUIRE(assembled == "streamed");
}

TEST_CASE("CurlTransport performs a real HTTPS GET", "[http][live]") {
  if (std::getenv("AGENTS_LIVE_HTTP") == nullptr) {
    SKIP("set AGENTS_LIVE_HTTP to run live HTTP tests");
  }
  CurlTransport transport;
  const auto response =
      transport.send(Request{"GET", "https://example.com", {}, {}}, std::chrono::seconds{30});
  REQUIRE(response.has_value());
  REQUIRE(response->status == 200);
  REQUIRE_FALSE(response->body.empty());
}
