#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "agents_framework/core/result.hpp"
#include "agents_framework/http/http_client.hpp"
#include "agents_framework/tools/http_tool.hpp"
#include "agents_framework/tools/registry.hpp"

namespace core = agents_framework::core;
namespace http = agents_framework::http;
namespace tools = agents_framework::tools;

namespace {

class MockTransport : public http::Transport {
 public:
  std::vector<core::Result<http::Response>> script;
  std::vector<http::Request> calls;

  core::Result<http::Response> send(const http::Request& request,
                                    std::chrono::milliseconds) override {
    calls.push_back(request);
    const std::size_t index = std::min(calls.size() - 1, script.size() - 1);
    return script[index];
  }

  core::Result<http::Response> send_stream(const http::Request& request,
                                           std::chrono::milliseconds,
                                           const http::StreamChunkCallback&) override {
    return send(request, std::chrono::milliseconds{0});
  }
};

struct Harness {
  std::shared_ptr<MockTransport> transport{std::make_shared<MockTransport>()};

  tools::HttpTool tool(tools::HttpToolOptions options) {
    return tools::HttpTool{std::move(options), http::HttpClient{{}, transport}};
  }
};

}

TEST_CASE("the http tool performs an allowed GET and returns status and body",
          "[tools][http_tool]") {
  Harness harness;
  harness.transport->script = {http::Response{200, {}, R"({"ok":true})"}};

  tools::HttpToolOptions options;
  options.allowed_url_prefixes = {"https://api.example.com/"};
  options.headers = {{"Accept", "application/json"}};
  auto tool = harness.tool(options);

  const auto result = tool.call({{"url", "https://api.example.com/status"}});
  REQUIRE(result);
  REQUIRE(*result == "HTTP 200\n{\"ok\":true}");

  REQUIRE(harness.transport->calls.size() == 1);
  const auto& request = harness.transport->calls.front();
  REQUIRE(request.method == "GET");
  REQUIRE(request.url == "https://api.example.com/status");
  REQUIRE(request.headers == http::Headers{{"Accept", "application/json"}});
}

TEST_CASE("urls outside the allowlist are rejected before any request", "[tools][http_tool]") {
  Harness harness;
  harness.transport->script = {http::Response{200, {}, "never"}};

  tools::HttpToolOptions options;
  options.allowed_url_prefixes = {"https://api.example.com/"};
  auto tool = harness.tool(options);

  const auto result = tool.call({{"url", "https://evil.example.net/steal"}});
  REQUIRE(!result);
  REQUIRE(result.error().message == "url is not covered by allowed_url_prefixes");
  REQUIRE(harness.transport->calls.empty());
}

TEST_CASE("prefix matching stops at url boundaries", "[tools][http_tool]") {
  Harness harness;
  harness.transport->script = {http::Response{200, {}, "ok"}};

  tools::HttpToolOptions options;
  options.allowed_url_prefixes = {"https://api.example.com"};
  auto tool = harness.tool(options);

  REQUIRE(tool.call({{"url", "https://api.example.com"}}));
  REQUIRE(tool.call({{"url", "https://api.example.com/path"}}));
  REQUIRE(tool.call({{"url", "https://api.example.com?q=1"}}));

  for (const std::string url : {"https://api.example.com.evil.io/steal",
                                "https://api.example.com:8080/steal",
                                "https://api.example.comX/steal"}) {
    const auto blocked = tool.call({{"url", url}});
    REQUIRE(!blocked);
    REQUIRE(blocked.error().message == "url is not covered by allowed_url_prefixes");
  }
}

TEST_CASE("urls carrying userinfo are rejected outright", "[tools][http_tool]") {
  Harness harness;
  harness.transport->script = {http::Response{200, {}, "ok"}};

  tools::HttpToolOptions options;
  options.allowed_url_prefixes = {"https://"};
  auto tool = harness.tool(options);

  const auto blocked = tool.call({{"url", "https://api.example.com@evil.io/steal"}});
  REQUIRE(!blocked);
  REQUIRE(blocked.error().message == "url must not contain userinfo");
  REQUIRE(harness.transport->calls.empty());

  REQUIRE(tool.call({{"url", "https://api.example.com/user@example.com/profile"}}));
}

TEST_CASE("methods outside the allowlist are rejected", "[tools][http_tool]") {
  Harness harness;
  harness.transport->script = {http::Response{200, {}, ""}};

  tools::HttpToolOptions options;
  options.allowed_url_prefixes = {"https://api.example.com/"};
  options.allowed_methods = {"GET", "POST"};
  auto tool = harness.tool(options);

  const auto rejected =
      tool.call({{"url", "https://api.example.com/x"}, {"method", "DELETE"}});
  REQUIRE(!rejected);
  REQUIRE(rejected.error().message == "method is not allowed for this tool");

  const auto posted = tool.call(
      {{"url", "https://api.example.com/x"}, {"method", "POST"}, {"body", "payload"}});
  REQUIRE(posted);
  REQUIRE(harness.transport->calls.back().method == "POST");
  REQUIRE(harness.transport->calls.back().body == "payload");
}

TEST_CASE("an unconfigured http tool fails closed", "[tools][http_tool]") {
  Harness harness;
  auto tool = harness.tool(tools::HttpToolOptions{});

  const auto result = tool.call({{"url", "https://api.example.com/"}});
  REQUIRE(!result);
  REQUIRE(result.error().code == core::ErrorCode::Config);
}

TEST_CASE("non-2xx responses are returned to the model, transport errors fail",
          "[tools][http_tool]") {
  Harness harness;
  harness.transport->script = {http::Response{404, {}, "not here"}};

  tools::HttpToolOptions options;
  options.allowed_url_prefixes = {"https://api.example.com/"};
  auto tool = harness.tool(options);

  const auto missing = tool.call({{"url", "https://api.example.com/gone"}});
  REQUIRE(missing);
  REQUIRE(*missing == "HTTP 404\nnot here");

  harness.transport->script = {core::fail(core::ErrorCode::Network, "connection refused")};
  const auto failed = tool.call({{"url", "https://api.example.com/down"}});
  REQUIRE(!failed);
  REQUIRE(failed.error().code == core::ErrorCode::Network);
}

TEST_CASE("large response bodies are truncated at the cap", "[tools][http_tool]") {
  Harness harness;
  harness.transport->script = {http::Response{200, {}, std::string(100, 'x')}};

  tools::HttpToolOptions options;
  options.allowed_url_prefixes = {"https://api.example.com/"};
  options.max_response_bytes = 10;
  auto tool = harness.tool(options);

  const auto result = tool.call({{"url", "https://api.example.com/big"}});
  REQUIRE(result);
  REQUIRE(*result == "HTTP 200\n" + std::string(10, 'x') + "\n[response truncated]");
}

TEST_CASE("the http tool registers with a schema the registry validates",
          "[tools][http_tool]") {
  Harness harness;
  harness.transport->script = {http::Response{200, {}, "ok"}};

  tools::HttpToolOptions options;
  options.allowed_url_prefixes = {"https://api.example.com/"};
  options.allowed_methods = {"GET", "POST"};
  tools::ToolRegistry registry;
  REQUIRE(registry.add(
      tools::make_http_tool(options, http::HttpClient{{}, harness.transport})));

  const auto defs = registry.defs();
  REQUIRE(defs.front().name == "http_request");
  REQUIRE(defs.front().input_schema.at("properties").contains("method"));
  REQUIRE(defs.front().input_schema.at("properties").contains("body"));

  const auto missing_url = registry.invoke("http_request", nlohmann::json::object());
  REQUIRE(!missing_url);
  REQUIRE(missing_url.error().message == "missing required argument 'url'");

  const auto bad_method = registry.invoke(
      "http_request", {{"url", "https://api.example.com/"}, {"method", "PATCH"}});
  REQUIRE(!bad_method);
  REQUIRE(bad_method.error().message.starts_with("argument 'method' must be one of"));

  REQUIRE(registry.invoke("http_request", {{"url", "https://api.example.com/"}}));
}
