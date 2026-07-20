#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "agents_framework/core/result.hpp"
#include "agents_framework/http/http_client.hpp"
#include "agents_framework/tools/tool.hpp"

namespace agents_framework::tools {

struct HttpToolOptions {
  std::string name{"http_request"};
  std::string description{
      "Make an HTTP request to an allowed URL and return the status line and body."};
  std::vector<std::string> allowed_url_prefixes;
  std::vector<std::string> allowed_methods{"GET"};
  http::Headers headers;
  std::size_t max_response_bytes{65536};
};

class HttpTool final : public Tool {
 public:
  explicit HttpTool(HttpToolOptions options, http::HttpClient client = http::HttpClient{});

  [[nodiscard]] const llm::ToolDef& def() const override { return def_; }
  core::Result<std::string> call(const nlohmann::json& args) override;

 private:
  HttpToolOptions options_;
  http::HttpClient client_;
  llm::ToolDef def_;
};

[[nodiscard]] inline std::unique_ptr<Tool> make_http_tool(
    HttpToolOptions options, http::HttpClient client = http::HttpClient{}) {
  return std::make_unique<HttpTool>(std::move(options), std::move(client));
}

}
