#include "agents_framework/tools/http_tool.hpp"

#include <algorithm>
#include <string_view>

namespace agents_framework::tools {

namespace {

bool prefix_allows(const std::string& url, const std::string& prefix) {
  if (!url.starts_with(prefix)) return false;
  if (prefix.ends_with('/') || url.size() == prefix.size()) return true;
  const char next = url[prefix.size()];
  return next == '/' || next == '?' || next == '#';
}

bool has_userinfo(const std::string& url) {
  const auto scheme = url.find("://");
  const std::size_t authority = scheme == std::string::npos ? 0 : scheme + 3;
  const auto authority_end = url.find_first_of("/?#", authority);
  const auto host = std::string_view{url}.substr(
      authority, (authority_end == std::string::npos ? url.size() : authority_end) - authority);
  return host.find('@') != std::string_view::npos;
}

}

HttpTool::HttpTool(HttpToolOptions options, http::HttpClient client)
    : options_(std::move(options)), client_(std::move(client)) {
  def_.name = options_.name;
  def_.description = options_.description;
  nlohmann::json properties = {
      {"url", {{"type", "string"}, {"description", "The URL to request."}}}};
  if (options_.allowed_methods.size() > 1) {
    properties["method"] = {{"type", "string"}, {"enum", options_.allowed_methods}};
  }
  const bool writable =
      std::any_of(options_.allowed_methods.begin(), options_.allowed_methods.end(),
                  [](const std::string& method) { return method != "GET" && method != "HEAD"; });
  if (writable) {
    properties["body"] = {{"type", "string"}, {"description", "The request body to send."}};
  }
  def_.input_schema = {{"type", "object"},
                       {"properties", properties},
                       {"required", nlohmann::json::array({"url"})},
                       {"additionalProperties", false}};
}

core::Result<std::string> HttpTool::call(const nlohmann::json& args) {
  if (options_.allowed_url_prefixes.empty()) {
    return core::fail(core::ErrorCode::Config, "http tool has no allowed_url_prefixes configured",
                      options_.name);
  }

  const auto url = args.at("url").get<std::string>();
  if (has_userinfo(url)) {
    return core::fail(core::ErrorCode::Invalid, "url must not contain userinfo", url);
  }
  const bool allowed =
      std::any_of(options_.allowed_url_prefixes.begin(), options_.allowed_url_prefixes.end(),
                  [&url](const std::string& prefix) { return prefix_allows(url, prefix); });
  if (!allowed) {
    return core::fail(core::ErrorCode::Invalid, "url is not covered by allowed_url_prefixes",
                      url);
  }

  std::string method = options_.allowed_methods.empty() ? std::string{"GET"}
                                                        : options_.allowed_methods.front();
  if (const auto requested = args.find("method"); requested != args.end()) {
    method = requested->get<std::string>();
    if (std::find(options_.allowed_methods.begin(), options_.allowed_methods.end(), method) ==
        options_.allowed_methods.end()) {
      return core::fail(core::ErrorCode::Invalid, "method is not allowed for this tool", method);
    }
  }

  http::Request request;
  request.method = method;
  request.url = url;
  request.headers = options_.headers;
  request.body = args.value("body", std::string{});

  AF_TRY(auto response, client_.send(request));

  std::string out = "HTTP " + std::to_string(response.status) + "\n";
  if (response.body.size() > options_.max_response_bytes) {
    out.append(response.body, 0, options_.max_response_bytes);
    out += "\n[response truncated]";
  } else {
    out += response.body;
  }
  return out;
}

}
