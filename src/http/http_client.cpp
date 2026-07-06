#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "agents_framework/http/http_client.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdlib>
#include <optional>
#include <random>
#include <thread>
#include <utility>

#include <curl/curl.h>

#include "agents_framework/core/rng.hpp"

namespace agents_framework::http {
namespace {


struct CurlHandle {
  CURL* handle{curl_easy_init()};
  ~CurlHandle() {
    if (handle) curl_easy_cleanup(handle);
  }
};

struct CurlSlist {
  curl_slist* list{nullptr};
  ~CurlSlist() {
    if (list) curl_slist_free_all(list);
  }
  void append(const std::string& header) { list = curl_slist_append(list, header.c_str()); }
};

void ensure_global_init() {
  static const int once = [] {
    curl_global_init(CURL_GLOBAL_DEFAULT);
    std::atexit([] { curl_global_cleanup(); });
    return 0;
  }();
  (void)once;
}

std::string trim(std::string_view s) {
  const auto begin = s.find_first_not_of(" \t\r\n");
  if (begin == std::string_view::npos) return {};
  const auto end = s.find_last_not_of(" \t\r\n");
  return std::string{s.substr(begin, end - begin + 1)};
}

std::size_t collect_body(char* ptr, std::size_t size, std::size_t nmemb, void* userdata) {
  const std::size_t total = size * nmemb;
  static_cast<std::string*>(userdata)->append(ptr, total);
  return total;
}

struct StreamState {
  const StreamChunkCallback* on_chunk{nullptr};
  bool cancelled{false};
};

std::size_t stream_body(char* ptr, std::size_t size, std::size_t nmemb, void* userdata) {
  auto* state = static_cast<StreamState*>(userdata);
  const std::size_t total = size * nmemb;
  if (!(*state->on_chunk)(std::string_view{ptr, total})) {
    state->cancelled = true;
    return 0;
  }
  return total;
}

std::size_t collect_header(char* buffer, std::size_t size, std::size_t nitems, void* userdata) {
  const std::size_t total = size * nitems;
  const std::string_view line{buffer, total};
  const auto colon = line.find(':');
  if (colon != std::string_view::npos) {
    auto* headers = static_cast<Headers*>(userdata);
    headers->emplace_back(trim(line.substr(0, colon)), trim(line.substr(colon + 1)));
  }
  return total;
}

core::Result<Response> curl_error(CURLcode code) {
  const core::ErrorCode ec =
      (code == CURLE_OPERATION_TIMEDOUT) ? core::ErrorCode::Timeout : core::ErrorCode::Network;
  return core::fail(ec, "curl request failed", curl_easy_strerror(code));
}

core::Result<Response> perform(const Request& request, std::chrono::milliseconds timeout,
                               const StreamChunkCallback* on_chunk) {
  ensure_global_init();
  CurlHandle curl;
  if (!curl.handle) {
    return core::fail(core::ErrorCode::Network, "failed to initialize curl handle");
  }

  Response response;
  StreamState stream{on_chunk, false};

  CurlSlist headers;
  for (const auto& [key, value] : request.headers) {
    headers.append(key + ": " + value);
  }

  curl_easy_setopt(curl.handle, CURLOPT_URL, request.url.c_str());
  if (headers.list) curl_easy_setopt(curl.handle, CURLOPT_HTTPHEADER, headers.list);
  curl_easy_setopt(curl.handle, CURLOPT_TIMEOUT_MS, static_cast<long>(timeout.count()));
  curl_easy_setopt(curl.handle, CURLOPT_NOSIGNAL, 1L);
  curl_easy_setopt(curl.handle, CURLOPT_ACCEPT_ENCODING, "");
  curl_easy_setopt(curl.handle, CURLOPT_HEADERFUNCTION, collect_header);
  curl_easy_setopt(curl.handle, CURLOPT_HEADERDATA, &response.headers);

  if (on_chunk) {
    curl_easy_setopt(curl.handle, CURLOPT_WRITEFUNCTION, stream_body);
    curl_easy_setopt(curl.handle, CURLOPT_WRITEDATA, &stream);
  } else {
    curl_easy_setopt(curl.handle, CURLOPT_WRITEFUNCTION, collect_body);
    curl_easy_setopt(curl.handle, CURLOPT_WRITEDATA, &response.body);
  }

  const std::string method = request.method.empty() ? "GET" : request.method;
  if (method == "POST") {
    curl_easy_setopt(curl.handle, CURLOPT_POST, 1L);
    curl_easy_setopt(curl.handle, CURLOPT_POSTFIELDS, request.body.data());
    curl_easy_setopt(curl.handle, CURLOPT_POSTFIELDSIZE_LARGE,
                     static_cast<curl_off_t>(request.body.size()));
  } else if (method != "GET") {
    curl_easy_setopt(curl.handle, CURLOPT_CUSTOMREQUEST, method.c_str());
    if (!request.body.empty()) {
      curl_easy_setopt(curl.handle, CURLOPT_POSTFIELDS, request.body.data());
      curl_easy_setopt(curl.handle, CURLOPT_POSTFIELDSIZE_LARGE,
                       static_cast<curl_off_t>(request.body.size()));
    }
  }

  const CURLcode code = curl_easy_perform(curl.handle);
  if (code != CURLE_OK) {
    if (code == CURLE_WRITE_ERROR && stream.cancelled) {
      return core::fail(core::ErrorCode::Cancelled, "stream cancelled by callback");
    }
    return curl_error(code);
  }

  long status = 0;
  curl_easy_getinfo(curl.handle, CURLINFO_RESPONSE_CODE, &status);
  response.status = status;
  return response;
}


bool is_retryable_status(long status) noexcept { return status == 429 || status >= 500; }

bool is_retryable_error(core::ErrorCode code) noexcept {
  return code == core::ErrorCode::Network || code == core::ErrorCode::Timeout;
}

std::optional<std::chrono::milliseconds> parse_retry_after(const Headers& headers) {
  for (const auto& [key, value] : headers) {
    if (key.size() == 11 &&
        std::equal(key.begin(), key.end(), "retry-after", [](char a, char b) {
          return std::tolower(static_cast<unsigned char>(a)) == b;
        })) {
      long long seconds = 0;
      const auto* first = value.data();
      const auto* last = value.data() + value.size();
      if (std::from_chars(first, last, seconds).ec == std::errc{} && seconds >= 0) {
        return std::chrono::seconds{seconds};
      }
      return std::nullopt;
    }
  }
  return std::nullopt;
}

std::chrono::milliseconds backoff(int attempt,
                                  std::optional<std::chrono::milliseconds> retry_after,
                                  const RetryPolicy& policy, core::Rng& rng) {
  if (retry_after) {
    return std::min(*retry_after, policy.max_delay);
  }
  long long delay = policy.base_delay.count();
  for (int i = 0; i < attempt && delay < policy.max_delay.count(); ++i) {
    delay <<= 1;
  }
  delay = std::min<long long>(delay, policy.max_delay.count());
  const double fraction = 0.5 + 0.5 * rng.next_double();
  return std::chrono::milliseconds{static_cast<long long>(static_cast<double>(delay) * fraction)};
}

}

CurlTransport::CurlTransport() { ensure_global_init(); }

core::Result<Response> CurlTransport::send(const Request& request,
                                           std::chrono::milliseconds timeout) {
  return perform(request, timeout, nullptr);
}

core::Result<Response> CurlTransport::send_stream(const Request& request,
                                                  std::chrono::milliseconds timeout,
                                                  const StreamChunkCallback& on_chunk) {
  return perform(request, timeout, &on_chunk);
}

core::ErrorCode error_code_for_status(long status) noexcept {
  if (status == 401 || status == 403) return core::ErrorCode::Auth;
  if (status == 404) return core::ErrorCode::NotFound;
  if (status == 408) return core::ErrorCode::Timeout;
  if (status == 429) return core::ErrorCode::RateLimited;
  if (status >= 400 && status < 500) return core::ErrorCode::Invalid;
  if (status >= 500) return core::ErrorCode::Network;
  return core::ErrorCode::Unknown;
}

HttpClient::HttpClient(ClientOptions options, std::shared_ptr<Transport> transport)
    : options_{std::move(options)},
      transport_{transport ? std::move(transport) : std::make_shared<CurlTransport>()} {}

core::Result<Response> HttpClient::send(const Request& request) {
  const std::uint64_t seed =
      options_.jitter_seed != 0 ? options_.jitter_seed : std::random_device{}();
  core::Rng rng{seed};

  for (int attempt = 0;; ++attempt) {
    auto result = transport_->send(request, options_.timeout);
    const bool exhausted = attempt >= options_.retry.max_retries;

    if (result) {
      if (!is_retryable_status(result->status)) {
        return result;
      }
      if (exhausted) {
        return core::fail(error_code_for_status(result->status),
                          "HTTP request failed after retries",
                          "status " + std::to_string(result->status));
      }
      std::this_thread::sleep_for(backoff(attempt, parse_retry_after(result->headers),
                                          options_.retry, rng));
    } else {
      if (!is_retryable_error(result.error().code) || exhausted) {
        return result;
      }
      std::this_thread::sleep_for(backoff(attempt, std::nullopt, options_.retry, rng));
    }
  }
}

core::Result<Response> HttpClient::send_stream(const Request& request,
                                               const StreamChunkCallback& on_chunk) {
  return transport_->send_stream(request, options_.timeout, on_chunk);
}

core::Result<nlohmann::json> HttpClient::post_json(std::string_view url,
                                                   const nlohmann::json& body,
                                                   Headers headers) {
  Request request;
  request.method = "POST";
  request.url = std::string{url};
  request.headers = std::move(headers);
  request.headers.emplace_back("Content-Type", "application/json");
  request.body = body.dump();

  AF_TRY(auto response, send(request));
  if (!response.ok()) {
    return core::fail(error_code_for_status(response.status),
                      "HTTP request returned an error status",
                      "status " + std::to_string(response.status));
  }
  try {
    return nlohmann::json::parse(response.body);
  } catch (const nlohmann::json::exception& e) {
    return core::fail(core::ErrorCode::Parse, "failed to parse JSON response", e.what());
  }
}

}
