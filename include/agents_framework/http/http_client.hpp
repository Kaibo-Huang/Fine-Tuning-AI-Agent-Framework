#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "agents_framework/core/result.hpp"

namespace agents_framework::http {

using Headers = std::vector<std::pair<std::string, std::string>>;

struct Request {
  std::string method{"GET"};
  std::string url;
  Headers headers;
  std::string body;
};

struct Response {
  long status{0};
  Headers headers;
  std::string body;

  [[nodiscard]] bool ok() const noexcept { return status >= 200 && status < 300; }
};

using StreamChunkCallback = std::function<bool(std::string_view)>;

class Transport {
 public:
  virtual ~Transport() = default;
  virtual core::Result<Response> send(const Request& request,
                                      std::chrono::milliseconds timeout) = 0;
  virtual core::Result<Response> send_stream(const Request& request,
                                             std::chrono::milliseconds timeout,
                                             const StreamChunkCallback& on_chunk) = 0;
};

class CurlTransport : public Transport {
 public:
  CurlTransport();
  core::Result<Response> send(const Request& request,
                              std::chrono::milliseconds timeout) override;
  core::Result<Response> send_stream(const Request& request,
                                     std::chrono::milliseconds timeout,
                                     const StreamChunkCallback& on_chunk) override;
};

struct RetryPolicy {
  int max_retries{3};
  std::chrono::milliseconds base_delay{500};
  std::chrono::milliseconds max_delay{20000};
};

struct ClientOptions {
  std::chrono::milliseconds timeout{30000};
  RetryPolicy retry;
  std::uint64_t jitter_seed{0};
};

class HttpClient {
 public:
  explicit HttpClient(ClientOptions options = {},
                      std::shared_ptr<Transport> transport = nullptr);

  core::Result<Response> send(const Request& request);
  core::Result<Response> send_stream(const Request& request,
                                     const StreamChunkCallback& on_chunk);

  core::Result<nlohmann::json> post_json(std::string_view url, const nlohmann::json& body,
                                         Headers headers = {});

  [[nodiscard]] const ClientOptions& options() const noexcept { return options_; }
  [[nodiscard]] Transport& transport() const noexcept { return *transport_; }

 private:
  ClientOptions options_;
  std::shared_ptr<Transport> transport_;
};

[[nodiscard]] core::ErrorCode error_code_for_status(long status) noexcept;

}
