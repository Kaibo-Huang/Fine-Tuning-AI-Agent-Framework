#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "agents_framework/core/result.hpp"
#include "agents_framework/http/http_client.hpp"
#include "agents_framework/llm/backend.hpp"
#include "agents_framework/llm/mock_backend.hpp"

namespace agents_framework::llm {

enum class BackendKind { Mock, Anthropic, OpenAi };

[[nodiscard]] std::string_view backend_kind_name(BackendKind kind) noexcept;
[[nodiscard]] std::optional<BackendKind> parse_backend_kind(std::string_view text) noexcept;

struct BackendSelection {
  BackendKind kind{BackendKind::Mock};
  std::string model;
  std::string base_url;
  std::string key_var;
  bool live{false};
  bool explicitly_requested{false};

  [[nodiscard]] std::string describe() const;
};

using EnvLookup = std::function<std::optional<std::string>(std::string_view)>;

[[nodiscard]] EnvLookup system_env();
[[nodiscard]] EnvLookup map_env(std::vector<std::pair<std::string, std::string>> entries);

struct BackendOptions {
  std::string model;
  std::string base_url;
  int max_tokens{1024};
  MockBackend::Handler mock_handler;
  std::shared_ptr<http::HttpClient> client;
};

[[nodiscard]] core::Result<BackendSelection> select_backend(const EnvLookup& env,
                                                            const BackendOptions& options = {});
[[nodiscard]] core::Result<BackendSelection> select_backend();

[[nodiscard]] core::Result<std::shared_ptr<LLMBackend>> make_backend(
    const BackendSelection& selection, BackendOptions options, const EnvLookup& env);

[[nodiscard]] core::Result<std::shared_ptr<LLMBackend>> backend_from_env(
    BackendOptions options = {});
[[nodiscard]] core::Result<std::shared_ptr<LLMBackend>> backend_from_env(BackendOptions options,
                                                                        const EnvLookup& env);

}
