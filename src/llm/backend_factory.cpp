#include "agents_framework/llm/backend_factory.hpp"

#include <algorithm>
#include <cctype>
#include <unordered_map>
#include <utility>

#include "agents_framework/core/dotenv.hpp"
#include "agents_framework/http/secrets.hpp"
#include "agents_framework/llm/anthropic_backend.hpp"
#include "agents_framework/llm/openai_backend.hpp"

namespace agents_framework::llm {
namespace {

constexpr std::string_view kAnthropicKeyVar = "ANTHROPIC_API_KEY";
constexpr std::string_view kOpenAiKeyVar = "OPENAI_API_KEY";

std::string lower(std::string_view text) {
  std::string out{text};
  std::transform(out.begin(), out.end(), out.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return out;
}

std::string_view key_var_for(BackendKind kind) noexcept {
  switch (kind) {
    case BackendKind::Anthropic: return kAnthropicKeyVar;
    case BackendKind::OpenAi:    return kOpenAiKeyVar;
    case BackendKind::Mock:      return {};
  }
  return {};
}

std::string default_model_for(BackendKind kind) {
  switch (kind) {
    case BackendKind::Anthropic: return AnthropicOptions{}.model;
    case BackendKind::OpenAi:    return OpenAiOptions{}.model;
    case BackendKind::Mock:      return "mock";
  }
  return {};
}

std::string default_base_url_for(BackendKind kind) {
  switch (kind) {
    case BackendKind::Anthropic: return AnthropicOptions{}.base_url;
    case BackendKind::OpenAi:    return OpenAiOptions{}.base_url;
    case BackendKind::Mock:      return {};
  }
  return {};
}

BackendSelection make_selection(BackendKind kind, const BackendOptions& options,
                                const EnvLookup& env, bool explicit_request) {
  BackendSelection selection;
  selection.kind = kind;
  selection.live = kind != BackendKind::Mock;
  selection.key_var = std::string{key_var_for(kind)};
  selection.explicitly_requested = explicit_request;

  if (const auto model = env("AF_MODEL"); model && !model->empty()) {
    selection.model = *model;
  } else if (!options.model.empty()) {
    selection.model = options.model;
  } else {
    selection.model = default_model_for(kind);
  }

  if (const auto base = env("AF_BASE_URL"); base && !base->empty()) {
    selection.base_url = *base;
  } else if (!options.base_url.empty()) {
    selection.base_url = options.base_url;
  } else {
    selection.base_url = default_base_url_for(kind);
  }
  return selection;
}

bool has_key(const EnvLookup& env, std::string_view var) {
  const auto value = env(var);
  return value && !value->empty();
}

}

std::string_view backend_kind_name(BackendKind kind) noexcept {
  switch (kind) {
    case BackendKind::Mock:      return "mock";
    case BackendKind::Anthropic: return "anthropic";
    case BackendKind::OpenAi:    return "openai";
  }
  return "mock";
}

std::optional<BackendKind> parse_backend_kind(std::string_view text) noexcept {
  const std::string value = lower(text);
  if (value == "mock") return BackendKind::Mock;
  if (value == "anthropic" || value == "claude") return BackendKind::Anthropic;
  if (value == "openai" || value == "chatgpt" || value == "gpt") return BackendKind::OpenAi;
  return std::nullopt;
}

std::string BackendSelection::describe() const {
  std::string out{backend_kind_name(kind)};
  out += " (model=";
  out += model;
  if (live) {
    out += ", base_url=";
    out += base_url;
    out += ", key from $";
    out += key_var;
  } else {
    out += ", offline";
  }
  out += ')';
  return out;
}

EnvLookup system_env() {
  return [](std::string_view name) { return core::get_env(name); };
}

EnvLookup map_env(std::vector<std::pair<std::string, std::string>> entries) {
  auto table = std::make_shared<std::unordered_map<std::string, std::string>>();
  for (auto& [key, value] : entries) {
    table->emplace(std::move(key), std::move(value));
  }
  return [table](std::string_view name) -> std::optional<std::string> {
    const auto it = table->find(std::string{name});
    if (it == table->end() || it->second.empty()) return std::nullopt;
    return it->second;
  };
}

core::Result<BackendSelection> select_backend(const EnvLookup& env,
                                              const BackendOptions& options) {
  const auto requested = env("AF_BACKEND");

  if (requested && !requested->empty() && lower(*requested) != "auto") {
    const auto kind = parse_backend_kind(*requested);
    if (!kind) {
      return core::fail(core::ErrorCode::Config,
                        "unknown AF_BACKEND value; expected mock, anthropic, openai, or auto",
                        *requested);
    }
    if (*kind != BackendKind::Mock && !has_key(env, key_var_for(*kind))) {
      return core::fail(core::ErrorCode::Auth,
                        std::string{"AF_BACKEND="} + std::string{backend_kind_name(*kind)} +
                            " was requested but its API key is not set",
                        std::string{"set $"} + std::string{key_var_for(*kind)});
    }
    return make_selection(*kind, options, env, true);
  }

  if (has_key(env, kAnthropicKeyVar)) {
    return make_selection(BackendKind::Anthropic, options, env, false);
  }
  if (has_key(env, kOpenAiKeyVar)) {
    return make_selection(BackendKind::OpenAi, options, env, false);
  }
  return make_selection(BackendKind::Mock, options, env, false);
}

core::Result<BackendSelection> select_backend() { return select_backend(system_env()); }

core::Result<std::shared_ptr<LLMBackend>> make_backend(const BackendSelection& selection,
                                                       BackendOptions options,
                                                       const EnvLookup& env) {
  auto client = options.client ? options.client : std::make_shared<http::HttpClient>();

  switch (selection.kind) {
    case BackendKind::Mock: {
      auto mock = std::make_shared<MockBackend>();
      if (options.mock_handler) mock->set_handler(std::move(options.mock_handler));
      return std::static_pointer_cast<LLMBackend>(mock);
    }
    case BackendKind::Anthropic: {
      const auto value = env(kAnthropicKeyVar);
      if (!value || value->empty()) {
        return core::fail(core::ErrorCode::Auth, "ANTHROPIC_API_KEY is not set");
      }
      AnthropicOptions anthropic;
      anthropic.model = selection.model;
      anthropic.base_url = selection.base_url;
      anthropic.max_tokens = options.max_tokens;
      return std::static_pointer_cast<LLMBackend>(std::make_shared<AnthropicBackend>(
          http::Secret{*value}, std::move(anthropic), std::move(client)));
    }
    case BackendKind::OpenAi: {
      const auto value = env(kOpenAiKeyVar);
      if (!value || value->empty()) {
        return core::fail(core::ErrorCode::Auth, "OPENAI_API_KEY is not set");
      }
      OpenAiOptions openai;
      openai.model = selection.model;
      openai.base_url = selection.base_url;
      openai.max_tokens = options.max_tokens;
      return std::static_pointer_cast<LLMBackend>(std::make_shared<OpenAiBackend>(
          http::Secret{*value}, std::move(openai), std::move(client)));
    }
  }
  return core::fail(core::ErrorCode::Invalid, "unhandled backend kind");
}

core::Result<std::shared_ptr<LLMBackend>> backend_from_env(BackendOptions options,
                                                           const EnvLookup& env) {
  AF_TRY(const auto selection, select_backend(env, options));
  return make_backend(selection, std::move(options), env);
}

core::Result<std::shared_ptr<LLMBackend>> backend_from_env(BackendOptions options) {
  return backend_from_env(std::move(options), system_env());
}

}
