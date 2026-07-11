#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <string>

#include "agents_framework/core/logging.hpp"
#include "agents_framework/llm/backend_factory.hpp"

using namespace agents_framework;
using Catch::Matchers::ContainsSubstring;

namespace {

constexpr const char* kFakeKey = "sk-fake-key-for-tests-do-not-use-0123456789";

}

TEST_CASE("auto selection falls back to mock with no keys", "[llm][factory]") {
  const auto env = llm::map_env({});
  const auto selection = llm::select_backend(env);
  REQUIRE(selection.has_value());
  CHECK(selection->kind == llm::BackendKind::Mock);
  CHECK_FALSE(selection->live);
  CHECK_FALSE(selection->explicitly_requested);
}

TEST_CASE("auto selection prefers anthropic, then openai", "[llm][factory]") {
  SECTION("both keys present") {
    const auto env = llm::map_env({{"ANTHROPIC_API_KEY", kFakeKey}, {"OPENAI_API_KEY", kFakeKey}});
    const auto selection = llm::select_backend(env);
    REQUIRE(selection.has_value());
    CHECK(selection->kind == llm::BackendKind::Anthropic);
  }
  SECTION("only openai present") {
    const auto env = llm::map_env({{"OPENAI_API_KEY", kFakeKey}});
    const auto selection = llm::select_backend(env);
    REQUIRE(selection.has_value());
    CHECK(selection->kind == llm::BackendKind::OpenAi);
    CHECK(selection->live);
    CHECK(selection->model == "gpt-4o-mini");
  }
}

TEST_CASE("AF_BACKEND=mock stays offline even when a key is present", "[llm][factory]") {
  const auto env = llm::map_env({{"AF_BACKEND", "mock"}, {"OPENAI_API_KEY", kFakeKey}});
  const auto selection = llm::select_backend(env);
  REQUIRE(selection.has_value());
  CHECK(selection->kind == llm::BackendKind::Mock);
  CHECK_FALSE(selection->live);
}

TEST_CASE("naming a live backend without its key is an error", "[llm][factory]") {
  const auto env = llm::map_env({{"AF_BACKEND", "openai"}});
  const auto selection = llm::select_backend(env);
  REQUIRE_FALSE(selection.has_value());
  CHECK(selection.error().code == core::ErrorCode::Auth);
}

TEST_CASE("an unknown AF_BACKEND value is rejected", "[llm][factory]") {
  const auto env = llm::map_env({{"AF_BACKEND", "gemini"}});
  const auto selection = llm::select_backend(env);
  REQUIRE_FALSE(selection.has_value());
  CHECK(selection.error().code == core::ErrorCode::Config);
}

TEST_CASE("AF_MODEL and AF_BASE_URL override defaults", "[llm][factory]") {
  const auto env = llm::map_env({{"AF_BACKEND", "openai"},
                                 {"OPENAI_API_KEY", kFakeKey},
                                 {"AF_MODEL", "gpt-4o"},
                                 {"AF_BASE_URL", "http://localhost:8000"}});
  const auto selection = llm::select_backend(env);
  REQUIRE(selection.has_value());
  CHECK(selection->model == "gpt-4o");
  CHECK(selection->base_url == "http://localhost:8000");
}

TEST_CASE("caller options supply defaults that the environment can override", "[llm][factory]") {
  llm::BackendOptions options;
  options.model = "caller-default";

  SECTION("used when AF_MODEL is unset") {
    const auto selection = llm::select_backend(llm::map_env({}), options);
    REQUIRE(selection.has_value());
    CHECK(selection->model == "caller-default");
  }
  SECTION("AF_MODEL wins") {
    const auto selection =
        llm::select_backend(llm::map_env({{"AF_MODEL", "from-env"}}), options);
    REQUIRE(selection.has_value());
    CHECK(selection->model == "from-env");
  }
}

TEST_CASE("backend_from_env builds the selected backend", "[llm][factory]") {
  SECTION("mock") {
    const auto backend = llm::backend_from_env({}, llm::map_env({}));
    REQUIRE(backend.has_value());
    CHECK((*backend)->name() == "mock");
  }
  SECTION("openai") {
    const auto env = llm::map_env({{"OPENAI_API_KEY", kFakeKey}});
    const auto backend = llm::backend_from_env({}, env);
    REQUIRE(backend.has_value());
    CHECK((*backend)->name() == "openai");
  }
  SECTION("mock handler is installed") {
    llm::BackendOptions options;
    options.mock_handler = [](const llm::ChatRequest&) {
      llm::ChatResponse response;
      response.content.push_back(llm::TextBlock{"canned"});
      return response;
    };
    const auto backend = llm::backend_from_env(std::move(options), llm::map_env({}));
    REQUIRE(backend.has_value());
    llm::ChatRequest request;
    request.messages = {llm::Message::user_text("hello")};
    const auto response = (*backend)->generate(request);
    REQUIRE(response.has_value());
    CHECK(response->text() == "canned");
  }
}

TEST_CASE("describe() names the key variable but never its value", "[llm][factory][secrets]") {
  const auto env = llm::map_env({{"OPENAI_API_KEY", kFakeKey}});
  const auto selection = llm::select_backend(env);
  REQUIRE(selection.has_value());

  const std::string described = selection->describe();
  CHECK_THAT(described, ContainsSubstring("OPENAI_API_KEY"));
  CHECK_THAT(described, !ContainsSubstring(kFakeKey));
}

TEST_CASE("a key routed through a backend is registered for log redaction",
          "[llm][factory][secrets]") {
  const auto env = llm::map_env({{"AF_BACKEND", "openai"}, {"OPENAI_API_KEY", kFakeKey}});
  const auto backend = llm::backend_from_env({}, env);
  REQUIRE(backend.has_value());

  const std::string leaked = std::string{"authorization: Bearer "} + kFakeKey;
  const std::string redacted = core::redact(leaked);
  CHECK_THAT(redacted, !ContainsSubstring(kFakeKey));
  CHECK_THAT(redacted, ContainsSubstring("***"));
}
