#include "agents_framework/http/secrets.hpp"

#include <cstdlib>
#include <utility>

#include "agents_framework/core/logging.hpp"

namespace agents_framework::http {

Secret::Secret(std::string value) : value_{std::move(value)} {
  core::add_redaction(value_);
}

core::Result<Secret> SecretStore::from_env(std::string_view var) {
  const std::string name{var};
  const char* value = std::getenv(name.c_str());
  if (value == nullptr || *value == '\0') {
    return core::fail(core::ErrorCode::Auth, "environment variable not set", name);
  }
  return Secret{std::string{value}};
}

}
