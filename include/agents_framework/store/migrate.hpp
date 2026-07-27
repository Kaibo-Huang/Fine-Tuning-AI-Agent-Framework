#pragma once

#include <span>
#include <string_view>

#include "agents_framework/core/result.hpp"
#include "agents_framework/store/db.hpp"

namespace agents_framework::store {

struct Migration {
  int version;
  std::string_view sql;
};

[[nodiscard]] core::Result<int> schema_version(Db& db, std::string_view component);

[[nodiscard]] core::Result<int> migrate(Db& db, std::string_view component,
                                        std::span<const Migration> migrations);

}
