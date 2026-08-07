#pragma once

#include <string_view>

namespace agents_framework {

[[nodiscard]] std::string_view version() noexcept;

[[nodiscard]] std::string_view commit() noexcept;

}
