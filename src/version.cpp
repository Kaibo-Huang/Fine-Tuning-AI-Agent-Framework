#include "agents_framework/version.hpp"

#ifndef AF_GIT_COMMIT
#define AF_GIT_COMMIT "unknown"
#endif

namespace agents_framework
{
  std::string_view version() noexcept
  {
    return "0.0.1";
  }

  std::string_view commit() noexcept
  {
    return AF_GIT_COMMIT;
  }
}
