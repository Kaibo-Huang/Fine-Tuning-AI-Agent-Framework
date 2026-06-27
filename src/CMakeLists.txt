add_library(agents_framework)

target_sources(agents_framework PRIVATE version.cpp)

target_include_directories(agents_framework
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)

target_compile_features(agents_framework PUBLIC cxx_std_23)

target_link_libraries(agents_framework
  PUBLIC
    nlohmann_json::nlohmann_json
  PRIVATE
    spdlog::spdlog
)

add_subdirectory(core)
