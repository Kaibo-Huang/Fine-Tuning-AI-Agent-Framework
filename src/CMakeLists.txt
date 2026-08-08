add_library(agents_framework)

target_sources(agents_framework PRIVATE version.cpp)

find_package(Git QUIET)
set(AF_GIT_COMMIT "unknown")
if(Git_FOUND)
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" rev-parse --short=12 HEAD
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    OUTPUT_VARIABLE AF_GIT_COMMIT_OUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE AF_GIT_COMMIT_RESULT
  )
  if(AF_GIT_COMMIT_RESULT EQUAL 0)
    set(AF_GIT_COMMIT "${AF_GIT_COMMIT_OUT}")
  endif()
endif()
set_property(SOURCE version.cpp APPEND PROPERTY
  COMPILE_DEFINITIONS "AF_GIT_COMMIT=\"${AF_GIT_COMMIT}\"")

target_include_directories(agents_framework
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)

target_compile_features(agents_framework PUBLIC cxx_std_23)

target_link_libraries(agents_framework
  PUBLIC
    nlohmann_json::nlohmann_json
  PRIVATE
    CURL::libcurl
    spdlog::spdlog
    unofficial::sqlite3::sqlite3
)

add_subdirectory(core)
add_subdirectory(http)
add_subdirectory(llm)
add_subdirectory(graph)
add_subdirectory(tools)
add_subdirectory(store)
add_subdirectory(trace)
