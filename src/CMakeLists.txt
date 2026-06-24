add_library(agents_framework
  version.cpp
)

target_include_directories(agents_framework
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)

target_compile_features(agents_framework PUBLIC cxx_std_23)
