cmake_minimum_required(VERSION 3.23)

project(agents-framework
  VERSION 0.0.1
  DESCRIPTION "Fine-Tuning AI Agent Framework"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CURL CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)

enable_testing()

add_subdirectory(src)
add_subdirectory(examples)
add_subdirectory(tests)
