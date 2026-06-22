cmake_minimum_required(VERSION 3.23)

project(agents-framework
  VERSION 0.0.1
  DESCRIPTION "Fine-Tuning AI Agent Framework"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_testing()

add_subdirectory(src)
add_subdirectory(tests)
