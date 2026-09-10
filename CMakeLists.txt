cmake_minimum_required(VERSION 3.23)

project(hustler)

add_library(tensor)

target_sources(tensor
  PRIVATE src/tensor.cpp
  PUBLIC
    FILE_SET HEADERS
    BASE_DIRS
      include
    FILES
      include/hustler/tensor.hpp
)

target_compile_features(tensor PUBLIC cxx_std_20)


add_executable(hustler src/main.cpp)

target_link_libraries(hustler PRIVATE tensor)
