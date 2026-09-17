cmake_minimum_required(VERSION 3.23)

project(hustler)

add_library(tensor)

target_sources(tensor
  PRIVATE
    src/dense_layer.cpp
    src/loss.cpp
    src/optimizer.cpp
    src/tensor.cpp
  PUBLIC
    FILE_SET HEADERS
    BASE_DIRS
      include
    FILES
      include/hustler/tensor.hpp
      include/hustler/dense_layer.hpp
      include/hustler/loss.hpp
      include/hustler/optimizer.hpp
)

target_compile_features(tensor PUBLIC cxx_std_20)


add_executable(hustler src/main.cpp)

target_link_libraries(hustler PRIVATE tensor)
