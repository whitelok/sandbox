cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

project(playground LANGUAGES CXX)

include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
  # for cute header
  include_directories(thirdparty/cutlass/include)

  # for GPU sm
  if(NOT DEFINED SM)
    list(APPEND CMAKE_CUDA_ARCHITECTURES 90)
  else()
    list(APPEND CMAKE_CUDA_ARCHITECTURES ${SM})
  endif()

  add_subdirectory(src)
endif()
