cmake_minimum_required(VERSION 3.20)

project(LearnCUDA101 LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}) 

add_subdirectory(vector_add)
add_subdirectory(image_blur)
add_subdirectory(convolution)
add_subdirectory(basic_vs_tiled_matrix_mul)
add_subdirectory(parallel_scan)
add_subdirectory(histogram)
add_subdirectory(sparse_matrix_compute)
