cmake_minimum_required(VERSION 3.20)

project(03_Convolution LANGUAGES CUDA CXX)

set(_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src) 

add_executable(
	${PROJECT_NAME} 
		${_SRC_DIR}/Convolution.h
		${_SRC_DIR}/CPUConvolution.cpp
		${_SRC_DIR}/GPUConvolution.cu
		${_SRC_DIR}/main.cpp
)

target_include_directories( 
	${PROJECT_NAME}  
		PRIVATE  
			${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
