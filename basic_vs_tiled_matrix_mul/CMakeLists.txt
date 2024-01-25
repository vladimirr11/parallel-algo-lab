cmake_minimum_required(VERSION 3.20)

project(04_MatrixMul LANGUAGES CUDA CXX)

set(_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src) 

add_executable(
	${PROJECT_NAME} 
		${_SRC_DIR}/main.cu
)

if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE -Xcompiler /openmp)
else()
    message("Please use Visual Studio generator to build the project")
endif()

target_include_directories( 
	${PROJECT_NAME}  
		PRIVATE  
			${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
