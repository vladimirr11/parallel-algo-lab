cmake_minimum_required(VERSION 3.20)

project(02_ImageBlur LANGUAGES CUDA CXX)

set(_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src) 

add_executable(
	${PROJECT_NAME} 
		${_SRC_DIR}/main.cu
)

target_include_directories( 
	${PROJECT_NAME}  
		PRIVATE  
			${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD     
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/data
    $<TARGET_FILE_DIR:${PROJECT_NAME}>/data
)
