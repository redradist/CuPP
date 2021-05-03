cmake_minimum_required(VERSION 3.18)
project(TestCuda C CXX CUDA)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR})

set(CMAKE_CXX_STANDARD 17)

find_package(CUDAToolkit REQUIRED)
file(GLOB_RECURSE TEST_CUDA_APP_SOURCE_FILES *.c *.cpp *.cu *.cuh)

set_source_files_properties(*.cu PROPERTIES LANGUAGE CUDA)
set_source_files_properties(*.cuh PROPERTIES LANGUAGE CUDA)

add_executable(${PROJECT_NAME} ${TEST_CUDA_APP_SOURCE_FILES})
target_include_directories(${PROJECT_NAME} PRIVATE "${CUDAToolkit_INCLUDE_DIRS}")
target_link_libraries(${PROJECT_NAME} CuPP CUDA::cudart CUDA::cuda_driver)
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 30)