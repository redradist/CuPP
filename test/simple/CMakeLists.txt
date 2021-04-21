cmake_minimum_required(VERSION 3.17)
project(TestCuda C CXX CUDA)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR})

set(CMAKE_CXX_STANDARD 20)

if(NOT EXISTS "${CMAKE_BINARY_DIR}/conan.cmake")
    message(STATUS "Downloading conan.cmake from https://github.com/conan-io/cmake-conan")
    file(DOWNLOAD "https://raw.githubusercontent.com/conan-io/cmake-conan/v0.16.1/conan.cmake"
         "${CMAKE_BINARY_DIR}/conan.cmake"
         TLS_VERIFY ON)
endif()

include(${CMAKE_BINARY_DIR}/conan.cmake)

conan_cmake_configure(REQUIRES ms-gsl/3.1.0
                      GENERATORS cmake_find_package)

conan_cmake_autodetect(settings)

conan_cmake_install(PATH_OR_REFERENCE .
                    BUILD missing
                    REMOTE conan-center
                    SETTINGS ${settings})

file(GLOB_RECURSE TEST_CUDA_APP_SOURCE_FILES *.c *.cpp *.cu *.cuh)

set_source_files_properties(*.cu PROPERTIES LANGUAGE CUDA)
set_source_files_properties(*.cuh PROPERTIES LANGUAGE CUDA)

add_executable(${PROJECT_NAME} ${TEST_CUDA_APP_SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Cudapp)
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 30)