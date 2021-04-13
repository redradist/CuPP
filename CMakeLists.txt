cmake_minimum_required(VERSION 3.17)
project(Cudapp C CXX CUDA)

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

file(GLOB_RECURSE CUDA_TEST_SOURCE_FILES src/*.c src/*.cpp src/*.cu src/*.cuh)

set_source_files_properties(*.cu PROPERTIES LANGUAGE CUDA)
set_source_files_properties(*.cuh PROPERTIES LANGUAGE CUDA)

add_library(${PROJECT_NAME} STATIC SHARED ${CUDA_TEST_SOURCE_FILES})
