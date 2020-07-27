cmake_minimum_required(VERSION 3.1.0)
project(cuda_image_processing LANGUAGES CXX CUDA)
set(CMAKE_BUILD_TYPE Debug)

include(CTest)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(cuda_image_processing source/main.cu)

set_property(TARGET cuda_image_processing PROPERTY CUDA_STANDARD 11)
target_link_libraries(cuda_image_processing ${OpenCV_LIBS})
