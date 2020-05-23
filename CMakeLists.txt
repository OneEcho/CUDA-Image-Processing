cmake_minimum_required(VERSION 3.0.0)
project(cuda-image-processing VERSION 0.1.0)

include(CTest)
enable_testing()
enable_language(CUDA)

add_executable(cuda-image-processing source/main.cu)

set_property(TARGET cuda-image-processing PROPERTY CUDA_STANDARD 11)
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
target_include_directories(cuda-image-processing PRIVATE images)

include(CPack)
