cmake_minimum_required(VERSION 3.10)
project(NAFNet_TensorRT)

set(CMAKE_CXX_STANDARD 14)

# Find CUDA
find_package(CUDA REQUIRED)

# TensorRT typical paths for Linux environments
find_path(TRT_INCLUDE_DIR NvInfer.h HINTS /usr/include/x86_64-linux-gnu /usr/local/tensorrt/include)
find_library(TRT_LIBRARY nvinfer HINTS /usr/lib/x86_64-linux-gnu /usr/local/tensorrt/lib)

include_directories(${TRT_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})

add_executable(infer infer.cpp)
target_link_libraries(infer ${TRT_LIBRARY} ${CUDA_LIBRARIES})