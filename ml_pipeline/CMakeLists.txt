cmake_minimum_required(VERSION 3.1)
project(my_pytorch)

set(CMAKE_CXX_STANDARD 14)  # Use C++14
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find pybind11
find_package(pybind11 REQUIRED)

# Find CUDA
find_package(CUDA REQUIRED)  # This will locate your CUDA installation

# Include directories for your header files
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CUDA_INCLUDE_DIRS})  # Add CUDA include directory

# Define the source files for your project
set(SOURCES
    src/tensor.cpp
    bindings/pybind_module.cpp
)


# Enable GPU support if CUDA is found
add_definitions(-DUSE_CUDA)

# Build the module as a shared library with Pybind11
pybind11_add_module(my_pytorch ${SOURCES})

# Link CUDA libraries using the keyword signature
target_link_libraries(my_pytorch PRIVATE ${CUDA_LIBRARIES})
