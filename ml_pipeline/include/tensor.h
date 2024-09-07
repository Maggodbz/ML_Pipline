#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <stdexcept>
#include <iostream>
#include <string>
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

class Tensor {
public:
    // Attributes
    std::vector<int> shape;       // Shape of the tensor
    int size;                     // Total number of elements
    std::vector<float> data;      // Data on the CPU
#ifdef USE_CUDA
    float* d_data = nullptr;      // Data on the GPU (if using CUDA)
#endif
    std::string device;           // Device type ("cpu" or "gpu")

    // Constructor
    Tensor(const std::vector<int>& shape, const std::string& device = "cpu");

    // Copy constructor
    Tensor(const Tensor& other);

    // Destructor
    ~Tensor();

    // Methods
    Tensor add(const Tensor& other);
    Tensor multiply(const Tensor& other);
    void zero();
    void random();
    float& operator()(int index);

    // Device management
    Tensor to(const std::string& device);
    bool is_on_gpu() const { return device == "gpu"; }

private:
#ifdef USE_CUDA
    void allocate_gpu_memory();  // Allocate memory on the GPU
    void copy_to_gpu();          // Copy data from CPU to GPU
    void copy_to_cpu();          // Copy data from GPU to CPU
#endif
};

#endif // TENSOR_H
