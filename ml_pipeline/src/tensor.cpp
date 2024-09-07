#include "tensor.h"
#include <cstdlib> // For rand()

// Constructor: Initializes the tensor on the specified device (CPU or GPU)
Tensor::Tensor(const std::vector<int>& shape, const std::string& device)
    : shape(shape), device(device) {
    size = 1;
    for (int dim : shape) {
        size *= dim;
    }
    data.resize(size);  // Allocate space on CPU

#ifdef USE_CUDA
    if (device == "gpu") {
        allocate_gpu_memory();  // Allocate space on GPU
    }
#endif
}

// Destructor: Free GPU memory if allocated
Tensor::~Tensor() {
#ifdef USE_CUDA
    if (d_data) {
        cudaFree(d_data);  // Free GPU memory if allocated
    }
#endif
}

// Copy constructor: Deep copy for both CPU and GPU data
Tensor::Tensor(const Tensor& other)
    : shape(other.shape), size(other.size), data(other.data), device(other.device) {
#ifdef USE_CUDA
    if (other.is_on_gpu()) {
        allocate_gpu_memory();
        cudaMemcpy(d_data, other.d_data, size * sizeof(float), cudaMemcpyDeviceToDevice);
    }
#endif
}

// Element-wise addition
Tensor Tensor::add(const Tensor& other) {
    if (size != other.size) {
        throw std::invalid_argument("Tensors must have the same size for addition.");
    }
    Tensor result(shape, device);  // Result is created on the same device
    if (device == "cpu") {
        for (int i = 0; i < size; ++i) {
            result.data[i] = data[i] + other.data[i];
        }
    } else {
#ifdef USE_CUDA
        // Add GPU kernel function here (omitted for simplicity)
#endif
    }
    return result;
}

// Element-wise multiplication
Tensor Tensor::multiply(const Tensor& other) {
    if (size != other.size) {
        throw std::invalid_argument("Tensors must have the same size for multiplication.");
    }
    Tensor result(shape, device);  // Result is created on the same device
    if (device == "cpu") {
        for (int i = 0; i < size; ++i) {
            result.data[i] = data[i] * other.data[i];
        }
    } else {
#ifdef USE_CUDA
        // Multiply GPU kernel function here (omitted for simplicity)
#endif
    }
    return result;
}

// Set all values to zero
void Tensor::zero() {
    if (device == "cpu") {
        for (int i = 0; i < size; ++i) {
            data[i] = 0.0f;
        }
    } else {
#ifdef USE_CUDA
        // Set GPU values to zero (GPU kernel code omitted)
#endif
    }
}

// Initialize with random values
void Tensor::random() {
    if (device == "cpu") {
        for (int i = 0; i < size; ++i) {
            data[i] = static_cast<float>(rand()) / RAND_MAX; // Random float between 0 and 1
        }
    } else {
#ifdef USE_CUDA
        // Initialize random values on GPU (GPU kernel code omitted)
#endif
    }
}

// Access the data using a 1D index (only valid on CPU)
float& Tensor::operator()(int index) {
    if (device == "gpu") {
        throw std::runtime_error("Cannot access GPU data directly. Please move to CPU first.");
    }
    if (index >= size) {
        throw std::out_of_range("Tensor index out of range.");
    }
    return data[index];
}

// Allocate memory on the GPU
#ifdef USE_CUDA
void Tensor::allocate_gpu_memory() {
    cudaMalloc(&d_data, size * sizeof(float));
}

// Copy data from CPU to GPU
void Tensor::copy_to_gpu() {
    if (device == "cpu") {
        if (!d_data) {
            allocate_gpu_memory();
        }
        cudaMemcpy(d_data, data.data(), size * sizeof(float), cudaMemcpyHostToDevice);
        device = "gpu";
    }
}

// Copy data from GPU to CPU
void Tensor::copy_to_cpu() {
    if (device == "gpu") {
        cudaMemcpy(data.data(), d_data, size * sizeof(float), cudaMemcpyDeviceToHost);
        device = "cpu";
    }
}
#endif

// Move tensor to a different device (CPU or GPU)
Tensor Tensor::to(const std::string& target_device) {
    if (target_device == device) {
        return *this;  // Already on the target device
    }

#ifdef USE_CUDA
    if (target_device == "gpu") {
        copy_to_gpu();
    } else if (target_device == "cpu") {
        copy_to_cpu();
    }
#endif

    return *this;
}
