#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <vector>

namespace py = pybind11;

// Function to add two tensors
py::array_t<float> add_tensors(py::array_t<float> tensor1, py::array_t<float> tensor2) {
    // Request buffers from the Python arrays
    py::buffer_info buf1 = tensor1.request();
    py::buffer_info buf2 = tensor2.request();

    if (buf1.size != buf2.size) {
        throw std::runtime_error("Tensors must have the same size");
    }

    // Access the data pointers
    float *ptr1 = static_cast<float *>(buf1.ptr);
    float *ptr2 = static_cast<float *>(buf2.ptr);

    // Create a new numpy array for the result
    py::array_t<float> result = py::array_t<float>(buf1.size);
    py::buffer_info buf_result = result.request();
    float *ptr_result = static_cast<float *>(buf_result.ptr);

    // Perform element-wise addition
    for (pybind11::ssize_t i = 0; i < buf1.size; ++i) {  // Use pybind11::ssize_t for size compatibility
        ptr_result[i] = ptr1[i] + ptr2[i];
    }

    return result;
}

// Pybind11 module definition
PYBIND11_MODULE(tensor_add, m) {
    m.def("add_tensors", &add_tensors, "A function that adds two tensors");
}
