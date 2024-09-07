#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "tensor.h"

namespace py = pybind11;

PYBIND11_MODULE(my_pytorch, m) {
    m.doc() = R"pbdoc(
        my_pytorch - A minimal tensor library with CPU/GPU support
        ---------------------------------------------------------
        This module provides basic tensor operations and device management similar to PyTorch.
    )pbdoc";  // Module docstring

    py::class_<Tensor>(m, "Tensor", R"pbdoc(
        A tensor object that supports operations on CPU and GPU.
        
        Args:
            shape (list[int]): Shape of the tensor.
            device (str, optional): Device where the tensor is allocated ('cpu' or 'gpu'). Defaults to 'cpu'.
        )pbdoc")
        .def(py::init<const std::vector<int>&, const std::string&>(), py::arg("shape"), py::arg("device") = "cpu", 
             R"pbdoc(
                Create a tensor with the given shape and device.

                Args:
                    shape (list[int]): Shape of the tensor.
                    device (str, optional): The device to allocate the tensor on ('cpu' or 'gpu'). Defaults to 'cpu'.
             )pbdoc")
        .def("add", &Tensor::add, R"pbdoc(
            Add two tensors element-wise.

            Args:
                other (Tensor): The tensor to add.

            Returns:
                Tensor: A new tensor with the element-wise sum.
        )pbdoc")
        .def("multiply", &Tensor::multiply, R"pbdoc(
            Multiply two tensors element-wise.

            Args:
                other (Tensor): The tensor to multiply with.

            Returns:
                Tensor: A new tensor with the element-wise product.
        )pbdoc")
        .def("random", &Tensor::random, R"pbdoc(
            Initialize the tensor with random values between 0 and 1.
        )pbdoc")
        .def("zero", &Tensor::zero, R"pbdoc(
            Set all elements of the tensor to zero.
        )pbdoc")
        .def("__call__", &Tensor::operator(), py::arg("index"), R"pbdoc(
            Access an element of the tensor by index.

            Args:
                index (int): Index of the element to access.

            Returns:
                float: The value at the specified index.
        )pbdoc")
        .def_readonly("size", &Tensor::size, R"pbdoc(
            The total number of elements in the tensor.
        )pbdoc")
        .def_readonly("shape", &Tensor::shape, R"pbdoc(
            The shape of the tensor as a list of integers.
        )pbdoc")
        .def("to", &Tensor::to, py::arg("device"), R"pbdoc(
            Move the tensor to the specified device ('cpu' or 'gpu').

            Args:
                device (str): The target device ('cpu' or 'gpu').

            Returns:
                Tensor: The tensor on the target device.
        )pbdoc")
        .def("to_numpy", [](Tensor &self) {
            if (self.is_on_gpu()) {
                self.to("cpu");  // Move data to CPU before conversion to NumPy
            }
            return py::array_t<float>(
                self.size,        // Shape (1D array, size is the total number of elements)
                self.data.data()  // Pointer to the underlying data (on CPU)
            );
        }, R"pbdoc(
            Convert the tensor to a NumPy array.

            Note: If the tensor is on the GPU, it will be moved to the CPU first.

            Returns:
                numpy.ndarray: The tensor as a NumPy array.
        )pbdoc");
}
