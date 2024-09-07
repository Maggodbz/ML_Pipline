

# ML Pipeline from Scratch in C++ with Python Interface (Inspired by PyTorch)

This project implements basic tensor operations on both CPU and GPU, such as tensor addition, multiplication, and device management. The core functionality is implemented in C++, and the Python interface is provided using `pybind11` to allow seamless interaction with Python scripts.


## Features

- **Tensor Operations**: Implement basic tensor functionality such as random initialization, zeroing out a tensor, and element-wise addition and multiplication.
- **Device Management**: Move tensors between CPU and GPU.
- **Python Interface**: Utilize the power of C++ in Python using `pybind11`.
- **Auto-generated Python Stubs**: Generate `.pyi` files for improved IDE support and type hinting.

## Getting Started

### Prerequisites

Ensure you have the following installed:
- C++ Compiler (with support for C++11 or higher)
- `cmake` for building the project
- Python 3.11 with `poetry` installed
- CUDA (if using GPU)

### Build Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/ml-pipeline.git
   cd ml-pipeline
   ```

2. **Make the build script executable**:
   ```bash
   chmod +x build.sh
   ```

3. **Run the build script**:
   ```bash
   ./build.sh
   ```

   This will:
   - Remove and recreate the `build/` directory.
   - Run `cmake` and `make` to compile the C++ code.
   - Auto-generate the `.pyi` stubs for Python.

### Running the Tests

After building the project, you can run the Python test script to verify everything works as expected:

```bash
python ml_pipeline/python/test.py
```


## Development Workflow

1. Modify the C++ code in `src/tensor.cpp` and `include/tensor.h`.
2. Update the Python bindings in `bindings/pybind_module.cpp`.
3. Rebuild the project using `./build.sh`.
4. Test with the Python script.

## Future Work

This is a foundational project. Potential extensions include:
- Adding more tensor operations (e.g., matrix multiplication, reshaping).
- Extending device support (e.g., adding multi-GPU).
- Incorporating automatic differentiation.
- and more!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
