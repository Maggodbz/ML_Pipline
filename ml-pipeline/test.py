import numpy as np
import time
import tensor_add  # Ensure your C++ module is correctly compiled and available

# Function to perform native Python list addition
def python_native_add(tensor1, tensor2):
    return [x + y for x, y in zip(tensor1, tensor2)]

# Function to benchmark NumPy, Python native list, and C++ addition multiple times and take the average
def benchmark_addition(num_iterations=100):
    # Create large tensors for benchmarking
    tensor1_large_numpy = np.random.rand(1000000).astype(np.float32)
    tensor2_large_numpy = np.random.rand(1000000).astype(np.float32)

    # Convert NumPy arrays to Python lists for the Python native version
    tensor1_large_python = tensor1_large_numpy.tolist()
    tensor2_large_python = tensor2_large_numpy.tolist()

    python_native_times = []
    numpy_times = []
    cpp_times = []

    # Run the benchmark num_iterations times
    for _ in range(num_iterations):
        # Benchmark NumPy addition
        start_time_numpy = time.time()
        result_numpy = tensor1_large_numpy + tensor2_large_numpy
        end_time_numpy = time.time()
        numpy_times.append(end_time_numpy - start_time_numpy)

        # Benchmark Python native addition
        start_time_python_native = time.time()
        result_python_native = python_native_add(tensor1_large_python, tensor2_large_python)
        end_time_python_native = time.time()
        python_native_times.append(end_time_python_native - start_time_python_native)

        # Benchmark C++ addition (using tensor_add module)
        start_time_cpp = time.time()
        result_cpp = tensor_add.add_tensors(tensor1_large_numpy, tensor2_large_numpy)
        end_time_cpp = time.time()
        cpp_times.append(end_time_cpp - start_time_cpp)

    # Calculate the average times
    avg_python_native_time = sum(python_native_times) / num_iterations
    avg_numpy_time = sum(numpy_times) / num_iterations
    avg_cpp_time = sum(cpp_times) / num_iterations

    return avg_python_native_time, avg_numpy_time, avg_cpp_time

# Run the benchmark and print the average times
avg_python_native_time, avg_numpy_time, avg_cpp_time = benchmark_addition(num_iterations=100)
print(f"Average Python native list addition time: {avg_python_native_time} seconds")
print(f"Average NumPy addition time: {avg_numpy_time} seconds")
print(f"Average C++ addition time: {avg_cpp_time} seconds")
