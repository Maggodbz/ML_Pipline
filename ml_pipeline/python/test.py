from ml_pipeline.build.my_pytorch import Tensor

# Create a tensor on CPU
tensor_cpu = Tensor([3])
tensor_cpu.random()
print("Tensor on CPU:", tensor_cpu.to_numpy())

# Move tensor to GPU
tensor_gpu = tensor_cpu.to("gpu")
print("Moved tensor to GPU.")

# Move it back to CPU and print the values
tensor_back_to_cpu = tensor_gpu.to("cpu")
print("Tensor moved back to CPU:", tensor_back_to_cpu.to_numpy())
