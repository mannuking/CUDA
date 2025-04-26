import torch
import time
import platform

# CUDA and device info check
print(f"Python version: {platform.python_version()}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"GPU Memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
else:
    print("CUDA is not available on this system.")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Matrix size
matrix_size = 32 * 512
x = torch.randn(matrix_size, matrix_size)
y = torch.randn(matrix_size, matrix_size)

print("************ CPU SPEED ***************")
start = time.time()
result = torch.matmul(x, y)
print(f"Time taken: {time.time() - start:.4f} seconds")
print("verify device:", result.device)

# Move tensors to GPU if available
if torch.cuda.is_available():
    x_gpu = x.to(device)
    y_gpu = y.to(device)
    torch.cuda.synchronize()
    for i in range(3):
        print("************ GPU SPEED ***************")
        start = time.time()
        result_gpu = torch.matmul(x_gpu, y_gpu)
        torch.cuda.synchronize()
        print(f"Time taken: {time.time() - start:.4f} seconds")
        print("verify device:", result_gpu.device)
else:
    print('CUDA is not available. Skipping GPU test.') 
