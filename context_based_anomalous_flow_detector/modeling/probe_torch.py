import torch

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Running on: {device}")
# Create a test tensor on your new device
x = torch.ones(1, 4).to(device)
print(x)

print(f"Is Torch using the GPU? {torch.cuda.is_available() or torch.backends.mps.is_available()}")

import torch
import sys

print(f"__Python VERSION: {sys.version}")
print(f"__pyTorch VERSION: {torch.__version__}")
print(f"__CUDA Runtime VERSION: {torch.version.cuda}")

try:
    print(f"__CUDA Available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        # This will give us a specific error code if possible
        print(f"__CUDA Error Count: {torch.cuda.device_count()}")
except Exception as e:
    print(f"__Error during CUDA check: {e}")

# Check if the binary was actually compiled for your 5090 architecture
print(f"__Supported Architectures: {torch.cuda.get_arch_list()}")