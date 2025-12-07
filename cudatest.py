import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")
print(f"Device Count: {torch.cuda.device_count()}")
# This line is the ultimate test:
test_tensor = torch.randn(10, 10).cuda()
print(f"Tensor on GPU: {test_tensor.device}")