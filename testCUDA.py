import torch

# 顯示 PyTorch 版本
print(f"PyTorch 版本: {torch.__version__}")

# 顯示 CUDA 是否可用
print(f"CUDA 可用: {torch.cuda.is_available()}")

# 顯示 CUDA 版本
print(f"CUDA 版本: {torch.version.cuda}")

# 顯示當前的 CUDA 設置
if torch.cuda.is_available():
    print(f"GPU 數量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"CUDA:{i} ({torch.cuda.get_device_name(i)}, {torch.cuda.get_device_properties(i).total_memory // 1024**2} MB)")
else:
    print("沒有可用的 CUDA 設置")

print(torch.backends.cudnn.version())