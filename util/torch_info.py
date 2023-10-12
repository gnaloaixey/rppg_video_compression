import torch

# 检查是否支持 CUDA（GPU 加速）
if torch.cuda.is_available():
    print('CUDA:\n---------------------------')
    # 获取当前可用的 CUDA 设备数量
    device_count = torch.cuda.device_count()
    print(f"PyTorch supports GPU and currently has {device_count} CUDA devices available.")

    # 获取当前默认的 CUDA 设备
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    print(f"The current default CUDA device is: {device_name}")
    print('---------------------------')
else:
    print("PyTorch does not support GPU acceleration.")
