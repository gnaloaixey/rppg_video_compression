import torch as __torch
from singleton_pattern import load_config as __load_config
# 检查是否支持 CUDA（GPU 加速）
def print_info():
    if __torch.cuda.is_available():
        print('CUDA:\n---------------------------')
        # 获取当前可用的 CUDA 设备数量
        device_count = __torch.cuda.device_count()
        print(f"PyTorch supports GPU and currently has {device_count} CUDA devices available.")

        # 获取当前默认的 CUDA 设备
        current_device = __torch.cuda.current_device()
        device_name = __torch.cuda.get_device_name(current_device)
        print(f"The current default CUDA device is: {device_name}")
        print('---------------------------')
    else:
        print("PyTorch does not support GPU acceleration.")
def get_device():
    cuda_index = __load_config.get_config()['cuda']
    if cuda_index == None:
        cuda_index = 0
    return __torch.device(f"cuda:{cuda_index}" if __torch.cuda.is_available() else "cpu")
