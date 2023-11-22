from torch.nn import Module

from singleton_pattern.load_config import get_config
class BaseMethod(Module):
    def __init__(self) -> None:
        super().__init__()
        config = get_config()
        data_format = config['data_format']
        self.slice_interval = data_format['slice_interval']
