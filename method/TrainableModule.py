from torch.nn import Module

from singleton_pattern.load_config import get_config

class TrainableModule(Module):
    def __init__(self) -> None:
        super().__init__()
        config = get_config()
        data_format = config['data_format']
        self.slice_interval = data_format['slice_interval']
        self.num_epochs = config.get('train',{}).get('num_epochs',10)
    def train_model(self,dataloader,num_epochs = None):
        pass