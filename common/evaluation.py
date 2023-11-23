import torch
from torch.utils.data import DataLoader
evaluation_name = 'result.csv'

def generate_evaluation(model:torch.nn.Module,dataset_loader:DataLoader):
    info_cols = ['compression_codec','resolution',]
    result_cols = ['compression_ratio','p']
    model.eval()
    for b_x,b_y in dataset_loader:
        pass
