import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from common.cuda_info import get_device
from loss.pearson import PearsonLoss
import numpy as np
evaluation_name = 'result.csv'



def generate_evaluation(model:torch.nn.Module,test_dataloader:DataLoader,write_type = 'a'):
    info_cols = ['compression_codec','resolution',]
    # evaluation
    calculate_mse = nn.MSELoss()
    evaluation_map = dict({
        'mse': calculate_mse,
        'rmse': lambda x,y:torch.sqrt(calculate_mse(x,y)),
        'pearson': PearsonLoss()
    })

    gpu_device = get_device()
    model.eval()
    model.to(gpu_device)

    evaluations = dict()
    for batch_X, batch_y in test_dataloader:
        batch_X = batch_X.to(gpu_device)
        batch_y = batch_y.to(gpu_device)
        outputs = model(batch_X)
        for calculater_name in evaluation_map.keys():
            calculater = evaluation_map.get(calculater_name)
            value = calculater(batch_y,outputs)
            temp_list = evaluations.get(calculater_name,None)
            if temp_list is not list:
                temp_list = list()
                evaluations.setdefault(calculater_name,temp_list)
            temp_list.append(value)
    for evaluation_name in evaluations.keys():
        evaluation = evaluations.get(evaluation_name)
        evaluations.setdefault(evaluation_name,np.array(evaluation,dtype=np.float32).mean())
    print(evaluations)