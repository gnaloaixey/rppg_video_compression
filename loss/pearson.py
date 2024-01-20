import torch
import torch.nn as nn


class PearsonLoss(nn.Module):
    def forward(self,predictions, targets):
        rst = 0
        targets = targets[:, :]
        for i in range(predictions.shape[0]):
            sum_x = torch.sum(predictions[i])  # x
            sum_y = torch.sum(targets[i])  # y
            sum_xy = torch.sum(predictions[i] * targets[i])  # xy
            sum_x2 = torch.sum(torch.pow(predictions[i], 2))  # x^2
            sum_y2 = torch.sum(torch.pow(targets[i], 2))  # y^2
            N = predictions.shape[1] if len(predictions.shape) > 1 else 1
            pearson = (N * sum_xy - sum_x * sum_y) / (
                torch.sqrt((N * sum_x2 - torch.pow(sum_x, 2)) * (N * sum_y2 - torch.pow(sum_y, 2))))

            rst += pearson

        rst = rst / predictions.shape[0]
        return rst

class NegPearsonLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pearson = PearsonLoss()
    def forward(self,predictions, targets):
        return 1 - self.pearson.forward(predictions, targets)