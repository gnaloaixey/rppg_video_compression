import torch.nn as nn

from loss.pearson import NegPearsonLoss

class PhyNetLoss(nn.Module):
    def __init__(self):
        super(PhyNetLoss, self).__init__()
        self.neg_pearson_Loss = NegPearsonLoss()
    def forward(self, predictions, targets):
        if len(predictions.shape) == 1:
            predictions = predictions.view(1, -1)
        if len(targets.shape) == 1:
            targets = targets.view(1, -1)
        return self.neg_pearson_Loss(predictions, targets)