import torch
import torch.nn as nn

class NegPearsonLoss(nn.Module):
    def forward(self,predictions, targets):
        # assert x.shape == y.shape, "Input tensors must have the same shape"
        # mean_x = torch.mean(x, dim=0, keepdim=True)
        # mean_y = torch.mean(y, dim=0, keepdim=True)
        # numerator = torch.sum((x - mean_x) * (y - mean_y), dim=0)
        # denominator_x = torch.sqrt(torch.sum((x - mean_x)**2, dim=0))
        # denominator_y = torch.sqrt(torch.sum((y - mean_y)**2, dim=0))
        # correlation = numerator / (denominator_x * denominator_y)

        # correlation = torch.mean(correlation)

        # loss = 1 - correlation

        # return loss

        rst = 0
        targets = targets[:, :]
        # predictions = torch.squeeze(predictions)
        # Pearson correlation can be performed on the premise of normalization of input data
        predictions = (predictions - torch.mean(predictions, dim=-1, keepdim=True)) / torch.std(predictions, dim=-1,
                                                                                                keepdim=True)
        targets = (targets - torch.mean(targets, dim=-1, keepdim=True)) / torch.std(targets, dim=-1, keepdim=True)

        for i in range(predictions.shape[0]):
            sum_x = torch.sum(predictions[i])  # x
            sum_y = torch.sum(targets[i])  # y
            sum_xy = torch.sum(predictions[i] * targets[i])  # xy
            sum_x2 = torch.sum(torch.pow(predictions[i], 2))  # x^2
            sum_y2 = torch.sum(torch.pow(targets[i], 2))  # y^2
            N = predictions.shape[1] if len(predictions.shape) > 1 else 1
            pearson = (N * sum_xy - sum_x * sum_y) / (
                torch.sqrt((N * sum_x2 - torch.pow(sum_x, 2)) * (N * sum_y2 - torch.pow(sum_y, 2))))

            rst += 1 - pearson

        rst = rst / predictions.shape[0]
        return rst