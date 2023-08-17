import torch
import torch.nn as nn


class TopKAccuracy(nn.Module):
    def __init__(self, k):
        super().__init__()

        self.k = k

    def forward(self, pred, gt):
        b, _ = pred.shape

        _, topk = torch.topk(pred, k=self.k, dim=1)
        corr = torch.eq(topk, gt.unsqueeze(1).repeat(1, 5))
        acc = corr.sum().item() / b
        return acc
