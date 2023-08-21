import torch
import torch.nn as nn


class TopKAccuracy(nn.Module):
    def __init__(self, k):
        super().__init__()

        self.k = k

    def forward(self, pred, gt):
        _, topk = torch.topk(pred, k=self.k, dim=1)
        corr = torch.eq(topk, gt.unsqueeze(1).repeat(1, self.k))
        acc = corr.sum(dim=1).float().mean().item()
        return acc


if __name__ == "__main__":
    metric = TopKAccuracy(k=1)
    pred = torch.randn(16, 100)
    gt = torch.argmax(pred, dim=1)
    metric(pred, gt)
