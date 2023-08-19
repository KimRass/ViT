import torch
import torch.nn as nn


class TopKAccuracy(nn.Module):
    def __init__(self, k):
        super().__init__()

        self.k = k

    def forward(self, pred, gt):
        b, _ = pred.shape

        _, topk = torch.topk(pred, k=self.k, dim=1)
        corr = torch.eq(topk, gt.unsqueeze(1).repeat(1, self.k))
        acc = corr.sum().item() / b
        return acc


if __name__ == "__main__":
    metric = TopKAccuracy(k=5)
    pred = torch.randn(16, 100)
    gt = torch.argmax(pred, dim=1)
    metric(pred, gt)

    crit = nn.CrossEntropyLoss()
    # pred = torch.Tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    # gt = torch.tensor([0, 1, 2, 3]).long()
    pred = torch.Tensor([[1, 0, 0, 0], [0, 1, 0, 0]])
    gt = torch.tensor([0, 1]).long()
    # pred.shape, gt.shape
    crit(pred, gt)
    pred, gt
    
    # Example of target with class indices
    loss = nn.CrossEntropyLoss()
    input = torch.randn(3, 5, requires_grad=True)
    # target = torch.empty(3, dtype=torch.long).random_(5)
    target = torch.randn(3, 5).softmax(dim=1)
    target
    input.shape, target.shape
    loss(input, target)

    