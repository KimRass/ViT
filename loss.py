# References:
    # https://github.com/omihub777/ViT-CIFAR/blob/main/criterions.py

import torch
import torch.nn as nn

import config


class ClassificationLoss(nn.Module):
    def __init__(self, n_classes, smoothing=0):
        super().__init__()

        assert 0 <= smoothing <= 1, "The argument `smoothing` must be between 0 and 1!"

        self.smoothing = smoothing
        self.n_classes = n_classes

        self.crit = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, pred, gt):
        new_gt = torch.full_like(pred, fill_value=self.smoothing / (self.n_classes - 1))
        new_gt.scatter_(1, gt.unsqueeze(1), 1 - self.smoothing)
        loss = self.crit(pred, new_gt)
        return loss


if __name__ == "__main__":
    # crit = ClassificationLoss(n_classes=config.N_CLASSES, smoothing=config.SMOOTHING)
    crit = ClassificationLoss(n_classes=config.N_CLASSES)
    crit(pred, gt)
