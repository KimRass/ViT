import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from model import ViT
from cifar100 import CIFAR100Dataset

ds = CIFAR100Dataset(config.DATA_DIR, split="train")
dl = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=True)

image, gt = next(iter(dl))
gt.shape, pred.shape

vit = ViT(
    img_size=config.IMG_SIZE,
    patch_size=config.PATCH_SIZE,
    n_layers=config.N_LAYERS,
    hidden_dim=config.HIDDEN_DIM,
    n_heads=config.N_HEADS,
    n_classes=config.N_CLASSES,
)
pred = vit(image)
pred = torch.argmax(output, dim=1)
pred

crit = nn.CrossEntropyLoss()
crit(pred, gt)