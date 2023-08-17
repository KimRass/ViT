# import sys
# sys.path.insert(0, "/Users/jongbeomkim/Desktop/workspace/vit_from_scratch")

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.cuda.amp import GradScaler

import config
from model import ViT
from cifar100 import CIFAR100Dataset

print(config.DEVICE)

train_ds = CIFAR100Dataset(config.DATA_DIR, split="train")
train_dl = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=True)

model = ViT(
    img_size=config.IMG_SIZE,
    patch_size=config.PATCH_SIZE,
    n_layers=config.N_LAYERS,
    hidden_dim=config.HIDDEN_DIM,
    n_heads=config.N_HEADS,
    n_classes=config.N_CLASSES,
)
if config.N_GPUS > 0:
    model = model.to(config.DEVICE)
    if config.MULTI_GPU:
        # model = DDP(model)
        model = nn.DataParallel(model)

crit = nn.CrossEntropyLoss()

# "Adam with $beta_{1} = 0.9$, $beta_{2}= 0.999$, a batch size of 4096 and apply a high weight decay
# of 0.1, which we found to be useful for transfer of all models."
BETA1 = 0.9
BETA2 = 0.999
WEIGHT_DECAY = 0.1
optim = Adam(model.parameters(), betas=(BETA1, BETA2), weight_decay=WEIGHT_DECAY)

scaler = GradScaler()

N_EPOCHS = 100

running_loss = 0
for epoch in range(1, N_EPOCHS + 1):
    running_loss = 0
    for step, (image, gt) in enumerate(train_dl, start=1):
        image = image.to(config.DEVICE)
        gt = gt.to(config.DEVICE)

        with torch.autocast(
            device_type=config.DEVICE.type, dtype=torch.float16, enabled=True if config.AUTOCAST else False
        ):
            pred = model(image)
            loss = crit(pred, gt)
            print(f"""{loss.item():.4f}""")
        running_loss += loss.item()

        optim.zero_grad()
        if config.AUTOCAST:
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            optim.step()