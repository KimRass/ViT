# import sys
# sys.path.insert(0, "/Users/jongbeomkim/Desktop/workspace/vit_from_scratch")

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.cuda.amp import GradScaler
from time import time

import config
from utils import get_elapsed_time
from model import ViT, ViTClsHead
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
)
head = ViTClsHead(hidden_dim=config.HIDDEN_DIM, n_classes=config.N_CLASSES)
if config.N_GPUS > 0:
    model = model.to(config.DEVICE)
    head = head.to(config.DEVICE)
    if config.MULTI_GPU:
        # model = DDP(model)
        model = nn.DataParallel(model)
        head = nn.DataParallel(head)

crit = nn.CrossEntropyLoss()

# "Adam with $beta_{1} = 0.9$, $beta_{2}= 0.999$, a batch size of 4096 and apply a high weight decay
# of 0.1, which we found to be useful for transfer of all models."
BETA1 = 0.9
BETA2 = 0.999
WEIGHT_DECAY = 0.1
optim = Adam(model.parameters(), betas=(BETA1, BETA2), weight_decay=WEIGHT_DECAY)

scaler = GradScaler()

running_loss = 0
for epoch in range(1, config.N_EPOCHS + 1):
    running_loss = 0
    for step, (image, gt) in enumerate(train_dl, start=1):
        image = image.to(config.DEVICE)
        gt = gt.to(config.DEVICE)

        with torch.autocast(
            device_type=config.DEVICE.type, dtype=torch.float16, enabled=True if config.AUTOCAST else False
        ):
            out = model(image)
            pred = head(out)
            loss = crit(pred, gt)
        running_loss += loss.item()

        optim.zero_grad()
        if config.AUTOCAST:
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            optim.step()

        if (step % config.N_PRINT_STEPS == 0) or (step == len(train_dl)):
            running_loss /= config.N_PRINT_STEPS
            print(f"""[ {epoch:,}/{config.N_EPOCHS} ][ {step:,}/{len(train_dl):,} ]""", end="")
            print(f"""[ {running_loss:.3f} ]""")
