# import sys
# sys.path.insert(0, "/Users/jongbeomkim/Desktop/workspace/vit_from_scratch")

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.cuda.amp import GradScaler
from time import time
from pathlib import Path

import config
from utils import get_elapsed_time
from model import ViT
from cifar100 import CIFAR100Dataset
from evaluate import TopKAccuracy

# torch.autograd.set_detect_anomaly(True)


def save_checkpoint(epoch, step, model, optim, scaler, save_path):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "epoch": epoch,
        "step": step,
        "optimizer": optim.state_dict(),
        "scaler": scaler.state_dict(),
    }
    if config.N_GPUS > 0 and config.MULTI_GPU:
        ckpt["model"] = model.module.state_dict()
    else:
        ckpt["model"] = model.state_dict()

    torch.save(ckpt, str(save_path))


def validate(test_dl, model, metric):
    model.eval()
    with torch.no_grad():
        sum_acc = 0
        for image, gt in test_dl:
            image = image.to(config.DEVICE)
            gt = gt.to(config.DEVICE)

            pred = model(image)
            acc = metric(pred=pred, gt=gt)
            sum_acc += acc
    avg_acc = sum_acc / len(test_dl)
    print(f"""Average accuracy: {avg_acc:.3f}""")

    model.train()


if __name__ == "__main__":
    print(f"""AUTOCAST = {config.AUTOCAST}""")
    print(f"""N_WORKES = {config.N_WORKERS}""")
    print(f"""BATCH_SIZE = {config.BATCH_SIZE}""")
    print(f"""DEVICE = {config.DEVICE}""")

    train_ds = CIFAR100Dataset(config.DATA_DIR, split="train")
    train_dl = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=True
    )

    # test_ds = CIFAR100Dataset(config.DATA_DIR, split="test")
    test_ds = CIFAR100Dataset(config.DATA_DIR, split="train")
    test_dl = DataLoader(
        test_ds, batch_size=config.BATCH_SIZE, shuffle=False, pin_memory=True, drop_last=False
    )

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
    metric = TopKAccuracy(k=5)

    optim = Adam(
        model.parameters(),
        lr=config.BASE_LR,
        betas=(config.BETA1, config.BETA2),
        weight_decay=config.WEIGHT_DECAY,
    )
    scaler = GradScaler()

    validate(test_dl=test_dl, model=model, metric=metric)

    start_time = time()
    running_loss = 0
    step_cnt = 0
    for epoch in range(1, config.N_EPOCHS + 1):
        for step, (image, gt) in enumerate(train_dl, start=1):
            image = image.to(config.DEVICE)
            gt = gt.to(config.DEVICE)

            with torch.autocast(
                device_type=config.DEVICE.type,
                dtype=torch.float16,
                enabled=True if config.AUTOCAST else False,
            ):
                pred = model(image)
                loss = crit(pred, gt)
            running_loss += loss.item()
            step_cnt += 1

            optim.zero_grad()
            if config.AUTOCAST:
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                optim.step()

        if (epoch % config.N_PRINT_EPOCHS == 0) or (epoch == config.N_EPOCHS):
            # loss = running_loss / (config.N_PRINT_EPOCHS * len(train_dl))
            loss = running_loss / step_cnt
            print(f"""[ {epoch:,}/{config.N_EPOCHS} ][ {step:,}/{len(train_dl):,} ]""", end="")
            print(f"""[ {get_elapsed_time(start_time)} ][ {loss:.3f} ]""")

            running_loss = 0
            step_cnt = 0
            start_time = time()

        if (epoch % config.N_VAL_EPOCHS == 0) or (epoch == config.N_EPOCHS):
            validate(test_dl=test_dl, model=model, metric=metric)

        if (epoch % config.N_CKPT_EPOCHS == 0) or (epoch == config.N_EPOCHS):
            save_checkpoint(
                epoch=epoch,
                step=step,
                model=model,
                optim=optim,
                scaler=scaler,
                save_path=config.CKPT_DIR/f"""{epoch}_{step}.pth""",
            )
            # print(f"""Saved checkpoint at epoch {epoch:,}/{config.N_EPOCHS}""")
            # print(f""" and step {step:,}/{len(train_dl):,}.""")
            print(f"""Saved checkpoint.""")
