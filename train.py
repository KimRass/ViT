# References:
    # https://github.com/omihub777/ViT-CIFAR

# import sys
# sys.path.insert(0, "/Users/jongbeomkim/Desktop/workspace/vit_from_scratch")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.cuda.amp import GradScaler
from time import time
from pathlib import Path
from tqdm.auto import tqdm

import config
from utils import get_elapsed_time
from model import ViT
from loss import ClassificationLoss
from cifar10 import CIFAR10Dataset
from cifar100 import CIFAR100Dataset
from evaluate import TopKAccuracy
from hide_and_seek import apply_hide_and_seek
from cutmix import apply_cutmix

torch.set_printoptions(linewidth=200, sci_mode=False)
torch.manual_seed(config.SEED)

# torch.autograd.set_detect_anomaly(True)


def save_checkpoint(epoch, model, optim, scaler, save_path):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "epoch": epoch,
        "optimizer": optim.state_dict(),
        "scaler": scaler.state_dict(),
    }
    if config.N_GPUS > 0 and config.MULTI_GPU:
        ckpt["model"] = model.module.state_dict()
    else:
        ckpt["model"] = model.state_dict()

    torch.save(ckpt, str(save_path))


@torch.no_grad()
def validate(test_dl, model, metric):
    model.eval()
    sum_acc = 0
    for image, gt in tqdm(test_dl):
        image = image.to(config.DEVICE)
        gt = gt.to(config.DEVICE)

        pred = model(image)
        acc = metric(pred=pred, gt=gt)
        sum_acc += acc
    avg_acc = sum_acc / len(test_dl)
    print(f"""Average accuracy: {avg_acc:.3f}""")
    model.train()
    return avg_acc


if __name__ == "__main__":
    print(f"""AUTOCAST = {config.AUTOCAST}""")
    print(f"""N_WORKERS = {config.N_WORKERS}""")
    print(f"""BATCH_SIZE = {config.BATCH_SIZE}""")
    print(f"""DEVICE = {config.DEVICE}""")

    train_ds = CIFAR10Dataset(config.DATA_DIR, split="train")
    train_dl = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=True,
    )

    test_ds = CIFAR10Dataset(config.DATA_DIR, split="test")
    test_dl = DataLoader(
        test_ds, batch_size=config.BATCH_SIZE, shuffle=False, pin_memory=True, drop_last=True,
    )

    model = ViT(
        img_size=config.IMG_SIZE,
        patch_size=config.PATCH_SIZE,
        n_layers=config.N_LAYERS,
        hidden_dim=config.HIDDEN_DIM,
        n_heads=config.N_HEADS,
        n_classes=config.N_CLASSES,
    )
    model.train()
    if config.N_GPUS > 0:
        model = model.to(config.DEVICE)
        if config.MULTI_GPU:
            # model = DDP(model)
            model = nn.DataParallel(model)

    crit = ClassificationLoss(n_classes=config.N_CLASSES, smoothing=config.SMOOTHING)
    metric = TopKAccuracy(k=1)

    optim = Adam(
        model.parameters(),
        lr=config.BASE_LR,
        betas=(config.BETA1, config.BETA2),
        weight_decay=config.WEIGHT_DECAY,
    )

    scaler = GradScaler(enabled=True if config.AUTOCAST else False)

    ### Resume
    if config.CKPT_PATH is not None:
        ckpt = torch.load(config.CKPT_PATH, map_location=config.DEVICE)
        if config.N_GPUS > 1 and config.MULTI_GPU:
            model.module.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt["model"])
        init_epoch = ckpt["epoch"]
        optim.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        print(f"""Resuming from checkpoint {config.CKPT_PATH}.""")

        prev_save_path = config.CKPT_PATH
    else:
        init_epoch = 0
        prev_save_path = ".pth"

    start_time = time()
    running_loss = 0
    step_cnt = 0
    best_avg_acc = 0
    for epoch in range(init_epoch + 1, config.N_EPOCHS + 1):
        for step, (image, gt) in enumerate(train_dl, start=1):
            image = image.to(config.DEVICE)
            gt = gt.to(config.DEVICE)

            if config.HIDE_AND_SEEK:
                image = apply_hide_and_seek(
                    image, patch_size=config.IMG_SIZE // 4, mean=(0.507, 0.487, 0.441)
                )
            if config.CUTMIX:
                image, gt = apply_cutmix(image=image, label=F.one_hot(gt, num_classes=config.N_CLASSES))

            with torch.autocast(
                device_type=config.DEVICE.type,
                dtype=torch.float16,
                enabled=True if config.AUTOCAST else False,
            ):
                pred = model(image)
                loss = crit(pred, gt)
            optim.zero_grad()
            if config.AUTOCAST:
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                optim.step()

            running_loss += loss.item()
            step_cnt += 1

        if (epoch % config.N_PRINT_EPOCHS == 0) or (epoch == config.N_EPOCHS):
            # loss = running_loss / (config.N_PRINT_EPOCHS * len(train_dl))
            loss = running_loss / step_cnt
            print(f"""[ {epoch:,}/{config.N_EPOCHS} ][ {step:,}/{len(train_dl):,} ]""", end="")
            print(f"""[ {get_elapsed_time(start_time)} ][ {int(loss):,} ]""")

            running_loss = 0
            step_cnt = 0
            start_time = time()

        if (epoch % config.N_VAL_EPOCHS == 0) or (epoch == config.N_EPOCHS):
            avg_acc = validate(test_dl=test_dl, model=model, metric=metric)
            if avg_acc > best_avg_acc:
                cur_save_path = config.CKPT_DIR/f"""epoch_{epoch}_avg_acc_{round(best_avg_acc, 3)}.pth"""
                save_checkpoint(
                    epoch=epoch,
                    model=model,
                    optim=optim,
                    scaler=scaler,
                    save_path=cur_save_path,
                )
                prev_save_path = Path(prev_save_path)
                if prev_save_path:
                    prev_save_path.unlink()
                print(f"""Saved checkpoint.""")

                best_avg_acc = avg_acc
                prev_save_path = cur_save_path
