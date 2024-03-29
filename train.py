# References:
    # https://github.com/omihub777/ViT-CIFAR
    # https://github.com/huggingface/pytorch-image-models/blob/main/timm/scheduler/cosine_lr.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.cuda.amp import GradScaler
from timm.scheduler import CosineLRScheduler
from time import time
from pathlib import Path

import config
from utils import get_elapsed_time
from model import ViT
from loss import CELossWithLabelSmoothing
from cifar10 import get_cifar10_dses
from cifar100 import get_cifar100_dses
from eval import TopKAccuracy
from hide_and_seek import apply_hide_and_seek
from cutmix import apply_cutmix
from cutout import apply_cutout

torch.set_printoptions(linewidth=200, sci_mode=False)
torch.manual_seed(config.SEED)


def save_checkpoint(epoch, model, optim, scaler, avg_acc, ckpt_path):
    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "epoch": epoch,
        "optimizer": optim.state_dict(),
        "scaler": scaler.state_dict(),
        "average_accuracy": avg_acc,
    }
    if config.N_GPUS > 0 and config.MULTI_GPU:
        ckpt["model"] = model.module.state_dict()
    else:
        ckpt["model"] = model.state_dict()

    torch.save(ckpt, str(ckpt_path))


@torch.no_grad()
def validate(dl, model, metric):
    print(f"""Validating...""")
    model.eval()
    sum_acc = 0
    for image, gt in dl:
        image = image.to(config.DEVICE)
        gt = gt.to(config.DEVICE)

        pred = model(image)
        acc = metric(pred=pred, gt=gt)
        sum_acc += acc
    avg_acc = sum_acc / len(dl)
    print(f"""Average accuracy: {avg_acc:.3f}""")

    model.train()
    return avg_acc


if __name__ == "__main__":
    print(f"""N_WORKERS = {config.N_WORKERS}""")
    print(f"""DEVICE = {config.DEVICE}""")
    print(f"""AUTOCAST = {config.AUTOCAST}""")
    print(f"""BATCH_SIZE = {config.BATCH_SIZE}""")

    train_ds, val_ds, test_ds = get_cifar10_dses(data_dir=config.DATA_DIR, val_ratio=config.VAL_RATIO)
    train_dl = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=config.BATCH_SIZE, shuffle=False, pin_memory=True, drop_last=True,
    )

    model = ViT(
        img_size=config.IMG_SIZE,
        patch_size=config.PATCH_SIZE,
        n_layers=config.N_LAYERS,
        hidden_size=config.HIDDEN_SIZE,
        mlp_size=config.MLP_SIZE,
        n_heads=config.N_HEADS,
        n_classes=config.N_CLASSES,
    )
    if config.N_GPUS > 0:
        model = model.to(config.DEVICE)
        if config.MULTI_GPU:
            model = nn.DataParallel(model)

    crit = CELossWithLabelSmoothing(n_classes=config.N_CLASSES, smoothing=config.SMOOTHING)
    metric = TopKAccuracy(k=1)

    optim = Adam(
        model.parameters(),
        lr=config.BASE_LR,
        betas=(config.BETA1, config.BETA2),
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = CosineLRScheduler(
        optimizer=optim,
        t_initial=config.N_EPOCHS,
        warmup_t=config.WARMUP_EPOCHS,
        warmup_lr_init=config.BASE_LR / 10,
        t_in_epochs=True,
    )

    scaler = GradScaler(enabled=True if config.AUTOCAST else False)

    ### Resume
    if config.CKPT_PATH is not None:
        ckpt = torch.load(config.CKPT_PATH, map_location=config.DEVICE)
        if config.N_GPUS > 1 and config.MULTI_GPU:
            model.module.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])

        init_epoch = ckpt["epoch"]
        best_avg_acc = ckpt["average_accuracy"]
        print(f"""Resuming from checkpoint '{config.CKPT_PATH}'...""")

        prev_ckpt_path = config.CKPT_PATH
    else:
        init_epoch = 0
        prev_ckpt_path = ".pth"
        best_avg_acc = 0

    start_time = time()
    running_loss = 0
    step_cnt = 0
    for epoch in range(init_epoch + 1, config.N_EPOCHS + 1):
        for step, (image, gt) in enumerate(train_dl, start=1):
            image = image.to(config.DEVICE)
            gt = gt.to(config.DEVICE)

            if config.HIDE_AND_SEEK:
                image = apply_hide_and_seek(
                    image, patch_size=config.IMG_SIZE // 4, mean=config.MEAN,
                )
            if config.CUTMIX:
                image, gt = apply_cutmix(image=image, gt=gt, n_classes=config.N_CLASSES)
            if config.CUTOUT:
                image = apply_cutout(image)

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
            scheduler.step_update(num_updates=epoch * len(train_dl))

            running_loss += loss.item()
            step_cnt += 1

        if (epoch % config.N_PRINT_EPOCHS == 0) or (epoch == config.N_EPOCHS):
            loss = running_loss / step_cnt
            lr = optim.param_groups[0]['lr']
            print(f"""[ {epoch:,}/{config.N_EPOCHS} ][ {step:,}/{len(train_dl):,} ]""", end="")
            print(f"""[ {lr:.5f} ][ {get_elapsed_time(start_time)} ][ {loss:.2f} ]""")

            running_loss = 0
            step_cnt = 0
            start_time = time()

        if (epoch % config.N_VAL_EPOCHS == 0) or (epoch == config.N_EPOCHS):
            avg_acc = validate(dl=val_dl, model=model, metric=metric)
            if avg_acc > best_avg_acc:
                cur_ckpt_path = config.CKPT_DIR/f"""epoch_{epoch}_avg_acc_{round(avg_acc, 3)}.pth"""
                save_checkpoint(
                    epoch=epoch,
                    model=model,
                    optim=optim,
                    scaler=scaler,
                    avg_acc=avg_acc,
                    ckpt_path=cur_ckpt_path,
                )
                print(f"""Saved checkpoint.""")
                prev_ckpt_path = Path(prev_ckpt_path)
                if prev_ckpt_path.exists():
                    prev_ckpt_path.unlink()

                best_avg_acc = avg_acc
                prev_ckpt_path = cur_ckpt_path

        scheduler.step(epoch + 1)
