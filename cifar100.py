# Source: https://www.cs.toronto.edu/~kriz/cifar.html

from pathlib import Path
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

import config
from cifar10 import CIFARDataset, get_cifar_mean_and_std


def get_cifar100_imgs_and_gts(data_path):
    with open(data_path, mode="rb") as f:
        data_dic = pickle.load(f, encoding="bytes")

    imgs = data_dic[b"data"]
    imgs = imgs.reshape(-1, 3, config.IMG_SIZE, config.IMG_SIZE)
    imgs = imgs.transpose(0, 2, 3, 1)

    gts = data_dic[b"fine_labels"]
    gts = np.array(gts)
    return imgs, gts


def get_all_cifar100_imgs_and_gts(data_dir, val_ratio):
    train_val_imgs, train_val_gts = get_cifar100_imgs_and_gts(Path(data_dir)/"train")
    train_imgs, val_imgs, train_gts, val_gts = train_test_split(
        train_val_imgs, train_val_gts, test_size=val_ratio,
    )
    test_imgs, test_gts = get_cifar100_imgs_and_gts(Path(data_dir)/"test")
    return train_imgs, train_gts, val_imgs, val_gts, test_imgs, test_gts


def get_cifar100_dses(data_dir, val_ratio=0.1):
    train_imgs, train_gts, val_imgs, val_gts, test_imgs, test_gts = get_all_cifar100_imgs_and_gts(
            data_dir=data_dir, val_ratio=val_ratio,
    )
    mean, std = get_cifar_mean_and_std(train_imgs)
    train_ds = CIFARDataset(imgs=train_imgs, gts=train_gts, mean=mean, std=std)
    val_ds = CIFARDataset(imgs=val_imgs, gts=val_gts, mean=mean, std=std)
    test_ds = CIFARDataset(imgs=test_imgs, gts=test_gts, mean=mean, std=std)
    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    data_dir = "/Users/jongbeomkim/Documents/datasets/cifar-100-python"
    train_ds, val_ds, test_ds = get_cifar100_dses(data_dir=data_dir, val_ratio=0.1)
