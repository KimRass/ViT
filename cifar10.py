# Source: https://www.cs.toronto.edu/~kriz/cifar.html

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

import config


def get_cifar10_imgs_and_gts(data_path):
    with open(data_path, mode="rb") as f:
        data_dic = pickle.load(f, encoding="bytes")

    imgs = data_dic[b"data"]
    imgs = imgs.reshape(-1, 3, config.IMG_SIZE, config.IMG_SIZE)
    imgs = imgs.transpose(0, 2, 3, 1)

    gts = data_dic[b"labels"]
    gts = np.array(gts)
    return imgs, gts
    

def get_cifar10_train_val_set(data_dir):
    imgs_ls = list()
    gts_ls = list()
    for idx in range(1, 6):
        imgs, gts = get_cifar10_imgs_and_gts(Path(data_dir)/f"data_batch_{idx}")
        imgs_ls.append(imgs)
        gts_ls.append(gts)
    imgs = np.concatenate(imgs_ls, axis=0)
    gts = np.concatenate(gts_ls, axis=0)
    return imgs, gts


def get_all_cifar10_imgs_and_gts(data_dir, val_ratio):
    train_val_imgs, train_val_gts = get_cifar10_train_val_set(data_dir)
    train_imgs, val_imgs, train_gts, val_gts = train_test_split(
        train_val_imgs, train_val_gts, test_size=val_ratio,
    )
    test_imgs, test_gts = get_cifar10_imgs_and_gts(Path(data_dir)/"test_batch")
    return train_imgs, train_gts, val_imgs, val_gts, test_imgs, test_gts


def get_cifar_mean_and_std(imgs):
    imgs = imgs.astype("float") / 255
    n_pixels = imgs.size // 3
    sum_ = imgs.reshape(-1, 3).sum(axis=0)
    sum_square = (imgs ** 2).reshape(-1, 3).sum(axis=0)
    mean = (sum_ / n_pixels).round(3)
    std = (((sum_square / n_pixels) - mean ** 2) ** 0.5).round(3)
    return mean, std


class CIFARDataset(Dataset):
    def __init__(self, imgs, gts, mean, std):
        super().__init__()

        self.imgs = imgs
        self.gts = gts

        self.transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomCrop(size=config.IMG_SIZE, padding=4, pad_if_needed=True),
            T.RandomApply(
                [T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)],
                p=0.4,
            ),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __len__(self):
        return len(self.gts)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        image = Image.fromarray(img, mode="RGB")
        image = self.transform(image)

        gt = self.gts[idx]
        gt = torch.tensor(gt).long()
        return image, gt


def get_cifar10_dses(data_dir, val_ratio=0.1):
    train_imgs, train_gts, val_imgs, val_gts, test_imgs, test_gts = get_all_cifar10_imgs_and_gts(
            data_dir=data_dir, val_ratio=val_ratio,
    )
    mean, std = get_cifar_mean_and_std(train_imgs)
    train_ds = CIFARDataset(imgs=train_imgs, gts=train_gts, mean=mean, std=std)
    val_ds = CIFARDataset(imgs=val_imgs, gts=val_gts, mean=mean, std=std)
    test_ds = CIFARDataset(imgs=test_imgs, gts=test_gts, mean=mean, std=std)
    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    data_dir = "/Users/jongbeomkim/Documents/datasets/cifar-10-batches-py"
    train_ds, val_ds, test_ds = get_cifar10_dses(data_dir=data_dir, val_ratio=0.1)
