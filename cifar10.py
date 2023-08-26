# Source: https://www.cs.toronto.edu/~kriz/cifar.html

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid
from PIL import Image
from pathlib import Path
import pickle
import numpy as np

import config


def _get_images_and_gts(data_path):
    with open(data_path, mode="rb") as f:
        data_dic = pickle.load(f, encoding="bytes")

    imgs = data_dic[b"data"]
    imgs = imgs.reshape(-1, 3, config.IMG_SIZE, config.IMG_SIZE)
    imgs = imgs.transpose(0, 2, 3, 1)

    gts = data_dic[b"labels"]
    return imgs, gts


def _get_cifar10_images_and_gts(data_dir, split="train"):
    if split == "train":
        imgs_ls = list()
        gts_ls = list()
        for idx in range(1, 6):
            imgs, gts = _get_images_and_gts(Path(data_dir)/f"""data_batch_{idx}""")
            imgs_ls.append(imgs)
            gts_ls.append(gts)
        imgs = np.concatenate(imgs_ls, axis=0)
        gts = np.concatenate(gts_ls, axis=0)
    elif split == "test":
        imgs, gts = _get_images_and_gts(Path(data_dir)/"test_batch")
    return imgs, gts


def get_cifar10_mean_and_std(data_dir, split="train"):
    imgs, _ = _get_cifar10_images_and_gts(data_dir=data_dir, split=split)

    imgs = imgs.astype("float") / 255
    n_pixels = imgs.size // 3
    sum_ = imgs.reshape(-1, 3).sum(axis=0)
    sum_square = (imgs ** 2).reshape(-1, 3).sum(axis=0)
    mean = (sum_ / n_pixels).round(3)
    std = (((sum_square / n_pixels) - mean ** 2) ** 0.5).round(3)
    return mean, std


class CIFAR10Dataset(Dataset):
    def __init__(self, data_dir, mean, std, split="train"):
        super().__init__()

        self.imgs, self.gts = _get_cifar10_images_and_gts(data_dir=data_dir, split=split)

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


if __name__ == "__main__":
    ds = CIFAR10Dataset(config.DATA_DIR, split="train")
    for _ in range(10):
        image, gt = ds[100]
    dl = DataLoader(ds, batch_size=1, shuffle=False)
    di = iter(dl)

    image, gt = next(di)
