import torch
import torchvision.transforms as T
from pathlib import Path
from tqdm.auto import tqdm
from PIL import Image
from time import time
from datetime import timedelta
import pickle

import config

# def get_image_dataset_mean_and_std(data_dir, ext="jpg"):
#     data_dir = Path(data_dir)

#     sum_rgb = 0
#     sum_rgb_square = 0
#     sum_resol = 0
#     for img_path in tqdm(list(data_dir.glob(f"""**/*.{ext}"""))):
#         pil_img = Image.open(img_path)
#         tensor = T.ToTensor()(pil_img)
        
#         sum_rgb += tensor.sum(dim=(1, 2))
#         sum_rgb_square += (tensor ** 2).sum(dim=(1, 2))
#         _, h, w = tensor.shape
#         sum_resol += h * w
#     mean = torch.round(sum_rgb / sum_resol, decimals=3)
#     std = torch.round((sum_rgb_square / sum_resol - mean ** 2) ** 0.5, decimals=3)
#     return mean, std


def _get_cifar100_images_and_gts(data_dir, split="train", img_size=32):
    with open(Path(data_dir)/split, mode="rb") as f:
        data_dic = pickle.load(f, encoding="bytes")

    imgs = data_dic[b"data"]
    imgs = imgs.reshape(-1, 3, img_size, img_size)
    imgs = imgs.transpose(0, 2, 3, 1)

    gts = data_dic[b"fine_labels"]
    return imgs, gts


def get_cifar100_mean_and_std(data_dir, split="train"):
    imgs, _ = _get_cifar100_images_and_gts(data_dir=data_dir, split=split)

    imgs = imgs.astype("float") / 255
    n_pixels = imgs.size // 3
    sum_ = imgs.reshape(-1, 3).sum(axis=0)
    sum_square = (imgs ** 2).reshape(-1, 3).sum(axis=0)
    mean = (sum_ / n_pixels).round(3)
    std = (((sum_square / n_pixels) - mean ** 2) ** 0.5).round(3)
    return mean, std
get_cifar100_mean_and_std(config.DATA_DIR)


def get_elapsed_time(start_time):
    return timedelta(seconds=round(time() - start_time))
