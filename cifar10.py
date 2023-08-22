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


class CIFAR10Dataset(Dataset):
    def __init__(self, data_dir, split="train"):
        super().__init__()

        self.imgs, self.gts = _get_cifar10_images_and_gts(data_dir=data_dir, split=split)

        self.transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            # T.RandomApply(
            #     [T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)],
            #     p=0.5,
            # ),
            T.ToTensor(),
            # get_cifar100_mean_and_std(config.DATA_DIR)
            T.Normalize(mean=(0.507, 0.487, 0.441), std=(0.267, 0.256, 0.276)),
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
    dl = DataLoader(ds, batch_size=4, shuffle=True)
    # for image, gt in dl:
    #     print(image.shape)
    # # print(len(dl))
    di = iter(dl)

    image, gt = next(di)
    # # grid = make_grid(image, nrow=1, normalize=True)
    # # TF.to_pil_image(grid).show()
    # # config.CIFAR100_CLASSES[gt]
