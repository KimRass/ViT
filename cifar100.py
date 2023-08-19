# Source: https://www.cs.toronto.edu/~kriz/cifar.html

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
import pickle

import config
from utils import _get_cifar100_images_and_gts, get_cifar100_mean_and_std


class CIFAR100Dataset(Dataset):
    def __init__(self, data_dir, split="train"):
        super().__init__()

        self.imgs, self.gts = _get_cifar100_images_and_gts(data_dir=data_dir, split=split)

        kernel_size = round(config.IMG_SIZE * 0.1) // 2 * 2 + 1
        self.transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(
                [T.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)],
                p=0.5,
            ),
            T.RandomApply(
                [T.GaussianBlur(kernel_size=(kernel_size, kernel_size), sigma=(0.1, 2))],
                p=0.5,
            ),
            T.ToTensor(),
            # get_cifar100_mean_and_std(config.DATA_DIR)
            T.Normalize(mean=(0.507, 0.487, 0.441), std=(0.267, 0.256, 0.276)),
        ])

    def __len__(self):
        return len(self.gts)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        image = Image.fromarray(img)
        image = self.transform(image)

        gt = self.gts[idx]
        return image, gt


if __name__ == "__main__":
    ds = CIFAR100Dataset(config.DATA_DIR, split="train")
    image, gt = ds[100]
    image.show()
    dl = DataLoader(ds, batch_size=1, shuffle=True)
    di = iter(dl)
    image, gt = next(di)
    image
    image.show()
    config.CIFAR100_CLASSES[gt]
