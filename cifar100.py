# Source: https://www.cs.toronto.edu/~kriz/cifar.html

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
import pickle

import config


class CIFAR100Dataset(Dataset):
    def __init__(self, data_dir, split="train"):
        super().__init__()

        with open(Path(data_dir)/split, mode="rb") as f:
            data_dic = pickle.load(f, encoding="bytes")
        self.imgs = data_dic[b"data"]
        self.gts = data_dic[b"fine_labels"]

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
            # T.ToTensor(),
            # T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return len(self.gts)

    def __getitem__(self, idx):
        arr = self.imgs[idx]
        img = arr.reshape(3, config.IMG_SIZE, config.IMG_SIZE).transpose(1, 2, 0)
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
