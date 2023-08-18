# Source: https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000

from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image
from pathlib import Path


ds = ImageFolder("/Users/jongbeomkim/Documents/datasets/imagenet-mini/train")
len(ds.classes)
len(ds)
ds[10300][0].show()
# class CIFAR100Dataset(Dataset):
#     def __init__(self, data_dir, split="train"):
#         super().__init__()
DATA_DIR = "/Users/jongbeomkim/Documents/datasets/imagenet-mini"
ds = ImageFolder(Path(DATA_DIR)/"train")
len(ds)
image, gt = ds[1000]
image.show()
gt


if __name__ == "__main__":