import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from PIL import Image
import random
import numpy as np
from pathlib import Path

from image_utils import denormalize_array, get_image_grid, save_image


def apply_cutmix(image, label):
    b, _, h, w = image.shape

    order = torch.randperm(b)
    shuffled_image = image[order]
    shuffled_label = label[order]

    lamb = random.random()
    region_x = random.randint(0, w)
    region_y = random.randint(0, h)
    region_w = region_h = (1 - lamb) ** 0.5

    xmin = max(0, int(region_x - region_w / 2))
    ymin = max(0, int(region_y - region_h / 2))
    xmax = max(w, int(region_x + region_w / 2))
    ymax = max(h, int(region_y + region_h / 2))

    image[:, :, ymin: ymax, xmin: xmax] = shuffled_image[:, :, ymin: ymax, xmin: xmax]
    lamb = 1 - (xmax - xmin) * (ymax - ymin) / (w * h)
    label = lamb * label + (1 - lamb) * shuffled_label
    return image, label


if __name__ == "__main__":
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Resize((224, 224)),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
    )
    ds = ImageFolder("/Users/jongbeomkim/Downloads/imagenet-mini/val", transform=transform)
    dl = DataLoader(dataset=ds, batch_size=16, shuffle=True, num_workers=4, drop_last=True)
    n_classes = len(ds.classes)
    for batch, (image, label) in enumerate(dl, start=1):
        label = F.one_hot(label, num_classes=n_classes)
        cutmixed_image, cutmixed_label = apply_cutmix(image=image, label=label)
        grid = get_image_grid(cutmixed_image)

        save_image(img=grid, path=f"""/Users/jongbeomkim/Desktop/workspace/machine_learning/computer_vision/image_data_augmentation/cutmix/samples/imagenet_mini{batch}.jpg""")