import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import random
import random

from image_utils import (
    get_image_dataset_mean_and_std,
    batched_image_to_grid,
    save_image,
)


def apply_hide_and_seek(image, patch_size=56, hide_prob=0.5, mean=(0, 0, 0)):
    b, _, h, w = image.shape
    assert h % patch_size == 0 and w % patch_size == 0,\
        "`patch_size` argument should be a multiple of both the width and height of the input image"

    mean_tensor = torch.Tensor(mean)[None, :, None, None].repeat(b, 1, patch_size, patch_size)

    copied_image = image.clone()
    for i in range(h // patch_size):
        for j in range(w // patch_size):
            if random.random() < hide_prob:
                    continue
            copied_image[
                ..., i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size
            ] = mean_tensor
    return copied_image


if __name__ == "__main__":
    data_dir = "/Users/jongbeomkim/Documents/datasets/imagenet-mini/val"
    mean, std = get_image_dataset_mean_and_std(data_dir)
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Resize((224, 224)),
            T.Normalize(
                mean=mean,
                std=std,
            ),
        ]
    )
    ds = ImageFolder(data_dir, transform=transform)
    dl = DataLoader(dataset=ds, batch_size=16, shuffle=True, num_workers=4, drop_last=True)
    for batch, (image, label) in enumerate(dl, start=1):
        image = apply_hide_and_seek(image, patch_size=56)
        grid = batched_image_to_grid(image, n_cols=4, normalize=True)
        
        # show_image(grid)
        save_image(grid, f"""/Users/jongbeomkim/Desktop/workspace/machine_learning/computer_vision/weakly_supervised_learning/hide_and_seek/samples/{batch}.jpg""")