import torch
import torch.nn.functional as F
import random
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T


from utils import image_to_grid


def apply_cutmix(image, gt, n_classes):
    if gt.ndim == 1:
        gt = F.one_hot(gt, num_classes=n_classes)

    b, _, h, w = image.shape

    lamb = random.random()
    region_x = random.randint(0, w)
    region_y = random.randint(0, h)
    region_w = region_h = (1 - lamb) ** 0.5

    xmin = max(0, int(region_x - region_w / 2))
    ymin = max(0, int(region_y - region_h / 2))
    xmax = max(w, int(region_x + region_w / 2))
    ymax = max(h, int(region_y + region_h / 2))

    indices = torch.randperm(b)
    image[:, :, ymin: ymax, xmin: xmax] = image[indices][:, :, ymin: ymax, xmin: xmax]
    lamb = 1 - (xmax - xmin) * (ymax - ymin) / (w * h)
    gt = lamb * gt + (1 - lamb) * gt[indices]
    return image, gt


if __name__ == "__main__":
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Resize((224, 224), antialias=True),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
    )
    ds = ImageFolder("/Users/jongbeomkim/Documents/datasets/imagenet-mini/val", transform=transform)
    dl = DataLoader(dataset=ds, batch_size=16, shuffle=True, num_workers=0, drop_last=True)
    n_classes = len(ds.classes)
    for batch, (image, gt) in enumerate(dl, start=1):
        gt = F.one_hot(gt, num_classes=n_classes)
        image, gt = apply_cutmix(image=image, gt=gt, n_classes=n_classes)
        grid = image_to_grid(
            image,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            n_cols=4,
        )
        grid.show()
        break

        # save_image(
        #     img=grid,
        #     path=f"""/Users/jongbeomkim/Desktop/workspace/vit_from_scratch/cutmix/examples/imagenet_mini{batch}.jpg"""
        # )