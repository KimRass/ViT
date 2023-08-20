from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import random

from image_utils import get_image_grid, save_image, show_image


def apply_cutout(image, cutout_size=16, mean=(0.485, 0.456, 0.406)):
    b, _, h, w = image.shape

    x = random.randint(0, w)
    y = random.randint(0, h)
    xmin = max(0, x - cutout_size // 2)
    ymin = max(0, y - cutout_size // 2)
    xmax = max(0, x + cutout_size // 2)
    ymax = max(0, y + cutout_size // 2)

    image[:, 0, ymin: ymax, xmin: xmax] = mean[0]
    image[:, 1, ymin: ymax, xmin: xmax] = mean[1]
    image[:, 2, ymin: ymax, xmin: xmax] = mean[2]
    return image


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
    for batch, (image, label) in enumerate(dl, start=1):
        cutouted_image = apply_cutout(image, cutout_size=112)
        grid = get_image_grid(cutouted_image)
        show_image(grid)

        save_image(img=grid, path=f"""/Users/jongbeomkim/Desktop/workspace/machine_learning/computer_vision/image_data_augmentation/cutout/samples/imagenet_mini{batch}.jpg""")