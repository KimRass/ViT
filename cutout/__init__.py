import random


def apply_cutout(image, cutout_size=16, mean=(0.485, 0.456, 0.406)):
    _, _, h, w = image.shape

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
