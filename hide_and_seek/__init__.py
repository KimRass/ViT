import torch
import random


def apply_hide_and_seek(image, patch_size, hide_prob=0.5, mean=(0.5, 0.5, 0.5)):
    b, _, h, w = image.shape
    assert h % patch_size == 0 and w % patch_size == 0,\
        "`patch_size` argument should be a multiple of both the width and height of the input image"

    mean_tensor = torch.Tensor(mean)[None, :, None, None].repeat(b, 1, patch_size, patch_size)

    copied = image.clone()
    for i in range(h // patch_size):
        for j in range(w // patch_size):
            if random.random() < hide_prob:
                    continue

            copied[
                ..., i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size
            ] = mean_tensor
    return copied
