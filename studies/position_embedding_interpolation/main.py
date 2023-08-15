# References
    # https://github.com/huggingface/pytorch-pos_embed-models/blob/main/timm/layers/pos_embed.py

import cv2
import torch
import torch.nn.functional as F
import torchvision
import ssl
import timm

from image_utils import save_image
from vit.studies.position_embedding_similarity.main import (
    get_position_embedding_similarity,
    visualize_position_embedding_similarity
)

ssl._create_default_https_context = ssl._create_unverified_context


def interpolate_position_embedding(pos_embed, old_grid_size, new_grid_size):
    pos_embed = pos_embed.reshape((1, old_grid_size[0], old_grid_size[1], -1))
    pos_embed = pos_embed.permute(0, 3, 1, 2)

    new_pos_embed = F.interpolate(pos_embed, size=new_grid_size, mode="bicubic")
    new_pos_embed = new_pos_embed.permute(0, 2, 3, 1)
    return new_pos_embed


if __name__ == "__main__":
    model = timm.create_model("deit_tiny_patch16_224", pretrained=True)
    old_grid_size = model.patch_embed.grid_size
    pos_embed = model.pos_embed[:, 1:, :]

    new_grid_size=(18, 22)
    new_pos_embed = interpolate_position_embedding(
        pos_embed=pos_embed, old_grid_size=old_grid_size, new_grid_size=new_grid_size
    )

    sim = get_position_embedding_similarity(pos_embed=new_pos_embed, grid_size=new_grid_size)
    vis = visualize_position_embedding_similarity(pos_embed=sim, n_cols=new_grid_size[1], upsample=3)
