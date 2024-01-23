# References
    # https://github.com/huggingface/pytorch-pos_embed-models/blob/624266148d8fa5ddb22a6f5e523a53aaf0e8a9eb/timm/models/vision_transformer.py#L297

import cv2
import torch
import torch.nn.functional as F
import torchvision
import ssl
import timm

from image_utils import save_image

ssl._create_default_https_context = ssl._create_unverified_context


def get_position_embedding_similarity(pos_embed, grid_size):
    pos_embed = pos_embed.reshape((1, grid_size[0], grid_size[1], -1))

    ls = list()
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            cos_sim = F.cosine_similarity(pos_embed[:, i: i + 1, j: j + 1], pos_embed, dim=3)
            ls.append(cos_sim)
    stacked = torch.stack(ls)
    return stacked


def visualize_position_embedding_similarity(pos_embed, n_cols, upsample=2, cmap="deepgreen"):
    b, _, h, w = pos_embed.shape
    assert b % n_cols == 0,\
        "The batch size should be a multiple of `n_cols` argument"
    pad = max(2, int(max(h, w) * 0.04))
    grid = torchvision.utils.make_grid(
         tensor=pos_embed,
         nrow=n_cols,
         padding=pad,
         normalize=True,
         value_range=(pos_embed.min(), pos_embed.max()),
         scale_each=True
    )
    grid *= 255
    grid = grid.clone().permute((1, 2, 0)).detach().cpu().numpy().astype("uint8")
    
    grid = cv2.applyColorMap(src=grid, colormap=eval(f"""cv2.COLORMAP_{cmap.upper()}"""))

    for k in range(n_cols + 1):
        grid[:, (pad + w) * k: (pad + w) * k + pad, :] = 255
    for k in range(b // n_cols + 1):
        grid[(pad + h) * k: (pad + h) * k + pad, :, :] = 255

    # Upsample as the image is too small.
    grid = cv2.resize(
        grid, dsize=(grid.shape[1] * upsample, grid.shape[0] * upsample), interpolation=cv2.INTER_NEAREST
    )
    return grid


if __name__ == "__main__":
    model = timm.create_model("deit_tiny_patch16_224", pretrained=True)
    grid_size = model.patch_embed.grid_size
    pos_embed = model.pos_embed[:, 1:, :]
    pos_embed.shape

    sim = get_position_embedding_similarity(pos_embed=pos_embed, grid_size=grid_size)
    vis = visualize_position_embedding_similarity(pos_embed=sim, n_cols=grid_size[1])

    save_image(img=vis, path="position_embedding_similarity.png")
