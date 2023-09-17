# References
    # https://github.com/jacobgil/vit-explain
    # https://jacobgil.github.io/deeplearning/vision-transformer-explainability#how-do-the-attention-activations-look-like-for-the-class-token-throughout-the-network-

import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
import cv2
import re
from itertools import product
from typing import Literal
import ssl

from image_utils import (
    load_image,
    show_image,
    save_image,
    _to_pil,
    _apply_jet_colormap,
    _blend_two_images,
    _rgba_to_rgb
)

ssl._create_default_https_context = ssl._create_unverified_context

IMG_SIZE = 224
PATCH_SIZE = 16
N_PATCHS = (IMG_SIZE // PATCH_SIZE) ** 2


class AttentionRollout:
    def __init__(
        self,
        model: nn.Module,
        attn_layer_regex: str=r"(.attn_drop)$",
    ):
        self.model = model
        self.attn_layer_regex = attn_layer_regex

        # Reference:
        # https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
        for name, module in model.named_modules():
            if re.search(pattern=r"(attn)$", string=name):
                module.fused_attn = False

        self.attn_mats = list()

        for name, module in self.model.named_modules():
            if re.search(pattern=attn_layer_regex, string=name):
                module.register_forward_hook(self._save_attention_matrices)

        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _save_attention_matrices(self, module, input, output):
        self.attn_mats.append(output.cpu())

    def _get_attention_matrices(self, image):
        self.model.eval()
        with torch.no_grad():
            self.model(image)
        return self.attn_mats

    def get_attention_map(
        self,
        img: np.ndarray,
        head_fusion: Literal["mean", "max", "min", "sum"]="min",
        discard_ratio: float=0.9
    ):
        image = self.transform(_to_pil(img)).unsqueeze(0)
        attn_mats = attn_rollout._get_attention_matrices(image)

        attn_map = torch.eye(attn_mats[0].squeeze().shape[1])
        for attn_mat in attn_mats:
            # At every Transformer block we get an attention matrix $A_{ij}$
            # that defines how much attention is going to flow from token $j$ in the previous layer
            # to token $i$ in the next layer.

            # The Attention rollout paper suggests taking the average of the heads.
            # It can also make sense using other choices: like the minimum, the maximum, or using different weights. 
            if head_fusion == "mean":
                attn_mat = attn_mat.mean(dim=1)
            elif head_fusion == "min":
                attn_mat = attn_mat.min(dim=1)[0]
            elif head_fusion == "max":
                attn_mat = attn_mat.max(dim=1)[0]
            # 제시된 방법은 아니지만 `sum()`도 사용할 수 있을 것입니다.
            elif head_fusion == "sum":
                attn_mat = attn_mat.sum(dim=1)

            # Without discarding low attention pixels the attention map is very noisy
            # and doesn’t seem to focus only on the interesting part of the image.
            # The more pixels we remove, we are able to better isolate the salient object in the image.
            flattened = torch.flatten(attn_mat.squeeze(), start_dim=0, end_dim=-1)
            sorted, _ = torch.sort(flattened, dim=0)
            ref_val = sorted[int(len(sorted) * discard_ratio)]
            attn_mat.masked_fill_(mask=(attn_mat < ref_val), value=0)

            # We also have the residual connections.
            # We can model them by adding the identity matrix $I$ to the layer attention matrix: $A_{ij} + I$.
            id_mat = torch.eye(attn_mat.shape[1])
            attn_mat = attn_mat + id_mat

            # If we look at the first row (shape 197), and discard the first value (left with shape 196=14x14) that’s how the inforattn_mapion flows from the different locations in the image to the class token.
            # We also have to normalize the rows, to keep the total attention flow $1$.
            # 따라서 각 행마다 합을 구해야 함 
            attn_mat /= attn_mat.sum(dim=2)

            # Recursively multiply the attention matrix
            # attn_map = torch.matmul(attn_map, attn_mat)
            attn_map = torch.matmul(attn_mat, attn_map)

        attn_map = attn_map.squeeze()[0, 1:]
        # attn_map = attn_map.view(int(attn_map.shape[0] ** 0.5), int(attn_map.shape[0] ** 0.5))
        attn_map = attn_map.reshape(int(attn_map.shape[0] ** 0.5), int(attn_map.shape[0] ** 0.5))

        attn_map = attn_map.detach().cpu().numpy()
        attn_map -= attn_map.min()
        attn_map /= attn_map.max()
        attn_map *= 255
        attn_map = attn_map.astype("uint8")

        h, w = img.shape[: 2]
        resized = cv2.resize(src=attn_map, dsize=(w, h))
        return resized


def apply_attention_map_to_image(img, attn_map, mode="jet"):
    if mode == "brightness":
        # attn_map = np.clip(attn_map.astype("uint16") * 1.4, 0, 255).astype("uint8")
        output = np.concatenate([img, attn_map[..., None]], axis=2)
        output = _rgba_to_rgb(output)
    elif mode == "jet":
        attn_map = _apply_jet_colormap(attn_map)
        output = _blend_two_images(img1=img, img2=attn_map, alpha=0.6)
    return output


if __name__ == "__main__":
    model = torch.hub.load("facebookresearch/deit:main", model="deit_tiny_patch16_224", pretrained=True)
    attn_rollout = AttentionRollout(model=model, attn_layer_regex=r"(.attn_drop)$")

    img = load_image("golden_retriever.jpg")
    # img = load_image("/Users/jongbeomkim/Desktop/workspace/explainable_ai/attention_rollout/golden_retriever.jpg")

    for discard_ratio in np.arange(0, 1, 0.05):
        attn_map = attn_rollout.get_attention_map(img=img, head_fusion="max", discard_ratio=discard_ratio)
        output = apply_attention_map_to_image(img=img, attn_map=attn_map, mode="jet")

        save_image(
            img1=output,
            path=f"""attention_map_examples/head_fusion_max_mode_jet/discard_ratio_{discard_ratio:.2f}.jpg"""
            # path=f"""/Users/jongbeomkim/Desktop/workspace/explainable_ai/attention_rollout/attention_map_examples/head_fusion_max_mode_jet/discard_ratio_{discard_ratio:.2f}.jpg"""
        )

    for discard_ratio in np.arange(0, 1, 0.05):
        attn_map = attn_rollout.get_attention_map(img=img, head_fusion="max", discard_ratio=discard_ratio)
        output = apply_attention_map_to_image(img=img, attn_map=attn_map, mode="brightness")

        save_image(
            img1=output,
            path=f"""attention_map_examples/head_fusion_max_mode_brightness/discard_ratio_{discard_ratio:.2f}.jpg"""
            # path=f"""/Users/jongbeomkim/Desktop/workspace/explainable_ai/attention_rollout/attention_map_examples/head_fusion_max_mode_brightness/discard_ratio_{discard_ratio:.2f}.jpg"""
        )
