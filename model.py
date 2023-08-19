# Reference
    # https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import config


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, hidden_dim, drop_prob=config.DROP_PROB):
        super().__init__()

        self.patch_size = patch_size
        dim = (patch_size ** 2) * 3

        self.norm1 = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, hidden_dim)
        self.drop = nn.Dropout(drop_prob)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = rearrange(
            x,
            pattern="b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
            p1=self.patch_size,
            p2=self.patch_size,
        )
        x = self.norm1(x) # Not in the paper
        x = self.proj(x)
        # "Dropout is applied after every dense layer except for the the qkv-projections
        # and directly after adding positional- to patch embeddings."
        x = self.drop(x)
        x = self.norm2(x) # Not in the paper
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, n_heads, drop_prob=config.DROP_PROB):
        super().__init__()

        self.n_heads = n_heads

        self.head_dim = hidden_dim // n_heads
        # "U_{qkv} \in \mathbb{R}^{D \times 3D_{h}}"
        self.qkv_proj = nn.Linear(hidden_dim, 3 * n_heads * self.head_dim, bias=False)
        self.drop = nn.Dropout(drop_prob)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def _get_attention_score(self, q, k):
        # "$qk^{T}$"
        attn_score = torch.einsum("bhnd,bhmd->bhnm", q, k)
        return attn_score

    def forward(self, x):
        # "$[q, k, v] = zU_{qkv}$"
        q, k, v = torch.split(
            self.qkv_proj(x), split_size_or_sections=self.n_heads * self.head_dim, dim=2,
        )
        q = rearrange(q, pattern="b n (h d) -> b h n d", h=self.n_heads, d=self.head_dim)
        k = rearrange(k, pattern="b n (h d) -> b h n d", h=self.n_heads, d=self.head_dim)
        v = rearrange(v, pattern="b n (h d) -> b h n d", h=self.n_heads, d=self.head_dim)
        attn_score = self._get_attention_score(q=q, k=k)
        # "$A = softmax(qk^{T}/\sqrt{D_{h}}), A \in \mathbb{R}^{N \times N}$"
        attn_weight = F.softmax(attn_score / (self.head_dim ** 0.5), dim=3)
        # attn_weight = self.drop(attn_weight)
        x = torch.einsum("bhnm,bhmd->bhnd", attn_weight, v)
         # "$U_{msa} \in \mathbb{R}^{k \cdot D_{h} \times D}$"
        x = rearrange(x, pattern="b h n d -> b n (h d)")
        x = self.out_proj(x)
        # "Dropout is applied after every dense layer except for the the qkv-projections
        # and directly after adding positional- to patch embeddings."
        x = self.drop(x)
        return x


class ResidualConnection(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.norm = nn.LayerNorm(hidden_dim) # "$LN$"

    def forward(self, x, sublayer):
        # "Layernorm (LN) is applied before every block, and residual connections after every block."
        # "$z'_{l} = MSA(LN(z_{l - 1})) + z_{l - 1}$", "$z_{l} = MLP(LN(z'_{l})) + z'_{l}$"
        skip = x.clone()
        x = self.norm(x)
        x = sublayer(x)
        x += skip
        return x


class MLP(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.mlp_dim = hidden_dim * 4

        self.proj1 = nn.Linear(hidden_dim, self.mlp_dim)
        self.drop1 = nn.Dropout(0.1)
        self.proj2 = nn.Linear(self.mlp_dim, hidden_dim)
        self.drop2 = nn.Dropout(0.1)

    def forward(self, x):
        x = self.proj1(x)
        x = F.gelu(x) # "The MLP contains two layers with a GELU non-linearity."
        # "Dropout is applied after every dense layer except for the the qkv-projections
        # and directly after adding positional- to patch embeddings."
        # Activation function 다음에 Dropout이 오도록!
        x = self.drop1(x)
        x = self.proj2(x)
        x = F.gelu(x)
        x = self.drop2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads):
        super().__init__()

        self.self_attn = MultiHeadAttention(hidden_dim=hidden_dim, n_heads=n_heads)
        self.self_attn_resid = ResidualConnection(hidden_dim=hidden_dim)
        self.mlp = MLP(hidden_dim=hidden_dim)
        self.mlp_resid = ResidualConnection(hidden_dim=hidden_dim)

    def forward(self, x):
        x = self.self_attn_resid(x=x, sublayer=self.self_attn)
        x = self.mlp_resid(x=x, sublayer=self.mlp)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, n_layers, hidden_dim, n_heads):
        super().__init__()

        self.enc_stack = nn.ModuleList(
            [TransformerEncoderLayer(hidden_dim=hidden_dim, n_heads=n_heads) for _ in range(n_layers)]
        )

    def forward(self, x):
        for enc_layer in self.enc_stack:
            x = enc_layer(x)
        return x


class ViT(nn.Module):
    # ViT-Base: `n_layers=12, hidden_dim=768, n_heads=12`
    # ViT-Large: `n_layers=24, hidden_dim=1024, n_heads=16`
    # ViT-Huge: `n_layers=32, hidden_dim=1280, n_heads=16`
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        n_layers=12,
        hidden_dim=768,
        n_heads=12,
        drop_prob=config.DROP_PROB,
        n_classes=0,
    ):
        super().__init__()

        self.n_classes = n_classes

        assert img_size % patch_size == 0, "`img_size` must be divisible by `patch_size`!"

        cell_size = img_size // patch_size
        n_patches = cell_size ** 2

        # $\textbf{E}$ of the equation 1 in the paper
        self.patch_embed = PatchEmbedding(patch_size=patch_size, hidden_dim=hidden_dim)
        self.cls_token = nn.Parameter(torch.randn((1, 1, hidden_dim))) # $x_{\text{class}}$
         # $\textbf{E}_\text{pos}$
        self.pos_embed = nn.Parameter(torch.randn((1, n_patches + 1, hidden_dim)))
        self.drop1 = nn.Dropout(drop_prob)
        self.tf_enc = TransformerEncoder(n_layers=n_layers, hidden_dim=hidden_dim, n_heads=n_heads)

        self.norm = nn.LayerNorm(hidden_dim) # "$LN$"
        self.proj = nn.Linear(hidden_dim, n_classes)
        self.drop2 = nn.Dropout(drop_prob)

    def forward(self, x):
        b, _, _, _ = x.shape

        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.repeat(b, 1, 1), x), dim=1)
        x += self.pos_embed
        # "Dropout is applied after every dense layer except for the the qkv-projections
        # and directly after adding positional- to patch embeddings."
        x = self.drop1(x)
        x = self.tf_enc(x)

        if self.n_classes == 0:
            x = x.mean(dim=1)
        else:
            x = x[:, 0, :] # $z^{0}_{L}$ of the equation 4 in the paper
            # "Layernorm (LN) is applied before every block."
            x = self.norm(x) # $y$
            x = self.proj(x)
            # "Dropout is applied after every dense layer except for the the qkv-projections
            # and directly after adding positional- to patch embeddings."
            x = self.drop2(x)
        return x


if __name__ == "__main__":
    image = torch.randn((4, 3, 32, 32))
    vit = ViT(
        img_size=32,
        patch_size=16,
        n_layers=12,
        hidden_dim=192,
        n_heads=12,
        n_classes=100,
    )
    out = vit(image)
    print(out.shape)


# "we optimize three basic regularization parameters – weight decay, dropout, and label smoothing. Figure"
