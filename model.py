# Reference
    # https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

import torch
import torch.nn as nn
from einops import rearrange


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, hidden_dim):
        super().__init__()

        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        dim = patch_size ** 2 * 3

        self.norm1 = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = rearrange(
            x, pattern="b c (gh p1) (gw p2) -> b (gh gw) (p1 p2 c)", p1=self.patch_size, p2=self.patch_size
        )
        x = self.norm1(x) # Not in the paper
        x = self.proj(x)
        x = self.norm2(x) # Not in the paper
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, n_heads):
        super().__init__()
    
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads

        self.head_dim = hidden_dim // n_heads

        self.w_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_v = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(0.1)
        self.w_o = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, q, k, v):
        b, l, _ = q.shape
        _, m, _ = k.shape

        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q = q.view(b, l, self.head_dim, self.n_heads)
        k = k.view(b, m, self.head_dim, self.n_heads)
        v = v.view(b, m, self.head_dim, self.n_heads)

        attn_score = torch.einsum("bldn,bmdn->blmn", q, k)
        attn_score /= (self.head_dim ** 0.5)

        attn_weight = self.softmax(attn_score)
        attn_weight = self.dropout(attn_weight)

        x = torch.einsum("blmn,bmdn->bldn", attn_weight, k)
        x = rearrange(x, pattern="b l d n -> b l (d n)")

        x = self.w_o(x)
        return x


class PositionwiseFeedForward(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.mlp_dim = hidden_dim * 4

        self.w1 = nn.Linear(hidden_dim, self.mlp_dim)
        self.gelu = nn.GELU()
        self.w2 = nn.Linear(self.mlp_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.w1(x)
        x = self.gelu(x)
        x = self.w2(x)
        x = self.dropout(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads

        self.self_attn = MultiHeadAttention(hidden_dim=hidden_dim, n_heads=n_heads)
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.ff = PositionwiseFeedForward(hidden_dim=hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        attn_output = self.self_attn(q=x, k=x, v=x)
        x += attn_output
        x = self.norm1(x)

        ff_output = self.ff(x)
        x += ff_output
        x = self.norm2(x)
        x = self.dropout(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, n_layers, hidden_dim, n_heads):
        super().__init__()

        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.enc_stack = nn.ModuleList(
            [EncoderLayer(hidden_dim=hidden_dim, n_heads=n_heads) for _ in range(n_layers)]
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
        n_classes=1000
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_classes = n_classes

        assert img_size % patch_size == 0,\
            "`img_size` must be divisible by `patch_size`!"

        grid_size = img_size // patch_size
        n_patches = grid_size ** 2

        # $\textbf{E}$ in the equation 1 of the paper
        self.patch_embed = PatchEmbedding(patch_size=patch_size, hidden_dim=hidden_dim)

        self.cls_token = nn.Parameter(torch.randn((1, 1, hidden_dim))) # $x_{\text{class}}$
        self.pos_embed = nn.Parameter(torch.randn((1, n_patches + 1, hidden_dim))) # $\textbf{E}_\text{pos}$
        self.dropout = nn.Dropout(0.5)

        self.tf_enc = TransformerEncoder(n_layers=n_layers, hidden_dim=hidden_dim, n_heads=n_heads)

        self.ln = nn.LayerNorm(hidden_dim)

        self.mlp = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        b, _, _, _ = x.shape

        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.repeat(b, 1, 1), x), dim=1)
        x += self.pos_embed
        x = self.dropout(x) # Not in the paper

        x = self.tf_enc(x)

        x = x[:, 0] # $z^{0}_{L}$ in the equation 4 of the paper
        x = self.ln(x) # $y$

        x = self.mlp(x)
        return x


if __name__ == "__main__":
    # image = torch.randn((4, 3, 32, 32))
    vit = ViT(
        img_size=32,
        patch_size=16,
        n_layers=6,
        hidden_dim=192,
        n_heads=6,
        n_classes=100
    )
    output = vit(image)
    pred = torch.argmax(output, dim=1)
    pred
