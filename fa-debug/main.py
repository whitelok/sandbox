# general imports
import os, time, functools

# torch imports
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

# timm imports
from timm.models.vision_transformer import VisionTransformer
from timm.layers import Mlp

IMG_SIZE = 224
BATCH_SIZE = 128

# Define ViT settings
NUM_HEADS = 16
HEAD_DIM = 64
DEPTH = 24
PATCH_SIZE = 16
SEQ_LEN = (IMG_SIZE // PATCH_SIZE)**2  # 196


class MyAttentionBlock(nn.Module):

    def __init__(self,
                 attn_fn,
                 format=None,
                 dim: int = 768,
                 num_heads: int = 12,
                 **kwargs) -> None:
        super().__init__()
        self.attn_fn = attn_fn
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=dim * 4,
        )
        permute = (2, 0, 3, 1, 4)
        self.permute_attn = functools.partial(torch.transpose, dim0=1, dim1=2)

        if format == 'bshd':
            permute = (2, 0, 1, 3, 4)
            self.permute_attn = nn.Identity()
        self.permute_qkv = functools.partial(torch.permute, dims=permute)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x_in)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        # permute tensor based on the specified format
        qkv = self.permute_qkv(qkv)
        q, k, v = qkv.unbind(0)
        # use the attention function specified by the user
        x = self.attn_fn(q, k, v)
        # permute output according to the specified format
        x = self.permute_attn(x).reshape(B, N, C)
        x = self.proj(x)
        x = x + x_in
        x = x + self.mlp(self.norm2(x))
        return x


class FakeDataset(Dataset):

    def __len__(self):
        return 1000000

    def __getitem__(self, index):
        rand_image = torch.randn([3, IMG_SIZE, IMG_SIZE], dtype=torch.float32)
        label = torch.tensor(data=index % 1000, dtype=torch.int64)
        return rand_image, label
