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


def train_fn(block_fn, compile):
    torch.random.manual_seed(0)
    device = torch.device("cuda:0")
    torch.set_float32_matmul_precision("high")

    # Create dataset and dataloader
    train_set = FakeDataset()
    train_loader = DataLoader(train_set,
                              batch_size=BATCH_SIZE,
                              num_workers=12,
                              pin_memory=True,
                              drop_last=True)

    model = VisionTransformer(img_size=IMG_SIZE,
                              patch_size=PATCH_SIZE,
                              embed_dim=NUM_HEADS * HEAD_DIM,
                              depth=DEPTH,
                              num_heads=NUM_HEADS,
                              class_token=False,
                              global_pool="avg",
                              block_fn=block_fn).to(device)

    if compile:
        model = torch.compile(model)

    # Define loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters())

    model.train()

    t0 = time.perf_counter()
    summ = 0
    count = 0
    for step, data in enumerate(train_loader):
        # Copy data to GPU
        inputs = data[0].to(device=device, non_blocking=True)
        label = data[1].to(device=device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
            outputs = model(inputs)
            loss = criterion(outputs, label)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Capture step time
        batch_time = time.perf_counter() - t0
        if step > 20:  # Skip first steps
            summ += batch_time
            count += 1
        t0 = time.perf_counter()
        if step > 100:
            break
    print(f'average step time: {summ / count}')


# define compiled and uncompiled variants of our train function
train = functools.partial(train_fn, compile=False)
train_compile = functools.partial(train_fn, compile=True)


def attn_fn(q, k, v):
    scale = HEAD_DIM**-0.5
    q = q * scale
    attn = q @ k.transpose(-2, -1)
    attn = attn.softmax(dim=-1)
    x = attn @ v
    return x


block_fn = functools.partial(MyAttentionBlock, attn_fn=attn_fn)

print('Default Attention')
train(block_fn)
print('Compiled Default Attention')
train_compile(block_fn)
