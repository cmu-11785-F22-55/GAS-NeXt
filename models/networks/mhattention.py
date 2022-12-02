# based on implementation in https://github.com/lucidrains/vit-pytorch
import torch
from einops import rearrange
from torch import Tensor, nn


class MHAttention(nn.Module):
    def __init__(
        self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0
    ):
        """Initiailization of Multi-Head attention Block

        Args:
            dim (int): input dimension
            heads (int, optional): number of heads. Defaults to 8.
            dim_head (int, optional): dimension of each head. Defaults to 64.
            dropout (float, optional): dropout rate. Defaults to 0.0.
        """
        super().__init__()
        inner_dim = dim_head * heads

        # final fully-connected layer
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, query: Tensor, key_value: Tensor):
        """
        Args:
            query (Tensor): Embeddings used to calculate queries
            kv (Tensor): Embeddings used to calculate keys and values
        Returns:
            Tensor: output from attention module
        """
        kv = self.to_kv(key_value).chunk(2, dim=-1)
        q = self.to_q(query)
        qkv = (q,) + kv
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)
