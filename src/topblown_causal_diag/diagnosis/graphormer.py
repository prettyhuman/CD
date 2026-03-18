from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphormerLayer(nn.Module):
    """A lightweight Graphormer-like Transformer block with shortest-path bias."""

    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.heads = int(heads)
        self.dim = int(dim)
        self.dk = dim // heads
        if dim % heads != 0:
            raise ValueError('dim must be divisible by heads')
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
        # x: (B,N,D), attn_bias: (B,H,N,N) or (1,H,N,N)
        B, N, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).view(B, N, 3, self.heads, self.dk).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B,H,N,dk)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.dk ** 0.5)  # (B,H,N,N)
        scores = scores + attn_bias
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (B,H,N,dk)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.proj(out)
        x = x + self.dropout(out)

        h2 = self.norm2(x)
        x = x + self.dropout(self.ffn(h2))
        return x


class GraphormerClassifier(nn.Module):
    """Graphormer for graph-level classification (pure PyTorch).

    We use shortest-path distances to build attention bias.
    """

    def __init__(
        self,
        n_nodes: int,
        in_dim: int,
        hidden: int,
        n_classes: int,
        dist: np.ndarray,
        heads: int = 4,
        layers: int = 3,
        max_dist: int = 10,
    ):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden)
        self.heads = int(heads)

        dist_clip = np.clip(dist, 0, max_dist).astype(np.int64)
        self.register_buffer('dist', torch.from_numpy(dist_clip))
        self.dist_emb = nn.Embedding(max_dist + 1, heads)

        self.tr_layers = nn.ModuleList([GraphormerLayer(hidden, heads=heads) for _ in range(layers)])
        self.readout = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,N,in_dim)
        h = self.in_proj(x)
        dist = self.dist.to(h.device)
        # (1,H,N,N)
        bias = self.dist_emb(dist).permute(2, 0, 1).unsqueeze(0)
        for layer in self.tr_layers:
            h = layer(h, bias)
        g = h.sum(dim=1)
        return self.readout(g)
