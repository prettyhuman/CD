from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    """Multi-head Graph Attention layer (dense mask version)."""

    def __init__(self, in_dim: int, out_dim: int, heads: int = 4):
        super().__init__()
        self.heads = int(heads)
        self.out_dim = int(out_dim)
        self.W = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.a_src = nn.Parameter(torch.empty(heads, out_dim))
        self.a_dst = nn.Parameter(torch.empty(heads, out_dim))
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, x: torch.Tensor, A_mask: torch.Tensor) -> torch.Tensor:
        # x: (B,N,F), A_mask: (N,N) binary adjacency (incl self)
        B, N, _ = x.shape
        h = self.W(x).view(B, N, self.heads, self.out_dim)  # (B,N,H,D)

        src = (h * self.a_src.view(1, 1, self.heads, self.out_dim)).sum(dim=-1)  # (B,N,H)
        dst = (h * self.a_dst.view(1, 1, self.heads, self.out_dim)).sum(dim=-1)

        logits = src.permute(0, 2, 1).unsqueeze(-1) + dst.permute(0, 2, 1).unsqueeze(-2)  # (B,H,N,N)
        logits = F.leaky_relu(logits, negative_slope=0.2)

        mask = A_mask.view(1, 1, N, N).to(logits.device)
        logits = logits.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(logits, dim=-1)

        h_head = h.permute(0, 2, 1, 3)  # (B,H,N,D)
        out = torch.matmul(attn, h_head)  # (B,H,N,D)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, self.heads * self.out_dim)
        return F.elu(out)


class GATClassifier(nn.Module):
    """GAT for graph-level classification (pure PyTorch)."""

    def __init__(
        self,
        n_nodes: int,
        in_dim: int,
        hidden: int,
        n_classes: int,
        A: np.ndarray,
        heads: int = 4,
        layers: int = 3,
    ):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden)
        self.heads = int(heads)

        Au = ((A + A.T) > 0).astype(np.int64)
        np.fill_diagonal(Au, 1)
        self.register_buffer('A_mask', torch.from_numpy(Au.astype(np.float32)))

        self.convs = nn.ModuleList([GATLayer(hidden, hidden // heads, heads=heads) for _ in range(layers)])
        self.post = nn.Linear(hidden, hidden)
        self.readout = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        for conv in self.convs:
            h = conv(h, self.A_mask)
        h = F.relu(self.post(h))
        g = h.sum(dim=1)
        return self.readout(g)
