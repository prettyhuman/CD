from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import batch_gather_nodes, make_adj_lists, pad_neighbors


class SAGEConv(nn.Module):
    """GraphSAGE mean aggregator."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim * 2, out_dim)

    def forward(self, x: torch.Tensor, neigh: torch.Tensor) -> torch.Tensor:
        # x: (B,N,F), neigh: (N, max_deg) with -1 padded
        B, N, _ = x.shape
        idx = neigh.unsqueeze(0).expand(B, -1, -1)
        mask = idx >= 0
        idx2 = idx.clamp(min=0)
        x_nei = batch_gather_nodes(x, idx2)
        x_nei = x_nei * mask.unsqueeze(-1)
        denom = mask.sum(dim=2).clamp(min=1).unsqueeze(-1)
        agg = x_nei.sum(dim=2) / denom
        h = torch.cat([x, agg], dim=-1)
        return F.relu(self.lin(h))


class GraphSAGEClassifier(nn.Module):
    """GraphSAGE for graph-level classification (pure PyTorch)."""

    def __init__(
        self,
        n_nodes: int,
        in_dim: int,
        hidden: int,
        n_classes: int,
        A: np.ndarray,
        layers: int = 3,
    ):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden)

        _, neigh_und = make_adj_lists(A)
        pad = pad_neighbors(neigh_und, n_nodes)
        self.register_buffer('neigh', torch.from_numpy(pad))

        self.convs = nn.ModuleList([SAGEConv(hidden, hidden) for _ in range(layers)])
        self.readout = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        neigh = self.neigh
        for conv in self.convs:
            h = conv(h, neigh)
        g = h.sum(dim=1)
        return self.readout(g)
