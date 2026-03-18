from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import batch_gather_nodes, make_adj_lists, pad_neighbors


class GINLayer(nn.Module):
    """GIN layer with sum aggregation."""

    def __init__(self, in_dim: int, out_dim: int, eps: float = 0.0):
        super().__init__()
        self.eps = nn.Parameter(torch.tensor(float(eps)))
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x: torch.Tensor, neigh: torch.Tensor) -> torch.Tensor:
        # x: (B,N,F), neigh: (N, max_deg) padded with -1
        B, N, _ = x.shape
        idx = neigh.unsqueeze(0).expand(B, -1, -1)  # (B,N,D)
        mask = idx >= 0
        idx2 = idx.clamp(min=0)
        x_nei = batch_gather_nodes(x, idx2)  # (B,N,D,F)
        x_nei = x_nei * mask.unsqueeze(-1)
        agg = x_nei.sum(dim=2)  # sum over neighbors
        return self.mlp((1.0 + self.eps) * x + agg)


class GINClassifier(nn.Module):
    """Graph Isomorphism Network for graph-level multi-class classification.

    This implementation is pure PyTorch (no PyG). It treats the process graph as
    fixed across samples; each sample provides node features.
    """

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

        # Use undirected neighbor list for message passing
        _, neigh_und = make_adj_lists(A)
        pad = pad_neighbors(neigh_und, n_nodes)
        self.register_buffer('neigh', torch.from_numpy(pad))

        self.convs = nn.ModuleList([GINLayer(hidden, hidden) for _ in range(layers)])
        self.readout = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,N,in_dim)
        h = self.in_proj(x)
        neigh = self.neigh
        for conv in self.convs:
            h = conv(h, neigh)
        g = h.sum(dim=1)
        return self.readout(g)
