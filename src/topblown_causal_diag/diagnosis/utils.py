from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch


def batch_gather_nodes(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather x along node dimension (dim=1) with per-batch indices.

    Args:
        x: (B, N, F)
        idx: (B, N, D) indices in [0, N-1]

    Returns:
        (B, N, D, F)
    """
    B, N, Fd = x.shape
    D = idx.shape[2]
    idx2 = idx.clamp(min=0)
    off = (torch.arange(B, device=x.device).view(B, 1, 1) * N).to(idx2.dtype)
    flat = (idx2 + off).reshape(-1)
    x_flat = x.reshape(B * N, Fd)
    out = x_flat[flat].reshape(B, N, D, Fd)
    return out


def make_adj_lists(A: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Create (directed, undirected) neighbor lists from an adjacency matrix."""
    d = A.shape[0]
    neigh = []
    for i in range(d):
        neigh.append(np.where(A[i] == 1)[0].astype(np.int64))  # outgoing
    neigh_und = []
    Au = ((A + A.T) > 0).astype(np.int64)
    for i in range(d):
        neigh_und.append(np.where(Au[i] == 1)[0].astype(np.int64))
    return neigh, neigh_und


def pad_neighbors(neigh_und: List[np.ndarray], n_nodes: int) -> np.ndarray:
    """Pad variable-length neighbor lists into a (N, max_deg) array with -1."""
    max_deg = max((len(n) for n in neigh_und), default=0)
    pad = -np.ones((n_nodes, max_deg), dtype=np.int64)
    for i, ns in enumerate(neigh_und):
        if len(ns) > 0:
            pad[i, : len(ns)] = ns
    return pad
