from __future__ import annotations

from typing import Optional

import numpy as np

from .gin import GINClassifier
from .graphsage import GraphSAGEClassifier
from .gat import GATClassifier
from .graphormer import GraphormerClassifier
from .cignn import CIGNNClassifier


def build_model(
    name: str,
    *,
    n_nodes: int,
    in_dim: int,
    hidden: int,
    n_classes: int,
    A: np.ndarray,
    dist: Optional[np.ndarray] = None,
    prior: Optional[np.ndarray] = None,
    n_disturb: int = 2,
    layers: int = 3,
    heads: int = 4,
    lambda_sparse: float = 1e-3,
    lambda_prior: float = 5e-3,
    lambda_ci: float = 1e-2,
):
    """Factory for diagnosis models.

    Args:
        name: gin / graphsage / gat / graphormer / ours(cignn)
        n_nodes: number of nodes (variables)
        in_dim: node feature dim
        hidden: hidden dim
        n_classes: number of fault classes
        A: adjacency (0/1), shape (N,N)
        dist: shortest path distance matrix (required for graphormer)
        prior: expert prior adjacency (optional, used by cignn)
    """
    key = name.lower()
    if key == 'gin':
        return GINClassifier(n_nodes=n_nodes, in_dim=in_dim, hidden=hidden, n_classes=n_classes, A=A, layers=layers)
    if key == 'graphsage':
        return GraphSAGEClassifier(n_nodes=n_nodes, in_dim=in_dim, hidden=hidden, n_classes=n_classes, A=A, layers=layers)
    if key == 'gat':
        return GATClassifier(n_nodes=n_nodes, in_dim=in_dim, hidden=hidden, n_classes=n_classes, A=A, heads=heads, layers=layers)
    if key == 'graphormer':
        if dist is None:
            raise ValueError('Graphormer requires `dist` (shortest-path distances).')
        return GraphormerClassifier(n_nodes=n_nodes, in_dim=in_dim, hidden=hidden, n_classes=n_classes, dist=dist, heads=heads, layers=layers)
    if key in ('ours', 'cignn'):
        return CIGNNClassifier(
            n_nodes=n_nodes,
            in_dim=in_dim,
            hidden=hidden,
            n_classes=n_classes,
            A=A,
            prior=prior,
            n_disturb=n_disturb,
            layers=layers,
            lambda_sparse=lambda_sparse,
            lambda_prior=lambda_prior,
            lambda_ci=lambda_ci,
        )
    raise ValueError(f'Unknown model name: {name}')
