from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CIGNNClassifier(nn.Module):
    """A lightweight CI-GNN-style graph classifier (Ours).

    Key components (paper-friendly):
    - Expert prior injection via prior-guided edge gates
    - Explicit disturbance nodes (learnable exogenous nodes U)
    - Residual-style node features
    - CI-inspired penalty on non-edge residual correlations

    This is a pure-PyTorch baseline you can evolve into your full CI-GNN.
    """

    def __init__(
        self,
        n_nodes: int,
        in_dim: int,
        hidden: int,
        n_classes: int,
        A: np.ndarray,
        prior: Optional[np.ndarray] = None,
        n_disturb: int = 2,
        layers: int = 3,
        lambda_sparse: float = 1e-3,
        lambda_prior: float = 5e-3,
        lambda_ci: float = 1e-2,
    ):
        super().__init__()
        self.n_obs = int(n_nodes)
        self.in_dim = int(in_dim)
        self.hidden = int(hidden)
        self.n_classes = int(n_classes)
        self.n_disturb = int(n_disturb)
        self.layers = int(layers)
        self.lambda_sparse = float(lambda_sparse)
        self.lambda_prior = float(lambda_prior)
        self.lambda_ci = float(lambda_ci)

        if prior is None:
            prior = np.zeros((n_nodes, n_nodes), dtype=np.int64)
        prior = (prior > 0).astype(np.int64)

        # Candidate edges = learned graph OR expert prior (undirected for message passing)
        Au = ((A + A.T) > 0).astype(np.int64)
        Pu = ((prior + prior.T) > 0).astype(np.int64)
        C = ((Au + Pu) > 0).astype(np.int64)
        np.fill_diagonal(C, 1)

        # Extend with disturbance nodes fully connected to observed nodes
        n_all = self.n_obs + self.n_disturb
        C_ext = np.zeros((n_all, n_all), dtype=np.int64)
        C_ext[: self.n_obs, : self.n_obs] = C
        if self.n_disturb > 0:
            C_ext[: self.n_obs, self.n_obs :] = 1
            C_ext[self.n_obs :, : self.n_obs] = 1
            np.fill_diagonal(C_ext, 1)

        self.register_buffer('cand_mask', torch.from_numpy(C_ext.astype(np.float32)))

        # Prior targets for gates (only observed-observed); disturbance edges are neutral (0)
        prior_ext = np.zeros((n_all, n_all), dtype=np.float32)
        prior_ext[: self.n_obs, : self.n_obs] = ((prior + prior.T) > 0).astype(np.float32)
        np.fill_diagonal(prior_ext, 1.0)
        self.register_buffer('prior_target', torch.from_numpy(prior_ext))

        # Edge gates (logits). We only train on candidate edges; others are effectively 0.
        init = np.zeros((n_all, n_all), dtype=np.float32)
        init[: self.n_obs, : self.n_obs] += prior_ext[: self.n_obs, : self.n_obs] * 1.0
        self.gate_logits = nn.Parameter(torch.from_numpy(init))

        # Residual modeling
        self.x_proj = nn.Linear(in_dim, hidden)
        self.pred_mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.pred_out = nn.Linear(hidden, in_dim)
        self.res_proj = nn.Linear(in_dim, hidden)

        # Disturbance node embeddings
        if self.n_disturb > 0:
            self.U = nn.Parameter(torch.randn(self.n_disturb, hidden) * 0.02)

        # Message passing (weighted SAGE-style)
        self.convs = nn.ModuleList([nn.Linear(hidden * 2, hidden) for _ in range(layers)])

        self.readout = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes)
        )

        # masks for CI penalty (observed only)
        C_obs = C.copy()
        np.fill_diagonal(C_obs, 1)
        non_edge = (C_obs == 0).astype(np.float32)
        self.register_buffer('non_edge_mask_obs', torch.from_numpy(non_edge))

        self._last_ci = None

    def _edge_weights(self) -> torch.Tensor:
        # sigmoid gates on candidate edges; enforce symmetry; enforce diag ~ 1
        g = torch.sigmoid(self.gate_logits)
        g = (g + g.t()) * 0.5
        g = g * self.cand_mask
        diag = torch.eye(g.shape[0], device=g.device)
        g = g * (1.0 - diag) + diag
        return g

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N_obs, in_dim)
        B, N, Fd = x.shape
        if N != self.n_obs or Fd != self.in_dim:
            raise ValueError(f'Expected x shape (B,{self.n_obs},{self.in_dim}), got {tuple(x.shape)}')

        # Project
        xh = self.x_proj(x)  # (B,N,H)

        # Weighted aggregate with current edge weights (observed submatrix)
        W = self._edge_weights()  # (N_all,N_all)
        W_obs = W[: self.n_obs, : self.n_obs]
        denom = W_obs.sum(dim=1, keepdim=True).clamp(min=1e-6)  # (N,1)
        agg = torch.einsum('ij,bjf->bif', W_obs, xh) / denom.unsqueeze(0)  # (B,N,H)

        # Predict x from aggregated context and compute residual
        pred_h = self.pred_mlp(agg)
        x_hat = self.pred_out(pred_h)
        resid = x - x_hat

        # CI-inspired penalty: penalize correlations between residual scalars on non-edges
        r = resid[..., 0]  # (B,N)
        r = (r - r.mean(dim=0, keepdim=True)) / (r.std(dim=0, keepdim=True) + 1e-6)
        corr = (r.t() @ r) / float(B)  # (N,N)
        non_edge = self.non_edge_mask_obs.to(corr.device)
        self._last_ci = (corr * non_edge).pow(2).mean()

        # Residual embedding
        h = self.res_proj(resid)  # (B,N,H)

        # Append disturbance nodes
        if self.n_disturb > 0:
            U = self.U.unsqueeze(0).expand(B, -1, -1)  # (B,K,H)
            h = torch.cat([h, U], dim=1)

        # Message passing with weighted adjacency
        W_all = W
        denom_all = W_all.sum(dim=1, keepdim=True).clamp(min=1e-6)
        for lin in self.convs:
            agg_all = torch.einsum('ij,bjf->bif', W_all, h) / denom_all.unsqueeze(0)
            h = torch.relu(lin(torch.cat([h, agg_all], dim=-1)))

        g = h.sum(dim=1)
        return self.readout(g)

    def reg_loss(self) -> torch.Tensor:
        """Regularizers used during training (called by trainer)."""
        W = self._edge_weights()
        diag = torch.eye(W.shape[0], device=W.device)
        off = (1.0 - diag) * self.cand_mask
        sparse = (W * off).mean()

        target = self.prior_target.to(W.device)
        mask = (self.cand_mask > 0.5)
        bce = F.binary_cross_entropy_with_logits(self.gate_logits[mask], target[mask])

        ci = self._last_ci if self._last_ci is not None else W.new_tensor(0.0)
        return self.lambda_sparse * sparse + self.lambda_prior * bce + self.lambda_ci * ci
