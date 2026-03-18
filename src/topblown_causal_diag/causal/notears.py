from __future__ import annotations

from typing import Tuple

import numpy as np
import scipy.linalg as sl
from scipy.optimize import minimize

from .score import bic_score
from ..graph_utils import is_dag


def _h(W: np.ndarray) -> float:
    # acyclicity constraint
    d = W.shape[0]
    return float(np.trace(sl.expm(W * W)) - d)


def _grad_h(W: np.ndarray) -> np.ndarray:
    E = sl.expm(W * W)
    return (E.T * W) * 2


def notears_linear(
    X: np.ndarray,
    lambda1: float = 0.01,
    max_iter: int = 100,
    h_tol: float = 1e-8,
    rho_max: float = 1e+16,
    w_threshold: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Linear NOTEARS. Return (A, W)."""
    n, d = X.shape

    def loss(W: np.ndarray) -> Tuple[float, np.ndarray]:
        W = W.reshape(d, d)
        np.fill_diagonal(W, 0.0)
        M = X @ W
        R = X - M
        f = 0.5 / n * np.sum(R * R)
        G = - (X.T @ R) / n
        return float(f), G

    def obj(w: np.ndarray, rho: float, alpha: float):
        W = w.reshape(d, d)
        f, G = loss(W)
        h = _h(W)
        obj = f + lambda1 * np.sum(np.abs(W)) + 0.5 * rho * h * h + alpha * h
        grad = G + lambda1 * np.sign(W) + (rho * h + alpha) * _grad_h(W)
        np.fill_diagonal(grad, 0.0)
        return obj, grad.reshape(-1)

    w_est = np.zeros((d, d), dtype=float)
    rho, alpha = 1.0, 0.0
    for _ in range(max_iter):
        sol = minimize(
            fun=lambda w: obj(w, rho, alpha)[0],
            x0=w_est.reshape(-1),
            jac=lambda w: obj(w, rho, alpha)[1],
            method='L-BFGS-B',
        )
        w_new = sol.x.reshape(d, d)
        np.fill_diagonal(w_new, 0.0)
        h_new = _h(w_new)
        if h_new <= h_tol or rho >= rho_max:
            w_est = w_new
            break
        # update
        alpha += rho * h_new
        rho *= 10
        w_est = w_new

    A = (np.abs(w_est) > w_threshold).astype(int)
    np.fill_diagonal(A, 0)
    # make DAG by removing edges if cycles remain (greedy)
    if not is_dag(A):
        # remove smallest weights on cycles
        Wabs = np.abs(w_est)
        A2 = A.copy()
        import networkx as nx
        G = nx.DiGraph(A2)
        while not nx.is_directed_acyclic_graph(G):
            cycle = nx.find_cycle(G, orientation='original')
            # pick edge with minimal |W|
            e = min([(u,v, Wabs[u,v]) for (u,v,_) in cycle], key=lambda x: x[2])
            A2[e[0], e[1]] = 0
            G = nx.DiGraph(A2)
        A = A2

    return A, w_est
