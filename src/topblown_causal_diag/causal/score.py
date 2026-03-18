from __future__ import annotations

import numpy as np


def _ols_rss(y: np.ndarray, Xp: np.ndarray) -> float:
    """OLS residual sum of squares with an intercept.

    Args:
        y: target vector, shape (n,)
        Xp: parent design matrix (without intercept), shape (n, k)

    Returns:
        rss (float)
    """
    n = y.shape[0]
    if Xp.size == 0:
        resid = y - y.mean()
        return float((resid ** 2).sum())

    X = np.column_stack([np.ones(n), Xp])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return float((resid ** 2).sum())


def bic_score(A: np.ndarray, X: np.ndarray, penalty_scale: float = 1.0) -> float:
    """Return negative BIC (higher is better).

    Linear-Gaussian SEM with per-node OLS regression.

    Note:
        Standard BIC uses the complexity penalty k*log(n). Here we expose
        `penalty_scale` to relax/strengthen the complexity penalty. Setting
        `penalty_scale < 1` tends to yield denser graphs (useful for RL search
        that otherwise may collapse to overly sparse solutions).
    """
    penalty_scale = float(penalty_scale)
    n, d = X.shape
    total_bic = 0.0
    for j in range(d):
        parents = np.where(A[:, j] == 1)[0]
        y = X[:, j]
        Xp = X[:, parents] if len(parents) else np.empty((n, 0))
        rss = _ols_rss(y, Xp)
        rss = max(rss, 1e-12)  # avoid log(0)
        k = len(parents) + 1  # + intercept
        bic_j = n * np.log(rss / n) + penalty_scale * k * np.log(n)
        total_bic += bic_j
    return -float(total_bic)
