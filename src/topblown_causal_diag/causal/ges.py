from __future__ import annotations

from typing import Tuple

import numpy as np

from .score import bic_score
from ..graph_utils import is_dag


def greedy_bic_search(
    X: np.ndarray,
    max_steps: int = 2000,
    verbose: bool = False,
) -> Tuple[np.ndarray, float]:
    """A practical GES-like baseline: greedy local search over DAG space using BIC.

    Operations: add, remove, reverse (if keeps DAG). Returns (A_best, score_best).
    """
    n, d = X.shape
    A = np.zeros((d,d), dtype=int)
    best = bic_score(A, X)

    def try_op(Acur: np.ndarray, op: str, i: int, j: int) -> Tuple[bool, np.ndarray, float]:
        Anew = Acur.copy()
        if op == 'add':
            if i==j or Anew[i,j]==1:
                return False, Acur, -1e18
            Anew[i,j]=1
        elif op == 'rem':
            if Anew[i,j]==0:
                return False, Acur, -1e18
            Anew[i,j]=0
        elif op == 'rev':
            if Anew[i,j]==0 or Anew[j,i]==1:
                return False, Acur, -1e18
            Anew[i,j]=0
            Anew[j,i]=1
        else:
            raise ValueError(op)
        if not is_dag(Anew):
            return False, Acur, -1e18
        sc = bic_score(Anew, X)
        return True, Anew, sc

    for step in range(max_steps):
        improved = False
        best_local = best
        best_A = A
        # enumerate ops
        for i in range(d):
            for j in range(d):
                if i==j:
                    continue
                for op in ('add','rem','rev'):
                    ok, Anew, sc = try_op(A, op, i, j)
                    if ok and sc > best_local + 1e-9:
                        best_local = sc
                        best_A = Anew
                        improved = True
        if improved:
            A = best_A
            best = best_local
            if verbose:
                print(f"step {step}: score {best:.3f}, edges {A.sum()}")
        else:
            break

    return A, float(best)
