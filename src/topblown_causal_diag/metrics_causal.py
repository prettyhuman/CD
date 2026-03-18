from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Set

import numpy as np


def _edge_set(A: np.ndarray) -> Set[Tuple[int,int]]:
    return {(i,j) for i in range(A.shape[0]) for j in range(A.shape[1]) if A[i,j]==1}


def shd(true_A: np.ndarray, pred_A: np.ndarray) -> int:
    """Structural Hamming Distance with reversal counted as 1."""
    T = _edge_set(true_A)
    P = _edge_set(pred_A)

    # reversals: i->j in P and j->i in T
    rev = set()
    for (i,j) in P:
        if (j,i) in T and (i,j) not in T:
            rev.add((i,j))
    # Count reversals once
    rev_cnt = len(rev)

    # additions: edges in P not in T and not a reversal counterpart
    add = {e for e in P if e not in T and (e[1], e[0]) not in T}
    # deletions: edges in T not in P and not reversed in P
    del_ = {e for e in T if e not in P and (e[1], e[0]) not in P}
    return rev_cnt + len(add) + len(del_)


def edge_scores(true_A: np.ndarray, pred_A: np.ndarray) -> Dict[str, float]:
    T = _edge_set(true_A)
    P = _edge_set(pred_A)
    tp = len(T & P)
    fp = len(P - T)
    fn = len(T - P)

    tpr = tp / (tp + fn + 1e-12)
    fdr = fp / (tp + fp + 1e-12)
    precision = tp / (tp + fp + 1e-12)
    recall = tpr
    f1 = 2*precision*recall / (precision + recall + 1e-12)

    return {
        'TP': float(tp),
        'FP': float(fp),
        'FN': float(fn),
        'TPR': float(tpr),
        'FDR': float(fdr),
        'Precision': float(precision),
        'Recall': float(recall),
        'F1': float(f1),
        'SHD': float(shd(true_A, pred_A)),
        'n_true_edges': float(len(T)),
        'n_pred_edges': float(len(P)),
    }
