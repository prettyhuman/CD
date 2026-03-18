from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
from scipy.stats import norm

from ..graph_utils import cpdag_to_dag, is_dag


def _partial_corr(X: np.ndarray, i: int, j: int, cond: List[int]) -> float:
    """Partial correlation r_{ij|cond} via precision matrix."""
    idx = [i, j] + list(cond)
    sub = X[:, idx]
    C = np.corrcoef(sub, rowvar=False)
    # regularize
    try:
        P = np.linalg.pinv(C)
    except Exception:
        P = np.linalg.pinv(C + 1e-6*np.eye(C.shape[0]))
    r = -P[0,1] / np.sqrt(max(P[0,0]*P[1,1], 1e-12))
    r = float(np.clip(r, -0.999999, 0.999999))
    return r


def _fisher_z_test(r: float, n: int, k: int) -> float:
    """Return p-value for partial correlation with Fisher Z transform."""
    z = 0.5 * np.log((1+r)/(1-r))
    stat = np.sqrt(max(n - k - 3, 1)) * abs(z)
    p = 2 * (1 - norm.cdf(stat))
    return float(p)


def pc_discovery(
    X: np.ndarray,
    alpha: float = 0.01,
    max_cond: int = 4,
    verbose: bool = False,
) -> Tuple[np.ndarray, Dict[Tuple[int,int], Set[int]]]:
    """PC algorithm (skeleton + v-structures) with Fisher-Z CI test.

    Returns a *directed* DAG adjacency by greedy completion of CPDAG.
    """
    n, d = X.shape
    # init complete undirected graph
    adj = np.ones((d, d), dtype=int) - np.eye(d, dtype=int)
    sepset: Dict[Tuple[int,int], Set[int]] = {}

    # neighbors function
    def neighbors(v: int) -> List[int]:
        return list(np.where((adj[v] == 1) | (adj[:, v] == 1))[0])

    l = 0
    cont = True
    while cont and l <= max_cond:
        cont = False
        pairs = [(i, j) for i in range(d) for j in range(i+1, d) if adj[i,j]==1 and adj[j,i]==1]
        for (i, j) in pairs:
            nbrs = [k for k in neighbors(i) if k != j]
            if len(nbrs) < l:
                continue
            for S in combinations(nbrs, l):
                r = _partial_corr(X, i, j, list(S))
                p = _fisher_z_test(r, n=n, k=l)
                if p > alpha:
                    # remove edge i--j
                    adj[i,j] = 0
                    adj[j,i] = 0
                    sepset[(i,j)] = set(S)
                    sepset[(j,i)] = set(S)
                    cont = True
                    if verbose:
                        print(f"remove {i}-{j} |S|={l} p={p:.3g}")
                    break
        l += 1

    # orient v-structures
    # represent directed edge i->j as adj[i,j]=1, adj[j,i]=0
    for i in range(d):
        for j in range(i+1, d):
            if adj[i,j] or adj[j,i]:
                continue
            # i and j non-adjacent
            # find common neighbors k where i-k and j-k exist (undirected)
            common = [k for k in range(d) if k not in (i,j) and adj[i,k]==1 and adj[k,i]==1 and adj[j,k]==1 and adj[k,j]==1]
            for k in common:
                S = sepset.get((i,j), set())
                if k not in S:
                    # orient i->k<-j
                    adj[i,k]=1; adj[k,i]=0
                    adj[j,k]=1; adj[k,j]=0

    # Meek rules (R1-R3)
    changed = True
    while changed:
        changed = False
        # R1: i->k, k-j undirected, i not adjacent j => k->j
        for i in range(d):
            for k in range(d):
                if adj[i,k]==1 and adj[k,i]==0:
                    for j in range(d):
                        if j in (i,k):
                            continue
                        if adj[k,j]==1 and adj[j,k]==1 and (adj[i,j]==0 and adj[j,i]==0):
                            adj[k,j]=1; adj[j,k]=0
                            changed = True
        # R2: i-k undirected, i->j, j->k => i->k
        for i in range(d):
            for k in range(d):
                if i==k:
                    continue
                if adj[i,k]==1 and adj[k,i]==1:
                    for j in range(d):
                        if j in (i,k):
                            continue
                        if adj[i,j]==1 and adj[j,i]==0 and adj[j,k]==1 and adj[k,j]==0:
                            adj[i,k]=1; adj[k,i]=0
                            changed = True
        # R3: i-k undirected, j->k, l->k, i-j undirected, i-l undirected, j and l not adjacent => i->k
        for i in range(d):
            for k in range(d):
                if i==k:
                    continue
                if not (adj[i,k]==1 and adj[k,i]==1):
                    continue
                # parents of k
                parents = [p for p in range(d) if adj[p,k]==1 and adj[k,p]==0]
                for j,l in combinations(parents, 2):
                    if adj[j,l] or adj[l,j]:
                        continue
                    if adj[i,j]==1 and adj[j,i]==1 and adj[i,l]==1 and adj[l,i]==1:
                        adj[i,k]=1; adj[k,i]=0
                        changed = True

    # complete to DAG
    dag = cpdag_to_dag(adj)
    return dag, sepset
