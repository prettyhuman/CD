from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Set, Iterable

import numpy as np
import pandas as pd
import networkx as nx


def edges_to_adjacency(edges: Iterable[Tuple[int,int]], d: int) -> np.ndarray:
    A = np.zeros((d, d), dtype=int)
    for i, j in edges:
        if i == j:
            continue
        A[i, j] = 1
    return A


def adjacency_to_edges(A: np.ndarray) -> List[Tuple[int,int]]:
    return [(i, j) for i in range(A.shape[0]) for j in range(A.shape[1]) if A[i, j] == 1]


def is_dag(A: np.ndarray) -> bool:
    G = nx.DiGraph(A)
    return nx.is_directed_acyclic_graph(G)


def topo_order(A: np.ndarray) -> List[int]:
    G = nx.DiGraph(A)
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError('Not a DAG')
    return list(nx.topological_sort(G))


def has_path(A: np.ndarray, u: int, v: int) -> bool:
    G = nx.DiGraph(A)
    return nx.has_path(G, u, v)


def try_orient_edge(A: np.ndarray, i: int, j: int) -> bool:
    """Try set i->j, return True if succeeds without cycle."""
    if A[j, i] == 1:
        return False
    A2 = A.copy()
    A2[i, j] = 1
    if is_dag(A2):
        A[:] = A2
        return True
    return False


def cpdag_to_dag(partial: np.ndarray) -> np.ndarray:
    """Convert a partially directed graph to a DAG by orienting remaining undirected edges greedily."""
    d = partial.shape[0]
    A = partial.copy().astype(int)
    # treat undirected as both directions 1; directed as single 1
    # We assume partial has 0/1 entries.

    # First, remove symmetric pairs by keeping both as 'undirected'
    undirected = [(i,j) for i in range(d) for j in range(i+1,d) if A[i,j]==1 and A[j,i]==1]
    # keep directed edges as is
    # Now orient undirected edges
    for i,j in undirected:
        # temporarily remove both
        A[i,j]=0; A[j,i]=0
    # ensure existing directed part is acyclic (if not, break cycles by removing weakest later)
    if not is_dag(A):
        # naive: remove edges until DAG
        G = nx.DiGraph(A)
        try:
            cycle = nx.find_cycle(G, orientation='original')
            while cycle:
                (u,v,_) = cycle[0]
                A[u,v]=0
                G = nx.DiGraph(A)
                cycle = nx.find_cycle(G, orientation='original')
        except Exception:
            pass

    for i,j in undirected:
        # try i->j else j->i
        if not try_orient_edge(A, i, j):
            try_orient_edge(A, j, i)
    return A


def shortest_path_distances(A: np.ndarray) -> np.ndarray:
    """Undirected shortest path distances for Graphormer bias."""
    d = A.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(d))
    edges = [(i,j) for i in range(d) for j in range(d) if A[i,j]==1]
    G.add_edges_from(edges)
    dist = np.full((d,d), fill_value=999, dtype=int)
    for i in range(d):
        dist[i,i]=0
    for i, lengths in nx.all_pairs_shortest_path_length(G):
        for j, l in lengths.items():
            dist[i,j]=l
    # clip
    return dist


def load_edge_csv(edge_csv: str, feature_names: List[str]) -> np.ndarray:
    df = pd.read_csv(edge_csv)
    idx = {n:i for i,n in enumerate(feature_names)}
    edges = [(idx[s], idx[t]) for s,t in zip(df['source'], df['target']) if s in idx and t in idx]
    return edges_to_adjacency(edges, len(feature_names))


def save_adjacency_csv(A: np.ndarray, path: str, feature_names: List[str]) -> None:
    df = pd.DataFrame(A, index=feature_names, columns=feature_names)
    df.to_csv(path, encoding='utf-8-sig')


def save_edges_csv(
    A: np.ndarray,
    path: str,
    feature_names: List[str],
    W: np.ndarray | None = None,
) -> None:
    """Save directed edges (with optional weights) into a CSV.

    Output columns:
        - source, target
        - source_idx, target_idx
        - weight (optional; if W provided)
    """
    edges = adjacency_to_edges(np.asarray(A))
    rows = []
    for i, j in edges:
        row = {
            'source': feature_names[i],
            'target': feature_names[j],
            'source_idx': int(i),
            'target_idx': int(j),
        }
        if W is not None:
            row['weight'] = float(W[i, j])
        rows.append(row)

    df = pd.DataFrame(rows)
    # stable ordering for easier diff/reading
    if not df.empty:
        df = df.sort_values(['source_idx', 'target_idx']).reset_index(drop=True)
    df.to_csv(path, index=False, encoding='utf-8-sig')
