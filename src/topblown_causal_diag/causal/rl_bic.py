from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .score import bic_score
from ..graph_utils import is_dag

@dataclass
class RLBICConfig:
    steps_per_episode: int = 40
    episodes: int = 800
    replay_size: int = 50000
    batch_size: int = 128
    gamma: float = 0.95
    lr: float = 1e-3
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: float = 0.995
    target_update: int = 50
    cycle_penalty: float = 10.0
    # --- densification (post-processing) ---
    # RL search tends to be conservative under BIC and can collapse to very sparse graphs.
    # We optionally "densify" the best DAG by greedily adding edges that (approximately)
    # improve the score under a relaxed BIC penalty.
    densify: bool = True
    densify_target_edges: int | None = None
    densify_max_edges: int | None = None
    densify_tolerance: float = -0.05  # allow slightly negative delta to reach target edges
    bic_penalty_scale: float = 0.3


class QNet(nn.Module):
    def __init__(self, d: int, n_actions: int, hidden: int = 512):
        super().__init__()
        self.fc1 = nn.Linear(d*d, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, n_actions)

    def forward(self, A_flat: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(A_flat))
        x = F.relu(self.fc2(x))
        return self.out(x)


class Replay:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buf = []
        self.pos = 0

    def push(self, s, a, r, s2, done):
        item = (s, a, r, s2, done)
        if len(self.buf) < self.capacity:
            self.buf.append(item)
        else:
            self.buf[self.pos] = item
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch: int):
        idx = np.random.choice(len(self.buf), size=batch, replace=False)
        s,a,r,s2,d = zip(*[self.buf[i] for i in idx])
        return np.stack(s), np.array(a), np.array(r, dtype=np.float32), np.stack(s2), np.array(d, dtype=np.float32)

    def __len__(self):
        return len(self.buf)


def _build_actions(d: int) -> List[Tuple[str,int,int]]:
    actions = []
    for i in range(d):
        for j in range(d):
            if i==j:
                continue
            actions.append(('add', i, j))
            actions.append(('rem', i, j))
            actions.append(('rev', i, j))
    return actions


def _apply_action(A: np.ndarray, action: Tuple[str,int,int]) -> Tuple[np.ndarray, bool]:
    op,i,j = action
    A2 = A.copy()
    if op == 'add':
        if A2[i,j]==1 or i==j:
            return A, False
        A2[i,j]=1
    elif op == 'rem':
        if A2[i,j]==0:
            return A, False
        A2[i,j]=0
    elif op == 'rev':
        if A2[i,j]==0 or A2[j,i]==1:
            return A, False
        A2[i,j]=0
        A2[j,i]=1
    else:
        raise ValueError(op)
    return A2, True


def _n_edges(A: np.ndarray) -> int:
    return int(A.sum())


def _greedy_densify(
    A0: np.ndarray,
    X: np.ndarray,
    penalty_scale: float,
    target_edges: int,
    max_edges: int,
    tolerance: float,
    prior: Optional[np.ndarray] = None,
    prior_weight: float = 0.0,
    forbid: Optional[np.ndarray] = None,
    forbid_penalty: float = 0.0,
) -> np.ndarray:
    """Greedily add edges to make the graph more complete.

    Motivation:
        Under standard BIC, score improvements from adding an edge can be outweighed
        by the complexity penalty, especially in RL where exploration/credit assignment
        is imperfect. This post-processing step helps recover a denser graph by
        (i) relaxing the BIC penalty (penalty_scale < 1), and
        (ii) greedily adding edges with the best score delta.

    Args:
        A0: initial DAG adjacency
        X: data
        penalty_scale: BIC complexity penalty multiplier
        target_edges: desired edge count (stop early if reached)
        max_edges: hard cap on edges
        tolerance: allow adding edges with delta >= tolerance (can be slightly negative)
        prior/prior_weight: optional bonus for adding prior edges
        forbid/forbid_penalty: optional penalty for adding forbidden edges
    """
    n, d = X.shape
    A = A0.copy().astype(int)
    prior = (prior > 0).astype(int) if prior is not None else None
    forbid = (forbid > 0).astype(int) if forbid is not None else None

    def shaped_score(Ax: np.ndarray) -> float:
        sc = bic_score(Ax, X, penalty_scale=penalty_scale)
        if prior is not None and prior_weight != 0.0:
            sc += float((Ax * prior).sum()) * prior_weight
        if forbid is not None and forbid_penalty != 0.0:
            sc -= float((Ax * forbid).sum()) * forbid_penalty
        return float(sc)

    cur = shaped_score(A)
    # greedy add loop
    while _n_edges(A) < min(max_edges, d * (d - 1)):
        if _n_edges(A) >= target_edges:
            break
        best_delta = -1e18
        best_edge = None
        for i in range(d):
            for j in range(d):
                if i == j or A[i, j] == 1:
                    continue
                if forbid is not None and forbid[i, j] == 1:
                    continue
                A2 = A.copy()
                A2[i, j] = 1
                if not is_dag(A2):
                    continue
                sc2 = shaped_score(A2)
                delta = sc2 - cur
                # small tie-breaker: prefer prior edges
                if prior is not None and prior[i, j] == 1:
                    delta += 1e-6
                if delta > best_delta:
                    best_delta = delta
                    best_edge = (i, j)
        if best_edge is None or best_delta < tolerance:
            break
        i, j = best_edge
        A[i, j] = 1
        cur = cur + best_delta
    return A


def rl_bic_search(
    X: np.ndarray,
    cfg: RLBICConfig,
    device: str = 'cpu',
    seed: int = 0,
    prior: Optional[np.ndarray] = None,
    prior_weight: float = 0.5,
    prior_neg_weight: float = 0.1,
    forbid: Optional[np.ndarray] = None,
    forbid_penalty: float = 5.0,
) -> Tuple[np.ndarray, float]:
    """DQN search for DAG maximizing BIC. Returns best DAG found."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    n, d = X.shape
    actions = _build_actions(d)
    n_actions = len(actions)

    q = QNet(d, n_actions).to(device)
    qt = QNet(d, n_actions).to(device)
    qt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=cfg.lr)
    rep = Replay(cfg.replay_size)

    # start from empty
    best_A = np.zeros((d,d), dtype=int)
    best_score = bic_score(best_A, X, penalty_scale=cfg.bic_penalty_scale)
    # optional expert prior / hard forbids
    if prior is None:
        prior = np.zeros((d,d), dtype=int)
    prior = (prior > 0).astype(int)
    if forbid is not None:
        forbid = (forbid > 0).astype(int)

    def prior_term(A: np.ndarray) -> float:
        # reward edges supported by prior; mildly penalize non-prior edges
        pos = float((A * prior).sum())
        neg = float((A * (1 - prior)).sum())
        return prior_weight * pos - prior_neg_weight * neg

    def forbid_term(A: np.ndarray) -> float:
        if forbid is None:
            return 0.0
        return float((A * forbid).sum())


    # recompute initial best with shaping terms
    best_score = best_score + prior_term(best_A) - forbid_penalty * forbid_term(best_A)

    eps = cfg.eps_start

    def reward(A_prev: np.ndarray, A_new: np.ndarray) -> float:
        # reward = delta BIC - cycle penalty
        sc_prev = bic_score(A_prev, X, penalty_scale=cfg.bic_penalty_scale)
        sc_new = bic_score(A_new, X, penalty_scale=cfg.bic_penalty_scale) if is_dag(A_new) else sc_prev
        r = sc_new - sc_prev
        # prior / forbid shaping
        r += (prior_term(A_new) - prior_term(A_prev))
        r -= forbid_penalty * (forbid_term(A_new) - forbid_term(A_prev))
        if not is_dag(A_new):
            r -= cfg.cycle_penalty
        return float(r)

    for ep in range(cfg.episodes):
        A = np.zeros((d,d), dtype=int)
        for t in range(cfg.steps_per_episode):
            s = A.reshape(-1).astype(np.float32)
            # epsilon-greedy
            if rng.random() < eps:
                a_idx = rng.integers(0, n_actions)
            else:
                with torch.no_grad():
                    qs = q(torch.from_numpy(s).to(device).unsqueeze(0))
                    a_idx = int(torch.argmax(qs, dim=1).item())

            A2, ok = _apply_action(A, actions[a_idx])
            if not ok:
                # small negative reward for invalid
                r = -0.1
                s2 = s
                done = 0.0
            else:
                r = reward(A, A2)
                s2 = A2.reshape(-1).astype(np.float32)
                done = 0.0
                A = A2

                # track best (must be DAG)
                if is_dag(A):
                    sc = bic_score(A, X, penalty_scale=cfg.bic_penalty_scale) + prior_term(A) - forbid_penalty * forbid_term(A)
                    if sc > best_score:
                        best_score = sc
                        best_A = A.copy()

            rep.push(s, a_idx, r, s2, done)

            # learn
            if len(rep) >= cfg.batch_size:
                sb, ab, rb, s2b, db = rep.sample(cfg.batch_size)
                sb_t = torch.from_numpy(sb).to(device)
                ab_t = torch.from_numpy(ab).to(device)
                rb_t = torch.from_numpy(rb).to(device)
                s2b_t = torch.from_numpy(s2b).to(device)
                db_t = torch.from_numpy(db).to(device)

                qv = q(sb_t).gather(1, ab_t.view(-1,1)).squeeze(1)
                with torch.no_grad():
                    q_next = qt(s2b_t).max(1).values
                    target = rb_t + cfg.gamma * q_next * (1.0 - db_t)
                loss = F.smooth_l1_loss(qv, target)
                opt.zero_grad()
                loss.backward()
                opt.step()

        # update target
        if (ep+1) % cfg.target_update == 0:
            qt.load_state_dict(q.state_dict())

        eps = max(cfg.eps_end, eps * cfg.eps_decay)

    # optional densification to avoid overly sparse graphs
    if cfg.densify:
        target_edges = cfg.densify_target_edges
        if target_edges is None:
            # heuristic target: ~1.7*d edges (for d=18 -> ~30)
            target_edges = max(0, int(round(1.7 * d)))
        max_edges = cfg.densify_max_edges
        if max_edges is None:
            max_edges = max(target_edges, int(round(2.0 * d)))
        best_A = _greedy_densify(
            best_A,
            X,
            penalty_scale=cfg.bic_penalty_scale,
            target_edges=target_edges,
            max_edges=max_edges,
            tolerance=cfg.densify_tolerance,
            prior=prior,
            prior_weight=prior_weight,
            forbid=forbid,
            forbid_penalty=forbid_penalty,
        )
        best_score = bic_score(best_A, X, penalty_scale=cfg.bic_penalty_scale) + prior_term(best_A) - forbid_penalty * forbid_term(best_A)

    return best_A, float(best_score)


def quick_config() -> RLBICConfig:
    return RLBICConfig(steps_per_episode=25, episodes=200, batch_size=64, target_update=25)
