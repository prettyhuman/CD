from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .rl_bic import rl_bic_search, RLBICConfig


def ours_discovery(
    X: np.ndarray,
    cfg: RLBICConfig,
    device: str = "cpu",
    seed: int = 0,
    prior: Optional[np.ndarray] = None,
    forbid: Optional[np.ndarray] = None,
    prior_weight: float = 0.5,
    prior_neg_weight: float = 0.02,
    forbid_penalty: float = 5.0,
) -> Tuple[np.ndarray, float]:
    """Our causal discovery method: Prior-injected RL-BIC.

    This wraps RL-BIC (DQN + BIC score) and injects expert knowledge by shaping the reward:
      + prior_weight for edges in `prior`
      - prior_neg_weight for edges not in `prior` (soft pull toward expert structure)
      - forbid_penalty for edges in `forbid` (soft/hard constraints)

    The RL config `cfg` also supports densification to recover a more complete graph under BIC.
    """
    A, score = rl_bic_search(
        X,
        cfg=cfg,
        device=device,
        seed=seed,
        prior=prior,
        prior_weight=prior_weight,
        prior_neg_weight=prior_neg_weight,
        forbid=forbid,
        forbid_penalty=forbid_penalty,
    )
    return A, float(score)
