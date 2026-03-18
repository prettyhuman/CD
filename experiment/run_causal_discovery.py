from __future__ import annotations

# ---- path bootstrap (so scripts work without installation) ----
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..'))
_SRC = os.path.join(_PROJECT_ROOT, 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
# --------------------------------------------------------------

import argparse
import json
from pathlib import Path

import numpy as np

from topblown_causal_diag.data import load_csv, prepare_splits
from topblown_causal_diag.graph_utils import load_edge_csv, save_adjacency_csv, save_edges_csv
from topblown_causal_diag.metrics_causal import edge_scores
from topblown_causal_diag.viz import plot_dag

from topblown_causal_diag.causal.pc import pc_discovery
from topblown_causal_diag.causal.ges import greedy_bic_search
from topblown_causal_diag.causal.notears import notears_linear
from topblown_causal_diag.causal.rl_bic import rl_bic_search, quick_config, RLBICConfig
from topblown_causal_diag.causal.ours import ours_discovery


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--use_normal_only', action='store_true')
    ap.add_argument('--alpha', type=float, default=0.01)
    ap.add_argument('--max_cond', type=int, default=4)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--rl_mode', choices=['quick','full'], default='quick')
    ap.add_argument('--prior_weight', type=float, default=0.5)
    # Penalize non-prior edges too hard => graph becomes unrealistically sparse.
    ap.add_argument('--prior_neg_weight', type=float, default=0.02)
    ap.add_argument('--bic_penalty_scale', type=float, default=0.3)
    ap.add_argument('--densify_target_edges', type=int, default=30)
    ap.add_argument('--densify_max_edges', type=int, default=36)
    ap.add_argument('--densify_tolerance', type=float, default=-0.05)
    ap.add_argument('--forbid_penalty', type=float, default=5.0)
    args = ap.parse_args()

    if args.device == 'auto':
        import torch
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    outdir = Path(args.outdir)
    df = load_csv(args.csv)
    pack = prepare_splits(df, use_normal_only_for_graph=args.use_normal_only)
    Xg = pack['X_graph']
    names = pack['feature_names']

    # ground truth
    true_A = load_edge_csv(str(Path('priors')/ 'ground_truth_edges.csv'), names)

    base = outdir / 'causal'
    _ensure_dir(base)

    results = {}

    # PC
    A_pc, _ = pc_discovery(Xg, alpha=args.alpha, max_cond=args.max_cond)
    m = edge_scores(true_A, A_pc)
    pdir = base / 'pc'
    _ensure_dir(pdir)
    save_adjacency_csv(A_pc, str(pdir / 'adjacency.csv'), names)
    save_edges_csv(A_pc, str(pdir / 'edges.csv'), names)
    plot_dag(A_pc, names, str(pdir / 'graph.png'), title='PC (completed DAG)')
    with open(pdir / 'metrics.json','w',encoding='utf-8') as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
    results['pc'] = m

    # GES-like
    A_ges, sc = greedy_bic_search(Xg)
    m = edge_scores(true_A, A_ges)
    m['score_bic'] = sc
    pdir = base / 'ges'
    _ensure_dir(pdir)
    save_adjacency_csv(A_ges, str(pdir / 'adjacency.csv'), names)
    save_edges_csv(A_ges, str(pdir / 'edges.csv'), names)
    plot_dag(A_ges, names, str(pdir / 'graph.png'), title='GES-like (Greedy BIC DAG search)')
    with open(pdir / 'metrics.json','w',encoding='utf-8') as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
    results['ges'] = m

    # NOTEARS
    A_no, W = notears_linear(Xg)
    m = edge_scores(true_A, A_no)
    pdir = base / 'notears'
    _ensure_dir(pdir)
    save_adjacency_csv(A_no, str(pdir / 'adjacency.csv'), names)
    # save edge list with weights for NOTEARS
    save_edges_csv(A_no, str(pdir / 'edges.csv'), names, W=W)
    plot_dag(A_no, names, str(pdir / 'graph.png'), title='NOTEARS (linear)')
    with open(pdir / 'metrics.json','w',encoding='utf-8') as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
    results['notears'] = m

    # RL-BIC
    if args.rl_mode == 'quick':
        cfg = quick_config()
    else:
        cfg = RLBICConfig()
    # tune RL-BIC density (default expects ~30 edges for d=18)
    cfg.bic_penalty_scale = float(args.bic_penalty_scale)
    cfg.densify_target_edges = int(args.densify_target_edges) if args.densify_target_edges > 0 else None
    cfg.densify_max_edges = int(args.densify_max_edges) if args.densify_max_edges > 0 else None
    cfg.densify_tolerance = float(args.densify_tolerance)
    A_rl, sc = rl_bic_search(Xg, cfg=cfg, device=args.device)
    m = edge_scores(true_A, A_rl)
    m['score_bic'] = sc
    pdir = base / 'rlbic'
    _ensure_dir(pdir)
    save_adjacency_csv(A_rl, str(pdir / 'adjacency.csv'), names)
    save_edges_csv(A_rl, str(pdir / 'edges.csv'), names)
    plot_dag(A_rl, names, str(pdir / 'graph.png'), title='RL-BIC (DQN search)')
    with open(pdir / 'metrics.json','w',encoding='utf-8') as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
    results['rlbic'] = m


    # OURS: Prior-injected RL-BIC (expert prior + BIC)
    prior_path = Path('priors') / 'expert_prior_edges.csv'
    forbid_path = Path('priors') / 'forbidden_edges.csv'
    prior = load_edge_csv(str(prior_path), names) if prior_path.exists() else None
    forbid = load_edge_csv(str(forbid_path), names) if forbid_path.exists() else None

    A_ours, sc = ours_discovery(
        Xg,
        cfg=cfg,
        device=args.device,
        seed=0,
        prior=prior,
        forbid=forbid,
        prior_weight=args.prior_weight,
        prior_neg_weight=args.prior_neg_weight,
        forbid_penalty=args.forbid_penalty,
    )
    m = edge_scores(true_A, A_ours)
    m['score_bic_plus_prior'] = sc
    pdir = base / 'ours'
    _ensure_dir(pdir)
    save_adjacency_csv(A_ours, str(pdir / 'adjacency.csv'), names)
    save_edges_csv(A_ours, str(pdir / 'edges.csv'), names)
    plot_dag(A_ours, names, str(pdir / 'graph.png'), title='Ours (Prior-injected RL-BIC)')
    with open(pdir / 'metrics.json','w',encoding='utf-8') as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
    results['ours'] = m
    

    with open(base / 'summary_metrics.json','w',encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
