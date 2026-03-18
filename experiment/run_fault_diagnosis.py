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
from topblown_causal_diag.graph_utils import load_edge_csv
from topblown_causal_diag.viz import plot_confusion_matrix

from topblown_causal_diag.diagnosis import build_model
from topblown_causal_diag.diagnosis.trainer import TrainConfig, train_model, evaluate
from topblown_causal_diag.graph_utils import shortest_path_distances


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--graph_source', choices=['truth','pc','ges','notears','rlbic','ours'], default='notears')
    ap.add_argument('--model', choices=['gin','graphsage','gat','graphormer','ours'], default='gin')
    ap.add_argument('--device', default='auto')
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--patience', type=int, default=12)
    ap.add_argument('--lambda_sparse', type=float, default=1e-3)
    ap.add_argument('--lambda_prior', type=float, default=5e-3)
    ap.add_argument('--lambda_ci', type=float, default=1e-2)
    ap.add_argument('--n_disturb', type=int, default=2)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    df = load_csv(args.csv)
    pack = prepare_splits(df, use_normal_only_for_graph=False)
    splits = pack['splits']
    names = pack['feature_names']

    n_classes = int(df['fault_id'].max()) + 1
    class_names = [f"{i}" for i in range(n_classes)]

    # graph adjacency
    if args.graph_source == 'truth':
        A = load_edge_csv(str(Path('priors') / 'ground_truth_edges.csv'), names)
    else:
        A_path = outdir / 'causal' / args.graph_source / 'adjacency.csv'
        if not A_path.exists():
            raise FileNotFoundError(f"Adjacency not found: {A_path}. Run causal discovery first.")
        import pandas as pd
        Adf = pd.read_csv(A_path, index_col=0, encoding='utf-8-sig')
        A = Adf.loc[names, names].to_numpy(dtype=int)

    dist = shortest_path_distances(A)

    # prepare node-feature tensor: (N_samples, N_nodes, 1)
    def to_node_tensor(X: np.ndarray) -> np.ndarray:
        return X.astype(np.float32).reshape(X.shape[0], X.shape[1], 1)

    X_tr = to_node_tensor(splits.X_train)
    X_va = to_node_tensor(splits.X_val)
    X_te = to_node_tensor(splits.X_test)

    device = args.device
    if device == 'auto':
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Expert prior edges (optional). Replace priors/expert_prior_edges.csv with your own.
    prior_path = Path('priors') / 'expert_prior_edges.csv'
    prior = load_edge_csv(str(prior_path), names) if prior_path.exists() else None

    model = build_model(
        args.model,
        n_nodes=len(names),
        in_dim=1,
        hidden=args.hidden,
        n_classes=n_classes,
        A=A,
        dist=dist,
        prior=prior,
        n_disturb=args.n_disturb,
        layers=args.layers,
        heads=args.heads,
        lambda_sparse=args.lambda_sparse,
        lambda_prior=args.lambda_prior,
        lambda_ci=args.lambda_ci,
    )

    cfg = TrainConfig(epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, batch_size=args.batch_size, patience=args.patience)
    train_info = train_model(model, X_tr, splits.y_train, X_va, splits.y_val, n_classes, device, cfg)
    te = evaluate(model, X_te, splits.y_test, n_classes, device)

    tag = f"{args.model}_{args.graph_source}"
    out = outdir / 'diagnosis' / tag
    out.mkdir(parents=True, exist_ok=True)

    metrics = {k:v for k,v in te.items() if k != 'confusion_matrix' and k != 'proba' and k != 'pred'}
    metrics.update(train_info)

    with open(out / 'metrics.json','w',encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # Save confusion matrix (counts + row-normalized %) for thesis-ready reporting
    cm = te['confusion_matrix']
    import pandas as pd
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(out / 'confusion_counts.csv', encoding='utf-8-sig')
    row_sum = cm_df.sum(axis=1).replace(0, 1)
    cm_pct = cm_df.div(row_sum, axis=0) * 100.0
    cm_pct.to_csv(out / 'confusion_percent_true.csv', encoding='utf-8-sig', float_format='%.4f')
    plot_confusion_matrix(cm, class_names, str(out / 'cm.png'), title=f"CM (%, row-normalized): {tag}", normalize='true', show_counts=False)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
