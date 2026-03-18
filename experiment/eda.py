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

import pandas as pd
import numpy as np

from topblown_causal_diag.data import load_csv
from topblown_causal_diag.config import FEATURE_COLS, LABEL_COL, LABEL_NAME_COL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    (outdir / 'reports').mkdir(parents=True, exist_ok=True)

    df = load_csv(args.csv)
    info = {
        'shape': list(df.shape),
        'n_missing': int(df.isna().sum().sum()),
        'class_counts': df[LABEL_COL].value_counts().sort_index().to_dict(),
        'class_names': df[LABEL_NAME_COL].value_counts().to_dict(),
    }

    # constant features
    const = {}
    for c in FEATURE_COLS:
        nun = df[c].nunique(dropna=False)
        if nun <= 1:
            const[c] = df[c].iloc[0]
    info['constant_features'] = const

    # correlations
    X = df[FEATURE_COLS].copy()
    X = X.loc[:, X.nunique()>1]
    cor = X.corr().abs()
    np.fill_diagonal(cor.values, 0)
    pairs = []
    cols = cor.columns
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            pairs.append((float(cor.iloc[i,j]), cols[i], cols[j]))
    pairs = sorted(pairs, reverse=True)[:15]
    info['top_abs_correlations'] = [{'abs_r':p[0],'a':p[1],'b':p[2]} for p in pairs]

    with open(outdir / 'reports' / 'dataset_profile.json', 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
