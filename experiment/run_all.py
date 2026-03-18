from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]):
    print("\n>>>", " ".join(cmd))
    subprocess.check_call(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--diag_graph_source', choices=['truth','pc','ges','notears','rlbic','ours'], default='ours')
    ap.add_argument('--run_diag_grid', action='store_true', help='Run diagnosis on (graph_source x model) grid (produces many confusion matrices).')
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) EDA / dataset profile
    run([sys.executable, 'experiment/eda.py', '--csv', args.csv, '--outdir', args.outdir])

    # 2) Causal discovery (normal-only by default, closer to industrial setting)
    run([
        sys.executable,
        'experiment/run_causal_discovery.py',
        '--csv', args.csv,
        '--outdir', args.outdir,
        '--use_normal_only',
        '--device', args.device,
    ])

    # 3) Fault diagnosis
    if args.run_diag_grid:
        graph_sources = ['truth','pc','ges','notears','rlbic','ours']
        models = ['gin','graphsage','gat','graphormer','ours']
        for gs in graph_sources:
            for m in models:
                run([
                    sys.executable,
                    'experiment/run_fault_diagnosis.py',
                    '--csv', args.csv,
                    '--outdir', args.outdir,
                    '--graph_source', gs,
                    '--model', m,
                    '--device', args.device,
                ])
    else:
        # Default: fixed graph, compare 5 models => exactly 5 confusion matrices.
        wrappers = {
            'gin': 'experiment/diagnosis_gin.py',
            'graphsage': 'experiment/diagnosis_graphsage.py',
            'gat': 'experiment/diagnosis_gat.py',
            'graphormer': 'experiment/diagnosis_graphormer.py',
            'ours': 'experiment/diagnosis_ours.py',
        }
        for m, script in wrappers.items():
            run([
                sys.executable,
                script,
                '--csv', args.csv,
                '--outdir', args.outdir,
                '--graph_source', args.diag_graph_source,
                '--device', args.device,
            ])


if __name__ == '__main__':
    main()
