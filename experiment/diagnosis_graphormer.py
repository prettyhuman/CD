from __future__ import annotations

import argparse
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--graph_source', choices=['truth','pc','ges','notears','rlbic','ours'], default='ours')
    ap.add_argument('--device', default='auto')
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--patience', type=int, default=12)
    args = ap.parse_args()

    cmd = [
        sys.executable,
        'experiment/run_fault_diagnosis.py',
        '--csv', args.csv,
        '--outdir', args.outdir,
        '--graph_source', args.graph_source,
        '--model', 'graphormer',
        '--device', args.device,        '--epochs', str(args.epochs),
        '--hidden', str(args.hidden),
        '--layers', str(args.layers),
        '--heads', str(args.heads),
        '--lr', str(args.lr),
        '--weight_decay', str(args.weight_decay),
        '--batch_size', str(args.batch_size),
        '--patience', str(args.patience),
    ]
    print('>>> ' + ' '.join(cmd))
    subprocess.check_call(cmd)


if __name__ == '__main__':
    main()
