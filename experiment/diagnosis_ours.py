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
    ap.add_argument('--lambda_sparse', type=float, default=1e-3)
    ap.add_argument('--lambda_prior', type=float, default=5e-3)
    ap.add_argument('--lambda_ci', type=float, default=1e-2)
    ap.add_argument('--n_disturb', type=int, default=2)
    args = ap.parse_args()

    cmd = [
        sys.executable,
        'experiment/run_fault_diagnosis.py',
        '--csv', args.csv,
        '--outdir', args.outdir,
        '--graph_source', args.graph_source,
        '--model', 'ours',
        '--device', args.device,
        '--epochs', str(args.epochs),
        '--lambda_sparse', str(args.lambda_sparse),
        '--lambda_prior', str(args.lambda_prior),
        '--lambda_ci', str(args.lambda_ci),
        '--n_disturb', str(args.n_disturb),
    ]
    print('>>> ' + ' '.join(cmd))
    subprocess.check_call(cmd)


if __name__ == '__main__':
    main()
