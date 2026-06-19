#!/usr/bin/env python3
"""Plot RQ3 frequency sweep from rq3_frequency_summary.csv."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--out_dir', required=True)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)

    metrics = [
        ('validation_loss', 'Validation loss'),
        ('token_accuracy', 'Token-level accuracy'),
        ('mean_comm_ms', 'Mean communication time (ms)'),
        ('step_time_ms', 'Step time (ms)'),
    ]
    for metric, ylabel in metrics:
        if metric not in df.columns:
            continue
        plt.figure()
        for residual, group in df.groupby('residual'):
            agg = group.groupby('release_period')[metric].mean().reset_index()
            plt.plot(agg['release_period'], agg[metric], marker='o', label=f'residual={residual}')
        plt.xlabel(r'$R_{low}$')
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f'rq3_{metric}.pdf')
        plt.close()


if __name__ == '__main__':
    main()
