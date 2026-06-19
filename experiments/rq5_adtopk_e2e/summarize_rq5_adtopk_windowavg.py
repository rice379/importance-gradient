#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize RQ5 ADTopk end-to-end runs."""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import pandas as pd


def read_run(run_dir: str, method: str, seed: int, burnin_steps: int) -> Dict[str, float]:
    metrics_path = os.path.join(run_dir, "metrics.csv")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(metrics_path)

    df = pd.read_csv(metrics_path)
    train = df[df["kind"] == "train"].copy()
    if "step" in train.columns:
        train = train[train["step"] >= burnin_steps]
    eval_df = df[df["kind"] == "eval"].copy()

    row: Dict[str, float] = {"method": method, "seed": seed}

    def mean_col(col: str):
        if col in train.columns and len(train) > 0:
            vals = pd.to_numeric(train[col], errors="coerce").dropna()
            if len(vals) > 0:
                row[col + "_mean"] = float(vals.mean())
                row[col + "_std"] = float(vals.std(ddof=0))

    for col in [
        "mean_step_time_ms",
        "tokens_per_sec",
        "adtopk_logical_sparse_bytes",
        "adtopk_selection_time_ms",
        "logical_sparse_comm_bytes",
        "bucket_pack_time_ms",
        "bucket_allreduce_time_ms",
        "bucket_unpack_time_ms",
        "bucket_total_comm_time_ms",
        "bucket_communicated_bytes",
        "bucket_imbalance_ratio",
        "bucket_imbalance_cv",
        "synced_bytes_est_dense",
        "residual_bytes_dense",
    ]:
        mean_col(col)

    if len(eval_df) > 0:
        last_eval = eval_df.sort_values("step").iloc[-1]
        for col in ["val_loss", "val_ppl", "val_acc"]:
            if col in eval_df.columns:
                row["final_" + col] = float(last_eval[col])

    best_path = os.path.join(run_dir, "best_summary.json")
    if os.path.exists(best_path):
        with open(best_path, "r", encoding="utf-8") as f:
            best = json.load(f)
        row["best_step"] = int(best.get("best_step", -1))
        row["best_val_loss"] = float(best.get("best_val_loss", float("nan")))
        if best.get("best_val_acc") is not None:
            row["best_val_acc"] = float(best.get("best_val_acc"))

    return row


def aggregate(df: pd.DataFrame, baseline_method: str) -> pd.DataFrame:
    metric_cols = [c for c in df.columns if c not in {"method", "seed"}]
    rows = []
    for method, g in df.groupby("method", sort=False):
        out = {"method": method, "num_seeds": len(g)}
        for col in metric_cols:
            vals = pd.to_numeric(g[col], errors="coerce").dropna()
            if len(vals) > 0:
                out[col + "_mean_over_seeds"] = float(vals.mean())
                out[col + "_std_over_seeds"] = float(vals.std(ddof=0))
        rows.append(out)
    agg = pd.DataFrame(rows)

    # Add reductions / improvements against baseline when possible.
    base = agg[agg["method"] == baseline_method]
    if len(base) == 1:
        base = base.iloc[0]
        for cost_col in [
            "logical_sparse_comm_bytes_mean_mean_over_seeds",
            "bucket_communicated_bytes_mean_mean_over_seeds",
            "bucket_total_comm_time_ms_mean_mean_over_seeds",
            "mean_step_time_ms_mean_mean_over_seeds",
        ]:
            if cost_col in agg.columns and cost_col in base and pd.notna(base[cost_col]) and base[cost_col] != 0:
                new_col = "reduction_vs_" + baseline_method + "__" + cost_col.replace("_mean_mean_over_seeds", "")
                agg[new_col] = 1.0 - agg[cost_col] / float(base[cost_col])

        thr_col = "tokens_per_sec_mean_mean_over_seeds"
        if thr_col in agg.columns and thr_col in base and pd.notna(base[thr_col]) and base[thr_col] != 0:
            agg["throughput_improvement_vs_" + baseline_method] = agg[thr_col] / float(base[thr_col]) - 1.0

    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_root", required=True)
    ap.add_argument("--methods", nargs="+", default=["adtopk_sync", "iad_adtopk", "ig_adtopk"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--baseline_method", default="adtopk_sync")
    ap.add_argument("--burnin_steps", type=int, default=100)
    ap.add_argument("--output_csv", default=None)
    ap.add_argument("--output_agg_csv", default=None)
    args = ap.parse_args()

    rows: List[Dict[str, float]] = []
    for seed in args.seeds:
        for method in args.methods:
            run_dir = os.path.join(args.run_root, f"seed{seed}", method)
            rows.append(read_run(run_dir, method=method, seed=seed, burnin_steps=args.burnin_steps))

    df = pd.DataFrame(rows)
    agg = aggregate(df, baseline_method=args.baseline_method)

    output_csv = args.output_csv or os.path.join(args.run_root, "rq5_adtopk_per_seed_summary.csv")
    output_agg_csv = args.output_agg_csv or os.path.join(args.run_root, "rq5_adtopk_mean_std_summary.csv")
    os.makedirs(args.run_root, exist_ok=True)
    df.to_csv(output_csv, index=False)
    agg.to_csv(output_agg_csv, index=False)
    print(f"[Summary] wrote {output_csv}")
    print(f"[Summary] wrote {output_agg_csv}")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
