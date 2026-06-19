#!/usr/bin/env python3
"""Summarize RQ4 packing-ablation runs into one CSV and JSON file."""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List

import pandas as pd


def summarize_one(path: str, warmup_steps: int) -> Dict[str, float | str | int]:
    df = pd.read_csv(path)
    if "kind" in df.columns:
        df = df[df["kind"] == "train"].copy()
    if "step" in df.columns:
        df = df[df["step"] >= warmup_steps].copy()
    if df.empty:
        raise ValueError(f"No train rows after warmup_steps={warmup_steps}: {path}")

    policy = str(df["policy"].dropna().iloc[0]) if "policy" in df.columns and len(df["policy"].dropna()) else os.path.basename(os.path.dirname(path))

    def mean_col(col: str) -> float:
        return float(pd.to_numeric(df[col], errors="coerce").dropna().mean()) if col in df.columns else float("nan")

    def std_col(col: str) -> float:
        return float(pd.to_numeric(df[col], errors="coerce").dropna().std(ddof=0)) if col in df.columns else float("nan")

    return {
        "policy": policy,
        "rows": int(len(df)),
        "mean_bucket_imbalance_ratio": mean_col("bucket_imbalance_ratio"),
        "std_bucket_imbalance_ratio": std_col("bucket_imbalance_ratio"),
        "mean_bucket_imbalance_cv": mean_col("bucket_imbalance_cv"),
        "std_bucket_imbalance_cv": std_col("bucket_imbalance_cv"),
        "mean_bucket_overflow_count": mean_col("bucket_overflow_count"),
        "mean_pack_time_ms": mean_col("pack_time_ms"),
        "mean_allreduce_time_ms": mean_col("allreduce_time_ms"),
        "mean_unpack_time_ms": mean_col("unpack_time_ms"),
        "mean_total_comm_time_ms": mean_col("total_comm_time_ms"),
        "mean_communicated_bytes": mean_col("communicated_bytes"),
        "mean_step_time_ms": mean_col("step_time_ms"),
        "mean_train_loss": mean_col("train_loss"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_root", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--warmup_steps", type=int, default=100)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(args.input_root, "*", "metrics.csv")))
    if not paths:
        raise FileNotFoundError(f"No metrics.csv found under {args.input_root}/*/metrics.csv")

    rows: List[Dict[str, float | str | int]] = []
    for p in paths:
        rows.append(summarize_one(p, args.warmup_steps))

    # Stable order for paper table/plot.
    order = {"direct_filtering": 0, "sequential": 1, "greedy": 2, "risk_aware": 3}
    rows = sorted(rows, key=lambda x: order.get(str(x["policy"]), 99))

    out_csv = os.path.join(args.output_dir, "rq4_packing_summary.csv")
    out_json = os.path.join(args.output_dir, "rq4_packing_summary.json")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print(pd.DataFrame(rows).to_string(index=False))
    print(f"[out] {out_csv}")


if __name__ == "__main__":
    main()
