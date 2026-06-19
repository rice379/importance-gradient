#!/usr/bin/env python3
"""
Analyze RQ1 cross-workload component-importance stability.

Input: profile CSV files exported by rq1_profile_workload_importance.py.
Output: pairwise CSV, per-model summary CSV/JSON, and heatmaps.

Metrics:
  - Jaccard overlap of important component sets
  - Top-k overlap, where k is the smaller important-set size of the pair
  - Spearman rank correlation over shared component scores
  - absolute threshold difference |Delta tau|
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def infer_label(path: str, df: pd.DataFrame, col: str, default: str) -> str:
    if col in df.columns and pd.notna(df[col].iloc[0]):
        return str(df[col].iloc[0])
    return default


def load_profile(path: str) -> dict:
    df = pd.read_csv(path)
    required = {"layer", "component", "final_score", "global_tau"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")

    df = df.copy()
    df["key"] = df["layer"].astype(int).astype(str) + ":" + df["component"].astype(str)
    tau = float(df["global_tau"].dropna().iloc[0])

    if "is_important" in df.columns:
        high = set(df.loc[df["is_important"].astype(bool), "key"].tolist())
    elif "is_high_priority" in df.columns:
        high = set(df.loc[df["is_high_priority"].astype(bool), "key"].tolist())
    else:
        high = set(df.loc[df["final_score"] >= tau, "key"].tolist())

    scores = {str(k): float(v) for k, v in zip(df["key"], df["final_score"])}
    model_name = infer_label(path, df, "model_name", "model")
    workload_name = infer_label(path, df, "workload_name", os.path.splitext(os.path.basename(path))[0])
    dataset_name = infer_label(path, df, "dataset_name", "dataset")
    profile_id = infer_label(path, df, "profile_id", os.path.splitext(os.path.basename(path))[0])

    return {
        "path": path,
        "profile_id": profile_id,
        "model_name": model_name,
        "workload_name": workload_name,
        "dataset_name": dataset_name,
        "tau": tau,
        "important": high,
        "scores": scores,
        "important_count": len(high),
        "component_count": len(scores),
    }


def rank_scores(scores: Dict[str, float], keys: List[str]) -> np.ndarray:
    ordered = sorted(keys, key=lambda k: (-scores.get(k, float("-inf")), k))
    rank = {k: i + 1 for i, k in enumerate(ordered)}
    return np.array([rank[k] for k in keys], dtype=np.float64)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt((x * x).sum()) * np.sqrt((y * y).sum()))
    if denom < 1e-12:
        return float("nan")
    return float((x * y).sum() / denom)


def pairwise_metrics(a: dict, b: dict) -> dict:
    keys = sorted(set(a["scores"]) & set(b["scores"]))
    imp_a, imp_b = a["important"], b["important"]
    union = imp_a | imp_b
    inter = imp_a & imp_b
    jaccard = len(inter) / max(len(union), 1)

    k = min(len(imp_a), len(imp_b))
    top_a = set(sorted(a["scores"], key=lambda kk: (-a["scores"][kk], kk))[:k]) if k > 0 else set()
    top_b = set(sorted(b["scores"], key=lambda kk: (-b["scores"][kk], kk))[:k]) if k > 0 else set()
    topk_overlap = len(top_a & top_b) / max(k, 1)

    rank_a = rank_scores(a["scores"], keys)
    rank_b = rank_scores(b["scores"], keys)
    spearman = pearson(rank_a, rank_b)

    return {
        "model_name": a["model_name"],
        "workload_a": a["workload_name"],
        "workload_b": b["workload_name"],
        "dataset_a": a["dataset_name"],
        "dataset_b": b["dataset_name"],
        "profile_a": a["profile_id"],
        "profile_b": b["profile_id"],
        "path_a": a["path"],
        "path_b": b["path"],
        "important_count_a": a["important_count"],
        "important_count_b": b["important_count"],
        "intersection": len(inter),
        "union": len(union),
        "jaccard_important": jaccard,
        "topk_overlap": topk_overlap,
        "spearman_rank": spearman,
        "tau_a": a["tau"],
        "tau_b": b["tau"],
        "tau_abs_diff": abs(a["tau"] - b["tau"]),
        "shared_components": len(keys),
    }


def save_matrix_figure(profiles: List[dict], pair_df: pd.DataFrame, output_path: str, metric: str, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] matplotlib unavailable, skip figure {output_path}: {exc}")
        return

    labels = [p["workload_name"] for p in profiles]
    mat = np.eye(len(labels), dtype=np.float64)
    idx = {label: i for i, label in enumerate(labels)}
    for _, row in pair_df.iterrows():
        i = idx[row["workload_a"]]
        j = idx[row["workload_b"]]
        val = float(row[metric])
        mat[i, j] = val
        mat[j, i] = val

    fig, ax = plt.subplots(figsize=(max(5, 0.8 * len(labels)), max(4, 0.7 * len(labels))))
    im = ax.imshow(mat, vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_score_rank_table(profiles: List[dict], output_csv: str) -> None:
    rows = []
    for p in profiles:
        ordered = sorted(p["scores"], key=lambda k: (-p["scores"][k], k))
        for rank, key in enumerate(ordered, start=1):
            rows.append({
                "model_name": p["model_name"],
                "workload_name": p["workload_name"],
                "component_key": key,
                "rank": rank,
                "score": p["scores"][key],
                "is_important": key in p["important"],
            })
    pd.DataFrame(rows).to_csv(output_csv, index=False)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--profile_glob", type=str, required=True, help="Glob pattern for profile CSV files")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--group_by", type=str, default="model_name", choices=["model_name", "all"], help="Compare workloads within each model or compare all profiles together")
    return p.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    paths = sorted(glob.glob(args.profile_glob))
    if len(paths) < 2:
        raise RuntimeError(f"Need at least 2 profiles, got {len(paths)} from {args.profile_glob}")

    profiles = [load_profile(p) for p in paths]

    groups: Dict[str, List[dict]] = {}
    if args.group_by == "all":
        groups["all"] = profiles
    else:
        for p in profiles:
            groups.setdefault(p["model_name"], []).append(p)

    all_rows: List[dict] = []
    summary_rows: List[dict] = []
    for group_name, group_profiles in groups.items():
        if len(group_profiles) < 2:
            print(f"[warn] skip group={group_name}: only {len(group_profiles)} profile")
            continue
        rows: List[dict] = []
        for i in range(len(group_profiles)):
            for j in range(i + 1, len(group_profiles)):
                rows.append(pairwise_metrics(group_profiles[i], group_profiles[j]))
        pair_df = pd.DataFrame(rows)
        all_rows.extend(rows)

        summary_rows.append({
            "group": group_name,
            "num_profiles": len(group_profiles),
            "mean_jaccard_important": float(pair_df["jaccard_important"].mean()),
            "std_jaccard_important": float(pair_df["jaccard_important"].std(ddof=0)),
            "mean_topk_overlap": float(pair_df["topk_overlap"].mean()),
            "std_topk_overlap": float(pair_df["topk_overlap"].std(ddof=0)),
            "mean_spearman_rank": float(pair_df["spearman_rank"].mean()),
            "std_spearman_rank": float(pair_df["spearman_rank"].std(ddof=0)),
            "mean_tau_abs_diff": float(pair_df["tau_abs_diff"].mean()),
            "std_tau_abs_diff": float(pair_df["tau_abs_diff"].std(ddof=0)),
        })

        safe_group = "".join(c if c.isalnum() or c in "._-" else "_" for c in group_name)
        save_matrix_figure(
            group_profiles,
            pair_df,
            os.path.join(args.output_dir, f"{safe_group}_jaccard_important_heatmap.pdf"),
            metric="jaccard_important",
            title=f"Important-component Jaccard ({group_name})",
        )
        save_matrix_figure(
            group_profiles,
            pair_df,
            os.path.join(args.output_dir, f"{safe_group}_spearman_rank_heatmap.pdf"),
            metric="spearman_rank",
            title=f"Component-score Spearman ({group_name})",
        )

    pair_all = pd.DataFrame(all_rows)
    pair_csv = os.path.join(args.output_dir, "rq1_workload_pairwise.csv")
    pair_all.to_csv(pair_csv, index=False)

    profile_summary = pd.DataFrame([
        {
            "model_name": p["model_name"],
            "workload_name": p["workload_name"],
            "dataset_name": p["dataset_name"],
            "profile_id": p["profile_id"],
            "path": p["path"],
            "tau": p["tau"],
            "important_count": p["important_count"],
            "component_count": p["component_count"],
        }
        for p in profiles
    ])
    profile_summary.to_csv(os.path.join(args.output_dir, "rq1_workload_profiles.csv"), index=False)
    save_score_rank_table(profiles, os.path.join(args.output_dir, "rq1_workload_component_ranks.csv"))

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(args.output_dir, "rq1_workload_summary_by_model.csv")
    summary_df.to_csv(summary_csv, index=False)
    summary = {
        "num_profiles": len(profiles),
        "num_groups": len(summary_rows),
        "pairwise_csv": pair_csv,
        "summary_csv": summary_csv,
        "groups": summary_rows,
    }
    with open(os.path.join(args.output_dir, "rq1_workload_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[out] pairwise metrics: {pair_csv}")
    print(f"[out] summary: {summary_csv}")


if __name__ == "__main__":
    main()
