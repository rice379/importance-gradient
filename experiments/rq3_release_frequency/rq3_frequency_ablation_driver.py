#!/usr/bin/env python3
"""
RQ3 frequency-ablation driver.

This script sweeps the release period of low-importance components and
compares two variants:
  1) with residual compensation
  2) without residual compensation

It is a launcher/collector. It assumes your actual training entry already
implements component-aware sparse synchronization and accepts the arguments
specified through --release-period-arg, --residual-on-args, and
--residual-off-args.

Example:
  python rq3_frequency_ablation_driver.py \
    --train_entry rq2_partition_ablation.py \
    --nproc_per_node 1 \
    --output_root runs/rq3_frequency_opt13b_pubmed \
    --release_periods 1,2,4,8,16 \
    --seeds 42,43,44 \
    --residual_on_args "--use_residual_compensation" \
    --residual_off_args "--disable_residual_compensation" \
    --train_args "--model_name facebook/opt-1.3b --dataset_path pubmed_data ..."
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def parse_csv_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def safe_name(s: str) -> str:
    return s.replace("/", "_").replace(" ", "_").replace("=", "-")


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def read_jsonl_last(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    last = None
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                last = obj
        except Exception:
            continue
    return last


def load_metrics(run_dir: Path) -> Dict[str, Any]:
    """Best-effort metric loader for common output formats.

    Your training script can write any of the following files:
      - summary.json
      - metrics.json
      - eval_results.json
      - metrics.jsonl
      - trainer_state.json
    The driver will merge what it can find.
    """
    metrics: Dict[str, Any] = {}
    for name in ["summary.json", "metrics.json", "eval_results.json"]:
        obj = read_json(run_dir / name)
        if obj:
            metrics.update(obj)

    obj = read_jsonl_last(run_dir / "metrics.jsonl")
    if obj:
        metrics.update(obj)

    trainer_state = read_json(run_dir / "trainer_state.json")
    if trainer_state and isinstance(trainer_state.get("log_history"), list):
        for item in trainer_state["log_history"]:
            if isinstance(item, dict):
                # keep the latest occurrence of each scalar
                metrics.update(item)
    return metrics


def normalize_metrics(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Map possible metric names to the names used in the paper."""
    aliases = {
        "validation_loss": ["validation_loss", "val_loss", "eval_loss", "loss"],
        "token_accuracy": ["token_accuracy", "accuracy", "eval_accuracy", "token_level_accuracy"],
        "communicated_bytes": ["communicated_bytes", "comm_bytes", "total_comm_bytes", "bytes"],
        "bytes_reduction": ["bytes_reduction", "byte_reduction", "comm_reduction"],
        "mean_comm_ms": ["mean_comm_ms", "comm_ms", "mean_communication_time_ms", "communication_time_ms"],
        "allreduce_ms": ["allreduce_ms", "all_reduce_ms", "mean_allreduce_ms"],
        "step_time_ms": ["step_time_ms", "mean_step_time_ms", "train_step_ms"],
        "tokens_per_sec": ["tokens_per_sec", "throughput", "train_tokens_per_second"],
        "comm_ratio": ["comm_ratio", "communication_ratio"],
    }
    out: Dict[str, Any] = {}
    for canonical, keys in aliases.items():
        for k in keys:
            if k in raw:
                out[canonical] = raw[k]
                break
    return out


def build_command(args: argparse.Namespace, run_dir: Path, release_period: int, seed: int, residual: bool, policy_name: str) -> List[str]:
    cmd: List[str] = []
    if args.use_torchrun:
        cmd += [
            "torchrun",
            f"--nproc_per_node={args.nproc_per_node}",
            f"--master_port={args.master_port}",
        ]
    else:
        cmd += [sys.executable]

    cmd.append(args.train_entry)
    cmd += shlex.split(args.train_args)

    cmd += [args.release_period_arg, str(release_period)]
    cmd += ["--output_dir", str(run_dir)]

    if args.seed_arg:
        cmd += [args.seed_arg, str(seed)]

    if args.policy_name_arg:
        cmd += [args.policy_name_arg, policy_name]

    residual_args = args.residual_on_args if residual else args.residual_off_args
    if residual_args.strip():
        cmd += shlex.split(residual_args)

    if args.extra_args.strip():
        cmd += shlex.split(args.extra_args)

    return cmd


def write_rows_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    all_keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in all_keys:
                all_keys.append(k)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def write_latex_table(path: Path, rows: List[Dict[str, Any]]) -> None:
    cols = [
        "release_period", "residual", "seed", "validation_loss", "token_accuracy",
        "communicated_bytes", "bytes_reduction", "mean_comm_ms", "step_time_ms", "tokens_per_sec"
    ]
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Effect of low-importance component release frequency.}")
    lines.append(r"\label{tab:rq3_frequency}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lcccccc}")
    lines.append(r"\toprule")
    lines.append(r"$R_{\mathrm{low}}$ & Residual & Seed & Val. Loss $\downarrow$ & Acc. $\uparrow$ & Comm. Time $\downarrow$ & Step Time $\downarrow$ \\")
    lines.append(r"\midrule")
    for r in rows:
        def fmt(x: Any) -> str:
            if x is None or x == "":
                return "--"
            if isinstance(x, float):
                return f"{x:.4f}"
            return str(x)
        lines.append(
            f"{fmt(r.get('release_period'))} & {fmt(r.get('residual'))} & {fmt(r.get('seed'))} & "
            f"{fmt(r.get('validation_loss'))} & {fmt(r.get('token_accuracy'))} & "
            f"{fmt(r.get('mean_comm_ms'))} & {fmt(r.get('step_time_ms'))} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    path.write_text("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train_entry", required=True, help="Actual training script, e.g., rq2_partition_ablation.py")
    p.add_argument("--output_root", required=True)
    p.add_argument("--release_periods", default="1,2,4,8,16")
    p.add_argument("--seeds", default="42")
    p.add_argument("--train_args", required=True, help="Quoted common args passed to the training script")
    p.add_argument("--extra_args", default="", help="Extra args appended to every training command")
    p.add_argument("--release_period_arg", default="--release_period")
    p.add_argument("--seed_arg", default="--seed")
    p.add_argument("--policy_name_arg", default="--rq3_policy_name")
    p.add_argument("--residual_on_args", default="--use_residual_compensation")
    p.add_argument("--residual_off_args", default="--disable_residual_compensation")
    p.add_argument("--use_torchrun", action="store_true")
    p.add_argument("--nproc_per_node", type=int, default=1)
    p.add_argument("--master_port", type=int, default=29501)
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    periods = parse_csv_ints(args.release_periods)
    seeds = parse_csv_ints(args.seeds)

    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        for r in periods:
            variants = [(True, "with_residual")]
            # R=1 has no skipped update; without-residual is redundant unless you explicitly want it.
            if r != 1:
                variants.append((False, "without_residual"))
            for residual, residual_name in variants:
                policy_name = f"R{r}_{residual_name}"
                run_dir = output_root / f"seed{seed}" / safe_name(policy_name)
                run_dir.mkdir(parents=True, exist_ok=True)

                cmd = build_command(args, run_dir, r, seed, residual, policy_name)
                cmd_str = " ".join(shlex.quote(x) for x in cmd)
                (run_dir / "command.sh").write_text(cmd_str + "\n")
                print(f"\n[Run] seed={seed} R_low={r} residual={residual}")
                print(cmd_str)

                if args.dry_run:
                    continue

                marker = run_dir / "_SUCCESS"
                if args.skip_existing and marker.exists():
                    print(f"[Skip] {run_dir}")
                else:
                    with (run_dir / "stdout_stderr.log").open("w") as logf:
                        ret = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
                    if ret.returncode != 0:
                        print(f"[Error] command failed. See {run_dir / 'stdout_stderr.log'}", file=sys.stderr)
                        raise SystemExit(ret.returncode)
                    marker.write_text("ok\n")

                raw = load_metrics(run_dir)
                norm = normalize_metrics(raw)
                row: Dict[str, Any] = {
                    "seed": seed,
                    "release_period": r,
                    "residual": "on" if residual else "off",
                    "policy": policy_name,
                    "run_dir": str(run_dir),
                }
                row.update(norm)
                rows.append(row)

    if not args.dry_run:
        csv_path = output_root / "rq3_frequency_summary.csv"
        tex_path = output_root / "rq3_frequency_summary.tex"
        write_rows_csv(csv_path, rows)
        write_latex_table(tex_path, rows)
        print(f"\n[Done] summary CSV: {csv_path}")
        print(f"[Done] LaTeX table: {tex_path}")


if __name__ == "__main__":
    main()
