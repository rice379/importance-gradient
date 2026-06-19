#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQ5 ADTopk end-to-end experiment runner.

This script is designed to reuse the support modules you already have:
  - real_bucket_comm.py
  - bucket_runtime_planner.py
  - periodic_sync_gate.py
  - metrics_schema.py

Main difference from the original train_importancecheck_real_bucket_ddp.py:
  1) It inserts an ADTopk sparse-gradient selector before the component gate.
  2) It can build an ADTopk-based component-importance profile.
  3) It supports the three RQ5 configurations under the same ADTopk sparsifier:
       adtopk_sync: all ADTopk-retained gradients synchronized every iteration
       iad_adtopk:  important components every iteration; low-importance delayed
       ig_adtopk:   iad_adtopk + released-payload selection + balanced packing
  4) It logs step time and throughput in addition to communication metrics.

Important implementation note:
  The existing bucket communicator in your code accepts dense tensor views. We
  therefore materialize ADTopk-retained gradients as dense tensors with zeros in
  unselected positions. The CSV logs both materialized communication bytes from
  the communicator and an estimated logical sparse payload size based on nonzero
  retained entries. This keeps the code compatible with your current runtime
  while making the ADTopk baseline explicit and reproducible.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler, RandomSampler, SequentialSampler
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from importance_gradient.bucket_runtime_planner import RiskAwareBucketPlanner
from importance_gradient.real_bucket_comm import BalancedBucketCommunicator, SyncTensorRef, group_allreduce_sync_tensors
from importance_gradient.periodic_sync_gate import PeriodicSyncGate, load_importance_profile, parse_opt_component


# -----------------------------------------------------------------------------
# Distributed utilities
# -----------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if is_dist():
        dist.barrier()


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def setup_device(args) -> Tuple[int, torch.device]:
    """Support both torchrun and single-process execution."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(args.local_rank)))
    if local_rank < 0:
        local_rank = 0

    if world_size > 1 and not is_dist():
        dist.init_process_group(backend="nccl")

    if not torch.cuda.is_available():
        raise RuntimeError("This experiment script expects CUDA GPUs.")

    visible = torch.cuda.device_count()
    if local_rank >= visible:
        raise RuntimeError(f"LOCAL_RANK={local_rank} but only {visible} CUDA device(s) are visible.")

    torch.cuda.set_device(local_rank)
    return local_rank, torch.device("cuda", local_rank)


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------


class TokenizedDataset(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            "labels": item["labels"],
        }


def build_text(example: Dict[str, Any]) -> str:
    for key in ["abstract", "text", "article", "content", "instruction", "output"]:
        if key in example and example[key] is not None:
            if key == "instruction":
                inst = str(example.get("instruction", "")).strip()
                out = str(example.get("output", "")).strip()
                return f"{inst} {out}".strip()
            return str(example[key]).strip()
    parts = []
    for v in example.values():
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
    return " ".join(parts).strip()


def tokenize_function(example, tokenizer, max_length: int):
    text = build_text(example)
    encoded = tokenizer(text, truncation=True, padding="max_length", max_length=max_length)
    labels = encoded["input_ids"].copy()
    labels = [token if mask == 1 else -100 for token, mask in zip(labels, encoded["attention_mask"])]
    encoded["labels"] = labels
    return encoded


def build_dataloaders(args, tokenizer, for_profile: bool = False):
    if is_main_process():
        print("[Data] loading dataset ...")

    if args.local_dataset_path is not None and os.path.exists(args.local_dataset_path):
        dataset = load_from_disk(args.local_dataset_path)
    else:
        dataset = load_dataset(
            args.dataset_name,
            name=args.dataset_config_name if args.dataset_config_name else None,
            split=args.dataset_split,
        )

    split = dataset.train_test_split(test_size=0.1, seed=args.seed)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    max_train = args.profile_max_train_samples if for_profile else args.max_train_samples
    if max_train is not None:
        train_dataset = train_dataset.select(range(min(max_train, len(train_dataset))))
    if args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(min(args.max_eval_samples, len(eval_dataset))))

    if is_main_process():
        print(f"[Data] train subset = {len(train_dataset):,}")
        print(f"[Data] eval subset  = {len(eval_dataset):,}")
        print("[Data] tokenizing ...")

    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_length=args.max_length),
        remove_columns=train_dataset.column_names,
    )
    eval_dataset = eval_dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_length=args.max_length),
        remove_columns=eval_dataset.column_names,
    )

    columns = ["input_ids", "attention_mask", "labels"]
    train_dataset.set_format(type="torch", columns=columns)
    eval_dataset.set_format(type="torch", columns=columns)

    if is_dist():
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=True,
            seed=args.seed,
            drop_last=False,
        )
        eval_sampler = DistributedSampler(
            eval_dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=False,
            drop_last=False,
        )
    else:
        train_sampler = RandomSampler(train_dataset)
        eval_sampler = SequentialSampler(eval_dataset)

    train_loader = DataLoader(
        TokenizedDataset(train_dataset),
        batch_size=args.per_device_train_batch_size,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=True,
    )
    eval_loader = DataLoader(
        TokenizedDataset(eval_dataset),
        batch_size=args.per_device_eval_batch_size,
        sampler=eval_sampler,
        num_workers=0,
        pin_memory=True,
    )
    return train_loader, eval_loader, train_sampler, eval_sampler


# -----------------------------------------------------------------------------
# ADTopk sparse-gradient selection
# -----------------------------------------------------------------------------


@dataclass
class ADTopkStats:
    total_numel: int = 0
    selected_numel: int = 0
    zeroed_numel: int = 0
    selected_value_bytes: int = 0
    selected_index_bytes: int = 0
    selection_time_ms: float = 0.0

    @property
    def logical_sparse_bytes(self) -> int:
        return int(self.selected_value_bytes + self.selected_index_bytes)


def _named_trainable_parameters(model: torch.nn.Module):
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            yield name, p


def apply_adtopk_to_model_grads(
    model: torch.nn.Module,
    ratio: float,
    residual_buffers: Optional[Dict[str, torch.Tensor]] = None,
    use_error_feedback: bool = False,
    index_bytes: int = 4,
) -> ADTopkStats:
    """
    All-dimension Top-k over each gradient tensor.

    For each trainable parameter tensor, flatten all dimensions and keep the
    largest |g| entries. Unselected entries are zeroed in-place, so the rest of
    your existing synchronization runtime can still consume dense tensor views.
    """
    ratio = float(ratio)
    if ratio <= 0.0 or ratio > 1.0:
        raise ValueError(f"--adtopk_ratio must be in (0, 1], got {ratio}")

    cuda_sync()
    t0 = time.perf_counter()

    stats = ADTopkStats()
    if residual_buffers is None:
        residual_buffers = {}

    for name, p in _named_trainable_parameters(model):
        g = p.grad.data
        if not torch.is_floating_point(g) or g.numel() == 0:
            continue

        if use_error_feedback:
            res = residual_buffers.get(name)
            if res is None or res.shape != g.shape or res.dtype != g.dtype or res.device != g.device:
                res = torch.zeros_like(g, memory_format=torch.preserve_format)
                residual_buffers[name] = res
            work = g.add(res)
        else:
            work = g

        flat = work.view(-1)
        numel = int(flat.numel())
        k = max(1, int(math.ceil(numel * ratio)))
        k = min(k, numel)

        stats.total_numel += numel
        stats.selected_numel += k
        stats.zeroed_numel += numel - k
        stats.selected_value_bytes += k * int(flat.element_size())
        stats.selected_index_bytes += k * int(index_bytes)

        if k >= numel:
            if use_error_feedback:
                residual_buffers[name].zero_()
                g.copy_(work)
            continue

        _, idx = torch.topk(flat.abs(), k, sorted=False)
        sparse_flat = torch.zeros_like(flat)
        sparse_flat[idx] = flat.index_select(0, idx)

        if use_error_feedback:
            residual_buffers[name].copy_((flat - sparse_flat).view_as(g))

        g.copy_(sparse_flat.view_as(g))

    cuda_sync()
    stats.selection_time_ms = (time.perf_counter() - t0) * 1000.0
    return stats


def estimate_logical_sparse_comm_bytes(sync_tensors: List[SyncTensorRef], index_bytes: int = 4) -> Tuple[int, int]:
    """Return (nnz, bytes) for the tensors selected for communication."""
    nnz_total = 0
    bytes_total = 0
    for ref in sync_tensors:
        if ref.tensor is None or ref.tensor.numel() == 0:
            continue
        # This is used only for logging. It introduces a GPU sync, so keep it
        # outside the timed communication section.
        nnz = int(torch.count_nonzero(ref.tensor).item())
        nnz_total += nnz
        bytes_total += nnz * (int(ref.tensor.element_size()) + int(index_bytes))
    return nnz_total, bytes_total


# -----------------------------------------------------------------------------
# Profile construction
# -----------------------------------------------------------------------------


def knee_threshold_desc(scores_desc: List[float]) -> float:
    """Simple knee detector on a descending score curve."""
    if not scores_desc:
        return 0.0
    if len(scores_desc) <= 2:
        return float(scores_desc[-1])

    y = torch.tensor(scores_desc, dtype=torch.float64)
    x = torch.arange(len(scores_desc), dtype=torch.float64)
    x = (x - x.min()) / (x.max() - x.min()).clamp_min(1e-12)
    y = (y - y.min()) / (y.max() - y.min()).clamp_min(1e-12)

    # distance from the line joining first and last point
    x1, y1 = x[0], y[0]
    x2, y2 = x[-1], y[-1]
    num = torch.abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
    den = torch.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2).clamp_min(1e-12)
    idx = int(torch.argmax(num / den).item())
    return float(scores_desc[idx])


def collect_component_activity(model: torch.nn.Module) -> Dict[Tuple[int, str], float]:
    sum_sq: Dict[Tuple[int, str], float] = {}
    numel: Dict[Tuple[int, str], int] = {}

    for name, p in _named_trainable_parameters(model):
        cid = parse_opt_component(name)
        if cid is None:
            continue
        g = p.grad.data
        val = float(torch.sum(g.float() * g.float()).item())
        sum_sq[cid] = sum_sq.get(cid, 0.0) + val
        numel[cid] = numel.get(cid, 0) + int(g.numel())

    activity = {}
    for cid, ss in sum_sq.items():
        activity[cid] = math.sqrt(max(ss, 0.0)) / math.sqrt(max(numel[cid], 1))
    return activity


def write_profile_csv(
    path: str,
    traces: List[Dict[Tuple[int, str], float]],
    alpha: float,
    threshold_method: str,
) -> None:
    ensure_dir(os.path.dirname(path))
    components = sorted({cid for tr in traces for cid in tr.keys()}, key=lambda x: (x[0], x[1]))
    if not components:
        raise RuntimeError("No Transformer components were observed during profiling.")

    # Dense activity matrix: T x M, missing component activity treated as 0.
    mat = []
    for tr in traces:
        mat.append([float(tr.get(cid, 0.0)) for cid in components])

    T = len(mat)
    M = len(components)
    avg = [sum(mat[t][j] for t in range(T)) / max(T, 1) for j in range(M)]
    mn, mx = min(avg), max(avg)
    mag = [(x - mn) / max(mx - mn, 1e-12) for x in avg]

    persistence = []
    for j in range(M):
        hit = 0
        for t in range(T):
            row = sorted(mat[t])
            median = row[len(row) // 2]
            if mat[t][j] >= median:
                hit += 1
        persistence.append(hit / max(T, 1))

    scores = [float(alpha) * mag[j] + (1.0 - float(alpha)) * persistence[j] for j in range(M)]
    sorted_scores = sorted(scores, reverse=True)

    if threshold_method == "knee":
        tau = knee_threshold_desc(sorted_scores)
    elif threshold_method.startswith("fixed_ratio:"):
        ratio = float(threshold_method.split(":", 1)[1])
        k = max(1, int(math.ceil(M * ratio)))
        tau = sorted_scores[min(k - 1, M - 1)]
    else:
        raise ValueError(f"Unsupported threshold_method={threshold_method}")

    with open(path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "layer", "component", "magnitude", "persistence", "importance",
            "global_tau", "is_important", "profile_steps", "alpha", "threshold_method",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cid, m, b, s in sorted(zip(components, mag, persistence, scores), key=lambda x: x[3], reverse=True):
            writer.writerow({
                "layer": cid[0],
                "component": cid[1],
                "magnitude": m,
                "persistence": b,
                "importance": s,
                "global_tau": tau,
                "is_important": int(s >= tau),
                "profile_steps": T,
                "alpha": alpha,
                "threshold_method": threshold_method,
            })


def run_profile(args) -> None:
    ensure_dir(args.output_dir)
    set_seed(args.seed)
    local_rank, device = setup_device(args)

    if is_dist() and get_world_size() != 1:
        raise RuntimeError("Profile construction should be run with one process/GPU for this script.")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_loader, _, train_sampler, _ = build_dataloaders(args, tokenizer, for_profile=True)

    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype_map[args.dtype],
        low_cpu_mem_usage=True,
        local_files_only=args.local_files_only,
    )
    model.config.use_cache = False
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_iter = iter(train_loader)
    traces: List[Dict[Tuple[int, str], float]] = []
    adtopk_residuals: Dict[str, torch.Tensor] = {}

    if is_main_process():
        print(f"[Profile] output_dir={args.output_dir}")
        print(f"[Profile] warmup={args.profile_warmup_steps}, steps={args.profile_steps}")

    total_steps = int(args.profile_warmup_steps + args.profile_steps)
    for step in range(1, total_steps + 1):
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(step)
        optimizer.zero_grad(set_to_none=True)
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        outputs = model(**batch)
        outputs.loss.backward()
        if args.max_grad_norm is not None and args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        apply_adtopk_to_model_grads(
            model,
            ratio=args.adtopk_ratio,
            residual_buffers=adtopk_residuals,
            use_error_feedback=args.use_sparsifier_residual,
            index_bytes=args.sparse_index_bytes,
        )

        if step > args.profile_warmup_steps:
            traces.append(collect_component_activity(model))

        optimizer.step()

        if is_main_process() and (step % args.log_interval == 0 or step == total_steps):
            print(f"[Profile] step={step}/{total_steps} collected={len(traces)}")

    score_csv = os.path.join(args.output_dir, "adtopk_component_scores.csv")
    write_profile_csv(score_csv, traces, alpha=args.score_alpha, threshold_method=args.threshold_method)

    summary = {
        "score_csv": score_csv,
        "profile_steps": len(traces),
        "adtopk_ratio": args.adtopk_ratio,
        "threshold_method": args.threshold_method,
        "score_alpha": args.score_alpha,
    }
    with open(os.path.join(args.output_dir, "profile_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if is_main_process():
        print(f"[Profile] wrote {score_csv}")

    if is_dist():
        dist.destroy_process_group()


# -----------------------------------------------------------------------------
# Training / evaluation
# -----------------------------------------------------------------------------


# A stable CSV schema is important because train/eval rows have different fields.
# The previous version wrote rows with different field orders, which can make
# pandas parse later rows incorrectly. Keep a single field list for the whole run.
RQ5_METRIC_FIELDNAMES = [
    "step", "kind", "method", "seed",
    "train_loss", "val_loss", "val_ppl", "val_acc", "val_tokens",
    "elapsed_sec",
    "window_steps", "mean_step_time_ms", "tokens_per_sec",

    # ADTopk sparsifier, averaged over the log window for train rows.
    "adtopk_total_numel", "adtopk_selected_numel", "adtopk_zeroed_numel",
    "adtopk_logical_sparse_bytes", "adtopk_selection_time_ms",

    # Gate-side statistics, averaged over the log window for train rows.
    "synced_groups", "synced_params", "synced_bytes_est_dense",
    "residual_groups", "residual_params", "residual_bytes_dense",
    "low_importance_synced_groups",

    # Logical sparse communication estimate, averaged over the log window.
    "logical_sparse_comm_nnz", "logical_sparse_comm_bytes",

    # Communication backend statistics, averaged over the log window.
    "bucket_mode_requested", "bucket_mode_effective",
    "bucket_global_uncertainty", "bucket_imbalance_ratio", "bucket_imbalance_cv",
    "bucket_overflow_count",
    "bucket_pack_time_ms", "bucket_allreduce_time_ms", "bucket_unpack_time_ms",
    "bucket_total_comm_time_ms",
    "bucket_count", "bucket_block_count",
    "bucket_communicated_numel", "bucket_communicated_bytes",
]


def append_csv_dict(path: str, row: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    exists = os.path.exists(path)
    clean = {k: row.get(k, None) for k in RQ5_METRIC_FIELDNAMES}
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RQ5_METRIC_FIELDNAMES, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(clean)


class LogWindowMeter:
    """Accumulate per-step metrics and emit log-window averages.

    This fixes the release-step sampling artifact: when log_interval is a
    multiple of low_importance_period, logging only the last step records only
    release steps. This meter averages all steps in the logging window, so
    delay steps and release steps are both represented.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.loss = 0.0
        self.tokens = 0.0
        self.step_time_ms = 0.0

        self.adtopk_total_numel = 0.0
        self.adtopk_selected_numel = 0.0
        self.adtopk_zeroed_numel = 0.0
        self.adtopk_logical_sparse_bytes = 0.0
        self.adtopk_selection_time_ms = 0.0

        self.synced_groups = 0.0
        self.synced_params = 0.0
        self.synced_bytes_est_dense = 0.0
        self.residual_groups = 0.0
        self.residual_params = 0.0
        self.residual_bytes_dense = 0.0
        self.low_importance_synced_groups = 0.0

        self.logical_sparse_comm_nnz = 0.0
        self.logical_sparse_comm_bytes = 0.0

        self.bucket_global_uncertainty = 0.0
        self.bucket_imbalance_ratio = 0.0
        self.bucket_imbalance_cv = 0.0
        self.bucket_overflow_count = 0.0
        self.bucket_pack_time_ms = 0.0
        self.bucket_allreduce_time_ms = 0.0
        self.bucket_unpack_time_ms = 0.0
        self.bucket_total_comm_time_ms = 0.0
        self.bucket_count = 0.0
        self.bucket_block_count = 0.0
        self.bucket_communicated_numel = 0.0
        self.bucket_communicated_bytes = 0.0

        self.last_bucket_mode_requested = None
        self.last_bucket_mode_effective = None

    def update(
        self,
        *,
        train_loss: float,
        global_tokens: float,
        step_time_ms: float,
        adtopk_stats: ADTopkStats,
        gate_stats,
        logical_nnz: int,
        logical_sparse_bytes: int,
        comm_stats,
    ) -> None:
        self.steps += 1
        self.loss += float(train_loss)
        self.tokens += float(global_tokens)
        self.step_time_ms += float(step_time_ms)

        self.adtopk_total_numel += float(adtopk_stats.total_numel)
        self.adtopk_selected_numel += float(adtopk_stats.selected_numel)
        self.adtopk_zeroed_numel += float(adtopk_stats.zeroed_numel)
        self.adtopk_logical_sparse_bytes += float(adtopk_stats.logical_sparse_bytes)
        self.adtopk_selection_time_ms += float(adtopk_stats.selection_time_ms)

        self.synced_groups += float(gate_stats.synced_groups)
        self.synced_params += float(gate_stats.synced_params)
        self.synced_bytes_est_dense += float(gate_stats.synced_bytes_est)
        self.residual_groups += float(gate_stats.residual_groups)
        self.residual_params += float(gate_stats.residual_params)
        self.residual_bytes_dense += float(gate_stats.residual_bytes)
        self.low_importance_synced_groups += float(gate_stats.low_importance_synced_groups)

        self.logical_sparse_comm_nnz += float(logical_nnz)
        self.logical_sparse_comm_bytes += float(logical_sparse_bytes)

        self.bucket_global_uncertainty += float(comm_stats.global_uncertainty)
        self.bucket_imbalance_ratio += float(comm_stats.bucket_imbalance_ratio)
        self.bucket_imbalance_cv += float(comm_stats.bucket_imbalance_cv)
        self.bucket_overflow_count += float(comm_stats.bucket_overflow_count)
        self.bucket_pack_time_ms += float(comm_stats.pack_time_ms)
        self.bucket_allreduce_time_ms += float(comm_stats.allreduce_time_ms)
        self.bucket_unpack_time_ms += float(comm_stats.unpack_time_ms)
        self.bucket_total_comm_time_ms += float(comm_stats.total_comm_time_ms)
        self.bucket_count += float(comm_stats.bucket_count)
        self.bucket_block_count += float(comm_stats.block_count)
        self.bucket_communicated_numel += float(comm_stats.communicated_numel)
        self.bucket_communicated_bytes += float(comm_stats.communicated_bytes)

        self.last_bucket_mode_requested = comm_stats.mode_requested
        self.last_bucket_mode_effective = comm_stats.mode_effective

    def row(self, *, global_step: int, method: str, seed: int, elapsed: float) -> Dict[str, Any]:
        n = max(1, self.steps)
        mean_step_time_ms = self.step_time_ms / n
        tokens_per_sec = self.tokens / max(self.step_time_ms / 1000.0, 1e-12)

        def avg(x):
            return float(x) / n

        return {
            "step": global_step,
            "kind": "train",
            "method": method,
            "seed": seed,
            "train_loss": avg(self.loss),
            "elapsed_sec": elapsed,
            "window_steps": self.steps,
            "mean_step_time_ms": mean_step_time_ms,
            "tokens_per_sec": tokens_per_sec,

            "adtopk_total_numel": avg(self.adtopk_total_numel),
            "adtopk_selected_numel": avg(self.adtopk_selected_numel),
            "adtopk_zeroed_numel": avg(self.adtopk_zeroed_numel),
            "adtopk_logical_sparse_bytes": avg(self.adtopk_logical_sparse_bytes),
            "adtopk_selection_time_ms": avg(self.adtopk_selection_time_ms),

            "synced_groups": avg(self.synced_groups),
            "synced_params": avg(self.synced_params),
            "synced_bytes_est_dense": avg(self.synced_bytes_est_dense),
            "residual_groups": avg(self.residual_groups),
            "residual_params": avg(self.residual_params),
            "residual_bytes_dense": avg(self.residual_bytes_dense),
            "low_importance_synced_groups": avg(self.low_importance_synced_groups),

            "logical_sparse_comm_nnz": avg(self.logical_sparse_comm_nnz),
            "logical_sparse_comm_bytes": avg(self.logical_sparse_comm_bytes),

            "bucket_mode_requested": self.last_bucket_mode_requested,
            "bucket_mode_effective": self.last_bucket_mode_effective,
            "bucket_global_uncertainty": avg(self.bucket_global_uncertainty),
            "bucket_imbalance_ratio": avg(self.bucket_imbalance_ratio),
            "bucket_imbalance_cv": avg(self.bucket_imbalance_cv),
            "bucket_overflow_count": avg(self.bucket_overflow_count),
            "bucket_pack_time_ms": avg(self.bucket_pack_time_ms),
            "bucket_allreduce_time_ms": avg(self.bucket_allreduce_time_ms),
            "bucket_unpack_time_ms": avg(self.bucket_unpack_time_ms),
            "bucket_total_comm_time_ms": avg(self.bucket_total_comm_time_ms),
            "bucket_count": avg(self.bucket_count),
            "bucket_block_count": avg(self.bucket_block_count),
            "bucket_communicated_numel": avg(self.bucket_communicated_numel),
            "bucket_communicated_bytes": avg(self.bucket_communicated_bytes),
        }


@torch.no_grad()
def evaluate(model, eval_loader, device) -> Dict[str, float]:
    model.eval()
    total_nll = torch.zeros(1, device=device, dtype=torch.float64)
    total_tokens = torch.zeros(1, device=device, dtype=torch.float64)
    total_correct = torch.zeros(1, device=device, dtype=torch.float64)

    for batch in eval_loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        logits = outputs.logits
        labels = batch["labels"]

        shift_logits = logits[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        valid_mask = shift_labels.ne(-100)
        valid_tokens = valid_mask.sum().item()
        if valid_tokens > 0:
            pred = shift_logits.argmax(dim=-1)
            correct = ((pred == shift_labels) & valid_mask).sum().item()
            total_correct += correct
            total_tokens += valid_tokens
            total_nll += loss.item() * valid_tokens

    if is_dist():
        dist.all_reduce(total_nll, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_correct, op=dist.ReduceOp.SUM)

    avg_loss = (total_nll / total_tokens.clamp_min(1.0)).item()
    avg_ppl = math.exp(min(avg_loss, 20.0))
    avg_acc = (total_correct / total_tokens.clamp_min(1.0)).item()
    model.train()
    return {"val_loss": avg_loss, "val_ppl": avg_ppl, "val_acc": avg_acc, "val_tokens": int(total_tokens.item())}


def build_optimizer_and_scheduler(args, model):
    no_decay = ["bias", "layer_norm", "layernorm", "ln_f", "norm"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n.lower() for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n.lower() for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    def lr_lambda(current_step: int):
        if current_step < args.warmup_steps:
            return float(current_step) / max(1, args.warmup_steps)
        progress = float(current_step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def run_train(args) -> None:
    ensure_dir(args.output_dir)
    set_seed(args.seed)
    local_rank, device = setup_device(args)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if is_main_process():
        print(json.dumps(vars(args), indent=2, ensure_ascii=False))
        print(f"[OutputDir] {args.output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_loader, eval_loader, train_sampler, eval_sampler = build_dataloaders(args, tokenizer, for_profile=False)

    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype_map[args.dtype],
        low_cpu_mem_usage=True,
        local_files_only=args.local_files_only,
    )
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.to(device)
    model.train()

    optimizer, scheduler = build_optimizer_and_scheduler(args, model)

    if args.score_csv is None or not os.path.exists(args.score_csv):
        raise FileNotFoundError(f"--score_csv is required for train mode and was not found: {args.score_csv}")
    score_dict, global_tau = load_importance_profile(args.score_csv)

    if args.method == "adtopk_sync":
        sync_mode = "full"
        use_residual = False
        comm_backend = "group_allreduce"
        payload_mode = "dense"
        bucket_cost_mode = "dense"
        effective_keep_ratio = 1.0
    elif args.method == "iad_adtopk":
        sync_mode = "periodic"
        use_residual = True
        comm_backend = "group_allreduce"
        payload_mode = "dense"
        bucket_cost_mode = "dense"
        effective_keep_ratio = 1.0
    elif args.method == "ig_adtopk":
        sync_mode = "periodic"
        use_residual = True
        comm_backend = "balanced_bucket_real"
        payload_mode = "effective_payload"
        bucket_cost_mode = "effective_sparse"
        effective_keep_ratio = args.effective_payload_low_keep_ratio
    else:
        raise ValueError(f"Unknown --method={args.method}")

    gate = PeriodicSyncGate(
        model=model,
        score_dict=score_dict,
        global_tau=global_tau,
        low_importance_period=args.low_importance_period,
        use_residual_accumulation=use_residual,
        always_sync_non_component=True,
        sync_mode=sync_mode,
    )

    planner = RiskAwareBucketPlanner(
        bucket_num=args.bucket_num,
        mode_requested=args.bucket_scheduler_mode,
        use_adaptive_switch=args.bucket_use_adaptive_switch,
        uncertainty_threshold=args.bucket_adapt_uncertainty_threshold,
        lambda_std=args.bucket_lambda_std,
        gamma_overflow=args.bucket_gamma_overflow,
        target_margin=args.bucket_target_margin,
    )
    bucket_communicator = BalancedBucketCommunicator(
        planner=planner,
        block_size_numel=args.bucket_block_size_numel,
        device=device,
        bucket_cost_mode=bucket_cost_mode,
        effective_low_cost_ratio=args.effective_low_cost_ratio,
        payload_mode=payload_mode,
        effective_payload_low_keep_ratio=effective_keep_ratio,
        effective_payload_rotation_interval=args.effective_payload_rotation_interval,
    )

    args_path = os.path.join(args.output_dir, "run_args.json")
    if is_main_process():
        with open(args_path, "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, ensure_ascii=False)

    metrics_path = os.path.join(args.output_dir, "metrics.csv")
    best_summary_path = os.path.join(args.output_dir, "best_summary.json")
    train_iter = iter(train_loader)
    adtopk_residuals: Dict[str, torch.Tensor] = {}

    global_step = 0
    start_time = time.time()
    window = LogWindowMeter()

    best = {"best_step": -1, "best_val_loss": float("inf"), "best_val_ppl": None, "best_val_acc": None}

    while global_step < args.max_steps:
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(global_step)

        cuda_sync()
        step_t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        micro_loss = 0.0
        local_tokens = 0

        for _ in range(args.gradient_accumulation_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            local_tokens += int(batch["attention_mask"].sum().item())
            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()
            micro_loss += float(loss.item())

        if args.max_grad_norm is not None and args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        adtopk_stats = apply_adtopk_to_model_grads(
            model,
            ratio=args.adtopk_ratio,
            residual_buffers=adtopk_residuals,
            use_error_feedback=args.use_sparsifier_residual,
            index_bytes=args.sparse_index_bytes,
        )

        sync_tensors, gate_stats = gate.prepare_sync_tensors(global_step + 1)
        logical_nnz, logical_sparse_bytes = estimate_logical_sparse_comm_bytes(sync_tensors, index_bytes=args.sparse_index_bytes)

        if comm_backend == "group_allreduce":
            comm_stats = group_allreduce_sync_tensors(sync_tensors, global_step + 1)
        elif comm_backend == "balanced_bucket_real":
            comm_stats = bucket_communicator.communicate(sync_tensors, global_step + 1)
        else:
            raise ValueError(comm_backend)

        gate.finalize_synced_tensors(sync_tensors)
        optimizer.step()
        scheduler.step()
        global_step += 1

        # Aggregate token count across ranks for throughput.
        token_tensor = torch.tensor([local_tokens], dtype=torch.float64, device=device)
        if is_dist():
            dist.all_reduce(token_tensor, op=dist.ReduceOp.SUM)
        global_tokens = float(token_tensor.item())

        cuda_sync()
        step_time_ms = (time.perf_counter() - step_t0) * 1000.0
        window.update(
            train_loss=micro_loss,
            global_tokens=global_tokens,
            step_time_ms=step_time_ms,
            adtopk_stats=adtopk_stats,
            gate_stats=gate_stats,
            logical_nnz=logical_nnz,
            logical_sparse_bytes=logical_sparse_bytes,
            comm_stats=comm_stats,
        )

        if global_step % args.log_interval == 0 and is_main_process():
            elapsed = time.time() - start_time
            row = window.row(global_step=global_step, method=args.method, seed=args.seed, elapsed=elapsed)
            append_csv_dict(metrics_path, row)
            print(
                f"[train] step={global_step} method={args.method} loss={row['train_loss']:.6f} "
                f"window_steps={row['window_steps']} "
                f"step_ms={row['mean_step_time_ms']:.2f} tok/s={row['tokens_per_sec']:.1f} "
                f"comm_ms(avg)={row['bucket_total_comm_time_ms']:.3f} "
                f"bytes(avg)={row['bucket_communicated_bytes']/1e6:.3f}MB "
                f"logical_sparse(avg)={row['logical_sparse_comm_bytes']/1e6:.3f}MB"
            )
            window.reset()

        if global_step % args.eval_interval == 0 or global_step == args.max_steps:
            barrier()
            if hasattr(eval_sampler, "set_epoch"):
                eval_sampler.set_epoch(global_step)
            eval_metrics = evaluate(model, eval_loader, device)
            barrier()
            if is_main_process():
                elapsed = time.time() - start_time
                row = {
                    "step": global_step,
                    "kind": "eval",
                    "method": args.method,
                    "seed": args.seed,
                    "val_loss": eval_metrics["val_loss"],
                    "val_ppl": eval_metrics["val_ppl"],
                    "val_acc": eval_metrics["val_acc"],
                    "val_tokens": eval_metrics["val_tokens"],
                    "elapsed_sec": elapsed,
                }
                append_csv_dict(metrics_path, row)
                print(
                    f"[eval] step={global_step} method={args.method} "
                    f"val_loss={eval_metrics['val_loss']:.6f} val_acc={eval_metrics['val_acc']:.4f}"
                )
                if eval_metrics["val_loss"] < best["best_val_loss"]:
                    best = {
                        "best_step": global_step,
                        "best_val_loss": eval_metrics["val_loss"],
                        "best_val_ppl": eval_metrics["val_ppl"],
                        "best_val_acc": eval_metrics["val_acc"],
                    }
                    with open(best_summary_path, "w", encoding="utf-8") as f:
                        json.dump(best, f, indent=2, ensure_ascii=False)

    # Flush a partial log window if max_steps is not divisible by log_interval.
    if window.steps > 0 and is_main_process():
        elapsed = time.time() - start_time
        row = window.row(global_step=global_step, method=args.method, seed=args.seed, elapsed=elapsed)
        append_csv_dict(metrics_path, row)
        print(
            f"[train-final-window] step={global_step} method={args.method} "
            f"window_steps={row['window_steps']} step_ms={row['mean_step_time_ms']:.2f} "
            f"comm_ms(avg)={row['bucket_total_comm_time_ms']:.3f}"
        )
        window.reset()

    barrier()
    if is_main_process():
        print("\nBest summary:")
        print(json.dumps(best, indent=2, ensure_ascii=False))

    if is_dist():
        dist.destroy_process_group()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_mode", type=str, choices=["profile", "train"], required=True)
    parser.add_argument("--local_rank", type=int, default=-1)

    # model / data
    parser.add_argument("--model_name_or_path", type=str, default="facebook/opt-1.3b")
    parser.add_argument("--dataset_name", type=str, default="uiyunkim-hub/pubmed-abstract")
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--local_dataset_path", type=str, default=None)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--max_train_samples", type=int, default=100000)
    parser.add_argument("--profile_max_train_samples", type=int, default=20000)
    parser.add_argument("--max_eval_samples", type=int, default=2000)

    # training
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # ADTopk sparsifier
    parser.add_argument("--adtopk_ratio", type=float, default=0.01)
    parser.add_argument("--sparse_index_bytes", type=int, default=4)
    parser.add_argument("--use_sparsifier_residual", action="store_true")

    # profiling
    parser.add_argument("--profile_warmup_steps", type=int, default=200)
    parser.add_argument("--profile_steps", type=int, default=400)
    parser.add_argument("--score_alpha", type=float, default=0.7)
    parser.add_argument("--threshold_method", type=str, default="knee")

    # RQ5 method / gate
    parser.add_argument("--method", type=str, default="adtopk_sync", choices=["adtopk_sync", "iad_adtopk", "ig_adtopk"])
    parser.add_argument("--score_csv", type=str, default=None)
    parser.add_argument("--low_importance_period", type=int, default=4)

    # bucket communication
    parser.add_argument("--bucket_num", type=int, default=4)
    parser.add_argument("--bucket_block_size_numel", type=int, default=262144)
    parser.add_argument("--bucket_scheduler_mode", type=str, default="risk_aware", choices=["round_robin", "lightest", "risk_aware"])
    parser.add_argument("--bucket_use_adaptive_switch", action="store_true")
    parser.add_argument("--bucket_adapt_uncertainty_threshold", type=float, default=0.15)
    parser.add_argument("--bucket_lambda_std", type=float, default=0.15)
    parser.add_argument("--bucket_gamma_overflow", type=float, default=1.0)
    parser.add_argument("--bucket_target_margin", type=float, default=1.05)
    parser.add_argument("--effective_low_cost_ratio", type=float, default=0.25)
    parser.add_argument("--effective_payload_low_keep_ratio", type=float, default=0.25)
    parser.add_argument("--effective_payload_rotation_interval", type=int, default=4)

    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.run_mode == "profile":
        run_profile(args)
    elif args.run_mode == "train":
        run_train(args)
    else:
        raise ValueError(args.run_mode)


if __name__ == "__main__":
    main()
