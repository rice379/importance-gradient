#!/usr/bin/env python3
"""
RQ1 workload-stability profiler for ImportanceGradient on OPT-6.7B.

This script runs a real post-training loop, profiles retained sparse-gradient
activity after a baseline Top-k sparsifier, and exports component importance
profiles. It is intended for RQ1:

  Does workload variation change component importance?

Typical use:
  torchrun --standalone --nproc_per_node=2 rq1_profile_workload_importance_pythia.py \
    --model_name_or_path facebook/opt-6.7b \
    --workload_name pubmed \
    --dataset_name uiyunkim-hub/pubmed-abstract \
    --dataset_split train \
    --max_steps 1600 \
    --profile_windows 400:1600 \
    --output_dir runs/rq1_workload/profile_pubmed

Memory note: OPT-6.7B is much larger than OPT-1.3B/Pythia-1.4B. The one-step script enables gradient checkpointing and uses max_length=128 by default.

Outputs:
  component_activity_trace.csv
  profiles/profile_<experiment>_w<start>_<end>.csv
  profile_paths.json
  run_args.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer


# ----------------------------- distributed utils -----------------------------

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
        if torch.cuda.is_available():
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()


def setup_distributed(args) -> torch.device:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(args.local_rank)))
    if local_rank < 0:
        local_rank = 0
    if world_size > 1 and not is_dist():
        dist.init_process_group(backend=args.dist_backend)
    if torch.cuda.is_available():
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(f"LOCAL_RANK={local_rank}, visible cuda devices={torch.cuda.device_count()}")
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def cleanup_distributed() -> None:
    if is_dist():
        dist.destroy_process_group()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_name(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return name.strip("_") or "unknown"


# ---------------------------- component parser -------------------------------

_COMPONENT_ORDER = {"Q": 0, "K": 1, "V": 2, "QKV": 3, "O": 4, "FC1": 5, "FC2": 6, "GATE": 7, "UP": 8, "DOWN": 9}


def parse_transformer_component(param_name: str) -> Optional[Tuple[int, str]]:
    """Map common HuggingFace LLM parameter names to Transformer components.

    The paper mainly uses Q/K/V/O and FC1/FC2 components. OPT-style models map
    directly to these names. GPT-NeoX/Pythia often uses a fused QKV projection,
    so the script records it as QKV. LLaMA/Qwen/Mistral-style models use
    gate/up/down MLP projections; gate/up are both treated as FC1-like expansion
    projections and down_proj as FC2.
    """
    parts = param_name.split(".")

    layer_id: Optional[int] = None
    for token in ("layers", "h", "block"):
        if token in parts:
            idx = parts.index(token)
            if idx + 1 < len(parts):
                try:
                    layer_id = int(parts[idx + 1])
                    break
                except Exception:
                    pass
    if layer_id is None:
        # BLOOM-style: transformer.h.<layer>.*
        if "transformer" in parts and "h" in parts:
            idx = parts.index("h")
            if idx + 1 < len(parts):
                try:
                    layer_id = int(parts[idx + 1])
                except Exception:
                    layer_id = None
    if layer_id is None:
        return None

    joined = ".".join(parts)

    # OPT / BART-style attention projections.
    if "q_proj" in parts:
        return layer_id, "Q"
    if "k_proj" in parts:
        return layer_id, "K"
    if "v_proj" in parts:
        return layer_id, "V"
    if "out_proj" in parts or "o_proj" in parts or "dense" in parts and "attention" in parts:
        return layer_id, "O"

    # GPT-NeoX/Pythia and BLOOM fused attention projection.
    if "query_key_value" in parts or "query_key_value" in joined or "qkv_proj" in parts:
        return layer_id, "QKV"

    # OPT feed-forward.
    if "fc1" in parts or "dense_h_to_4h" in parts:
        return layer_id, "FC1"
    if "fc2" in parts or "dense_4h_to_h" in parts:
        return layer_id, "FC2"

    # LLaMA/Qwen/Mistral feed-forward. Keep paper-level two-way mapping.
    if "gate_proj" in parts or "up_proj" in parts:
        return layer_id, "FC1"
    if "down_proj" in parts:
        return layer_id, "FC2"

    return None


def component_sort_key(row) -> Tuple[int, int, str]:
    comp = str(row["component"])
    return int(row["layer"]), _COMPONENT_ORDER.get(comp, 99), comp


# ------------------------------- dataset utils -------------------------------

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


def build_text(example: dict, text_fields: Optional[str] = None) -> str:
    if text_fields:
        fields = [f.strip() for f in text_fields.split(",") if f.strip()]
        vals = []
        for f in fields:
            v = example.get(f, None)
            if v is not None:
                vals.append(str(v).strip())
        return " ".join([v for v in vals if v]).strip()

    # Common datasets used in the paper and in HF examples.
    if "instruction" in example:
        inst = str(example.get("instruction", "")).strip()
        inp = str(example.get("input", "")).strip()
        out = str(example.get("output", "")).strip()
        return " ".join([x for x in [inst, inp, out] if x]).strip()
    for key in ["abstract", "text", "article", "content", "sentence", "document"]:
        if key in example and example[key] is not None:
            return str(example[key]).strip()
    parts = []
    for _, value in example.items():
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    return " ".join(parts).strip()


def tokenize_function(example, tokenizer, max_length: int, text_fields: Optional[str]):
    text = build_text(example, text_fields=text_fields)
    encoded = tokenizer(text, truncation=True, padding="max_length", max_length=max_length)
    labels = encoded["input_ids"].copy()
    labels = [tok if mask == 1 else -100 for tok, mask in zip(labels, encoded["attention_mask"])]
    encoded["labels"] = labels
    return encoded


def build_dataloaders(args, tokenizer):
    if is_main_process():
        print(f"[Data] loading workload={args.workload_name} dataset={args.dataset_name}", flush=True)
    if args.local_dataset_path and os.path.exists(args.local_dataset_path):
        dataset = load_from_disk(args.local_dataset_path)
        if isinstance(dataset, dict) and args.dataset_split in dataset:
            dataset = dataset[args.dataset_split]
    else:
        dataset = load_dataset(
            args.dataset_name,
            name=args.dataset_config_name if args.dataset_config_name else None,
            split=args.dataset_split,
        )

    split = dataset.train_test_split(test_size=args.eval_ratio, seed=args.seed)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    if args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(min(args.max_train_samples, len(train_dataset))))
    if args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(min(args.max_eval_samples, len(eval_dataset))))

    if is_main_process():
        print(f"[Data] train={len(train_dataset):,}, eval={len(eval_dataset):,}", flush=True)
        print("[Data] tokenizing ...", flush=True)

    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, args.max_length, args.text_fields),
        remove_columns=train_dataset.column_names,
    )
    eval_dataset = eval_dataset.map(
        lambda x: tokenize_function(x, tokenizer, args.max_length, args.text_fields),
        remove_columns=eval_dataset.column_names,
    )

    columns = ["input_ids", "attention_mask", "labels"]
    train_dataset.set_format(type="torch", columns=columns)
    eval_dataset.set_format(type="torch", columns=columns)

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=get_world_size(),
        rank=get_rank(),
        shuffle=True,
        seed=args.seed,
        drop_last=False,
    ) if get_world_size() > 1 else None
    eval_sampler = DistributedSampler(
        eval_dataset,
        num_replicas=get_world_size(),
        rank=get_rank(),
        shuffle=False,
        seed=args.seed,
        drop_last=False,
    ) if get_world_size() > 1 else None

    train_loader = DataLoader(
        TokenizedDataset(train_dataset),
        batch_size=args.per_device_train_batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    eval_loader = DataLoader(
        TokenizedDataset(eval_dataset),
        batch_size=args.per_device_eval_batch_size,
        sampler=eval_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, eval_loader, train_sampler, eval_sampler


# ------------------------ profiling and importance logic ----------------------

def parse_profile_windows(raw: str) -> List[Tuple[int, int]]:
    windows: List[Tuple[int, int]] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid window '{part}'. Use start:end, e.g. 400:1600")
        a, b = part.split(":", 1)
        start, end = int(a), int(b)
        if end <= start:
            raise ValueError(f"Invalid window {part}: end must be larger than start")
        windows.append((start, end))
    if not windows:
        raise ValueError("No valid profile windows provided")
    return windows


def topk_squared_norm_and_nnz(grad: torch.Tensor, topk_ratio: float, min_k: int = 1) -> Tuple[float, int]:
    if grad is None or grad.numel() == 0:
        return 0.0, 0
    flat = grad.detach().view(-1)
    numel = flat.numel()
    k = int(math.ceil(numel * float(topk_ratio)))
    k = max(min_k, min(k, numel))
    if k >= numel:
        vals = flat.float()
        return float(torch.sum(vals * vals).item()), int(numel)
    vals = torch.topk(flat.abs(), k=k, largest=True, sorted=False).values.float()
    return float(torch.sum(vals * vals).item()), int(k)


def topk_squared_norms_for_fused_qkv(
    grad: torch.Tensor,
    topk_ratio: float,
    min_k: int = 1,
) -> Dict[str, Tuple[float, int, int]]:
    """Apply Top-k over a fused QKV tensor and attribute retained entries to Q/K/V.

    GPT-NeoX/Pythia stores Q, K, and V in one query_key_value parameter. This
    helper keeps the Top-k decision at the fused tensor level, then maps each
    retained entry back to Q, K, or V according to its row segment. This avoids
    treating the whole fused tensor as a single coarse QKV component while still
    respecting the baseline tensor-level Top-k selection.
    """
    if grad is None or grad.numel() == 0:
        return {"Q": (0.0, 0, 0), "K": (0.0, 0, 0), "V": (0.0, 0, 0)}

    if grad.dim() == 0 or grad.shape[0] % 3 != 0:
        sq, nnz = topk_squared_norm_and_nnz(grad, topk_ratio=topk_ratio, min_k=min_k)
        return {"QKV": (sq, nnz, int(grad.numel()))}

    out_dim = int(grad.shape[0])
    part = out_dim // 3
    row_stride = int(grad.numel() // out_dim)
    flat_abs = grad.detach().abs().reshape(-1)
    numel = int(flat_abs.numel())
    k = int(math.ceil(numel * float(topk_ratio)))
    k = max(min_k, min(k, numel))

    if k >= numel:
        idx = torch.arange(numel, device=flat_abs.device)
        vals = flat_abs.float()
    else:
        vals, idx = torch.topk(flat_abs, k=k, largest=True, sorted=False)
        vals = vals.float()

    rows = torch.div(idx, row_stride, rounding_mode="floor")
    out: Dict[str, Tuple[float, int, int]] = {}
    for comp, lo, hi in (("Q", 0, part), ("K", part, 2 * part), ("V", 2 * part, 3 * part)):
        mask = (rows >= lo) & (rows < hi)
        if bool(mask.any().item()):
            comp_vals = vals[mask]
            sq = float(torch.sum(comp_vals * comp_vals).item())
            nnz = int(mask.sum().item())
        else:
            sq = 0.0
            nnz = 0
        out[comp] = (sq, nnz, int(part * row_stride))
    return out


def collect_component_activity(model: torch.nn.Module, topk_ratio: float, split_fused_qkv: bool = True) -> List[dict]:
    sq_by_comp: Dict[Tuple[int, str], float] = defaultdict(float)
    retained_by_comp: Dict[Tuple[int, str], int] = defaultdict(int)
    numel_by_comp: Dict[Tuple[int, str], int] = defaultdict(int)

    for name, param in model.named_parameters():
        if not param.requires_grad or param.grad is None:
            continue
        cid = parse_transformer_component(name)
        if cid is None:
            continue
        layer, comp = cid
        if comp == "QKV" and split_fused_qkv:
            parts = topk_squared_norms_for_fused_qkv(param.grad.data, topk_ratio=topk_ratio)
            for sub_comp, (sq, retained, numel) in parts.items():
                sub_cid = (layer, sub_comp)
                sq_by_comp[sub_cid] += sq
                retained_by_comp[sub_cid] += retained
                numel_by_comp[sub_cid] += numel
        else:
            sq, retained = topk_squared_norm_and_nnz(param.grad.data, topk_ratio=topk_ratio)
            sq_by_comp[cid] += sq
            retained_by_comp[cid] += retained
            numel_by_comp[cid] += param.numel()

    rows: List[dict] = []
    for (layer, comp), sq_sum in sorted(sq_by_comp.items(), key=lambda x: (x[0][0], _COMPONENT_ORDER.get(x[0][1], 99), x[0][1])):
        raw_activity = math.sqrt(max(sq_sum, 0.0))
        n_m = max(numel_by_comp[(layer, comp)], 1)
        d_activity = raw_activity / math.sqrt(n_m)
        rows.append({
            "layer": int(layer),
            "component": str(comp),
            "raw_activity": float(raw_activity),
            "d_activity": float(d_activity),
            "numel": int(n_m),
            "retained_numel": int(retained_by_comp[(layer, comp)]),
        })
    return rows

def allreduce_gradients(model: torch.nn.Module) -> None:
    if not is_dist() or get_world_size() == 1:
        return
    world = get_world_size()
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
            p.grad.data.div_(world)


def minmax_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    lo, hi = float(np.min(values)), float(np.max(values))
    if hi - lo < 1e-12:
        return np.ones_like(values)
    return (values - lo) / (hi - lo)


def geometric_knee_threshold(scores: np.ndarray) -> Tuple[int, float]:
    z = np.asarray(scores, dtype=np.float64)
    if len(z) == 0:
        raise ValueError("Empty score list")
    if len(z) <= 2:
        return 0, float(z[0])
    x = np.linspace(0.0, 1.0, len(z))
    y = minmax_normalize(z)
    points = np.stack([x, y], axis=1)
    a = points[0]
    b = points[-1]
    ab = b - a
    denom = np.linalg.norm(ab)
    if denom < 1e-12:
        idx = int(np.argmax(z))
        return idx, float(z[idx])
    ap = points - a
    distances = np.abs(ab[0] * ap[:, 1] - ab[1] * ap[:, 0]) / denom
    idx = int(np.argmax(distances))
    return idx, float(z[idx])


def build_profile_from_trace(
    trace_csv: str,
    output_csv: str,
    window: Tuple[int, int],
    alpha: float,
    seed: int,
    profile_id: str,
    model_name: str,
    workload_name: str,
    dataset_name: str,
) -> pd.DataFrame:
    df = pd.read_csv(trace_csv)
    start, end = window
    sub = df[(df["step"] >= start) & (df["step"] < end)].copy()
    if sub.empty:
        raise ValueError(f"No trace rows in window {start}:{end} from {trace_csv}")

    # Fill missing component-step pairs with zero activity so persistence is
    # measured against the full profiling window, not only observed rows.
    all_steps = sorted(sub["step"].unique().tolist())
    components = sub[["layer", "component"]].drop_duplicates().copy()
    grid = pd.MultiIndex.from_product(
        [all_steps, range(len(components))], names=["step", "component_idx"]
    ).to_frame(index=False)
    comp_lookup = components.reset_index(drop=True).reset_index().rename(columns={"index": "component_idx"})
    grid = grid.merge(comp_lookup, on="component_idx", how="left").drop(columns=["component_idx"])
    sub = grid.merge(sub, on=["step", "layer", "component"], how="left")
    sub["d_activity"] = sub["d_activity"].fillna(0.0)
    sub["raw_activity"] = sub["raw_activity"].fillna(0.0)
    sub["numel"] = sub["numel"].fillna(0).astype(int)
    sub["retained_numel"] = sub["retained_numel"].fillna(0).astype(int)

    # Per-step median over components. Persistence is the fraction of profiled
    # steps where a component is above that step's median retained activity.
    step_median = sub.groupby("step")["d_activity"].median().rename("step_median")
    sub = sub.merge(step_median, on="step", how="left")
    sub["above_median"] = sub["d_activity"] > sub["step_median"]

    rows: List[dict] = []
    for (layer, comp), g in sub.groupby(["layer", "component"]):
        d = g["d_activity"].astype(float).to_numpy()
        mean_activity = float(np.mean(d))
        energy = float(np.mean(d ** 2))
        persistence_raw = float(g["above_median"].astype(float).mean())
        rows.append({
            "layer": int(layer),
            "component": str(comp),
            "mean_d_activity": mean_activity,
            "energy": energy,
            "persistence_raw": persistence_raw,
            "trace_steps": int(len(g)),
        })

    prof = pd.DataFrame(rows)
    prof["magnitude"] = minmax_normalize(prof["mean_d_activity"].to_numpy())
    prof["persistence"] = prof["persistence_raw"].astype(float)
    prof["final_score"] = float(alpha) * prof["magnitude"] + (1.0 - float(alpha)) * prof["persistence"]

    prof = prof.sort_values("final_score", ascending=False).reset_index(drop=True)
    sorted_scores = prof["final_score"].to_numpy()
    knee_idx, tau = geometric_knee_threshold(sorted_scores)
    prof["global_tau"] = float(tau)
    prof["is_important"] = prof["final_score"] >= tau
    # Legacy-compatible column for existing analysis scripts. Do not use this
    # term in paper text.
    prof["is_high_priority"] = prof["is_important"]
    prof["rank"] = np.arange(1, len(prof) + 1)
    prof["knee_rank"] = int(knee_idx + 1)
    prof["window_start"] = int(start)
    prof["window_end"] = int(end)
    prof["seed"] = int(seed)
    prof["profile_id"] = str(profile_id)
    prof["model_name"] = str(model_name)
    prof["workload_name"] = str(workload_name)
    prof["dataset_name"] = str(dataset_name)
    prof["key"] = prof["layer"].astype(int).astype(str) + ":" + prof["component"].astype(str)

    prof.to_csv(output_csv, index=False)
    return prof


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
        valid_tokens = int(valid_mask.sum().item())
        if valid_tokens > 0:
            pred = shift_logits.argmax(dim=-1)
            correct = int(((pred == shift_labels) & valid_mask).sum().item())
            total_correct += correct
            total_tokens += valid_tokens
            total_nll += float(loss.item()) * valid_tokens

    if is_dist():
        dist.all_reduce(total_nll, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_correct, op=dist.ReduceOp.SUM)

    avg_loss = (total_nll / total_tokens.clamp_min(1.0)).item()
    avg_acc = (total_correct / total_tokens.clamp_min(1.0)).item()
    model.train()
    return {"val_loss": avg_loss, "val_acc": avg_acc, "val_ppl": math.exp(min(avg_loss, 20.0))}


# ------------------------------------ main -----------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--local_rank", type=int, default=-1)
    p.add_argument("--dist_backend", type=str, default="nccl")

    # model / data
    p.add_argument("--model_name_or_path", type=str, default="facebook/opt-6.7b")
    p.add_argument("--model_label", type=str, default=None)
    p.add_argument("--workload_name", type=str, required=True)
    p.add_argument("--dataset_name", type=str, required=True)
    p.add_argument("--dataset_config_name", type=str, default=None)
    p.add_argument("--dataset_split", type=str, default="train")
    p.add_argument("--local_dataset_path", type=str, default=None)
    p.add_argument("--text_fields", type=str, default=None, help="Comma-separated text fields to concatenate")
    p.add_argument("--local_files_only", dest="local_files_only", action="store_true", default=True)
    p.add_argument("--no_local_files_only", dest="local_files_only", action="store_false")
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--max_train_samples", type=int, default=100000)
    p.add_argument("--max_eval_samples", type=int, default=2000)
    p.add_argument("--eval_ratio", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=0)

    # training
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=1600)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--gradient_checkpointing", action="store_true")

    # profiling
    p.add_argument("--topk_ratio", type=float, default=0.01)
    p.add_argument("--split_fused_qkv", dest="split_fused_qkv", action="store_true", default=True,
                   help="Split GPT-NeoX/Pythia fused query_key_value tensors into Q/K/V components after Top-k selection")
    p.add_argument("--no_split_fused_qkv", dest="split_fused_qkv", action="store_false")
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--profile_windows", type=str, default="400:1600")
    p.add_argument("--trace_every", type=int, default=1)
    p.add_argument("--log_interval", type=int, default=20)
    p.add_argument("--eval_interval", type=int, default=0, help="0 disables eval during profiling")
    p.add_argument("--experiment_id", type=str, default=None)

    # output
    p.add_argument("--output_dir", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    ensure_dir(os.path.join(args.output_dir, "profiles"))
    set_seed(args.seed)
    windows = parse_profile_windows(args.profile_windows)

    model_label = args.model_label or args.model_name_or_path
    exp_id = args.experiment_id or f"{safe_name(model_label)}_{safe_name(args.workload_name)}_seed{args.seed}"

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = setup_distributed(args)

    if is_main_process():
        with open(os.path.join(args.output_dir, "run_args.json"), "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, ensure_ascii=False)
        print(json.dumps(vars(args), indent=2, ensure_ascii=False), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        local_files_only=args.local_files_only,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_loader, eval_loader, train_sampler, eval_sampler = build_dataloaders(args, tokenizer)

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
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-8)

    def lr_lambda(current_step: int):
        if current_step < args.warmup_steps:
            return float(current_step) / max(1, args.warmup_steps)
        progress = float(current_step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    trace_path = os.path.join(args.output_dir, "component_activity_trace.csv")
    train_metrics_path = os.path.join(args.output_dir, "train_eval_metrics.csv")
    if is_main_process():
        with open(trace_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "step", "seed", "model_name", "workload_name", "dataset_name",
                "layer", "component", "raw_activity", "d_activity", "numel", "retained_numel",
            ])
            writer.writeheader()
        with open(train_metrics_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["step", "kind", "loss", "val_loss", "val_acc", "val_ppl", "elapsed_sec"])
            writer.writeheader()

    global_step = 0
    train_iter = iter(train_loader)
    start_time = time.time()

    while global_step < args.max_steps:
        if train_sampler is not None:
            train_sampler.set_epoch(global_step)
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0

        for _ in range(args.gradient_accumulation_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()
            loss_sum += float(loss.item())

        if args.max_grad_norm and args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        # Match data-parallel averaged gradients before profiling retained sparse activity.
        allreduce_gradients(model)

        next_step = global_step + 1
        if next_step % args.trace_every == 0:
            rows = collect_component_activity(model, topk_ratio=args.topk_ratio, split_fused_qkv=args.split_fused_qkv)
            if is_main_process():
                with open(trace_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=[
                        "step", "seed", "model_name", "workload_name", "dataset_name",
                        "layer", "component", "raw_activity", "d_activity", "numel", "retained_numel",
                    ])
                    for row in rows:
                        writer.writerow({
                            "step": next_step,
                            "seed": args.seed,
                            "model_name": model_label,
                            "workload_name": args.workload_name,
                            "dataset_name": args.dataset_name,
                            **row,
                        })

        optimizer.step()
        scheduler.step()
        global_step = next_step

        if is_main_process() and (global_step % args.log_interval == 0 or global_step == 1):
            elapsed = time.time() - start_time
            print(f"[profile-train] workload={args.workload_name} step={global_step} loss={loss_sum:.6f} elapsed={elapsed:.1f}s", flush=True)
            with open(train_metrics_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["step", "kind", "loss", "val_loss", "val_acc", "val_ppl", "elapsed_sec"])
                writer.writerow({"step": global_step, "kind": "train", "loss": loss_sum, "elapsed_sec": elapsed})

        if args.eval_interval and args.eval_interval > 0 and (global_step % args.eval_interval == 0):
            barrier()
            if eval_sampler is not None:
                eval_sampler.set_epoch(global_step)
            metrics = evaluate(model, eval_loader, device)
            barrier()
            if is_main_process():
                elapsed = time.time() - start_time
                print(f"[eval] workload={args.workload_name} step={global_step} val_loss={metrics['val_loss']:.6f} val_acc={metrics['val_acc']:.6f}", flush=True)
                with open(train_metrics_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=["step", "kind", "loss", "val_loss", "val_acc", "val_ppl", "elapsed_sec"])
                    writer.writerow({"step": global_step, "kind": "eval", "val_loss": metrics["val_loss"], "val_acc": metrics["val_acc"], "val_ppl": metrics["val_ppl"], "elapsed_sec": elapsed})

    barrier()

    if is_main_process():
        profile_paths = []
        for start, end in windows:
            profile_id = f"{exp_id}_w{start}_{end}"
            out_csv = os.path.join(args.output_dir, "profiles", f"profile_{profile_id}.csv")
            prof = build_profile_from_trace(
                trace_csv=trace_path,
                output_csv=out_csv,
                window=(start, end),
                alpha=args.alpha,
                seed=args.seed,
                profile_id=profile_id,
                model_name=model_label,
                workload_name=args.workload_name,
                dataset_name=args.dataset_name,
            )
            profile_paths.append(out_csv)
            imp_count = int(prof["is_important"].sum())
            tau = float(prof["global_tau"].iloc[0])
            print(f"[profile] {profile_id}: tau={tau:.6f}, important_count={imp_count}, path={out_csv}", flush=True)
        with open(os.path.join(args.output_dir, "profile_paths.json"), "w", encoding="utf-8") as f:
            json.dump(profile_paths, f, indent=2, ensure_ascii=False)

    barrier()
    cleanup_distributed()


if __name__ == "__main__":
    main()
