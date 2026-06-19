#!/usr/bin/env python3
"""
RQ2 partition-method ablation for ImportanceGradient.

Goal:
  Compare different component-importance partition methods, e.g., Knee, Otsu,
  GMM, KMeans, fixed ratio, and percentile, under the same online policy.

What this script measures:
  - Model quality: validation loss, perplexity, token-level accuracy.
  - System metrics: logical sparse bytes, sparse-sync time, mean step time,
    throughput.

Important design choice:
  This script does NOT freeze low-importance components. Low-importance
  components are delayed rather than dropped. Their retained sparse gradients
  are accumulated in residual buffers and released every R_low iterations.

Launch with torchrun. Example:
  torchrun --nproc_per_node=4 rq2_partition_ablation.py \
    --model_name facebook/opt-1.3b \
    --dataset_path alpaca_data \
    --threshold_output_dir threshold_compare_outputs \
    --methods knee otsu gmm_2 kmeans_2 percentile_30 fixed_ratio_20 \
    --output_dir rq2_outputs
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
import torch.distributed as dist
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer


Component = Tuple[int, str]


# -----------------------------------------------------------------------------
# Distributed helpers
# -----------------------------------------------------------------------------


def init_distributed() -> Tuple[int, int, int, torch.device]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world_size = dist.get_world_size()

    return rank, local_rank, world_size, device


def is_main_process() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def main_print(*args, **kwargs) -> None:
    if is_main_process():
        print(*args, **kwargs, flush=True)


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def reduce_sum_float(value: float, device: torch.device) -> float:
    t = torch.tensor([float(value)], device=device, dtype=torch.float64)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item())


def reduce_max_float(value: float, device: torch.device) -> float:
    t = torch.tensor([float(value)], device=device, dtype=torch.float64)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


# -----------------------------------------------------------------------------
# Reproducibility and component parsing
# -----------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_opt_layer_component(param_name: str) -> Tuple[Optional[int], Optional[str]]:
    """Map OPT parameter names to Transformer components.

    Components follow the paper notation: Q, K, V, O, FC1, FC2.
    Non-component parameters, e.g., embeddings and lm_head, return (None, None).
    """
    if "layers." not in param_name:
        return None, None

    try:
        layer = int(param_name.split("layers.", 1)[1].split(".", 1)[0])
    except Exception:
        return None, None

    if "q_proj" in param_name:
        comp = "Q"
    elif "k_proj" in param_name:
        comp = "K"
    elif "v_proj" in param_name:
        comp = "V"
    elif "out_proj" in param_name:
        comp = "O"
    elif ".fc1." in param_name:
        comp = "FC1"
    elif ".fc2." in param_name:
        comp = "FC2"
    else:
        return None, None

    return layer, comp


def collect_model_components(model: torch.nn.Module) -> Set[Component]:
    comps: Set[Component] = set()
    for name, _ in model.named_parameters():
        layer, comp = parse_opt_layer_component(name)
        if layer is not None and comp is not None:
            comps.add((layer, comp))
    return comps


# -----------------------------------------------------------------------------
# Selection files
# -----------------------------------------------------------------------------


def _items_to_component_set(items: Sequence[dict]) -> Set[Component]:
    out: Set[Component] = set()
    for item in items:
        if "layer" not in item or "component" not in item:
            continue
        out.add((int(item["layer"]), str(item["component"])))
    return out


def load_selection_json(path: str, all_components: Set[Component], candidate_key_role: str) -> Dict[str, object]:
    """Load one threshold-selection JSON file.

    Expected schema, compatible with previous threshold scripts:
      {
        "tau": 0.3046,
        "candidates": [{"layer": 0, "component": "Q"}, ...],
        "important": [{"layer": 0, "component": "K"}, ...],   # optional
        "info": {...}
      }

    In the user's earlier freezing script, `candidates` are the component set
    selected by the threshold method and then frozen. For ImportanceGradient RQ2,
    this usually corresponds to low-importance components. Therefore the default
    role is: candidates -> low-importance components.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    candidates = _items_to_component_set(data.get("candidates", []))
    important_from_file = _items_to_component_set(data.get("important", []))

    if important_from_file:
        important = important_from_file & all_components
        low = all_components - important
    else:
        if candidate_key_role == "low":
            low = candidates & all_components
            important = all_components - low
        elif candidate_key_role == "important":
            important = candidates & all_components
            low = all_components - important
        else:
            raise ValueError(f"Unknown candidate_key_role={candidate_key_role}")

    return {
        "tau": data.get("tau", None),
        "info": data.get("info", {}),
        "important_components": sorted(list(important)),
        "low_components": sorted(list(low)),
        "num_important": len(important),
        "num_low": len(low),
        "num_total_components": len(all_components),
    }


def build_method_policies(
    methods: Sequence[str],
    threshold_output_dir: str,
    all_components: Set[Component],
    candidate_key_role: str,
    include_topk_sync: bool,
) -> List[Dict[str, object]]:
    policies: List[Dict[str, object]] = []

    if include_topk_sync:
        policies.append({
            "method": "topk_sync",
            "json_path": None,
            "tau": None,
            "important_components": sorted(list(all_components)),
            "low_components": [],
            "num_important": len(all_components),
            "num_low": 0,
            "num_total_components": len(all_components),
            "info": {"description": "all components synchronized every iteration"},
        })

    for method in methods:
        path = os.path.join(threshold_output_dir, f"{method}_selection.json")
        if not os.path.exists(path):
            main_print(f"[Warning] selection file not found for method={method}: {path}")
            continue
        item = load_selection_json(path, all_components, candidate_key_role)
        item["method"] = method
        item["json_path"] = path
        policies.append(item)

    return policies


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------


def infer_text(example: dict, dataset_format: str) -> str:
    if dataset_format == "alpaca" or (dataset_format == "auto" and "instruction" in example and "output" in example):
        instr = str(example.get("instruction", ""))
        inp = str(example.get("input", ""))
        out = str(example.get("output", ""))
        if inp.strip():
            return f"Instruction: {instr}\nInput: {inp}\nOutput: {out}"
        return f"Instruction: {instr}\nOutput: {out}"

    if dataset_format == "wikitext" or (dataset_format == "auto" and "text" in example):
        return str(example.get("text", ""))

    if dataset_format == "pubmed" or (dataset_format == "auto" and "abstract" in example):
        title = str(example.get("title", ""))
        abstract = str(example.get("abstract", ""))
        return f"{title}\n{abstract}" if title.strip() else abstract

    # fallback: concatenate all scalar fields
    vals = []
    for v in example.values():
        if isinstance(v, (str, int, float)):
            vals.append(str(v))
    return " ".join(vals)


def tokenize_batch(batch: dict, tokenizer, max_length: int, dataset_format: str) -> dict:
    size = len(next(iter(batch.values())))
    texts = []
    for i in range(size):
        ex = {k: v[i] for k, v in batch.items()}
        texts.append(infer_text(ex, dataset_format))

    encoded = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    labels = []
    for ids, mask in zip(encoded["input_ids"], encoded["attention_mask"]):
        labels.append([tok if m == 1 else -100 for tok, m in zip(ids, mask)])
    encoded["labels"] = labels
    return encoded


def build_dataloaders(args, tokenizer, rank: int, world_size: int):
    main_print(f"Loading dataset from {args.dataset_path}")
    if os.path.isdir(args.dataset_path):
        dataset = load_from_disk(args.dataset_path)
    else:
        # Supports names like wikitext or local files supported by datasets.
        dataset = load_dataset(args.dataset_path)

    if "train" in dataset and "test" in dataset:
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]
    elif "train" in dataset and "validation" in dataset:
        train_dataset = dataset["train"]
        eval_dataset = dataset["validation"]
    else:
        if hasattr(dataset, "train_test_split"):
            split = dataset.train_test_split(test_size=0.1, seed=42)
        else:
            # DatasetDict without explicit train/test. Use first split.
            first = next(iter(dataset.values()))
            split = first.train_test_split(test_size=0.1, seed=42)
        train_dataset = split["train"]
        eval_dataset = split["test"]

    if args.limit_train_samples > 0:
        train_dataset = train_dataset.select(range(min(args.limit_train_samples, len(train_dataset))))
    if args.limit_eval_samples > 0:
        eval_dataset = eval_dataset.select(range(min(args.limit_eval_samples, len(eval_dataset))))

    cols_to_remove = list(train_dataset.column_names)
    train_dataset = train_dataset.map(
        lambda b: tokenize_batch(b, tokenizer, args.max_length, args.dataset_format),
        batched=True,
        remove_columns=cols_to_remove,
        desc="Tokenizing train set",
    )
    eval_cols_to_remove = list(eval_dataset.column_names)
    eval_dataset = eval_dataset.map(
        lambda b: tokenize_batch(b, tokenizer, args.max_length, args.dataset_format),
        batched=True,
        remove_columns=eval_cols_to_remove,
        desc="Tokenizing eval set",
    )

    columns = ["input_ids", "attention_mask", "labels"]
    train_dataset.set_format(type="torch", columns=columns)
    eval_dataset.set_format(type="torch", columns=columns)

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    ) if world_size > 1 else None

    eval_sampler = DistributedSampler(
        eval_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    ) if world_size > 1 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        sampler=eval_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_loader, eval_loader, train_sampler


def infinite_loader(loader: DataLoader, sampler: Optional[DistributedSampler]):
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


# -----------------------------------------------------------------------------
# Sparse communication primitives
# -----------------------------------------------------------------------------


@dataclass
class SyncStats:
    logical_sparse_bytes: int = 0
    actual_padded_bytes: int = 0
    nnz_sent: int = 0
    sync_time_s: float = 0.0
    synced_tensors: int = 0

    def reset(self) -> None:
        self.logical_sparse_bytes = 0
        self.actual_padded_bytes = 0
        self.nnz_sent = 0
        self.sync_time_s = 0.0
        self.synced_tensors = 0


VALUE_BYTES = 4       # values are communicated as fp32 in this reference script
INDEX_BYTES = 4       # logical sparse index bytes; actual all_gather uses int64 padding


def topk_indices_values(tensor: torch.Tensor, ratio: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return Top-k sparse entries of tensor as flat indices and fp32 values."""
    if ratio <= 0:
        empty_idx = torch.empty(0, device=tensor.device, dtype=torch.long)
        empty_val = torch.empty(0, device=tensor.device, dtype=torch.float32)
        return empty_idx, empty_val

    flat = tensor.detach().reshape(-1)
    numel = flat.numel()
    if numel == 0:
        empty_idx = torch.empty(0, device=tensor.device, dtype=torch.long)
        empty_val = torch.empty(0, device=tensor.device, dtype=torch.float32)
        return empty_idx, empty_val

    k = max(1, int(math.ceil(numel * ratio)))
    k = min(k, numel)
    _, idx = torch.topk(flat.abs(), k=k, largest=True, sorted=False)
    val = flat.index_select(0, idx).to(torch.float32)

    # Avoid communicating padded zero entries when a tensor has many zeros.
    nz = val != 0
    idx = idx[nz]
    val = val[nz]
    return idx.contiguous(), val.contiguous()


def dense_from_sparse(indices: torch.Tensor, values: torch.Tensor, shape: torch.Size, dtype: torch.dtype) -> torch.Tensor:
    out = torch.zeros(int(math.prod(shape)), device=values.device, dtype=torch.float32)
    if indices.numel() > 0:
        out.index_add_(0, indices.to(torch.long), values.to(torch.float32))
    return out.view(shape).to(dtype)


def sparse_average(
    indices: torch.Tensor,
    values: torch.Tensor,
    shape: torch.Size,
    dtype: torch.dtype,
    stats: SyncStats,
) -> torch.Tensor:
    """Average sparse entries across workers and return a dense gradient tensor.

    This reference primitive all-gathers variable-length sparse tensors and
    reconstructs the averaged dense gradient locally. It is intentionally simple
    and correct for experiments; an optimized runtime can replace this function.
    """
    device = values.device
    local_nnz = int(indices.numel())
    stats.logical_sparse_bytes += local_nnz * (VALUE_BYTES + INDEX_BYTES)
    stats.nnz_sent += local_nnz
    stats.synced_tensors += 1

    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
        return dense_from_sparse(indices, values, shape, dtype)

    world_size = dist.get_world_size()
    local_len = torch.tensor([local_nnz], device=device, dtype=torch.long)
    lengths = [torch.zeros_like(local_len) for _ in range(world_size)]
    dist.all_gather(lengths, local_len)
    lengths_int = [int(x.item()) for x in lengths]
    max_len = max(lengths_int) if lengths_int else 0

    if max_len == 0:
        return torch.zeros(shape, device=device, dtype=dtype)

    pad_idx = torch.zeros(max_len, device=device, dtype=torch.long)
    pad_val = torch.zeros(max_len, device=device, dtype=torch.float32)
    if local_nnz > 0:
        pad_idx[:local_nnz] = indices.to(torch.long)
        pad_val[:local_nnz] = values.to(torch.float32)

    gathered_idx = [torch.empty_like(pad_idx) for _ in range(world_size)]
    gathered_val = [torch.empty_like(pad_val) for _ in range(world_size)]
    dist.all_gather(gathered_idx, pad_idx)
    dist.all_gather(gathered_val, pad_val)

    # Actual padded bytes sent per rank in the reference all_gather path.
    stats.actual_padded_bytes += max_len * (8 + VALUE_BYTES)

    flat_out = torch.zeros(int(math.prod(shape)), device=device, dtype=torch.float32)
    for r, n in enumerate(lengths_int):
        if n <= 0:
            continue
        flat_out.index_add_(0, gathered_idx[r][:n], gathered_val[r][:n])
    flat_out.div_(world_size)
    return flat_out.view(shape).to(dtype)


# -----------------------------------------------------------------------------
# ImportanceGradient synchronization policy
# -----------------------------------------------------------------------------


def sync_gradients_importance_policy(
    model: torch.nn.Module,
    low_components: Set[Component],
    residuals: Dict[str, torch.Tensor],
    step: int,
    args,
    stats: SyncStats,
) -> None:
    """Apply component-aware sparse synchronization to model gradients.

    Important components: synchronize retained Top-k sparse gradients every step.
    Low-importance components: accumulate retained Top-k sparse gradients in a
    residual buffer and release every R_low steps. On release, select a fixed
    fraction of the delayed payload; unselected entries remain in the residual.
    """
    release_now = (args.release_period <= 1) or ((step + 1) % args.release_period == 0)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        grad = param.grad.detach()
        layer, comp = parse_opt_layer_component(name)
        component_key = (layer, comp) if layer is not None and comp is not None else None
        is_low = component_key in low_components if component_key is not None else False

        if not is_low:
            # Important and non-component tensors stay on the every-iteration path.
            idx, val = topk_indices_values(grad, args.topk_ratio)
            synced = sparse_average(idx, val, grad.shape, grad.dtype, stats)
            param.grad.copy_(synced)
            continue

        # Low-importance component path.
        if name not in residuals:
            residuals[name] = torch.zeros_like(grad, memory_format=torch.preserve_format)

        # Accumulate the current retained sparse update into the residual.
        idx, val = topk_indices_values(grad, args.topk_ratio)
        if idx.numel() > 0:
            residuals[name].view(-1).index_add_(0, idx.to(torch.long), val.to(residuals[name].dtype))

        if not release_now:
            # Delay rather than drop: no optimizer update for this component now;
            # its retained sparse update stays in the residual buffer.
            param.grad.zero_()
            continue

        # Release a fraction of the accumulated delayed payload.
        payload = residuals[name]
        rel_idx, rel_val = topk_indices_values(payload, args.delayed_keep_ratio)
        selected_local = dense_from_sparse(rel_idx, rel_val, payload.shape, payload.dtype)

        synced = sparse_average(rel_idx, rel_val, grad.shape, grad.dtype, stats)
        param.grad.copy_(synced)

        # Keep unselected delayed entries for future releases.
        residuals[name].sub_(selected_local)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    stats.sync_time_s += time.perf_counter() - t0


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------


@torch.no_grad()
def evaluate(model: torch.nn.Module, eval_loader: DataLoader, device: torch.device) -> Tuple[float, float, float]:
    model.eval()
    total_loss_weighted = 0.0
    total_tokens = 0.0
    correct = 0.0
    acc_tokens = 0.0

    for batch in eval_loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        valid_tokens = (labels != -100).sum().item()
        total_loss_weighted += float(loss.item()) * valid_tokens
        total_tokens += float(valid_tokens)

        logits = outputs.logits
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        preds = shift_logits.argmax(dim=-1)
        mask = shift_labels != -100
        correct += float(((preds == shift_labels) & mask).sum().item())
        acc_tokens += float(mask.sum().item())

    # Global reduction.
    total_loss_weighted = reduce_sum_float(total_loss_weighted, device)
    total_tokens = reduce_sum_float(total_tokens, device)
    correct = reduce_sum_float(correct, device)
    acc_tokens = reduce_sum_float(acc_tokens, device)

    model.train()
    if total_tokens <= 0:
        return float("inf"), float("inf"), 0.0
    val_loss = total_loss_weighted / total_tokens
    ppl = math.exp(val_loss) if val_loss < 50 else float("inf")
    accuracy = correct / acc_tokens if acc_tokens > 0 else 0.0
    return val_loss, ppl, accuracy




def resolve_model_dtype(dtype_name: str):
    """Return dtype for from_pretrained.

    The reference script uses a plain PyTorch AdamW optimizer. Loading OPT in
    fp16 can make AdamW update fp16 parameters/states directly and may produce
    NaNs because the default epsilon is too small for fp16. The default is
    therefore float32 for numerical sanity.
    """
    dtype_name = str(dtype_name).lower()
    if dtype_name in {"float32", "fp32"}:
        return torch.float32
    if dtype_name in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if dtype_name in {"float16", "fp16", "half"}:
        return torch.float16
    if dtype_name == "auto":
        return "auto"
    raise ValueError(f"Unsupported --model_dtype: {dtype_name}")


# -----------------------------------------------------------------------------
# One method run
# -----------------------------------------------------------------------------


def write_csv_header_if_needed(path: str, fieldnames: Sequence[str]) -> None:
    if not is_main_process():
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()


def append_csv_row(path: str, fieldnames: Sequence[str], row: dict) -> None:
    if not is_main_process():
        return
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)


def run_one_policy(policy: Dict[str, object], seed: int, args, rank: int, world_size: int, device: torch.device) -> None:
    method = str(policy["method"])
    main_print("=" * 100)
    main_print(f"Running method={method}, seed={seed}")
    main_print(f"important={policy['num_important']} low={policy['num_low']} total={policy['num_total_components']} tau={policy.get('tau')}")
    main_print("=" * 100)

    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_dtype = resolve_model_dtype(args.model_dtype)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=model_dtype)
    model.config.use_cache = False
    model.to(device)
    if args.model_dtype.lower() in {"float32", "fp32"}:
        model.float()
    model.train()

    train_loader, eval_loader, train_sampler = build_dataloaders(args, tokenizer, rank, world_size)
    train_iter = infinite_loader(train_loader, train_sampler)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        eps=args.adam_eps,
    )

    low_components: Set[Component] = set(tuple(x) for x in policy["low_components"])  # type: ignore[arg-type]
    residuals: Dict[str, torch.Tensor] = {}
    stats = SyncStats()

    fieldnames = [
        "method", "seed", "step", "elapsed_s", "val_loss", "ppl", "accuracy",
        "global_logical_sparse_bytes", "global_actual_padded_bytes", "global_nnz_sent",
        "sync_time_ms_max", "mean_step_time_ms_max", "throughput_tokens_per_s",
        "num_important", "num_low", "tau",
    ]
    progress_csv = os.path.join(args.output_dir, "rq2_progress.csv")
    write_csv_header_if_needed(progress_csv, fieldnames)

    start_time = time.perf_counter()
    interval_step_times: List[float] = []

    if args.eval_before_train:
        val_loss0, ppl0, accuracy0 = evaluate(model, eval_loader, device)
        if is_main_process():
            print(
                f"[{method} seed={seed}] pretrain_eval "
                f"val_loss={val_loss0:.6f} acc={accuracy0:.4f} ppl={ppl0:.4f}"
            )

    for step in range(args.total_steps):
        batch = next(train_iter)
        step_t0 = time.perf_counter()

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        if not torch.isfinite(loss).all():
            raise RuntimeError(
                f"Non-finite training loss before backward: method={method}, "
                f"seed={seed}, step={step + 1}, loss={loss.item()}"
            )
        loss.backward()

        if args.debug_numerics:
            for n, p in model.named_parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    raise RuntimeError(
                        f"Non-finite gradient before sync: method={method}, "
                        f"seed={seed}, step={step + 1}, param={n}"
                    )

        sync_gradients_importance_policy(model, low_components, residuals, step, args, stats)
        optimizer.step()

        if args.debug_numerics:
            for n, p in model.named_parameters():
                if not torch.isfinite(p).all():
                    raise RuntimeError(
                        f"Non-finite parameter after optimizer.step: method={method}, "
                        f"seed={seed}, step={step + 1}, param={n}"
                    )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        step_time_s = time.perf_counter() - step_t0
        interval_step_times.append(step_time_s)

        do_eval = (step == 0) or ((step + 1) % args.eval_interval == 0) or (step + 1 == args.total_steps)
        if do_eval:
            val_loss, ppl, accuracy = evaluate(model, eval_loader, device)

            global_logical_bytes = reduce_sum_float(float(stats.logical_sparse_bytes), device)
            global_actual_padded_bytes = reduce_sum_float(float(stats.actual_padded_bytes), device)
            global_nnz_sent = reduce_sum_float(float(stats.nnz_sent), device)
            sync_time_ms_max = reduce_max_float(stats.sync_time_s * 1000.0, device)
            mean_step_s_local = sum(interval_step_times) / max(1, len(interval_step_times))
            mean_step_s_max = reduce_max_float(mean_step_s_local, device)
            throughput_tokens = (args.train_batch_size * world_size * args.max_length) / max(mean_step_s_max, 1e-12)
            elapsed = time.perf_counter() - start_time

            row = {
                "method": method,
                "seed": seed,
                "step": step + 1,
                "elapsed_s": f"{elapsed:.3f}",
                "val_loss": f"{val_loss:.6f}",
                "ppl": f"{ppl:.6f}",
                "accuracy": f"{accuracy:.6f}",
                "global_logical_sparse_bytes": int(global_logical_bytes),
                "global_actual_padded_bytes": int(global_actual_padded_bytes),
                "global_nnz_sent": int(global_nnz_sent),
                "sync_time_ms_max": f"{sync_time_ms_max:.3f}",
                "mean_step_time_ms_max": f"{mean_step_s_max * 1000.0:.3f}",
                "throughput_tokens_per_s": f"{throughput_tokens:.3f}",
                "num_important": int(policy["num_important"]),
                "num_low": int(policy["num_low"]),
                "tau": policy.get("tau"),
            }
            append_csv_row(progress_csv, fieldnames, row)
            main_print(
                f"[{method} seed={seed}] step={step+1:5d} "
                f"val_loss={val_loss:.4f} acc={accuracy:.4f} "
                f"step_ms={mean_step_s_max*1000.0:.2f} sync_ms={sync_time_ms_max:.2f} "
                f"bytes={int(global_logical_bytes)} throughput={throughput_tokens:.1f} tok/s"
            )

            stats.reset()
            interval_step_times.clear()

    barrier()
    del model, optimizer, residuals, train_loader, eval_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    barrier()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="RQ2 component partition ablation")
    parser.add_argument("--model_name", type=str, default="facebook/opt-1.3b")
    parser.add_argument("--dataset_path", type=str, default="alpaca_data")
    parser.add_argument("--dataset_format", type=str, default="auto", choices=["auto", "alpaca", "wikitext", "pubmed"])
    parser.add_argument("--threshold_output_dir", type=str, default="threshold_compare_outputs")
    parser.add_argument("--methods", nargs="+", default=["knee", "otsu", "gmm_2", "kmeans_2", "percentile_30", "fixed_ratio_20"])
    parser.add_argument("--candidate_key_role", type=str, default="low", choices=["low", "important"])
    parser.add_argument("--output_dir", type=str, default="rq2_partition_outputs")

    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--limit_train_samples", type=int, default=0)
    parser.add_argument("--limit_eval_samples", type=int, default=0)

    parser.add_argument("--total_steps", type=int, default=5000)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument("--model_dtype", type=str, default="float32", choices=["auto", "float32", "fp32", "bfloat16", "bf16", "float16", "fp16"], help="Load model dtype. Default float32 avoids fp16 AdamW NaNs in this reference script.")
    parser.add_argument("--eval_before_train", action="store_true", help="Run one evaluation before the first optimizer step.")
    parser.add_argument("--debug_numerics", action="store_true", help="Stop immediately if loss/gradients/parameters become non-finite.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])

    parser.add_argument("--topk_ratio", type=float, default=0.001, help="Baseline retained sparse-gradient ratio per tensor.")
    parser.add_argument("--release_period", type=int, default=4, help="R_low for low-importance components.")
    parser.add_argument("--delayed_keep_ratio", type=float, default=0.25, help="Fraction of delayed payload released at a release iteration.")
    parser.add_argument("--include_topk_sync", action="store_true", help="Also run Top-k Sync internal baseline.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size, device = init_distributed()
    args.seed = args.seeds[0] if len(args.seeds) > 0 else args.seed

    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "args.json"), "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2)

    main_print(f"Distributed setup: rank={rank}, local_rank={local_rank}, world_size={world_size}, device={device}")

    # Load a temporary model once to infer all OPT components.
    set_seed(args.seed)
    tmp_model = AutoModelForCausalLM.from_pretrained(args.model_name)
    all_components = collect_model_components(tmp_model)
    del tmp_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    policies = build_method_policies(
        methods=args.methods,
        threshold_output_dir=args.threshold_output_dir,
        all_components=all_components,
        candidate_key_role=args.candidate_key_role,
        include_topk_sync=args.include_topk_sync,
    )

    if is_main_process():
        policy_json = []
        for p in policies:
            item = dict(p)
            item["important_components"] = [{"layer": x[0], "component": x[1]} for x in item["important_components"]]
            item["low_components"] = [{"layer": x[0], "component": x[1]} for x in item["low_components"]]
            policy_json.append(item)
        with open(os.path.join(args.output_dir, "rq2_policies.json"), "w", encoding="utf-8") as f:
            json.dump(policy_json, f, indent=2)

    if len(policies) == 0:
        raise RuntimeError("No valid policies found. Check --threshold_output_dir and --methods.")

    for seed in args.seeds:
        for policy in policies:
            run_one_policy(policy, seed, args, rank, world_size, device)

    barrier()
    main_print(f"Done. Results saved to {args.output_dir}/rq2_progress.csv")
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
