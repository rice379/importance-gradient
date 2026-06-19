#!/usr/bin/env python3
"""
RQ4: Balanced payload materialization / packing ablation.

This script runs post-training with the same importance gate and residual
semantics as ImportanceGradient, but varies how the released/selected payload
is materialized into communication buffers:

  direct_filtering : keep each selected block in its original pre-filter bucket
  sequential       : repack selected blocks sequentially into K buffers
  greedy           : sort selected blocks by size and place into the lightest buffer
  risk_aware       : use RiskAwareBucketPlanner as the ImportanceGradient policy

It logs bucket load skew, packing overhead, all-reduce time, total communication
time, communicated bytes, and training quality proxies.

Expected local files in the same directory:
  - periodic_sync_gate.py
  - bucket_runtime_planner.py
  - real_bucket_comm.py
  - metrics_schema.py
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
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from importance_gradient.bucket_runtime_planner import BlockMeta, RiskAwareBucketPlanner
from importance_gradient.periodic_sync_gate import PeriodicSyncGate, load_importance_profile
from importance_gradient.real_bucket_comm import SyncTensorRef, split_tensor_ranges


# -----------------------------
# distributed helpers
# -----------------------------

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


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def ddp_setup(args) -> torch.device:
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    if local_rank < 0:
        local_rank = 0
    if not is_dist():
        dist.init_process_group(backend=args.dist_backend)
    if torch.cuda.device_count() == 0:
        raise RuntimeError("No visible CUDA devices found.")
    torch.cuda.set_device(local_rank)
    return torch.device("cuda", local_rank)


def cleanup_dist() -> None:
    if is_dist():
        dist.destroy_process_group()


# -----------------------------
# data helpers
# -----------------------------

class TokenizedDataset(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
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
    parts = [str(v).strip() for v in example.values() if isinstance(v, str) and v.strip()]
    return " ".join(parts).strip()


def tokenize_function(example: Dict[str, Any], tokenizer, max_length: int) -> Dict[str, Any]:
    text = build_text(example)
    encoded = tokenizer(text, truncation=True, padding="max_length", max_length=max_length)
    labels = encoded["input_ids"].copy()
    labels = [tok if mask == 1 else -100 for tok, mask in zip(labels, encoded["attention_mask"])]
    encoded["labels"] = labels
    return encoded


def build_dataloaders(args, tokenizer):
    if is_main_process():
        print("[Data] loading dataset ...")

    if args.local_dataset_path and os.path.exists(args.local_dataset_path):
        dataset = load_from_disk(args.local_dataset_path)
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
        print(f"[Data] train={len(train_dataset):,}, eval={len(eval_dataset):,}")
        print("[Data] tokenizing ...")

    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, args.max_length),
        remove_columns=train_dataset.column_names,
    )
    eval_dataset = eval_dataset.map(
        lambda x: tokenize_function(x, tokenizer, args.max_length),
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
    )
    eval_sampler = DistributedSampler(
        eval_dataset,
        num_replicas=get_world_size(),
        rank=get_rank(),
        shuffle=False,
        seed=args.seed,
        drop_last=False,
    )

    train_loader = DataLoader(
        TokenizedDataset(train_dataset),
        batch_size=args.per_device_train_batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    eval_loader = DataLoader(
        TokenizedDataset(eval_dataset),
        batch_size=args.per_device_eval_batch_size,
        sampler=eval_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_loader, eval_loader, train_sampler, eval_sampler


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
            total_correct += ((pred == shift_labels) & valid_mask).sum().item()
            total_tokens += valid_tokens
            total_nll += float(loss.item()) * valid_tokens

    if is_dist():
        dist.all_reduce(total_nll, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_correct, op=dist.ReduceOp.SUM)

    avg_loss = (total_nll / total_tokens.clamp_min(1.0)).item()
    avg_ppl = math.exp(min(avg_loss, 20.0))
    avg_acc = (total_correct / total_tokens.clamp_min(1.0)).item()
    model.train()
    return {"val_loss": avg_loss, "val_ppl": avg_ppl, "val_acc": avg_acc}


# -----------------------------
# materialization communicator
# -----------------------------

@dataclass
class GradBlockRef:
    block_id: int
    param_name: str
    group_id: Any
    is_low_importance: bool
    flat_view: torch.Tensor
    start: int
    end: int
    numel: int


@dataclass
class PackingStats:
    step: int
    policy: str
    bucket_count: int
    block_count: int
    communicated_numel: int
    communicated_bytes: int
    bucket_imbalance_ratio: float
    bucket_imbalance_cv: float
    bucket_overflow_count: int
    pack_time_ms: float
    allreduce_time_ms: float
    unpack_time_ms: float
    total_comm_time_ms: float


class PackingAblationCommunicator:
    """
    Real communication path for RQ4 packing ablation.

    The communicator first applies the same effective-payload selection to
    released low-priority blocks for every policy. It then varies only how the
    selected blocks are assigned to communication buffers.
    """

    def __init__(
        self,
        policy: str,
        bucket_num: int = 4,
        block_size_numel: int = 262144,
        payload_low_keep_ratio: float = 0.25,
        payload_rotation_interval: int = 4,
        enable_cuda_timing_sync: bool = True,
        enable_async_allreduce: bool = True,
        risk_lambda_std: float = 0.15,
        risk_gamma_overflow: float = 1.0,
        risk_target_margin: float = 1.05,
        risk_use_adaptive_switch: bool = True,
    ):
        valid = {"direct_filtering", "sequential", "greedy", "risk_aware"}
        if policy not in valid:
            raise ValueError(f"Unknown policy={policy}; expected one of {sorted(valid)}")
        self.policy = policy
        self.bucket_num = int(bucket_num)
        self.block_size_numel = int(block_size_numel)
        self.payload_low_keep_ratio = float(payload_low_keep_ratio)
        self.payload_rotation_interval = max(1, int(payload_rotation_interval))
        self.enable_cuda_timing_sync = bool(enable_cuda_timing_sync)
        self.enable_async_allreduce = bool(enable_async_allreduce)
        self._payload_residuals: Dict[Tuple[str, int, int], torch.Tensor] = {}
        self._payload_residual_active: Dict[Tuple[str, int, int], bool] = {}
        self._bucket_buffers: Dict[int, torch.Tensor] = {}
        self.risk_planner = RiskAwareBucketPlanner(
            bucket_num=self.bucket_num,
            mode_requested="risk_aware",
            use_adaptive_switch=risk_use_adaptive_switch,
            lambda_std=risk_lambda_std,
            gamma_overflow=risk_gamma_overflow,
            target_margin=risk_target_margin,
        )

    def _cuda_sync(self) -> None:
        if self.enable_cuda_timing_sync and torch.cuda.is_available():
            torch.cuda.synchronize()

    def _payload_slots(self) -> int:
        keep = max(0.0, min(1.0, self.payload_low_keep_ratio))
        if keep >= 1.0:
            return 1
        if keep <= 0.0:
            return 10**9
        return max(1, int(round(1.0 / keep)))

    def _phase_mod(self, step: int, slots: int) -> int:
        if slots <= 1:
            return 0
        return int((int(step) // self.payload_rotation_interval) % slots)

    @staticmethod
    def _res_key(br: GradBlockRef) -> Tuple[str, int, int]:
        return (str(br.param_name), int(br.start), int(br.end))

    def _get_residual(self, br: GradBlockRef) -> torch.Tensor:
        key = self._res_key(br)
        res = self._payload_residuals.get(key)
        if res is None or res.numel() != br.numel or res.dtype != br.flat_view.dtype or res.device != br.flat_view.device:
            res = torch.zeros(br.numel, dtype=br.flat_view.dtype, device=br.flat_view.device)
            self._payload_residuals[key] = res
            self._payload_residual_active[key] = False
        return res

    def build_blocks(self, sync_tensors: List[SyncTensorRef]) -> List[GradBlockRef]:
        blocks: List[GradBlockRef] = []
        bid = 0
        for ref in sync_tensors:
            if ref.tensor is None or ref.tensor.numel() == 0:
                continue
            flat = ref.tensor.view(-1)
            for start, end in split_tensor_ranges(flat.numel(), self.block_size_numel):
                blocks.append(
                    GradBlockRef(
                        block_id=bid,
                        param_name=str(ref.name),
                        group_id=ref.group_id,
                        is_low_importance=bool(ref.is_low_importance),
                        flat_view=flat,
                        start=int(start),
                        end=int(end),
                        numel=int(end - start),
                    )
                )
                bid += 1
        return blocks

    def apply_effective_payload_filter(self, blocks: List[GradBlockRef], step: int) -> List[GradBlockRef]:
        slots = self._payload_slots()
        phase = self._phase_mod(step, slots)
        kept: List[GradBlockRef] = []

        for br in blocks:
            seg = br.flat_view[br.start:br.end]
            if not br.is_low_importance or slots <= 1:
                kept.append(br)
                continue

            keep = ((int(br.block_id) + phase) % slots) == 0
            key = self._res_key(br)
            if keep:
                if self._payload_residual_active.get(key, False):
                    res = self._get_residual(br)
                    seg.add_(res)
                    res.zero_()
                    self._payload_residual_active[key] = False
                kept.append(br)
            else:
                res = self._get_residual(br)
                if self._payload_residual_active.get(key, False):
                    res.add_(seg)
                else:
                    res.copy_(seg)
                    self._payload_residual_active[key] = True
                # Prevent unsynchronized local update for skipped payload.
                seg.zero_()
        return kept

    @staticmethod
    def block_cost(br: GradBlockRef) -> int:
        return int(br.numel * br.flat_view.element_size())

    def _direct_original_bucket_assignment(self, base_blocks: List[GradBlockRef]) -> Dict[int, int]:
        """
        Build equal dense buckets on the full pre-filter layout, then keep the
        original bucket id after filtering. This models direct filtering: payload
        shrinks, but the original bucket layout is not rebalanced.
        """
        total = sum(self.block_cost(b) for b in base_blocks)
        target = total / max(1, self.bucket_num)
        assignment: Dict[int, int] = {}
        bucket_id = 0
        current = 0.0
        for br in base_blocks:
            cost = self.block_cost(br)
            if bucket_id < self.bucket_num - 1 and current > 0 and current + cost > target:
                bucket_id += 1
                current = 0.0
            assignment[int(br.block_id)] = int(bucket_id)
            current += cost
        return assignment

    def _sequential_assignment(self, blocks: List[GradBlockRef]) -> Dict[int, int]:
        total = sum(self.block_cost(b) for b in blocks)
        target = total / max(1, self.bucket_num)
        assignment: Dict[int, int] = {}
        bucket_id = 0
        current = 0.0
        for br in blocks:
            cost = self.block_cost(br)
            if bucket_id < self.bucket_num - 1 and current > 0 and current + cost > target:
                bucket_id += 1
                current = 0.0
            assignment[int(br.block_id)] = int(bucket_id)
            current += cost
        return assignment

    def _greedy_assignment(self, blocks: List[GradBlockRef]) -> Dict[int, int]:
        loads = [0 for _ in range(self.bucket_num)]
        assignment: Dict[int, int] = {}
        # Large blocks first gives a strong deterministic greedy baseline.
        for br in sorted(blocks, key=lambda x: self.block_cost(x), reverse=True):
            bid = min(range(self.bucket_num), key=lambda i: loads[i])
            assignment[int(br.block_id)] = int(bid)
            loads[bid] += self.block_cost(br)
        return assignment

    def _risk_aware_assignment(self, blocks: List[GradBlockRef], step: int) -> Dict[int, int]:
        metas: List[BlockMeta] = []
        for br in blocks:
            cost = float(self.block_cost(br))
            metas.append(
                BlockMeta(
                    block_id=int(br.block_id),
                    param_name=str(br.param_name),
                    group_id=br.group_id,
                    start=int(br.start),
                    end=int(br.end),
                    numel=int(br.numel),
                    est_cost_mean=cost,
                    est_cost_std=0.0,
                    nnz_proxy=int(br.numel),
                )
            )
        plan = self.risk_planner.plan(metas, step=step)
        return {int(k): int(v) for k, v in plan.assignment.items()}

    def assign_buckets(self, base_blocks: List[GradBlockRef], kept_blocks: List[GradBlockRef], step: int) -> Dict[int, int]:
        if self.policy == "direct_filtering":
            original = self._direct_original_bucket_assignment(base_blocks)
            return {int(br.block_id): int(original[int(br.block_id)]) for br in kept_blocks}
        if self.policy == "sequential":
            return self._sequential_assignment(kept_blocks)
        if self.policy == "greedy":
            return self._greedy_assignment(kept_blocks)
        if self.policy == "risk_aware":
            return self._risk_aware_assignment(kept_blocks, step)
        raise ValueError(self.policy)

    def _compute_load_metrics(self, blocks: List[GradBlockRef], assignment: Dict[int, int]) -> Tuple[List[int], float, float, int]:
        loads = [0 for _ in range(self.bucket_num)]
        for br in blocks:
            loads[int(assignment[int(br.block_id)])] += self.block_cost(br)
        mx = max(loads) if loads else 0
        mn = min(loads) if loads else 0
        mean = sum(loads) / max(1, len(loads))
        ratio = float(mx / max(mn, 1e-12)) if mx > 0 else 1.0
        if mean > 0 and len(loads) > 1:
            var = sum((x - mean) ** 2 for x in loads) / len(loads)
            cv = float(math.sqrt(var) / mean)
        else:
            cv = 0.0
        overflow_count = sum(1 for x in loads if mean > 0 and x > mean * 1.05)
        return loads, ratio, cv, overflow_count

    def _get_bucket_buffer(self, bucket_id: int, total_numel: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        old = self._bucket_buffers.get(bucket_id)
        if old is None or old.numel() < total_numel or old.dtype != dtype or old.device != device:
            old = torch.empty(total_numel, dtype=dtype, device=device)
            self._bucket_buffers[bucket_id] = old
        return old[:total_numel]

    def _pack_bucket(self, bucket_id: int, refs: List[GradBlockRef]) -> Tuple[torch.Tensor, List[Tuple[int, int, GradBlockRef]]]:
        total_numel = sum(br.numel for br in refs)
        dtype = refs[0].flat_view.dtype
        device = refs[0].flat_view.device
        flat = self._get_bucket_buffer(bucket_id, total_numel, dtype, device)
        mapping: List[Tuple[int, int, GradBlockRef]] = []
        cur = 0
        for br in refs:
            seg = br.flat_view[br.start:br.end]
            flat[cur:cur + br.numel].copy_(seg)
            mapping.append((cur, cur + br.numel, br))
            cur += br.numel
        return flat, mapping

    def _unpack_bucket(self, flat: torch.Tensor, mapping: List[Tuple[int, int, GradBlockRef]]) -> None:
        for s, e, br in mapping:
            br.flat_view[br.start:br.end].copy_(flat[s:e])

    def communicate(self, sync_tensors: List[SyncTensorRef], step: int) -> PackingStats:
        if not sync_tensors:
            return PackingStats(step, self.policy, self.bucket_num, 0, 0, 0, 1.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0)

        self._cuda_sync()
        t0 = time.perf_counter()

        base_blocks = self.build_blocks(sync_tensors)
        kept_blocks = self.apply_effective_payload_filter(base_blocks, step)
        if not kept_blocks:
            return PackingStats(step, self.policy, self.bucket_num, 0, 0, 0, 1.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0)

        assignment = self.assign_buckets(base_blocks, kept_blocks, step)
        loads, ratio, cv, overflow_count = self._compute_load_metrics(kept_blocks, assignment)

        bucket_to_blocks: Dict[int, List[GradBlockRef]] = {i: [] for i in range(self.bucket_num)}
        for br in kept_blocks:
            bucket_to_blocks[int(assignment[int(br.block_id)])].append(br)

        self._cuda_sync()
        t_plan = time.perf_counter()

        packed: List[Tuple[int, torch.Tensor, List[Tuple[int, int, GradBlockRef]]]] = []
        total_numel = 0
        total_bytes = 0

        self._cuda_sync()
        t_pack0 = time.perf_counter()
        for bucket_id in range(self.bucket_num):
            refs = bucket_to_blocks[bucket_id]
            if not refs:
                continue
            flat, mapping = self._pack_bucket(bucket_id, refs)
            total_numel += int(flat.numel())
            total_bytes += int(flat.numel() * flat.element_size())
            packed.append((bucket_id, flat, mapping))
        self._cuda_sync()
        t_pack1 = time.perf_counter()

        self._cuda_sync()
        t_ar0 = time.perf_counter()
        if is_dist() and self.enable_async_allreduce:
            works = []
            for _, flat, _ in packed:
                works.append((dist.all_reduce(flat, op=dist.ReduceOp.SUM, async_op=True), flat))
            for work, _ in works:
                work.wait()
            world = get_world_size()
            for _, flat in works:
                flat.div_(world)
        elif is_dist():
            world = get_world_size()
            for _, flat, _ in packed:
                dist.all_reduce(flat, op=dist.ReduceOp.SUM)
                flat.div_(world)
        self._cuda_sync()
        t_ar1 = time.perf_counter()

        self._cuda_sync()
        t_up0 = time.perf_counter()
        for _, flat, mapping in packed:
            self._unpack_bucket(flat, mapping)
            # Update cost history for risk-aware mode.
            if self.policy == "risk_aware":
                for _, _, br in mapping:
                    self.risk_planner.update_history(int(br.block_id), float(self.block_cost(br)))
        self._cuda_sync()
        t_up1 = time.perf_counter()

        self._cuda_sync()
        t_end = time.perf_counter()

        pack_ms = (t_pack1 - t_pack0 + t_plan - t0) * 1000.0
        ar_ms = (t_ar1 - t_ar0) * 1000.0
        up_ms = (t_up1 - t_up0) * 1000.0
        total_ms = (t_end - t0) * 1000.0

        return PackingStats(
            step=int(step),
            policy=self.policy,
            bucket_count=int(self.bucket_num),
            block_count=int(len(kept_blocks)),
            communicated_numel=int(total_numel),
            communicated_bytes=int(total_bytes),
            bucket_imbalance_ratio=float(ratio),
            bucket_imbalance_cv=float(cv),
            bucket_overflow_count=int(overflow_count),
            pack_time_ms=float(pack_ms),
            allreduce_time_ms=float(ar_ms),
            unpack_time_ms=float(up_ms),
            total_comm_time_ms=float(total_ms),
        )


# -----------------------------
# logging
# -----------------------------

def append_dict_csv(path: str, row: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


# -----------------------------
# training
# -----------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--dist_backend", type=str, default="nccl")

    # model/data
    parser.add_argument("--model_name_or_path", type=str, default="facebook/opt-1.3b")
    parser.add_argument("--dataset_name", type=str, default="uiyunkim-hub/pubmed-abstract")
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--local_dataset_path", type=str, default=None)
    parser.add_argument("--local_files_only", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--max_train_samples", type=int, default=100000)
    parser.add_argument("--max_eval_samples", type=int, default=2000)
    parser.add_argument("--eval_ratio", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=0)

    # training
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--eval_interval", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # ImportanceGradient gate
    parser.add_argument("--score_csv", type=str, required=True)
    parser.add_argument("--sync_mode", type=str, default="periodic", choices=["full", "periodic"])
    parser.add_argument("--low_importance_period", type=int, default=4)
    parser.add_argument("--use_residual_accumulation", action="store_true")

    # RQ4 materialization
    parser.add_argument("--materialization_policy", type=str, required=True,
                        choices=["direct_filtering", "sequential", "greedy", "risk_aware"])
    parser.add_argument("--bucket_num", type=int, default=4)
    parser.add_argument("--bucket_block_size_numel", type=int, default=262144)
    parser.add_argument("--payload_low_keep_ratio", type=float, default=0.25)
    parser.add_argument("--payload_rotation_interval", type=int, default=4)
    parser.add_argument("--risk_lambda_std", type=float, default=0.15)
    parser.add_argument("--risk_gamma_overflow", type=float, default=1.0)
    parser.add_argument("--risk_target_margin", type=float, default=1.05)
    parser.add_argument("--risk_use_adaptive_switch", action="store_true")

    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = ddp_setup(args)

    if is_main_process():
        print(json.dumps(vars(args), indent=2, ensure_ascii=False))
        with open(os.path.join(args.output_dir, "run_args.json"), "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, ensure_ascii=False)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        local_files_only=bool(args.local_files_only),
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
        local_files_only=bool(args.local_files_only),
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

    def lr_lambda(step: int):
        if step < args.warmup_steps:
            return float(step) / max(1, args.warmup_steps)
        progress = float(step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    score_dict, global_tau = load_importance_profile(args.score_csv)
    gate = PeriodicSyncGate(
        model=model,
        score_dict=score_dict,
        global_tau=global_tau,
        low_importance_period=args.low_importance_period,
        use_residual_accumulation=args.use_residual_accumulation,
        always_sync_non_component=True,
        sync_mode=args.sync_mode,
    )
    communicator = PackingAblationCommunicator(
        policy=args.materialization_policy,
        bucket_num=args.bucket_num,
        block_size_numel=args.bucket_block_size_numel,
        payload_low_keep_ratio=args.payload_low_keep_ratio,
        payload_rotation_interval=args.payload_rotation_interval,
        risk_lambda_std=args.risk_lambda_std,
        risk_gamma_overflow=args.risk_gamma_overflow,
        risk_target_margin=args.risk_target_margin,
        risk_use_adaptive_switch=args.risk_use_adaptive_switch,
    )

    metrics_path = os.path.join(args.output_dir, "metrics.csv")
    summary_path = os.path.join(args.output_dir, "summary.json")

    global_step = 0
    train_iter = iter(train_loader)
    start_time = time.time()
    prev_step_time = time.perf_counter()
    loss_accum = 0.0
    loss_count = 0

    best = {"best_step": -1, "best_val_loss": float("inf"), "best_val_acc": None, "best_val_ppl": None}

    while global_step < args.max_steps:
        train_sampler.set_epoch(global_step)
        optimizer.zero_grad(set_to_none=True)

        micro_loss = 0.0
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
            micro_loss += float(loss.item())

        if args.max_grad_norm and args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        sync_tensors, gate_stats = gate.prepare_sync_tensors(global_step + 1)
        comm_stats = communicator.communicate(sync_tensors, global_step + 1)
        gate.finalize_synced_tensors(sync_tensors)

        optimizer.step()
        scheduler.step()
        global_step += 1

        loss_accum += micro_loss
        loss_count += 1

        if global_step % args.log_interval == 0 and is_main_process():
            now = time.perf_counter()
            step_time_ms = (now - prev_step_time) * 1000.0 / max(1, args.log_interval)
            prev_step_time = now
            elapsed = time.time() - start_time
            row = {
                "step": global_step,
                "kind": "train",
                "policy": args.materialization_policy,
                "train_loss": loss_accum / max(1, loss_count),
                "elapsed_sec": elapsed,
                "step_time_ms": step_time_ms,
                "synced_groups": gate_stats.synced_groups,
                "residual_groups": gate_stats.residual_groups,
                "low_importance_synced_groups": gate_stats.low_importance_synced_groups,
                **asdict(comm_stats),
            }
            append_dict_csv(metrics_path, row)
            print(
                f"[train] step={global_step} policy={args.materialization_policy} "
                f"loss={row['train_loss']:.6f} cv={comm_stats.bucket_imbalance_cv:.4f} "
                f"ratio={comm_stats.bucket_imbalance_ratio:.3f} comm_ms={comm_stats.total_comm_time_ms:.3f}"
            )
            loss_accum = 0.0
            loss_count = 0

        if args.eval_interval > 0 and (global_step % args.eval_interval == 0 or global_step == args.max_steps):
            barrier()
            eval_sampler.set_epoch(global_step)
            eval_metrics = evaluate(model, eval_loader, device)
            barrier()
            if is_main_process():
                if eval_metrics["val_loss"] < best["best_val_loss"]:
                    best = {
                        "best_step": global_step,
                        "best_val_loss": eval_metrics["val_loss"],
                        "best_val_acc": eval_metrics["val_acc"],
                        "best_val_ppl": eval_metrics["val_ppl"],
                    }
                append_dict_csv(metrics_path, {
                    "step": global_step,
                    "kind": "eval",
                    "policy": args.materialization_policy,
                    "val_loss": eval_metrics["val_loss"],
                    "val_ppl": eval_metrics["val_ppl"],
                    "val_acc": eval_metrics["val_acc"],
                    "elapsed_sec": time.time() - start_time,
                })
                print(f"[eval] step={global_step} loss={eval_metrics['val_loss']:.6f} acc={eval_metrics['val_acc']:.4f}")

    if is_main_process():
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({"args": vars(args), "best": best}, f, indent=2, ensure_ascii=False)
        print(f"[done] metrics={metrics_path}")

    barrier()
    cleanup_dist()


if __name__ == "__main__":
    main()
