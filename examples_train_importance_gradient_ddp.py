from __future__ import annotations

import os
import csv
import json
import math
import time
import random
import argparse
from typing import Dict, Optional

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM

from importance_gradient.bucket_runtime_planner import RiskAwareBucketPlanner
from importance_gradient.real_bucket_comm import BalancedBucketCommunicator, group_allreduce_sync_tensors
from importance_gradient.periodic_sync_gate import PeriodicSyncGate, load_importance_profile
from importance_gradient.metrics_schema import TrainEvalRow


def set_seed(seed: int):
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


def barrier():
    if is_dist():
        dist.barrier()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def ddp_setup(args):
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    if local_rank < 0:
        local_rank = 0

    if not is_dist():
        dist.init_process_group(backend="nccl")

    visible = torch.cuda.device_count()
    if visible == 0:
        raise RuntimeError("No visible CUDA devices found.")
    if local_rank >= visible:
        raise RuntimeError(f"LOCAL_RANK={local_rank} but only {visible} visible GPUs.")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    return local_rank, device


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


def build_text(example):
    for key in ["abstract", "text", "article", "content", "instruction", "output"]:
        if key in example and example[key] is not None:
            if key == "instruction":
                out = str(example.get("output", "")).strip()
                inst = str(example.get("instruction", "")).strip()
                return f"{inst} {out}".strip()
            return str(example[key]).strip()
    parts = []
    for _, v in example.items():
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
    return " ".join(parts).strip()


def tokenize_function(example, tokenizer, max_length=256):
    text = build_text(example)
    encoded = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    labels = encoded["input_ids"].copy()
    labels = [
        token if mask == 1 else -100
        for token, mask in zip(labels, encoded["attention_mask"])
    ]
    encoded["labels"] = labels
    return encoded


def build_dataloaders(args, tokenizer):
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

    if args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(min(args.max_train_samples, len(train_dataset))))
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


@torch.no_grad()
def evaluate(model, eval_loader, device):
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
    return {
        "val_loss": avg_loss,
        "val_ppl": avg_ppl,
        "val_acc": avg_acc,
        "val_tokens": int(total_tokens.item()),
    }


def append_csv(csv_path: str, row: TrainEvalRow):
    row_dict = row.to_dict()
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--local_rank", type=int, default=-1)

    # model / data
    parser.add_argument("--model_name_or_path", type=str, default="facebook/opt-1.3b")
    parser.add_argument("--dataset_name", type=str, default="uiyunkim-hub/pubmed-abstract")
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--local_dataset_path", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--max_train_samples", type=int, default=100000)
    parser.add_argument("--max_eval_samples", type=int, default=2000)

    # training
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # innovation 1: periodic + residual
    parser.add_argument("--score_csv", type=str, required=True)
    parser.add_argument("--sync_mode", type=str, default="periodic", choices=["full", "periodic"])
    parser.add_argument("--low_importance_period", type=int, default=4)
    parser.add_argument("--use_residual_accumulation", action="store_true")

    # innovation 2: real bucket communication
    parser.add_argument("--comm_backend", type=str, default="group_allreduce",
                        choices=["group_allreduce", "balanced_bucket_real"])
    parser.add_argument("--bucket_num", type=int, default=4)
    parser.add_argument("--bucket_block_size_numel", type=int, default=262144)
    parser.add_argument("--bucket_scheduler_mode", type=str, default="risk_aware",
                        choices=["round_robin", "lightest", "risk_aware"])
    parser.add_argument("--bucket_use_adaptive_switch", action="store_true")
    parser.add_argument("--bucket_adapt_uncertainty_threshold", type=float, default=0.15)
    parser.add_argument("--bucket_lambda_std", type=float, default=0.15)
    parser.add_argument("--bucket_gamma_overflow", type=float, default=1.0)
    parser.add_argument("--bucket_target_margin", type=float, default=1.05)

    # D2/D3: effective cost / effective payload
    parser.add_argument("--bucket_cost_mode", type=str, default="dense",
                        choices=["dense", "effective_sparse"])
    parser.add_argument("--effective_low_cost_ratio", type=float, default=0.25)

    parser.add_argument("--payload_mode", type=str, default="dense",
                        choices=["dense", "effective_payload"])
    parser.add_argument("--effective_payload_low_keep_ratio", type=float, default=1.0)
    parser.add_argument("--effective_payload_rotation_interval", type=int, default=4)

    # output
    parser.add_argument("--output_dir", type=str, required=True)

    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    set_seed(args.seed)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    local_rank, device = ddp_setup(args)

    if is_main_process():
        print(json.dumps(vars(args), indent=2, ensure_ascii=False))
        print(f"[OutputDir] {args.output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        local_files_only=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_loader, eval_loader, train_sampler, eval_sampler = build_dataloaders(args, tokenizer)

    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    load_dtype = dtype_map[args.dtype]

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=load_dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.to(device)
    model.train()

    no_decay = ["bias", "layer_norm", "layernorm", "ln_f", "norm"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad and not any(nd in n.lower() for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad and any(nd in n.lower() for nd in no_decay)
            ],
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
        bucket_cost_mode=args.bucket_cost_mode,
        effective_low_cost_ratio=args.effective_low_cost_ratio,
        payload_mode=args.payload_mode,
        effective_payload_low_keep_ratio=args.effective_payload_low_keep_ratio,
        effective_payload_rotation_interval=args.effective_payload_rotation_interval,
    )

    csv_path = os.path.join(args.output_dir, "metrics.csv")
    best_summary_path = os.path.join(args.output_dir, "best_summary.json")
    args_path = os.path.join(args.output_dir, "run_args.json")

    if is_main_process():
        with open(args_path, "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, ensure_ascii=False)

    global_step = 0
    train_iter = iter(train_loader)
    step_loss_accum = 0.0
    step_loss_count = 0
    start_time = time.time()

    best = {
        "best_step": -1,
        "best_val_loss": float("inf"),
        "best_val_ppl": None,
        "best_val_acc": None,
    }

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
            micro_loss += loss.item()

        if args.max_grad_norm is not None and args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        # Innovation 1: decide which gradients are really communicated this step
        sync_tensors, gate_stats = gate.prepare_sync_tensors(global_step + 1)

        # Innovation 2: choose the real communication organization
        if args.comm_backend == "group_allreduce":
            comm_stats = group_allreduce_sync_tensors(sync_tensors, global_step + 1)
        elif args.comm_backend == "balanced_bucket_real":
            comm_stats = bucket_communicator.communicate(sync_tensors, global_step + 1)
        else:
            raise ValueError(args.comm_backend)

        gate.finalize_synced_tensors(sync_tensors)

        optimizer.step()
        scheduler.step()
        global_step += 1

        avg_train_loss = micro_loss
        step_loss_accum += avg_train_loss
        step_loss_count += 1

        if global_step % args.log_interval == 0 and is_main_process():
            elapsed = time.time() - start_time
            train_loss_mean = step_loss_accum / max(1, step_loss_count)

            row = TrainEvalRow(
                step=global_step,
                kind="train",
                train_loss=train_loss_mean,
                elapsed_sec=elapsed,

                synced_groups=gate_stats.synced_groups,
                synced_params=gate_stats.synced_params,
                synced_bytes_est=gate_stats.synced_bytes_est,
                residual_groups=gate_stats.residual_groups,
                residual_params=gate_stats.residual_params,
                residual_bytes=gate_stats.residual_bytes,
                low_importance_synced_groups=gate_stats.low_importance_synced_groups,

                bucket_mode_requested=comm_stats.mode_requested,
                bucket_mode_effective=comm_stats.mode_effective,
                bucket_global_uncertainty=comm_stats.global_uncertainty,
                bucket_imbalance_ratio=comm_stats.bucket_imbalance_ratio,
                bucket_imbalance_cv=comm_stats.bucket_imbalance_cv,
                bucket_overflow_count=comm_stats.bucket_overflow_count,
                bucket_pack_time_ms=comm_stats.pack_time_ms,
                bucket_allreduce_time_ms=comm_stats.allreduce_time_ms,
                bucket_unpack_time_ms=comm_stats.unpack_time_ms,
                bucket_total_comm_time_ms=comm_stats.total_comm_time_ms,
                bucket_count=comm_stats.bucket_count,
                bucket_block_count=comm_stats.block_count,
                bucket_communicated_numel=comm_stats.communicated_numel,
                bucket_communicated_bytes=comm_stats.communicated_bytes,
            )
            append_csv(csv_path, row)

            print(
                f"[train] step={global_step} loss={train_loss_mean:.6f} "
                f"sync_groups={gate_stats.synced_groups} res_groups={gate_stats.residual_groups} "
                f"comm={args.comm_backend} "
                f"bucket_mode={comm_stats.mode_effective} "
                f"bucket_comm_ms={comm_stats.total_comm_time_ms:.3f}"
            )
            step_loss_accum = 0.0
            step_loss_count = 0

        if global_step % args.eval_interval == 0 or global_step == args.max_steps:
            barrier()
            eval_sampler.set_epoch(global_step)
            eval_metrics = evaluate(model, eval_loader, device)
            barrier()

            if is_main_process():
                elapsed = time.time() - start_time
                row = TrainEvalRow(
                    step=global_step,
                    kind="eval",
                    val_loss=eval_metrics["val_loss"],
                    val_ppl=eval_metrics["val_ppl"],
                    val_acc=eval_metrics["val_acc"],
                    elapsed_sec=elapsed,
                )
                append_csv(csv_path, row)

                print(
                    f"[eval] step={global_step} val_loss={eval_metrics['val_loss']:.6f} "
                    f"val_ppl={eval_metrics['val_ppl']:.4f} "
                    f"val_acc={eval_metrics['val_acc']:.4f}"
                )

                if eval_metrics["val_loss"] < best["best_val_loss"]:
                    best["best_step"] = global_step
                    best["best_val_loss"] = eval_metrics["val_loss"]
                    best["best_val_ppl"] = eval_metrics["val_ppl"]
                    best["best_val_acc"] = eval_metrics["val_acc"]
                    with open(best_summary_path, "w", encoding="utf-8") as f:
                        json.dump(best, f, indent=2, ensure_ascii=False)

    barrier()
    if is_main_process():
        print("\nBest summary:")
        print(json.dumps(best, indent=2, ensure_ascii=False))

    if is_dist():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
