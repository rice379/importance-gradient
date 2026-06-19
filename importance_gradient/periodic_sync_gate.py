from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Any

import pandas as pd
import torch

from importance_gradient.component_mapper import parse_opt_component
from importance_gradient.metrics_schema import GateStats
from importance_gradient.real_bucket_comm import SyncTensorRef


def load_importance_profile(csv_path: str) -> Tuple[Dict[Tuple[int, str], float], float]:
    df = pd.read_csv(csv_path)
    fieldnames = list(df.columns)

    def pick(cols):
        for c in cols:
            if c in fieldnames:
                return c
        return None

    layer_col = pick(["layer", "layer_id"])
    comp_col = pick(["component", "comp"])
    score_col = pick(["keep_score", "score", "final_score", "importance", "final_value"])
    tau_col = pick(["global_tau", "tau", "threshold_tau"])

    if layer_col is None or comp_col is None or score_col is None:
        raise ValueError(f"CSV columns do not match requirement: {fieldnames}")
    if tau_col is None:
        raise ValueError(f"CSV must contain a global tau column: {fieldnames}")

    tau_series = df[tau_col].dropna()
    if len(tau_series) == 0:
        raise ValueError("Failed to read global_tau from CSV")
    global_tau = float(tau_series.iloc[0])

    score_dict: Dict[Tuple[int, str], float] = {}
    for _, row in df.iterrows():
        layer = int(row[layer_col])
        comp = str(row[comp_col]).strip().upper()
        score = float(row[score_col])
        score_dict[(layer, comp)] = score

    return score_dict, global_tau


class PeriodicSyncGate:
    """
    Gate module for the first innovation:
      - merge residuals
      - decide sync vs defer
      - return sync tensors
    It does NOT perform real communication.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        score_dict: Dict[Tuple[int, str], float],
        global_tau: float,
        low_importance_period: int = 4,
        use_residual_accumulation: bool = True,
        always_sync_non_component: bool = True,
        sync_mode: str = "periodic",
    ):
        self.score_dict = score_dict
        self.global_tau = global_tau
        self.low_importance_period = max(1, int(low_importance_period))
        self.use_residual_accumulation = use_residual_accumulation
        self.always_sync_non_component = always_sync_non_component
        self.sync_mode = sync_mode

        self.group_to_param_names: Dict[Any, List[str]] = {}
        self.name_to_param: Dict[str, torch.nn.Parameter] = {}
        self.group_period: Dict[Any, int] = {}
        self.group_is_low_importance: Dict[Any, bool] = {}
        self.residual_buffers: Dict[str, torch.Tensor] = {}

        self._build(model)

    def _build(self, model: torch.nn.Module):
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue

            cid = parse_opt_component(name)
            group_id = cid if cid is not None else ("OTHER", name)

            self.group_to_param_names.setdefault(group_id, []).append(name)
            self.name_to_param[name] = p

        for group_id in self.group_to_param_names:
            if self.sync_mode == "full":
                self.group_period[group_id] = 1
                self.group_is_low_importance[group_id] = False
                continue

            if isinstance(group_id, tuple) and len(group_id) == 2 and isinstance(group_id[0], int):
                score = self.score_dict.get(group_id, None)
                is_low = (score is None) or (score < self.global_tau)
                self.group_is_low_importance[group_id] = is_low
                self.group_period[group_id] = self.low_importance_period if is_low else 1
            else:
                self.group_is_low_importance[group_id] = False
                self.group_period[group_id] = 1 if self.always_sync_non_component else self.low_importance_period

        if self.use_residual_accumulation:
            for group_id, names in self.group_to_param_names.items():
                if self.group_period[group_id] <= 1:
                    continue
                for n in names:
                    p = self.name_to_param[n]
                    self.residual_buffers[n] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

    def group_should_sync(self, group_id: Any, global_step: int) -> bool:
        period = self.group_period[group_id]
        if period <= 1:
            return True
        return (global_step % period) == 0

    def prepare_sync_tensors(self, global_step: int) -> tuple[List[SyncTensorRef], GateStats]:
        stats = GateStats(step=global_step)
        sync_tensors: List[SyncTensorRef] = []

        for group_id, names in self.group_to_param_names.items():
            need_sync = self.group_should_sync(group_id, global_step)
            is_low = self.group_is_low_importance.get(group_id, False)

            active = False
            for name in names:
                p = self.name_to_param[name]
                if p.grad is None:
                    continue

                active = True
                stats.numel_seen += p.grad.numel()

                # merge residual before any sync/defer decision
                if name in self.residual_buffers:
                    p.grad.data.add_(self.residual_buffers[name])

                if need_sync:
                    sync_tensors.append(
                        SyncTensorRef(
                            name=name,
                            group_id=group_id,
                            tensor=p.grad.data,
                            is_low_importance=is_low,
                        )
                    )
                    stats.synced_params += 1
                    stats.synced_bytes_est += p.grad.numel() * p.grad.element_size()
                else:
                    stats.residual_params += 1
                    stats.residual_bytes += p.grad.numel() * p.grad.element_size()
                    if name in self.residual_buffers:
                        self.residual_buffers[name].copy_(p.grad.data)
                    p.grad.data.zero_()

            if not active:
                continue

            if need_sync:
                stats.synced_groups += 1
                if is_low:
                    stats.low_importance_synced_groups += 1
            else:
                stats.residual_groups += 1

        return sync_tensors, stats

    def finalize_synced_tensors(self, sync_tensors: List[SyncTensorRef]) -> None:
        for ref in sync_tensors:
            if ref.name in self.residual_buffers:
                self.residual_buffers[ref.name].zero_()
