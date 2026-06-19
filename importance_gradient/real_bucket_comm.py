# real_bucket_comm.py
# v6 optimized effective-payload version:
#   1) rank0-only bucket planning + broadcast assignment to all ranks
#   2) layout/bucket-size checks are cached by repeated layout pattern
#   3) persistent per-bucket buffers avoid torch.empty every step
#   4) CUDA synchronize around timing sections
#   5) per-block history update

from __future__ import annotations

import json
import hashlib
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.distributed as dist

from importance_gradient.bucket_runtime_planner import (
    BlockMeta,
    RiskAwareBucketPlanner,
)
from importance_gradient.metrics_schema import BucketCommStats


@dataclass
class SyncTensorRef:
    """
    A gradient tensor selected by PeriodicSyncGate for communication.

    name:
        Full parameter name.
    group_id:
        Component/group id, e.g., (layer, "Q") or ("OTHER", name).
    tensor:
        The gradient tensor to be communicated. This is usually p.grad.data.
    is_low_importance:
        Whether this tensor belongs to a low-importance group.
    """
    name: str
    group_id: Any
    tensor: torch.Tensor
    is_low_importance: bool


@dataclass
class GradBlockRef:
    """
    A block view into a gradient tensor.

    The actual communicated data is tensor.view(-1)[start:end].
    """
    block_id: int
    param_name: str
    group_id: Any
    is_low_importance: bool

    tensor: torch.Tensor
    flat_view: torch.Tensor
    start: int
    end: int
    numel: int


def split_tensor_ranges(numel: int, block_size_numel: int) -> List[Tuple[int, int]]:
    """
    Split a flat tensor of length numel into [start, end) ranges.
    """
    if block_size_numel <= 0:
        raise ValueError(f"block_size_numel must be positive, got {block_size_numel}")

    ranges: List[Tuple[int, int]] = []
    start = 0
    while start < numel:
        end = min(start + block_size_numel, numel)
        ranges.append((start, end))
        start = end
    return ranges


class BalancedBucketCommunicator:
    """
    Real bucketized communication path.

    Pipeline:
      1. Build GradBlockRef from selected SyncTensorRef.
      2. Check all ranks have identical block layout.
      3. Let rank0 build the bucket plan.
      4. Broadcast the bucket assignment to all ranks.
      5. Pack blocks in each planned bucket.
      6. Run real dist.all_reduce(flat_bucket).
      7. Unpack all-reduced bucket data back to original grad views.

    This class intentionally does not modify NCCL/DeepSpeed internals.
    It reorganizes the communication buffers at the Python reducer layer.
    """

    def __init__(
        self,
        planner: RiskAwareBucketPlanner,
        block_size_numel: int = 262144,
        device: Optional[torch.device] = None,
        enable_layout_checks: bool = True,
        enable_bucket_numel_checks: bool = True,
        enable_cuda_timing_sync: bool = True,
        layout_check_once_per_pattern: bool = True,
        bucket_numel_check_once_per_pattern: bool = True,
        enable_persistent_buffers: bool = True,
        enable_async_allreduce: bool = True,
        enable_plan_cache: bool = True,
        plan_cache_after_step: int = 5,

        # D2: cost model used by the planner.
        # dense: previous behavior.
        # effective_sparse: low-importance blocks use a smaller expected cost.
        bucket_cost_mode: str = "dense",
        effective_low_cost_ratio: float = 0.25,

        # D3: real payload mode.
        # dense: previous behavior.
        # effective_payload: rotate low-importance blocks through block-level residuals.
        payload_mode: str = "dense",
        effective_payload_low_keep_ratio: float = 1.0,
        effective_payload_rotation_interval: int = 4,
    ):
        self.planner = planner
        self.block_size_numel = int(block_size_numel)
        self.device = device

        # Debug/correctness guards. In performance mode, checks are cached
        # after the first occurrence of each repeated layout pattern.
        self.enable_layout_checks = bool(enable_layout_checks)
        self.enable_bucket_numel_checks = bool(enable_bucket_numel_checks)
        self.layout_check_once_per_pattern = bool(layout_check_once_per_pattern)
        self.bucket_numel_check_once_per_pattern = bool(bucket_numel_check_once_per_pattern)

        # Persistent bucket buffers: bucket_id -> capacity tensor.
        self.enable_persistent_buffers = bool(enable_persistent_buffers)
        self.enable_async_allreduce = bool(enable_async_allreduce)
        self._bucket_buffers: Dict[int, torch.Tensor] = {}

        # Plan cache keyed by layout hash + planner config.
        self.enable_plan_cache = bool(enable_plan_cache)
        self.plan_cache_after_step = int(plan_cache_after_step)
        self._plan_cache: Dict[str, Dict[str, Any]] = {}

        # Expensive check caches.
        self._checked_layout_hashes: set[str] = set()
        self._checked_bucket_numel_keys: set[Tuple[str, int]] = set()

        # For accurate timing of CUDA/NCCL work.
        self.enable_cuda_timing_sync = bool(enable_cuda_timing_sync)

        # D2: cost model for bucket planning.
        self.bucket_cost_mode = str(bucket_cost_mode)
        if self.bucket_cost_mode not in {"dense", "effective_sparse"}:
            raise ValueError(f"Unknown bucket_cost_mode={self.bucket_cost_mode}")
        self.effective_low_cost_ratio = float(effective_low_cost_ratio)

        # D3: real payload filtering.
        self.payload_mode = str(payload_mode)
        if self.payload_mode not in {"dense", "effective_payload"}:
            raise ValueError(f"Unknown payload_mode={self.payload_mode}")
        self.effective_payload_low_keep_ratio = float(effective_payload_low_keep_ratio)
        self.effective_payload_rotation_interval = max(1, int(effective_payload_rotation_interval))

        # Block-level residuals used only by D3/D4 effective_payload.
        # Key: (param_name, start, end) -> residual tensor.
        self._payload_residuals: Dict[Tuple[str, int, int], torch.Tensor] = {}

        # D4 optimization:
        # Track whether a residual buffer is active on the Python side.
        # This avoids torch.count_nonzero(res).item(), which causes a GPU-CPU sync.
        self._payload_residual_active: Dict[Tuple[str, int, int], bool] = {}

        # D4 optimization:
        # Cache low-importance keep decisions for repeated layouts/phases.
        # Key: (base_layout_hash, slots, phase_mod, keep_ratio)
        # Value: set(block_id) that should be transmitted.
        self._payload_keep_cache: Dict[Tuple[str, int, int, float], set[int]] = {}

    # ---------------------------------------------------------------------
    # Utility    # ---------------------------------------------------------------------
    # Utility
    # ---------------------------------------------------------------------

    def _cuda_sync(self) -> None:
        """
        Synchronize CUDA work for accurate timing.

        Without this, dist.all_reduce and tensor copies may only be enqueued,
        and measured time can under-report actual communication/copy latency.
        """
        if self.enable_cuda_timing_sync and torch.cuda.is_available():
            torch.cuda.synchronize()

    @staticmethod
    def _world_size() -> int:
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
        return 1

    @staticmethod
    def _rank() -> int:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
        return 0

    # ---------------------------------------------------------------------
    # Block construction
    # ---------------------------------------------------------------------

    def build_blocks_from_sync_tensors(self, sync_tensors: List[SyncTensorRef]) -> List[GradBlockRef]:
        """
        Split every selected tensor into fixed-size flat blocks.

        Important:
            This function must be deterministic and produce identical block_id
            ordering across ranks. That requires sync_tensors to be built in
            identical order across ranks.
        """
        block_refs: List[GradBlockRef] = []
        next_block_id = 0

        for ref in sync_tensors:
            if ref.tensor is None:
                continue
            if ref.tensor.numel() == 0:
                continue

            flat = ref.tensor.view(-1)
            for start, end in split_tensor_ranges(flat.numel(), self.block_size_numel):
                block_refs.append(
                    GradBlockRef(
                        block_id=next_block_id,
                        param_name=ref.name,
                        group_id=ref.group_id,
                        is_low_importance=bool(ref.is_low_importance),
                        tensor=ref.tensor,
                        flat_view=flat,
                        start=start,
                        end=end,
                        numel=end - start,
                    )
                )
                next_block_id += 1

        return block_refs

    def materialize_block_metas(self, block_refs: List[GradBlockRef]) -> List[BlockMeta]:
        """
        Convert GradBlockRef to planner-side BlockMeta.

        First runnable version:
            cost = dense bytes = numel * element_size

        Later, this can be replaced with effective sparse cost:
            cost ~= nnz * (value_bytes + index_bytes)
        """
        metas: List[BlockMeta] = []

        for br in block_refs:
            value_bytes = br.flat_view.element_size()
            dense_cost = float(br.numel * value_bytes)

            if self.bucket_cost_mode == "dense":
                est_cost_mean = dense_cost
            elif self.bucket_cost_mode == "effective_sparse":
                # D2: planner sees a smaller expected load for low-importance blocks.
                # This changes bucket planning only. Real bytes are reduced only when
                # payload_mode="effective_payload".
                ratio = self.effective_low_cost_ratio if br.is_low_importance else 1.0
                est_cost_mean = dense_cost * float(ratio)
            else:
                raise ValueError(f"Unknown bucket_cost_mode={self.bucket_cost_mode}")

            # The planner may override this with history if available.
            # Keep initial std=0 so risk_aware can safely fallback to lightest
            # before enough history is accumulated.
            est_cost_std = 0.0

            metas.append(
                BlockMeta(
                    block_id=int(br.block_id),
                    param_name=str(br.param_name),
                    group_id=br.group_id,
                    start=int(br.start),
                    end=int(br.end),
                    numel=int(br.numel),
                    est_cost_mean=est_cost_mean,
                    est_cost_std=est_cost_std,
                    nnz_proxy=int(br.numel),
                )
            )

        return metas

    # ---------------------------------------------------------------------
    # Cross-rank safety checks
    # ---------------------------------------------------------------------

    def _block_signature(self, block_refs: List[GradBlockRef]) -> List[Dict[str, Any]]:
        """
        Build a structural signature of the local block layout.

        It intentionally ignores gradient values and only compares layout.
        """
        sig: List[Dict[str, Any]] = []
        for br in block_refs:
            sig.append(
                {
                    "block_id": int(br.block_id),
                    "param_name": str(br.param_name),
                    "start": int(br.start),
                    "end": int(br.end),
                    "numel": int(br.numel),
                    "dtype": str(br.flat_view.dtype),
                    "is_low_importance": bool(br.is_low_importance),
                }
            )
        return sig


    @staticmethod
    def _hash_signature(sig: List[Dict[str, Any]]) -> str:
        sig_json = json.dumps(sig, sort_keys=True)
        return hashlib.sha1(sig_json.encode("utf-8")).hexdigest()

    def _layout_signature_and_hash(self, block_refs: List[GradBlockRef]) -> Tuple[List[Dict[str, Any]], str]:
        sig = self._block_signature(block_refs)
        return sig, self._hash_signature(sig)

    def _assert_same_block_layout_across_ranks(
        self,
        block_refs: List[GradBlockRef],
        step: int,
        sig: Optional[List[Dict[str, Any]]] = None,
        sig_hash: Optional[str] = None,
    ) -> str:
        """
        Fail early if ranks build different block layouts.

        v3: if this layout hash has already been checked once, skip the
        all_gather_object check to reduce per-step overhead.
        """
        if sig is None or sig_hash is None:
            sig, sig_hash = self._layout_signature_and_hash(block_refs)

        if not self.enable_layout_checks:
            return sig_hash
        if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
            return sig_hash

        if self.layout_check_once_per_pattern and sig_hash in self._checked_layout_hashes:
            return sig_hash

        local_info = {
            "rank": int(dist.get_rank()),
            "step": int(step),
            "num_blocks": int(len(sig)),
            "total_numel": int(sum(x["numel"] for x in sig)),
            "hash": sig_hash,
            "head": sig[:3],
            "tail": sig[-3:] if len(sig) >= 3 else sig,
        }

        gathered: List[Any] = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, local_info)

        hashes = set(x["hash"] for x in gathered)
        if len(hashes) != 1:
            msg = [
                f"[BucketLayoutMismatch] step={step}",
                "Different ranks constructed different block layouts before bucket planning.",
                "This would cause NCCL all_reduce size/order mismatch.",
                json.dumps(gathered, indent=2, ensure_ascii=False),
            ]
            raise RuntimeError("\n".join(msg))

        self._checked_layout_hashes.add(sig_hash)
        return sig_hash

    def _assert_same_bucket_numel(
        self,
        flat_buffer: torch.Tensor,
        bucket_id: int,
        step: int,
        layout_hash: str,
    ) -> None:
        """
        Fail early if a planned bucket has different tensor sizes across ranks.

        v3: check once for each (layout_hash, bucket_id) pair by default.
        """
        if not self.enable_bucket_numel_checks:
            return
        if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
            return

        key = (layout_hash, int(bucket_id))
        if self.bucket_numel_check_once_per_pattern and key in self._checked_bucket_numel_keys:
            return

        local = torch.tensor(
            [int(flat_buffer.numel())],
            device=flat_buffer.device,
            dtype=torch.long,
        )
        gathered = [torch.zeros_like(local) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, local)

        nums = [int(x.item()) for x in gathered]
        if len(set(nums)) != 1:
            raise RuntimeError(
                f"[BucketNumelMismatch] step={step}, bucket_id={bucket_id}, "
                f"rank={dist.get_rank()}, all_numel={nums}. "
                "All ranks must call all_reduce with the same tensor size."
            )

        self._checked_bucket_numel_keys.add(key)

    # ---------------------------------------------------------------------
    # Plan broadcasting    # ---------------------------------------------------------------------
    # Plan broadcasting
    # ---------------------------------------------------------------------

    def _plan_cache_key(self, layout_hash: str) -> str:
        return (
            f"layout={layout_hash}|"
            f"bucket_num={int(self.planner.bucket_num)}|"
            f"mode={str(self.planner.mode_requested)}|"
            f"adaptive={bool(getattr(self.planner, 'use_adaptive_switch', False))}|"
            f"threshold={float(getattr(self.planner, 'uncertainty_threshold', 0.0))}|"
            f"block_size={int(self.block_size_numel)}|"
            f"cost_mode={self.bucket_cost_mode}|"
            f"low_cost_ratio={self.effective_low_cost_ratio}|"
            f"payload_mode={self.payload_mode}|"
            f"payload_keep={self.effective_payload_low_keep_ratio}"
        )

    def _broadcast_plan_from_rank0(
        self,
        block_metas: List[BlockMeta],
        step: int,
        layout_hash: str,
    ) -> Dict[str, Any]:
        """
        Build bucket plan only on rank0 and broadcast it to all ranks.

        v3: optionally cache repeated-layout plans after plan_cache_after_step.
        """
        cache_key = self._plan_cache_key(layout_hash)
        if self.enable_plan_cache and cache_key in self._plan_cache:
            return self._plan_cache[cache_key]

        if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
            plan = self.planner.plan(block_metas, step=step)
            imbalance_ratio, cv, overflow_count = self.planner.compute_plan_metrics(plan)
            payload = {
                "mode_requested": plan.mode_requested,
                "mode_effective": plan.mode_effective,
                "global_uncertainty": float(plan.global_uncertainty),
                "assignment": {int(k): int(v) for k, v in plan.assignment.items()},
                "bucket_loads": [float(x.load_mean) for x in plan.bucket_states],
                "bucket_imbalance_ratio": float(imbalance_ratio),
                "bucket_imbalance_cv": float(cv),
                "bucket_overflow_count": int(overflow_count),
            }
            if self.enable_plan_cache and step >= self.plan_cache_after_step:
                self._plan_cache[cache_key] = payload
            return payload

        rank = dist.get_rank()
        if rank == 0:
            plan = self.planner.plan(block_metas, step=step)
            imbalance_ratio, cv, overflow_count = self.planner.compute_plan_metrics(plan)
            payload = {
                "mode_requested": plan.mode_requested,
                "mode_effective": plan.mode_effective,
                "global_uncertainty": float(plan.global_uncertainty),
                "assignment": {int(k): int(v) for k, v in plan.assignment.items()},
                "bucket_loads": [float(x.load_mean) for x in plan.bucket_states],
                "bucket_imbalance_ratio": float(imbalance_ratio),
                "bucket_imbalance_cv": float(cv),
                "bucket_overflow_count": int(overflow_count),
            }
        else:
            payload = None

        obj = [payload]
        dist.broadcast_object_list(obj, src=0)
        payload = obj[0]

        if self.enable_plan_cache and step >= self.plan_cache_after_step:
            self._plan_cache[cache_key] = payload

        return payload

    # ---------------------------------------------------------------------
    # Persistent buffers
    # ---------------------------------------------------------------------

    def _get_persistent_bucket_buffer(
        self,
        bucket_id: int,
        total_numel: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Return a flat buffer view with length total_numel.
        Reuses self._bucket_buffers[bucket_id] if capacity is sufficient.
        """
        bucket_id = int(bucket_id)
        total_numel = int(total_numel)

        if total_numel <= 0:
            raise ValueError("total_numel must be positive")

        if not self.enable_persistent_buffers:
            return torch.empty(total_numel, dtype=dtype, device=device)

        old = self._bucket_buffers.get(bucket_id, None)
        need_new = (
            old is None
            or old.numel() < total_numel
            or old.dtype != dtype
            or old.device != device
        )

        if need_new:
            self._bucket_buffers[bucket_id] = torch.empty(total_numel, dtype=dtype, device=device)

        return self._bucket_buffers[bucket_id][:total_numel]

    # ---------------------------------------------------------------------
    # Pack / all-reduce / unpack    # ---------------------------------------------------------------------
    # Pack / all-reduce / unpack
    # ---------------------------------------------------------------------

    def pack_bucket(
        self,
        bucket_id: int,
        bucket_block_refs: List[GradBlockRef],
    ) -> Tuple[torch.Tensor, List[Tuple[int, int, GradBlockRef]]]:
        """
        Copy bucket blocks into one contiguous flat buffer.
        """
        if not bucket_block_refs:
            raise ValueError("Cannot pack an empty bucket")

        total_numel = sum(x.numel for x in bucket_block_refs)
        dtype = bucket_block_refs[0].flat_view.dtype
        device = bucket_block_refs[0].flat_view.device

        flat_buffer = self._get_persistent_bucket_buffer(
            bucket_id=bucket_id,
            total_numel=total_numel,
            dtype=dtype,
            device=device,
        )
        mapping: List[Tuple[int, int, GradBlockRef]] = []

        cursor = 0
        for br in bucket_block_refs:
            segment = br.flat_view[br.start:br.end]
            flat_buffer[cursor:cursor + br.numel].copy_(segment)
            mapping.append((cursor, cursor + br.numel, br))
            cursor += br.numel

        return flat_buffer, mapping

    def allreduce_bucket(self, flat_buffer: torch.Tensor) -> None:
        """
        Real collective communication for one planned bucket.
        """
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            dist.all_reduce(flat_buffer, op=dist.ReduceOp.SUM)
            flat_buffer.div_(world_size)

    def launch_allreduce_bucket_async(self, flat_buffer: torch.Tensor):
        """
        Launch one bucket all-reduce asynchronously.

        The caller must wait on the returned work handle and divide the
        buffer by world_size after completion.
        """
        if dist.is_available() and dist.is_initialized():
            return dist.all_reduce(flat_buffer, op=dist.ReduceOp.SUM, async_op=True)
        return None

    def unpack_bucket(
        self,
        flat_buffer: torch.Tensor,
        mapping: List[Tuple[int, int, GradBlockRef]],
    ) -> None:
        """
        Scatter all-reduced bucket buffer back into original gradient views.
        """
        for buf_start, buf_end, br in mapping:
            br.flat_view[br.start:br.end].copy_(flat_buffer[buf_start:buf_end])

    # ---------------------------------------------------------------------
    # D4 optimized effective payload filtering
    # ---------------------------------------------------------------------

    @staticmethod
    def _payload_residual_key(br: GradBlockRef) -> Tuple[str, int, int]:
        return (str(br.param_name), int(br.start), int(br.end))

    def _get_payload_residual(self, br: GradBlockRef) -> torch.Tensor:
        key = self._payload_residual_key(br)
        res = self._payload_residuals.get(key)
        if (
            res is None
            or res.numel() != br.numel
            or res.dtype != br.flat_view.dtype
            or res.device != br.flat_view.device
        ):
            res = torch.zeros(br.numel, dtype=br.flat_view.dtype, device=br.flat_view.device)
            self._payload_residuals[key] = res
            self._payload_residual_active[key] = False
        return res

    def _payload_slots(self) -> int:
        keep_ratio = max(0.0, min(1.0, float(self.effective_payload_low_keep_ratio)))
        if keep_ratio >= 1.0:
            return 1
        if keep_ratio <= 0.0:
            # Effectively never transmit low-importance blocks.
            # Keep a very large slot count to make the rule deterministic.
            return 10**9
        return max(1, int(round(1.0 / keep_ratio)))

    def _payload_phase_mod(self, step: int, slots: int) -> int:
        if slots <= 1:
            return 0
        phase = int(step) // self.effective_payload_rotation_interval
        return int(phase % slots)

    def _compute_keep_block_ids_cached(
        self,
        base_layout_hash: str,
        block_refs: List[GradBlockRef],
        step: int,
    ) -> set[int]:
        """
        Return the set of block_ids that should be transmitted.

        High-importance blocks are always kept.
        Low-importance blocks are kept by a deterministic rotating schedule.

        D4 caches the decision set for repeated layout/phase patterns, avoiding
        repeated Python decision work. More importantly, the residual merge logic
        below no longer uses torch.count_nonzero(...).item().
        """
        if self.payload_mode == "dense":
            return {int(br.block_id) for br in block_refs}

        slots = self._payload_slots()
        if slots <= 1:
            return {int(br.block_id) for br in block_refs}

        phase_mod = self._payload_phase_mod(step, slots)
        keep_ratio_key = round(float(self.effective_payload_low_keep_ratio), 6)
        cache_key = (str(base_layout_hash), int(slots), int(phase_mod), float(keep_ratio_key))

        cached = self._payload_keep_cache.get(cache_key)
        if cached is not None:
            return cached

        keep_ids: set[int] = set()
        for br in block_refs:
            bid = int(br.block_id)
            if not br.is_low_importance:
                keep_ids.add(bid)
            else:
                if ((bid + phase_mod) % slots) == 0:
                    keep_ids.add(bid)

        self._payload_keep_cache[cache_key] = keep_ids
        return keep_ids

    def _apply_effective_payload_filter(
        self,
        block_refs: List[GradBlockRef],
        step: int,
        base_layout_hash: str,
    ) -> List[GradBlockRef]:
        """
        Apply D4 effective payload filtering and block-level residual accumulation.

        Main D4 optimization:
            The previous D3 version used torch.count_nonzero(res).item() to test
            whether a residual was active. That forces GPU-CPU synchronization.
            D4 replaces it with a Python-side active flag.

        Semantics:
            - High-importance blocks are always transmitted.
            - Low-importance blocks are transmitted according to a rotating schedule.
            - Skipped low-importance blocks are accumulated in residual buffers.
            - When a skipped block is eventually transmitted, the residual is merged.
        """
        if self.payload_mode == "dense":
            return block_refs

        keep_ids = self._compute_keep_block_ids_cached(
            base_layout_hash=base_layout_hash,
            block_refs=block_refs,
            step=step,
        )

        kept: List[GradBlockRef] = []

        for br in block_refs:
            bid = int(br.block_id)

            # High-importance blocks are kept and never need payload residual buffers.
            if not br.is_low_importance:
                kept.append(br)
                continue

            key = self._payload_residual_key(br)
            seg = br.flat_view[br.start:br.end]

            if bid in keep_ids:
                # Merge delayed payload residual only if the Python-side flag says it exists.
                if self._payload_residual_active.get(key, False):
                    res = self._get_payload_residual(br)
                    seg.add_(res)
                    res.zero_()
                    self._payload_residual_active[key] = False
                kept.append(br)
            else:
                # Skip this low-importance block in the current step.
                # Accumulate current gradient into residual. If a residual is already
                # active, add current segment; otherwise initialize it with current segment.
                res = self._get_payload_residual(br)
                if self._payload_residual_active.get(key, False):
                    res.add_(seg)
                else:
                    res.copy_(seg)
                    self._payload_residual_active[key] = True

                # Avoid unsynchronized local updates for skipped payload.
                seg.zero_()

        return kept

    def _clear_payload_residual_for_mapping(self, mapping: List[Tuple[int, int, GradBlockRef]]) -> None:
        """
        Clear residual state for blocks that were transmitted.

        This is mostly redundant with the merge-time clearing, but keeps the
        state safe if a residual is externally materialized before communication.
        """
        if self.payload_mode == "dense":
            return
        for _, _, br in mapping:
            if not br.is_low_importance:
                continue
            key = self._payload_residual_key(br)
            res = self._payload_residuals.get(key)
            if res is not None:
                res.zero_()
            self._payload_residual_active[key] = False

    # ---------------------------------------------------------------------
    # Main entry
    # ---------------------------------------------------------------------

    def communicate(self, sync_tensors: List[SyncTensorRef], step: int) -> BucketCommStats:
        """
        Execute real balanced bucket communication for the selected tensors.
        """
        if not sync_tensors:
            return BucketCommStats(step=int(step))

        self._cuda_sync()
        t0 = time.perf_counter()

        block_refs = self.build_blocks_from_sync_tensors(sync_tensors)
        if not block_refs:
            return BucketCommStats(step=int(step))

        # D4: compute the full pre-filter layout hash once. This is used to cache
        # payload keep decisions across repeated normal/release layouts.
        _, base_layout_hash = self._layout_signature_and_hash(block_refs)

        # D4: optionally reduce the real communicated payload for low-importance blocks.
        # This happens before filtered-layout hashing/planning so every rank communicates
        # the same filtered block layout.
        block_refs = self._apply_effective_payload_filter(
            block_refs=block_refs,
            step=step,
            base_layout_hash=base_layout_hash,
        )
        if not block_refs:
            return BucketCommStats(step=int(step))

        sig, layout_hash = self._layout_signature_and_hash(block_refs)
        self._assert_same_block_layout_across_ranks(
            block_refs=block_refs,
            step=step,
            sig=sig,
            sig_hash=layout_hash,
        )

        block_metas = self.materialize_block_metas(block_refs)
        plan_payload = self._broadcast_plan_from_rank0(
            block_metas=block_metas,
            step=step,
            layout_hash=layout_hash,
        )

        assignment = plan_payload["assignment"]
        block_ref_map = {int(x.block_id): x for x in block_refs}
        bucket_to_blocks: Dict[int, List[GradBlockRef]] = {
            i: [] for i in range(int(self.planner.bucket_num))
        }

        for raw_block_id, raw_bucket_id in assignment.items():
            block_id = int(raw_block_id)
            bucket_id = int(raw_bucket_id)

            if block_id not in block_ref_map:
                raise RuntimeError(
                    f"[PlanBlockMissing] step={step}, rank={self._rank()}, "
                    f"block_id={block_id} not found in local block_ref_map"
                )
            if bucket_id not in bucket_to_blocks:
                raise RuntimeError(
                    f"[InvalidBucketId] step={step}, rank={self._rank()}, "
                    f"bucket_id={bucket_id}, bucket_num={self.planner.bucket_num}"
                )

            bucket_to_blocks[bucket_id].append(block_ref_map[block_id])

        self._cuda_sync()
        t1 = time.perf_counter()

        total_comm_numel = 0
        total_comm_bytes = 0
        total_pack_time = 0.0
        total_allreduce_time = 0.0
        total_unpack_time = 0.0

        # -------------------------------------------------------------
        # v4 async path:
        #   1) pack all non-empty buckets
        #   2) launch all bucket all-reduces with async_op=True
        #   3) wait all handles
        #   4) average buffers
        #   5) unpack all buckets
        # -------------------------------------------------------------

        packed_buckets: List[Tuple[int, torch.Tensor, List[Tuple[int, int, GradBlockRef]]]] = []

        # Pack all buckets first.
        self._cuda_sync()
        t_pack0 = time.perf_counter()

        for bucket_id in range(int(self.planner.bucket_num)):
            refs = bucket_to_blocks[bucket_id]
            if not refs:
                continue

            flat_buffer, mapping = self.pack_bucket(bucket_id, refs)

            self._assert_same_bucket_numel(
                flat_buffer=flat_buffer,
                bucket_id=bucket_id,
                step=step,
                layout_hash=layout_hash,
            )

            total_comm_numel += int(flat_buffer.numel())
            total_comm_bytes += int(flat_buffer.numel() * flat_buffer.element_size())

            packed_buckets.append((bucket_id, flat_buffer, mapping))

        self._cuda_sync()
        t_pack1 = time.perf_counter()
        total_pack_time += (t_pack1 - t_pack0)

        # Launch all all-reduces before waiting.
        self._cuda_sync()
        t_ar0 = time.perf_counter()

        if self.enable_async_allreduce and dist.is_available() and dist.is_initialized():
            works: List[Tuple[Any, torch.Tensor]] = []
            for bucket_id, flat_buffer, mapping in packed_buckets:
                work = self.launch_allreduce_bucket_async(flat_buffer)
                works.append((work, flat_buffer))

            for work, flat_buffer in works:
                if work is not None:
                    work.wait()

            world_size = dist.get_world_size()
            for _, flat_buffer in works:
                flat_buffer.div_(world_size)
        else:
            # Synchronous fallback.
            for bucket_id, flat_buffer, mapping in packed_buckets:
                self.allreduce_bucket(flat_buffer)

        self._cuda_sync()
        t_ar1 = time.perf_counter()
        total_allreduce_time += (t_ar1 - t_ar0)

        # Unpack all buckets after all communication finishes.
        self._cuda_sync()
        t_up0 = time.perf_counter()

        for bucket_id, flat_buffer, mapping in packed_buckets:
            self.unpack_bucket(flat_buffer, mapping)
            self._clear_payload_residual_for_mapping(mapping)

            for _, _, br in mapping:
                observed_cost = float(br.numel * br.flat_view.element_size())
                self.planner.update_history(int(br.block_id), observed_cost)

        self._cuda_sync()
        t_up1 = time.perf_counter()
        total_unpack_time += (t_up1 - t_up0)

        self._cuda_sync()
        t2 = time.perf_counter()

        plan_overhead_ms = (t1 - t0) * 1000.0
        pack_time_ms = total_pack_time * 1000.0
        allreduce_time_ms = total_allreduce_time * 1000.0
        unpack_time_ms = total_unpack_time * 1000.0
        total_comm_time_ms = (t2 - t0) * 1000.0

        # Keep backward compatibility with the current CSV schema:
        # pack_time_ms includes non-allreduce overhead, i.e., plan/check + pack.
        pack_plus_plan_ms = pack_time_ms + plan_overhead_ms

        return BucketCommStats(
            step=int(step),
            mode_requested=str(plan_payload["mode_requested"]),
            mode_effective=str(plan_payload["mode_effective"]),
            global_uncertainty=float(plan_payload["global_uncertainty"]),
            bucket_imbalance_ratio=float(plan_payload["bucket_imbalance_ratio"]),
            bucket_imbalance_cv=float(plan_payload["bucket_imbalance_cv"]),
            bucket_overflow_count=int(plan_payload["bucket_overflow_count"]),
            pack_time_ms=float(pack_plus_plan_ms),
            allreduce_time_ms=float(allreduce_time_ms),
            unpack_time_ms=float(unpack_time_ms),
            total_comm_time_ms=float(total_comm_time_ms),
            bucket_count=int(self.planner.bucket_num),
            block_count=int(len(block_refs)),
            communicated_numel=int(total_comm_numel),
            communicated_bytes=int(total_comm_bytes),
        )



# -------------------------------------------------------------------------
# Baseline group all-reduce helper

# -------------------------------------------------------------------------
# Baseline group all-reduce helper
# -------------------------------------------------------------------------

def group_allreduce_sync_tensors(sync_tensors: List[SyncTensorRef], step: int = 0) -> BucketCommStats:
    """
    Baseline communication path used by train_importancecheck_real_bucket_ddp.py
    when --comm_backend group_allreduce is selected.

    It flattens all selected gradient tensors into one contiguous buffer,
    performs one real all_reduce, averages by world size, and scatters the
    result back to the original gradient tensors.

    This helper is intentionally kept in real_bucket_comm.py so the training
    script can import:

        from importance_gradient.real_bucket_comm import BalancedBucketCommunicator, group_allreduce_sync_tensors

    Returns:
        BucketCommStats with mode fields set to "group_allreduce".
    """
    if not sync_tensors:
        return BucketCommStats(
            step=int(step),
            mode_requested="group_allreduce",
            mode_effective="group_allreduce",
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    valid_refs = [ref for ref in sync_tensors if ref.tensor is not None and ref.tensor.numel() > 0]
    if not valid_refs:
        return BucketCommStats(
            step=int(step),
            mode_requested="group_allreduce",
            mode_effective="group_allreduce",
        )

    dtype = valid_refs[0].tensor.dtype
    device = valid_refs[0].tensor.device
    total_numel = sum(ref.tensor.numel() for ref in valid_refs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_pack0 = time.perf_counter()

    flat_buffer = torch.empty(total_numel, dtype=dtype, device=device)
    mapping: List[Tuple[int, int, SyncTensorRef]] = []

    cursor = 0
    for ref in valid_refs:
        n = ref.tensor.numel()
        flat_buffer[cursor:cursor + n].copy_(ref.tensor.view(-1))
        mapping.append((cursor, cursor + n, ref))
        cursor += n

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_pack1 = time.perf_counter()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_ar0 = time.perf_counter()

    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        dist.all_reduce(flat_buffer, op=dist.ReduceOp.SUM)
        flat_buffer.div_(world_size)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_ar1 = time.perf_counter()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_unpack0 = time.perf_counter()

    for start, end, ref in mapping:
        ref.tensor.view(-1).copy_(flat_buffer[start:end])

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_unpack1 = time.perf_counter()

    total_ms = (t_unpack1 - t0) * 1000.0
    pack_ms = (t_pack1 - t_pack0) * 1000.0
    ar_ms = (t_ar1 - t_ar0) * 1000.0
    unpack_ms = (t_unpack1 - t_unpack0) * 1000.0

    return BucketCommStats(
        step=int(step),
        mode_requested="group_allreduce",
        mode_effective="group_allreduce",
        global_uncertainty=0.0,
        bucket_imbalance_ratio=1.0,
        bucket_imbalance_cv=0.0,
        bucket_overflow_count=0,
        pack_time_ms=float(pack_ms),
        allreduce_time_ms=float(ar_ms),
        unpack_time_ms=float(unpack_ms),
        total_comm_time_ms=float(total_ms),
        bucket_count=1,
        block_count=len(valid_refs),
        communicated_numel=int(total_numel),
        communicated_bytes=int(total_numel * flat_buffer.element_size()),
    )
