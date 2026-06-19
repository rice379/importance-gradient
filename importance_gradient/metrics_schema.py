from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class GateStats:
    step: int
    synced_groups: int = 0
    synced_params: int = 0
    synced_bytes_est: int = 0
    residual_groups: int = 0
    residual_params: int = 0
    residual_bytes: int = 0
    low_importance_synced_groups: int = 0
    numel_seen: int = 0


@dataclass
class BucketCommStats:
    step: int
    mode_requested: str = "group_allreduce"
    mode_effective: str = "group_allreduce"
    global_uncertainty: float = 0.0

    bucket_imbalance_ratio: float = 1.0
    bucket_imbalance_cv: float = 0.0
    bucket_overflow_count: int = 0

    pack_time_ms: float = 0.0
    allreduce_time_ms: float = 0.0
    unpack_time_ms: float = 0.0
    total_comm_time_ms: float = 0.0

    bucket_count: int = 0
    block_count: int = 0
    communicated_numel: int = 0
    communicated_bytes: int = 0


@dataclass
class TrainEvalRow:
    step: int
    kind: str  # train / eval

    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    val_ppl: Optional[float] = None
    val_acc: Optional[float] = None

    elapsed_sec: Optional[float] = None

    # gate-side
    synced_groups: Optional[int] = None
    synced_params: Optional[int] = None
    synced_bytes_est: Optional[int] = None
    residual_groups: Optional[int] = None
    residual_params: Optional[int] = None
    residual_bytes: Optional[int] = None
    low_importance_synced_groups: Optional[int] = None

    # bucket-side / comm-side
    bucket_mode_requested: Optional[str] = None
    bucket_mode_effective: Optional[str] = None
    bucket_global_uncertainty: Optional[float] = None
    bucket_imbalance_ratio: Optional[float] = None
    bucket_imbalance_cv: Optional[float] = None
    bucket_overflow_count: Optional[int] = None
    bucket_pack_time_ms: Optional[float] = None
    bucket_allreduce_time_ms: Optional[float] = None
    bucket_unpack_time_ms: Optional[float] = None
    bucket_total_comm_time_ms: Optional[float] = None
    bucket_count: Optional[int] = None
    bucket_block_count: Optional[int] = None
    bucket_communicated_numel: Optional[int] = None
    bucket_communicated_bytes: Optional[int] = None

    def to_dict(self):
        return asdict(self)
