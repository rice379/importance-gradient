from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import math
import statistics


@dataclass
class BlockMeta:
    block_id: int
    param_name: str
    group_id: object
    start: int
    end: int
    numel: int
    est_cost_mean: float
    est_cost_std: float
    nnz_proxy: int


@dataclass
class BucketState:
    bucket_id: int
    load_mean: float = 0.0
    load_var: float = 0.0
    blocks: List[int] = field(default_factory=list)

    @property
    def load_std(self) -> float:
        return math.sqrt(max(self.load_var, 0.0))


@dataclass
class BucketPlan:
    step: int
    mode_requested: str
    mode_effective: str
    global_uncertainty: float
    bucket_states: List[BucketState]
    assignment: Dict[int, int]  # block_id -> bucket_id


class CostHistory:
    def __init__(self, max_history: int = 32):
        self.max_history = max_history
        self._history: Dict[int, List[float]] = {}

    def update(self, block_id: int, observed_cost: float) -> None:
        hist = self._history.setdefault(block_id, [])
        hist.append(float(observed_cost))
        if len(hist) > self.max_history:
            del hist[0]

    def get_mean_std(self, block_id: int) -> Tuple[float, float]:
        hist = self._history.get(block_id)
        if not hist:
            return 0.0, 0.0
        if len(hist) == 1:
            return hist[0], 0.0
        return statistics.mean(hist), statistics.pstdev(hist)


class RiskAwareBucketPlanner:
    """
    Your method:
      - requested mode can be risk_aware / lightest / round_robin
      - if requested mode == risk_aware and uncertainty is low,
        the effective mode degrades to lightest.
    """

    def __init__(
        self,
        bucket_num: int = 4,
        mode_requested: str = "risk_aware",
        use_adaptive_switch: bool = True,
        uncertainty_threshold: float = 0.15,
        lambda_std: float = 0.15,
        gamma_overflow: float = 1.0,
        target_margin: float = 1.05,
        history_size: int = 32,
    ):
        self.bucket_num = bucket_num
        self.mode_requested = mode_requested
        self.use_adaptive_switch = use_adaptive_switch
        self.uncertainty_threshold = uncertainty_threshold
        self.lambda_std = lambda_std
        self.gamma_overflow = gamma_overflow
        self.target_margin = target_margin
        self.cost_history = CostHistory(max_history=history_size)

    def update_history(self, block_id: int, observed_cost: float) -> None:
        self.cost_history.update(block_id, observed_cost)

    def estimate_cost(self, block: BlockMeta) -> Tuple[float, float]:
        hist_mean, hist_std = self.cost_history.get_mean_std(block.block_id)
        if hist_mean > 0:
            return hist_mean, hist_std
        return block.est_cost_mean, block.est_cost_std

    def compute_global_uncertainty(self, blocks: List[BlockMeta]) -> float:
        ratios = []
        for b in blocks:
            mu, sigma = self.estimate_cost(b)
            if mu > 0:
                ratios.append(sigma / mu)
        if not ratios:
            return 0.0
        return float(sum(ratios) / len(ratios))

    def choose_bucket_round_robin(self, block: BlockMeta, bucket_states: List[BucketState]) -> int:
        return block.block_id % len(bucket_states)

    def choose_bucket_lightest(self, block: BlockMeta, bucket_states: List[BucketState]) -> int:
        best_idx = 0
        best_score = float("inf")
        for i, st in enumerate(bucket_states):
            if st.load_mean < best_score:
                best_score = st.load_mean
                best_idx = i
        return best_idx

    def _overflow_penalty(self, mu: float, bucket: BucketState, current_target: float) -> float:
        next_load = bucket.load_mean + mu
        return 1.0 if next_load > current_target * self.target_margin else 0.0

    def choose_bucket_risk_aware(
        self,
        block: BlockMeta,
        bucket_states: List[BucketState],
        current_target: float,
    ) -> int:
        mu, sigma = self.estimate_cost(block)
        best_idx = 0
        best_score = float("inf")

        for i, st in enumerate(bucket_states):
            penalty = self._overflow_penalty(mu, st, current_target)
            score = st.load_mean + self.lambda_std * (st.load_std + sigma) + self.gamma_overflow * penalty
            if score < best_score:
                best_score = score
                best_idx = i
        return best_idx

    def plan(self, blocks: List[BlockMeta], step: int) -> BucketPlan:
        bucket_states = [BucketState(bucket_id=i) for i in range(self.bucket_num)]
        blocks = sorted(blocks, key=lambda x: x.est_cost_mean, reverse=True)

        U = self.compute_global_uncertainty(blocks)
        mode_effective = self.mode_requested

        if self.mode_requested == "risk_aware" and self.use_adaptive_switch:
            if U < self.uncertainty_threshold:
                mode_effective = "lightest"

        total_mu = sum(self.estimate_cost(b)[0] for b in blocks)
        current_target = total_mu / max(len(bucket_states), 1)

        assignment: Dict[int, int] = {}
        for block in blocks:
            if mode_effective == "round_robin":
                bid = self.choose_bucket_round_robin(block, bucket_states)
            elif mode_effective == "lightest":
                bid = self.choose_bucket_lightest(block, bucket_states)
            elif mode_effective == "risk_aware":
                bid = self.choose_bucket_risk_aware(block, bucket_states, current_target)
            else:
                raise ValueError(f"Unknown bucket mode: {mode_effective}")

            mu, sigma = self.estimate_cost(block)
            st = bucket_states[bid]
            st.blocks.append(block.block_id)
            st.load_mean += mu
            st.load_var += sigma * sigma
            assignment[block.block_id] = bid

        return BucketPlan(
            step=step,
            mode_requested=self.mode_requested,
            mode_effective=mode_effective,
            global_uncertainty=U,
            bucket_states=bucket_states,
            assignment=assignment,
        )

    @staticmethod
    def compute_plan_metrics(plan: BucketPlan) -> Tuple[float, float, int]:
        loads = [b.load_mean for b in plan.bucket_states]
        if not loads:
            return 1.0, 0.0, 0

        mx = max(loads)
        mn = min(loads)
        mean = sum(loads) / len(loads)

        imbalance_ratio = mx / max(mn, 1e-12)
        cv = 0.0
        if mean > 0.0 and len(loads) > 1:
            var = sum((x - mean) ** 2 for x in loads) / len(loads)
            cv = math.sqrt(var) / mean

        threshold = mean * 1.05
        overflow_count = sum(1 for x in loads if x > threshold)
        return imbalance_ratio, cv, overflow_count
