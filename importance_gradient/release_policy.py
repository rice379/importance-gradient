"""
Minimal runtime-policy patch for RQ3.

Integrate the logic below into the point where your code already has retained
sparse gradients after Top-k and before collective communication.

Required CLI args in your training entry:
  parser.add_argument('--release_period', type=int, default=4)
  parser.add_argument('--use_residual_compensation', action='store_true')
  parser.add_argument('--disable_residual_compensation', action='store_true')
  parser.add_argument('--rq3_policy_name', type=str, default='')

Expected behavior:
  - important components: synchronize every iteration
  - low-importance components: synchronize every R_low iterations
  - with residual compensation: accumulate skipped retained updates and merge on release
  - without residual compensation: discard skipped updates
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, MutableMapping, Tuple

import torch


@dataclass
class RQ3ReleasePolicy:
    important_components: set[str]
    low_importance_components: set[str]
    release_period: int = 4
    use_residual_compensation: bool = True
    residuals: MutableMapping[str, torch.Tensor] = field(default_factory=dict)

    def is_important(self, component_id: str) -> bool:
        return component_id in self.important_components

    def is_low_importance(self, component_id: str) -> bool:
        return component_id in self.low_importance_components

    def should_release_low(self, global_step: int) -> bool:
        if self.release_period <= 1:
            return True
        # Use 1-based release semantics: release at steps R, 2R, 3R, ...
        return (global_step + 1) % self.release_period == 0

    @torch.no_grad()
    def process_component_update(
        self,
        component_id: str,
        retained_update: torch.Tensor,
        global_step: int,
    ) -> Tuple[bool, torch.Tensor | None]:
        """Return (should_communicate, update_to_communicate_or_None)."""
        if self.is_important(component_id):
            return True, retained_update

        if not self.is_low_importance(component_id):
            # Conservative fallback: communicate unknown components every iteration.
            return True, retained_update

        if self.should_release_low(global_step):
            if self.use_residual_compensation:
                residual = self.residuals.pop(component_id, None)
                if residual is not None:
                    retained_update = retained_update + residual.to(retained_update.device)
            return True, retained_update

        # Not a release iteration for this low-importance component.
        if self.use_residual_compensation:
            if component_id not in self.residuals:
                self.residuals[component_id] = retained_update.detach().clone().cpu()
            else:
                self.residuals[component_id].add_(retained_update.detach().cpu())
        # Without residual compensation, the deferred update is dropped.
        return False, None


def example_integration_loop(policy: RQ3ReleasePolicy, retained_by_component: Mapping[str, torch.Tensor], global_step: int):
    """Pseudo-integration for your sparse-gradient communication path."""
    to_communicate: Dict[str, torch.Tensor] = {}
    for component_id, retained_update in retained_by_component.items():
        should_comm, update = policy.process_component_update(component_id, retained_update, global_step)
        if should_comm and update is not None:
            to_communicate[component_id] = update

    # Your existing code should pack/synchronize `to_communicate` here.
    # For the RQ3 frequency sweep, balanced packing can be disabled unless you
    # want to measure the full system; keep it fixed across all R_low settings.
    return to_communicate
