"""Cluster-level allocation planning and actuator capability boundary.

The forward model can calculate a six-cluster allocation.  That does not mean
the field can execute six independent liquid commands.  This module keeps the
two concepts explicit:

* ``plan_cluster_allocation`` always works as a model/simulation output;
* ``execute_cluster_allocation`` refuses unless the configured actuator
  advertises ``direct_cluster_flow=True``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


class ClusterControlUnavailable(RuntimeError):
    """Raised when a direct cluster command has no physical actuator."""


@dataclass(frozen=True)
class ClusterActuatorCapabilities:
    direct_cluster_flow: bool = False
    stage_total_flow: bool = True
    supports_feedback_ack: bool = False
    actuator_name: str = "none"


@dataclass(frozen=True)
class ClusterAllocationPlan:
    stage_id: str
    total_rate_m3_s: float
    weights: tuple[float, ...]
    cluster_rates_m3_s: tuple[float, ...]
    mode: str = "model_only"


def normalize_cluster_weights(weights: Iterable[float], n_clusters: int | None = None) -> np.ndarray:
    values = np.asarray(list(weights), dtype=float)
    if n_clusters is not None and len(values) != int(n_clusters):
        raise ValueError("cluster weight count does not match n_clusters")
    values = np.clip(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    total = float(values.sum())
    if total <= 1.0e-12:
        return np.full(len(values), 1.0 / max(len(values), 1), dtype=float)
    return values / total


class ClusterAllocationController:
    def __init__(self, capabilities: ClusterActuatorCapabilities | None = None) -> None:
        self.capabilities = capabilities or ClusterActuatorCapabilities()

    def plan_cluster_allocation(self, stage_id: str, total_rate_m3_s: float, weights: Iterable[float]) -> ClusterAllocationPlan:
        normalized = normalize_cluster_weights(weights)
        total = max(float(total_rate_m3_s), 0.0)
        return ClusterAllocationPlan(
            stage_id=str(stage_id),
            total_rate_m3_s=total,
            weights=tuple(float(v) for v in normalized),
            cluster_rates_m3_s=tuple(float(v) for v in total * normalized),
            mode="direct_cluster_command" if self.capabilities.direct_cluster_flow else "model_only",
        )

    def execute_cluster_allocation(self, plan: ClusterAllocationPlan, *, approved: bool = False) -> dict[str, Any]:
        if not self.capabilities.direct_cluster_flow:
            raise ClusterControlUnavailable(
                "direct cluster liquid control requires a field actuator; "
                "current capability is model_only and can only plan/visualize allocation"
            )
        if not approved:
            raise ClusterControlUnavailable("manual approval is required before a cluster command")
        # The actual vendor implementation must be injected here.  Returning
        # an acknowledged command without a vendor adapter would be unsafe.
        if self.capabilities.actuator_name == "none":
            raise ClusterControlUnavailable("capability claims direct flow but no actuator adapter is configured")
        return {
            "accepted": True,
            "stage_id": plan.stage_id,
            "cluster_rates_m3_s": list(plan.cluster_rates_m3_s),
            "actuator": self.capabilities.actuator_name,
        }
