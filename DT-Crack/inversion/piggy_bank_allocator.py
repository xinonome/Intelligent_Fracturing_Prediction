"""Conservative, auditable stage-level liquid advisory.

The original prototype represented Piggy-Bank as a direct six-cluster liquid
transfer.  That is not an available actuator when the field can only change
the total surface pumping rate.  This module therefore keeps the cluster
response analysis, but only recommends a next-window *stage total liquid*
decrease, hold, or increase.  Cluster weights are retained as diagnostics;
they are not field-control commands and no already injected liquid is moved.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PiggyBankConfig:
    """Safety and numerical settings for the temporary allocation layer."""

    # Kept as a compatibility alias for existing command lines.  It now
    # limits the proposed stage-total change, not a cluster-level change.
    max_single_cluster_adjustment_fraction: float = 0.10
    max_stage_adjustment_fraction: float | None = None
    max_single_cluster_volume_m3: float = 1.0e9
    max_total_reserve_m3: float = 1.0e9
    minimum_cluster_weight: float = 1.0e-6
    score_threshold: float = 0.20
    oscillation_damping: float = 0.50
    max_consecutive_same_direction: int = 4
    require_geometry_valid: bool = True


def normalize_weights(values: Iterable[float], minimum: float = 1.0e-6) -> np.ndarray:
    values = np.asarray(list(values), dtype=float)
    if values.size == 0:
        raise ValueError("at least one cluster weight is required")
    values = np.clip(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), minimum, None)
    return values / max(float(values.sum()), 1.0e-12)


def _safe_float(value: object, default: float = np.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


class PiggyBankAllocator:
    """Compute a conservation-aware next-window stage-liquid advisory."""

    def __init__(self, config: PiggyBankConfig | None = None) -> None:
        self.config = config or PiggyBankConfig()
        self.bank_balance_m3 = 0.0
        self.previous_transfer_m3: np.ndarray | None = None
        self.consecutive_direction = 0

    def reset(self) -> None:
        self.bank_balance_m3 = 0.0
        self.previous_transfer_m3 = None
        self.consecutive_direction = 0

    def snapshot(self) -> dict[str, object]:
        return {
            # There is no physical or executed reserve in a total-rate-only
            # advisory.  Keep the legacy field for file compatibility, but
            # never present it as an accumulated field liquid volume.
            "bank_balance_m3": 0.0,
            "previous_transfer_m3": None if self.previous_transfer_m3 is None else self.previous_transfer_m3.tolist(),
            "consecutive_direction": int(self.consecutive_direction),
        }

    def restore(self, snapshot: dict[str, object]) -> None:
        # Legacy snapshots may contain a cluster-transfer reserve.  It is not
        # an executable field volume in the total-rate-only advisory, so do
        # not restore it as a live balance.
        self.bank_balance_m3 = 0.0
        previous = snapshot.get("previous_transfer_m3")
        self.previous_transfer_m3 = None if previous is None else np.asarray(previous, dtype=float)
        self.consecutive_direction = max(0, int(snapshot.get("consecutive_direction", 0)))

    def _hold(
        self,
        base: np.ndarray,
        total_volume_m3: float,
        reason: str,
        states: list[str] | None = None,
        scores: np.ndarray | None = None,
    ) -> dict[str, object]:
        n_clusters = len(base)
        zero = np.zeros(n_clusters, dtype=float)
        state_values = states or ["unknown"] * n_clusters
        score_values = np.full(n_clusters, np.nan) if scores is None else scores
        return self._result(
            base,
            base,
            zero,
            zero,
            total_volume_m3,
            reason,
            state_values,
            score_values,
            transfer_total=0.0,
            released_candidate=0.0,
            received_candidate=0.0,
        )

    def _result(
        self,
        base: np.ndarray,
        recommended: np.ndarray,
        transfer: np.ndarray,
        bias: np.ndarray,
        total_volume_m3: float,
        status: str,
        states: list[str],
        scores: np.ndarray,
        *,
        transfer_total: float,
        released_candidate: float,
        received_candidate: float,
    ) -> dict[str, object]:
        return {
            "status": status,
            "base_weights": base.tolist(),
            "recommended_weights": recommended.tolist(),
            "piggy_bank_bias": bias.tolist(),
            "transferred_volume_m3": transfer.tolist(),
            "recommended_volume_m3": (recommended * total_volume_m3).tolist(),
            "base_volume_m3": (base * total_volume_m3).tolist(),
            # Keep the legacy field for file compatibility, but never present
            # it as an accumulated physical reserve.
            "bank_balance_m3": 0.0,
            "transfer_total_m3": float(transfer_total),
            "released_candidate_m3": float(released_candidate),
            "received_candidate_m3": float(received_candidate),
            "total_volume_m3": float(total_volume_m3),
            "cluster_states": list(states),
            "control_scores": np.asarray(scores, dtype=float).tolist(),
            "total_weight_error": float(abs(float(recommended.sum()) - 1.0)),
            "physical_parameters_modified": False,
            "control_layer": "stage_total_liquid_advisory",
            "cluster_allocation_actionable": False,
            "stage_adjustment_fraction": 0.0,
            "stage_liquid_delta_m3": 0.0,
            "recommended_stage_volume_m3": float(total_volume_m3),
            "recommended_stage_rate_m3_s": 0.0,
            "virtual_stage_reserve_m3": 0.0,
            "recommendation_direction": "hold",
            "recommendation_basis": str(status),
        }

    def allocate(
        self,
        base_weights: Iterable[float],
        total_rate_m3_s: float,
        window_seconds: float,
        response_metrics: pd.DataFrame | None = None,
        *,
        target_shares: Iterable[float] | None = None,
        pressure_high: bool = False,
        uncertainty_high: bool = False,
        data_quality_valid: bool = True,
        geometry_valid: bool = True,
        manual_hold: bool = False,
    ) -> dict[str, object]:
        """Recommend a next-window total liquid change for the current stage.

        ``response_metrics`` must contain one row per cluster, preferably the
        output of :func:`compute_segment_response_metrics`.  A positive
        ``control_score`` means that the cluster is slow/under-allocated and a
        negative score means fast/dominant.  If the signal is mixed, the
        allocator holds because total-rate control cannot target one cluster.
        Cluster weights remain diagnostics only; this method does not issue a
        six-cluster field allocation command.
        """

        base = normalize_weights(base_weights, self.config.minimum_cluster_weight)
        n_clusters = len(base)
        total_rate = max(_safe_float(total_rate_m3_s, 0.0), 0.0)
        duration = max(_safe_float(window_seconds, 0.0), 0.0)
        total_volume = total_rate * duration

        def hold(reason: str, states: list[str] | None = None, scores: np.ndarray | None = None) -> dict[str, object]:
            result = self._hold(base, total_volume, reason, states, scores)
            result["recommended_stage_rate_m3_s"] = total_rate
            return result
        if target_shares is not None:
            target = normalize_weights(target_shares, self.config.minimum_cluster_weight)
            if len(target) != n_clusters:
                raise ValueError("target_shares must have same length as base_weights")
        else:
            target = np.full(n_clusters, 1.0 / n_clusters, dtype=float)

        if manual_hold:
            return hold("manual_hold")
        if pressure_high:
            return hold("pressure_safety_hold")
        if uncertainty_high:
            return hold("uncertainty_hold")
        if not data_quality_valid:
            return hold("observation_quality_hold")
        if self.config.require_geometry_valid and not geometry_valid:
            return hold("geometry_quality_hold")
        if response_metrics is None or response_metrics.empty:
            return hold("no_response_metrics")

        metrics = response_metrics.copy()
        if "cluster_id" in metrics.columns:
            metrics = metrics.sort_values("cluster_id")
        if len(metrics) != n_clusters:
            return hold("incomplete_cluster_metrics")
        states = metrics.get("control_state", pd.Series(["unknown"] * n_clusters)).astype(str).tolist()
        scores = pd.to_numeric(metrics.get("control_score"), errors="coerce").to_numpy(dtype=float)
        if len(scores) != n_clusters or not np.isfinite(scores).any():
            return hold("no_valid_control_score", states, scores)
        if any(state == "unknown" for state in states):
            return hold("unknown_cluster_response", states, scores)

        scores = np.clip(np.nan_to_num(scores, nan=0.0), -1.0, 1.0)
        threshold = max(float(self.config.score_threshold), 0.0)
        positive = scores[scores > threshold]
        negative = scores[scores < -threshold]
        if total_volume <= 0.0:
            return hold("no_stage_volume", states, scores)
        if len(positive) and len(negative):
            return hold("mixed_cluster_signal_total_rate_not_targetable", states, scores)
        if not len(positive) and not len(negative):
            return hold("no_stage_direction_signal", states, scores)

        net_score = float(np.clip(np.nanmean(scores), -1.0, 1.0))
        max_adjustment = (
            float(self.config.max_stage_adjustment_fraction)
            if self.config.max_stage_adjustment_fraction is not None
            else float(self.config.max_single_cluster_adjustment_fraction)
        )
        adjustment_fraction = float(np.clip(net_score * max(max_adjustment, 0.0), -max_adjustment, max_adjustment))
        if abs(adjustment_fraction) <= 1.0e-12:
            return hold("stage_adjustment_below_threshold", states, scores)

        recommended_volume = total_volume * (1.0 + adjustment_fraction)
        stage_delta = recommended_volume - total_volume
        direction = "increase" if stage_delta > 0.0 else "decrease"
        status = "stage_total_liquid_increase_recommended" if direction == "increase" else "stage_total_liquid_decrease_recommended"
        result = self._result(
            base,
            base,
            np.zeros(n_clusters, dtype=float),
            np.zeros(n_clusters, dtype=float),
            total_volume,
            status,
            states,
            scores,
            transfer_total=0.0,
            released_candidate=0.0,
            received_candidate=0.0,
        )
        result.update(
            {
                "stage_adjustment_fraction": adjustment_fraction,
                "stage_liquid_delta_m3": stage_delta,
                "recommended_stage_volume_m3": recommended_volume,
                "recommended_stage_rate_m3_s": recommended_volume / max(duration, 1.0e-12),
                "virtual_stage_reserve_m3": max(-stage_delta, 0.0),
                "recommendation_direction": direction,
                "recommendation_basis": "unidirectional_cluster_response_signal",
            }
        )
        return result

    @staticmethod
    def audit_frame(result: dict[str, object], cluster_ids: Iterable[int] | None = None, *, as_of_step: int | None = None) -> pd.DataFrame:
        """Convert one allocator result into the machine-readable audit rows."""

        base = np.asarray(result.get("base_weights", []), dtype=float)
        recommended = np.asarray(result.get("recommended_weights", []), dtype=float)
        bias = np.asarray(result.get("piggy_bank_bias", np.zeros(len(base))), dtype=float)
        transfer = np.asarray(result.get("transferred_volume_m3", np.zeros(len(base))), dtype=float)
        states = list(result.get("cluster_states", ["unknown"] * len(base)))
        scores = np.asarray(result.get("control_scores", [np.nan] * len(base)), dtype=float)
        ids = list(cluster_ids) if cluster_ids is not None else list(range(1, len(base) + 1))
        if len(ids) != len(base):
            raise ValueError("cluster_ids length does not match allocator result")
        rows = []
        for idx, cluster_id in enumerate(ids):
            rows.append({
                "as_of_step": None if as_of_step is None else int(as_of_step),
                "cluster_id": int(cluster_id),
                "cluster_state": states[idx] if idx < len(states) else "unknown",
                "response_score": float(scores[idx]) if idx < len(scores) and np.isfinite(scores[idx]) else np.nan,
                "base_weight": float(base[idx]),
                "piggy_bank_bias": float(bias[idx]),
                "recommended_weight": float(recommended[idx]),
                "base_volume_m3": float(result.get("base_volume_m3", [0.0] * len(base))[idx]),
                "recommended_volume_m3": float(result.get("recommended_volume_m3", [0.0] * len(base))[idx]),
                "transferred_volume_m3": float(transfer[idx]),
                "bank_balance_m3": 0.0,
                "virtual_stage_reserve_m3": float(result.get("virtual_stage_reserve_m3", 0.0)),
                "stage_liquid_delta_m3": float(result.get("stage_liquid_delta_m3", 0.0)),
                "recommended_stage_volume_m3": float(result.get("recommended_stage_volume_m3", 0.0)),
                "recommendation_direction": str(result.get("recommendation_direction", "hold")),
                "cluster_allocation_actionable": bool(result.get("cluster_allocation_actionable", False)),
                "status": str(result.get("status", "unknown")),
                "constraint_status": str(result.get("status", "unknown")),
                "physical_parameters_modified": bool(result.get("physical_parameters_modified", False)),
            })
        return pd.DataFrame(rows)
