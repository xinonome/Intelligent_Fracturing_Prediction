"""Quantify how stage-total rate changes affect predicted cluster balance.

The control boundary in the current system is the *stage total* rate.  This
module therefore does not pretend that a pump can command one cluster.  It
scans several total-rate scenarios through the same PKN allocation operator,
then measures the resulting six-cluster shares and balance indices.

The result is deliberately diagnostic rather than prescriptive: if changing
the total rate barely changes the predicted shares, the report says so.  A
Piggy-Bank recommendation must not be justified by a rate effect that the
forward model does not actually identify.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .physics import PhysicalEnKFConfig, pkn_with_carter_leakoff
from .segment_response_metrics import balance_indices


@dataclass(frozen=True)
class RateBalanceResponseConfig:
    """Configuration for a total-rate sensitivity scan."""

    duration_s: float = 300.0
    n_clusters: int = 6
    base_shares: tuple[float, ...] | None = None
    target_shares: tuple[float, ...] | None = None
    minimum_identifiable_balance_range: float = 1.0e-3


def _normalise(values: Iterable[float], n: int) -> np.ndarray:
    array = np.nan_to_num(np.asarray(list(values), dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if array.size != n:
        raise ValueError(f"expected {n} cluster values, received {array.size}")
    array = np.clip(array, 0.0, None)
    total = float(array.sum())
    if total <= 1.0e-12:
        return np.full(n, 1.0 / max(n, 1), dtype=float)
    return array / total


def scan_rate_balance_response(
    state: np.ndarray,
    total_rates_m3_s: Iterable[float],
    *,
    physics_config: PhysicalEnKFConfig | None = None,
    config: RateBalanceResponseConfig | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Run a total-rate scan and return rows plus a machine-readable summary.

    ``state`` is one PKN/EnKF physical state.  ``total_rates_m3_s`` contains
    stage-total rates; each scenario starts from the same baseline cluster
    shares and lets the forward operator calculate the final shares.
    """

    config = config or RateBalanceResponseConfig()
    physics_config = physics_config or PhysicalEnKFConfig()
    n = int(config.n_clusters)
    if n <= 0:
        raise ValueError("n_clusters must be positive")
    state = np.asarray(state, dtype=float)
    if state.ndim != 1:
        raise ValueError("state must be one-dimensional")
    rates = np.asarray(list(total_rates_m3_s), dtype=float)
    if rates.size < 2:
        raise ValueError("at least two total-rate scenarios are required")
    if not np.isfinite(rates).all() or (rates <= 0.0).any():
        raise ValueError("total rates must be finite and positive")
    base_shares = (
        _normalise(config.base_shares, n)
        if config.base_shares is not None
        else np.full(n, 1.0 / n, dtype=float)
    )
    target_shares = (
        _normalise(config.target_shares, n)
        if config.target_shares is not None
        else np.full(n, 1.0 / n, dtype=float)
    )

    rows: list[dict[str, object]] = []
    for rate in rates:
        q = float(rate) * base_shares
        result = pkn_with_carter_leakoff(
            state,
            q,
            float(config.duration_s),
            physics_config,
            q_current_m3_s=q,
        )
        allocation = np.asarray(result["cluster_allocation"], dtype=float)
        metrics = balance_indices(allocation, target_shares)
        row: dict[str, object] = {
            "total_rate_m3_s": float(rate),
            "duration_s": float(config.duration_s),
            "balance_degree": float(metrics["balance_degree"]),
            "gini": float(metrics["gini"]),
            "entropy_normalized": float(metrics["entropy_normalized"]),
            "max_cluster_share": float(metrics["max_share"]),
            "min_cluster_share": float(metrics["min_share"]),
            "net_pressure_mpa": float(result["net_pressure_mpa"]),
            "leakoff_fraction": float(result["leakoff_fraction"]),
            "rate_conservation_error": float(result["rate_conservation_error"]),
            "cluster_allocation": allocation.tolist(),
        }
        for index, share in enumerate(allocation, start=1):
            row[f"cluster_{index}_share"] = float(share)
        rows.append(row)

    # Sort only for finite-difference calculation; preserve the caller's
    # order in the returned rows so the output remains easy to compare with a
    # requested scenario list.
    ordered = sorted(rows, key=lambda item: float(item["total_rate_m3_s"]))
    for index, row in enumerate(ordered):
        if len(ordered) == 2:
            left, right = ordered[0], ordered[1]
        elif index == 0:
            left, right = ordered[0], ordered[1]
        elif index == len(ordered) - 1:
            left, right = ordered[-2], ordered[-1]
        else:
            left, right = ordered[index - 1], ordered[index + 1]
        dq = float(right["total_rate_m3_s"]) - float(left["total_rate_m3_s"])
        db = float(right["balance_degree"]) - float(left["balance_degree"])
        row["local_d_balance_d_rate"] = float(db / dq) if abs(dq) > 1.0e-12 else 0.0
    sensitivity_by_rate = {float(row["total_rate_m3_s"]): row for row in ordered}
    for row in rows:
        row["local_d_balance_d_rate"] = sensitivity_by_rate[float(row["total_rate_m3_s"])] ["local_d_balance_d_rate"]

    balance_values = np.asarray([float(row["balance_degree"]) for row in rows], dtype=float)
    balance_range = float(np.max(balance_values) - np.min(balance_values))
    identifiable = bool(balance_range >= max(float(config.minimum_identifiable_balance_range), 0.0))
    summary: dict[str, object] = {
        "status": "rate_effect_identified" if identifiable else "rate_effect_not_identified",
        "scientific_status": "model_sensitivity_scan",
        "n_scenarios": int(len(rows)),
        "total_rate_min_m3_s": float(np.min(rates)),
        "total_rate_max_m3_s": float(np.max(rates)),
        "duration_s": float(config.duration_s),
        "balance_degree_min": float(np.min(balance_values)),
        "balance_degree_max": float(np.max(balance_values)),
        "balance_degree_range": balance_range,
        "target_shares": target_shares.tolist(),
        "base_shares": base_shares.tolist(),
        "interpretation": (
            "在当前物理参数和分配方程下，改变总排量能够产生可辨识的均衡度变化；"
            "仍需用现场分段验证确认方向。"
            if identifiable
            else
            "在当前扫描范围内，总排量对均衡度的影响小于识别阈值；"
            "不能仅凭总排量建议宣称均衡度会改善，应优先补充观测或重新校准分配参数。"
        ),
        "limitations": [
            "这是模型情景敏感性，不是现场因果证明。",
            "总排量控制不能直接指定某一个簇的液量。",
            "真实控制仍需通过阶段总液量建议、现场执行量和后续观测验证。",
        ],
    }
    return rows, summary


def rows_to_csv(rows: list[dict[str, object]], path: str) -> None:
    """Write scan rows without requiring a dataframe dependency."""

    import csv

    if not rows:
        raise ValueError("rows cannot be empty")
    fieldnames = [key for key, value in rows[0].items() if not isinstance(value, list)]
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})
