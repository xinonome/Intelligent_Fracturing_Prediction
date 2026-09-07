from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class IntegratedRewardConfig:
    effectiveness_weight: float = 3.0
    # Cluster balance is a production-effectiveness signal.  It is deliberately
    # kept below the pressure/abnormal-risk terms and is only active when a
    # measured or explicitly model-derived balance series is available.
    cluster_balance_weight: float = 1.0
    cluster_balance_improvement_scale: float = 0.05
    fracture_width_weight: float = 0.75
    fracture_volume_weight: float = 0.25
    fracture_geometry_improvement_scale: float = 0.10
    pressure_safety_weight: float = 3.0
    abnormal_risk_weight: float = 4.0
    construction_cost_weight: float = 1.0
    bottomhole_pressure_min_mpa: float = 45.0
    bottomhole_pressure_max_mpa: float = 110.0
    net_pressure_min_mpa: float = 0.0
    net_pressure_max_mpa: float = 35.0
    target_posterior_error: float = 0.15
    high_sand_ratio_warning_percent: float = 10.0
    high_sand_ratio_weight: float = 2.0
    action_change_weight: float = 0.5

    def to_dict(self) -> dict:
        return asdict(self)


def _resample(values: np.ndarray, size: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return np.full(size, np.nan)
    values = pd.Series(values).interpolate(limit_direction="both").to_numpy(dtype=float)
    if len(values) == size:
        return values
    return np.interp(np.linspace(0.0, 1.0, size), np.linspace(0.0, 1.0, len(values)), values)


def _first_available_column(frame: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    return None


def _load_cluster_balance_series(path: Path) -> np.ndarray:
    """Read a balance series or derive one from per-cluster share exports.

    The latter is useful for ``cluster_share_history.csv``.  Normalized
    entropy is used because it is 1 for equal allocation and approaches 0 as
    allocation concentrates in one cluster.  This is a derived diagnostic,
    not raw DAS amplitude.
    """

    if path.suffix.lower() in {".txt", ".log"}:
        values: list[float] = []
        with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("FracMonitor"):
                    continue
                main = line.split("#", 1)[0].split(",")
                if len(main) >= 5:
                    try:
                        values.append(float(main[3]))
                    except (TypeError, ValueError):
                        values.append(np.nan)
        return np.asarray(values, dtype=float)

    frame = pd.read_csv(path)
    direct = _first_available_column(
        frame,
        ("cluster_balance_degree", "balance_degree", "fiber_balance_degree", "cumulative_balance_degree"),
    )
    if direct:
        return pd.to_numeric(frame[direct], errors="coerce").to_numpy(dtype=float)

    share = _first_available_column(
        frame,
        ("observed_liquid_share", "fiber_liquid_allocation", "posterior_liquid_share", "allocation_weight"),
    )
    if not share:
        raise ValueError(
            "Cluster balance CSV must contain balance_degree or a per-cluster share column"
        )
    values = pd.to_numeric(frame[share], errors="coerce")
    group_col = _first_available_column(frame, ("sequence_index", "step", "time_s", "time"))
    if not group_col:
        groups = [(0, values.to_numpy(dtype=float))]
    else:
        groups = (
            (key, group["_share"].to_numpy(dtype=float))
            for key, group in frame.assign(_share=values).groupby(group_col, sort=True)
        )

    result: list[float] = []
    for _, raw_values in groups:
        finite = np.asarray(raw_values, dtype=float)
        finite = finite[np.isfinite(finite) & (finite >= 0.0)]
        if not len(finite) or float(finite.sum()) <= 1.0e-12:
            result.append(np.nan)
            continue
        probabilities = finite / float(finite.sum())
        denominator = np.log(float(len(probabilities))) if len(probabilities) > 1 else 1.0
        entropy = -float(np.sum(probabilities * np.log(np.maximum(probabilities, 1.0e-12))))
        result.append(float(np.clip(entropy / denominator, 0.0, 1.0)) if len(probabilities) > 1 else 1.0)
    return np.asarray(result, dtype=float)


def _aggregate_dt(frame: pd.DataFrame) -> pd.DataFrame:
    if "time_s" not in frame:
        raise ValueError("Digital-twin context must contain time_s")
    numeric = frame.select_dtypes(include=[np.number]).columns.tolist()
    aggregations = {col: "mean" for col in numeric if col != "time_s"}
    for col in ["half_length_m", "prior_half_length_m", "area_m2", "volume_m3"]:
        if col in aggregations:
            aggregations[col] = "sum"
    return frame.groupby("time_s", as_index=False).agg(aggregations).sort_values("time_s")


def load_reward_context(
    size: int,
    dt_context_csv: str | None = None,
    abnormal_probability_csv: str | None = None,
    alignment_mode: str = "normalized_progress",
    cluster_balance_csv: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    context = pd.DataFrame(index=np.arange(size))
    provenance = {
        "alignment_mode": alignment_mode,
        "scientific_status": "demo_only" if alignment_mode == "normalized_progress" else "same_stage_time_aligned",
        "digital_twin_source": dt_context_csv,
        "abnormal_probability_source": abnormal_probability_csv,
        "cluster_balance_source": cluster_balance_csv,
        "available_components": [],
    }
    if dt_context_csv:
        dt_path = Path(dt_context_csv)
        dt = _aggregate_dt(pd.read_csv(dt_path))
        history_path = dt_path.with_name("enkf_history.csv")
        if "posterior_error" not in dt and history_path.exists():
            history = pd.read_csv(history_path)
            if "time_s" in history and "posterior_error" in history:
                history = history[["time_s", "posterior_error"]].drop_duplicates("time_s")
                dt = pd.merge_asof(
                    dt.sort_values("time_s"),
                    history.sort_values("time_s"),
                    on="time_s",
                    direction="nearest",
                )
                provenance["enkf_history_source"] = str(history_path)
        column_map = {
            "half_length_m": "posterior_total_half_length_m",
            "prior_half_length_m": "prior_total_half_length_m",
            "area_m2": "fracture_area_m2",
            "volume_m3": "fracture_volume_m3",
            "posterior_error": "posterior_error",
            "bottomhole_pressure_mpa": "bottomhole_pressure_mpa",
            "net_pressure_mpa": "net_pressure_mpa",
            "fracture_volume_m3": "fracture_volume_m3",
        }
        for source, target in column_map.items():
            if source in dt:
                context[target] = _resample(dt[source].to_numpy(dtype=float), size)
        balance_source = _first_available_column(
            dt,
            ("cluster_balance_degree", "balance_degree", "fiber_balance_degree", "cumulative_balance_degree"),
        )
        if balance_source:
            context["cluster_balance_degree"] = _resample(
                dt[balance_source].to_numpy(dtype=float), size
            )
            provenance["available_components"].append("cluster_balance")
        width_source = _first_available_column(
            dt,
            ("fracture_width_m", "maximum_width_m", "max_width_m", "width_m", "fracture_width_mm", "max_aperture_mm"),
        )
        if width_source:
            width = pd.to_numeric(dt[width_source], errors="coerce").to_numpy(dtype=float)
            if width_source.endswith("_mm") or width_source == "max_aperture_mm":
                width = width / 1000.0
            context["fracture_width_m"] = _resample(width, size)
            provenance["available_components"].append("fracture_width")
        if {
            "posterior_total_half_length_m",
            "fracture_width_m",
            "fracture_volume_m3",
        } & set(context.columns):
            provenance["available_components"].append("fracture_effectiveness")
        if {"bottomhole_pressure_mpa", "net_pressure_mpa"} & set(context.columns):
            provenance["available_components"].append("pressure_safety")

    if cluster_balance_csv:
        balance_path = Path(cluster_balance_csv)
        balance = _load_cluster_balance_series(balance_path)
        context["cluster_balance_degree"] = _resample(balance, size)
        provenance["available_components"].append("cluster_balance")

    if abnormal_probability_csv:
        probs = pd.read_csv(Path(abnormal_probability_csv))
        normal_cols = [col for col in probs if col.lower() in {"prob_next_normal", "prob_normal", "normal_probability"}]
        abnormal_cols = [col for col in probs if col.startswith("prob_") and col not in normal_cols]
        if normal_cols:
            abnormal = 1.0 - pd.to_numeric(probs[normal_cols[0]], errors="coerce").to_numpy(dtype=float)
        elif abnormal_cols:
            abnormal = probs[abnormal_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1).to_numpy(dtype=float)
        else:
            raise ValueError("Abnormal probability CSV has no prob_* columns")
        context["abnormal_probability"] = np.clip(_resample(abnormal, size), 0.0, 1.0)
        sand_cols = [col for col in probs if "砂堵" in col]
        if sand_cols:
            context["sand_plug_probability"] = np.clip(
                _resample(pd.to_numeric(probs[sand_cols[0]], errors="coerce").to_numpy(dtype=float), size), 0.0, 1.0
            )
        provenance["available_components"].append("abnormal_risk")
    provenance["available_components"] = list(dict.fromkeys(provenance["available_components"]))
    return context, provenance


def calculate_integrated_reward(
    flow: np.ndarray,
    sand_ratio: np.ndarray,
    current_flow: np.ndarray,
    current_sand_ratio: np.ndarray,
    context: pd.DataFrame,
    max_flow: float,
    max_sand_ratio: float,
    config: IntegratedRewardConfig,
) -> dict[str, np.ndarray]:
    size = len(flow)
    zeros = np.zeros(size, dtype=float)

    def relative_improvement(column: str) -> tuple[np.ndarray, np.ndarray]:
        improvement = np.full(size, np.nan, dtype=float)
        reward = zeros.copy()
        if column not in context or size == 0:
            return improvement, reward
        values = pd.to_numeric(context[column], errors="coerce").to_numpy(dtype=float)
        previous = np.r_[values[0], values[:-1]]
        valid = (
            np.isfinite(values)
            & np.isfinite(previous)
            & (values >= 0.0)
            & (previous >= 0.0)
        )
        improvement[valid] = (values[valid] - previous[valid]) / np.maximum(
            np.abs(previous[valid]), 1.0e-6
        )
        scale = max(float(config.fracture_geometry_improvement_scale), 1.0e-6)
        reward[valid] = np.clip(improvement[valid] / scale, -1.0, 1.0)
        return improvement, reward

    effectiveness = zeros.copy()
    effectiveness_available = False
    if "posterior_total_half_length_m" in context:
        length = context["posterior_total_half_length_m"].to_numpy(dtype=float)
        length_gain = np.maximum(length - np.r_[length[0], length[:-1]], 0.0)
        scale = max(float(np.nanpercentile(length_gain, 95)), 1e-6)
        effectiveness += np.clip(length_gain / scale, 0.0, 1.0)
        effectiveness_available = True
    if "posterior_error" in context:
        error = context["posterior_error"].to_numpy(dtype=float)
        effectiveness += np.clip(1.0 - error / config.target_posterior_error, -1.0, 1.0)
        effectiveness_available = True
    if effectiveness_available:
        effectiveness *= config.effectiveness_weight / (2.0 if "posterior_error" in context else 1.0)

    fracture_width_improvement, width_signal = relative_improvement("fracture_width_m")
    fracture_volume_improvement, volume_signal = relative_improvement("fracture_volume_m3")
    fracture_width_reward = config.fracture_width_weight * width_signal
    fracture_volume_reward = config.fracture_volume_weight * volume_signal
    if "fracture_width_m" in context:
        effectiveness += fracture_width_reward
        effectiveness_available = True
    if "fracture_volume_m3" in context:
        # Volume remains an internal physical-effectiveness signal.  It is
        # intentionally not exposed as a production/production-rate KPI.
        effectiveness += fracture_volume_reward
        effectiveness_available = True

    # Reward improvement in observed/model-declared cluster balance separately
    # from fracture growth.  A flat balance receives no artificial positive
    # reward, deterioration is penalized, and missing balance data contributes
    # exactly zero.  This keeps the term interpretable as an improvement
    # reward, rather than silently turning unavailable cluster observations
    # into a fabricated classifier or target.
    cluster_balance_reward = zeros.copy()
    cluster_balance_improvement = np.full(size, np.nan, dtype=float)
    cluster_balance_available = False
    if "cluster_balance_degree" in context:
        balance = pd.to_numeric(context["cluster_balance_degree"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(balance) & (balance >= 0.0) & (balance <= 1.0)
        previous = np.r_[balance[0], balance[:-1]] if size else np.asarray([], dtype=float)
        transition_valid = valid & np.isfinite(previous) & (previous >= 0.0) & (previous <= 1.0)
        cluster_balance_improvement[transition_valid] = balance[transition_valid] - previous[transition_valid]
        scale = max(float(config.cluster_balance_improvement_scale), 1.0e-6)
        cluster_balance_reward[transition_valid] = config.cluster_balance_weight * np.clip(
            cluster_balance_improvement[transition_valid] / scale,
            -1.0,
            1.0,
        )
        cluster_balance_available = bool(transition_valid.any())
        effectiveness += cluster_balance_reward

    pressure_penalty = zeros.copy()
    pressure_available = False
    if "bottomhole_pressure_mpa" in context:
        bhp = context["bottomhole_pressure_mpa"].to_numpy(dtype=float)
        span = max(config.bottomhole_pressure_max_mpa - config.bottomhole_pressure_min_mpa, 1e-6)
        pressure_penalty += np.maximum(config.bottomhole_pressure_min_mpa - bhp, 0.0) / span
        pressure_penalty += np.maximum(bhp - config.bottomhole_pressure_max_mpa, 0.0) / span
        pressure_available = True
    if "net_pressure_mpa" in context:
        net = context["net_pressure_mpa"].to_numpy(dtype=float)
        span = max(config.net_pressure_max_mpa - config.net_pressure_min_mpa, 1e-6)
        pressure_penalty += np.maximum(config.net_pressure_min_mpa - net, 0.0) / span
        pressure_penalty += np.maximum(net - config.net_pressure_max_mpa, 0.0) / span
        pressure_available = True
    pressure_penalty *= config.pressure_safety_weight

    abnormal_penalty = zeros.copy()
    abnormal_available = "abnormal_probability" in context
    if abnormal_available:
        abnormal_penalty = config.abnormal_risk_weight * context["abnormal_probability"].to_numpy(dtype=float)
        if "sand_plug_probability" in context:
            abnormal_penalty += config.abnormal_risk_weight * context["sand_plug_probability"].to_numpy(dtype=float)

    q = np.asarray(flow, dtype=float)
    s = np.asarray(sand_ratio, dtype=float)
    q0 = np.asarray(current_flow, dtype=float)
    s0 = np.asarray(current_sand_ratio, dtype=float)
    construction_cost = config.construction_cost_weight * (
        0.35 * np.clip(q / max(max_flow, 1e-6), 0.0, 2.0)
        + 0.35 * np.clip(s / max(max_sand_ratio, 1e-6), 0.0, 2.0)
        + 0.15 * np.abs(q - q0) / max(max_flow, 1e-6)
        + 0.15 * np.abs(s - s0) / max(max_sand_ratio, 1e-6)
    )
    # High sand ratio is a safety concern, not an effectiveness target.  Keep
    # this penalty independent from the schedule projector so a policy cannot
    # receive a positive learning signal merely by repeatedly requesting the
    # largest permitted sand value.
    high_sand_penalty = config.high_sand_ratio_weight * np.maximum(
        s - config.high_sand_ratio_warning_percent, 0.0
    ) / max(max_sand_ratio - config.high_sand_ratio_warning_percent, 1e-6)
    action_change_penalty = config.action_change_weight * (
        np.abs(q - q0) / max(max_flow, 1e-6)
        + np.abs(s - s0) / max(max_sand_ratio, 1e-6)
    )
    safety_action_penalty = high_sand_penalty + action_change_penalty
    total = effectiveness - pressure_penalty - abnormal_penalty - construction_cost - safety_action_penalty
    return {
        "integrated_reward": total,
        "effectiveness_reward": effectiveness,
        "fracture_width_reward": fracture_width_reward,
        "fracture_volume_reward": fracture_volume_reward,
        "fracture_width_improvement": fracture_width_improvement,
        "fracture_volume_improvement": fracture_volume_improvement,
        "fracture_width_m": (
            pd.to_numeric(context["fracture_width_m"], errors="coerce").to_numpy(dtype=float)
            if "fracture_width_m" in context else np.full(size, np.nan, dtype=float)
        ),
        "fracture_volume_m3": (
            pd.to_numeric(context["fracture_volume_m3"], errors="coerce").to_numpy(dtype=float)
            if "fracture_volume_m3" in context else np.full(size, np.nan, dtype=float)
        ),
        "cluster_balance_reward": cluster_balance_reward,
        "cluster_balance_improvement": cluster_balance_improvement,
        "cluster_balance_degree": (
            pd.to_numeric(context["cluster_balance_degree"], errors="coerce").to_numpy(dtype=float)
            if "cluster_balance_degree" in context
            else np.full(size, np.nan, dtype=float)
        ),
        "pressure_safety_penalty": pressure_penalty,
        "abnormal_risk_penalty": abnormal_penalty,
        "construction_cost_penalty": construction_cost,
        "high_sand_penalty": high_sand_penalty,
        "action_change_penalty": action_change_penalty,
        "safety_action_penalty": safety_action_penalty,
        "effectiveness_available": np.full(size, effectiveness_available),
        "pressure_available": np.full(size, pressure_available),
        "abnormal_probability_available": np.full(size, abnormal_available),
        "cluster_balance_available": np.full(size, cluster_balance_available),
    }
