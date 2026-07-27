from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class IntegratedRewardConfig:
    effectiveness_weight: float = 3.0
    pressure_safety_weight: float = 3.0
    abnormal_risk_weight: float = 4.0
    construction_cost_weight: float = 1.0
    bottomhole_pressure_min_mpa: float = 45.0
    bottomhole_pressure_max_mpa: float = 110.0
    net_pressure_min_mpa: float = 0.0
    net_pressure_max_mpa: float = 35.0
    target_posterior_error: float = 0.15

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
) -> tuple[pd.DataFrame, dict]:
    context = pd.DataFrame(index=np.arange(size))
    provenance = {
        "alignment_mode": alignment_mode,
        "scientific_status": "demo_only" if alignment_mode == "normalized_progress" else "same_stage_time_aligned",
        "digital_twin_source": dt_context_csv,
        "abnormal_probability_source": abnormal_probability_csv,
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
        }
        for source, target in column_map.items():
            if source in dt:
                context[target] = _resample(dt[source].to_numpy(dtype=float), size)
        if {"posterior_total_half_length_m", "fracture_area_m2"} & set(context.columns):
            provenance["available_components"].append("fracture_effectiveness")
        if {"bottomhole_pressure_mpa", "net_pressure_mpa"} & set(context.columns):
            provenance["available_components"].append("pressure_safety")

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
    total = effectiveness - pressure_penalty - abnormal_penalty - construction_cost
    return {
        "integrated_reward": total,
        "effectiveness_reward": effectiveness,
        "pressure_safety_penalty": pressure_penalty,
        "abnormal_risk_penalty": abnormal_penalty,
        "construction_cost_penalty": construction_cost,
        "effectiveness_available": np.full(size, effectiveness_available),
        "pressure_available": np.full(size, pressure_available),
        "abnormal_probability_available": np.full(size, abnormal_available),
    }
