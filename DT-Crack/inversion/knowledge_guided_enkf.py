"""Knowledge-graph guided prior construction for the physical EnKF.

The knowledge graph is deliberately kept upstream of the filter.  It may
change the prior mean, spread, covariance structure, observation confidence,
and the maximum parameter update, but it never replaces the EnKF innovation
update and never writes a fracture length directly.  The physical forward
operator remains the authority for the predicted state.

The mappings in this module are explicit engineering bridges.  They are
reported in the run metadata and must be recalibrated with held-out wells
before being treated as production knowledge.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_RULES_PATH = (
    Path(__file__).resolve().parents[2]
    / "FSL-Expert"
    / "rule_fusion"
    / "rule_fusion"
    / "fused_sand_plug_rules.json"
)


@dataclass(frozen=True)
class KnowledgeGuidedPriorConfig:
    """Controls for the optional KG-EnKF prior bridge.

    ``uncertainty_only`` preserves the original experiment.  ``soft_prior``
    adds a small, auditable mean shift.  ``soft_correlated`` additionally
    creates a rule-informed covariance, relaxes pressure observation noise in
    ambiguous high-risk contexts, and limits one assimilation step.
    """

    enabled: bool = False
    strength: float = 0.35
    rules_path: str | None = None
    mode: str = "uncertainty_only"
    mean_shift_scale: float = 1.0
    covariance_scale: float = 1.0
    observation_noise_scale: float = 0.25
    max_update_scale: float = 2.0


VALID_MODES = {"off", "uncertainty_only", "soft_prior", "soft_correlated"}
STATE_PARAMETER_NAMES = [
    "log_eprime_scale",
    "log_leakoff_scale",
    "log_viscosity_scale",
    "min_horizontal_stress_mpa",
    "log_fracture_toughness_scale",
]


def _load_rule_ids(path: str | Path | None) -> tuple[set[str], dict[str, Any]]:
    rule_path = Path(path) if path else DEFAULT_RULES_PATH
    if not rule_path.exists():
        return set(), {"path": str(rule_path), "loaded": False, "reason": "file_not_found"}
    try:
        payload = json.loads(rule_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return set(), {"path": str(rule_path), "loaded": False, "reason": str(exc)}
    rules = payload.get("rules", payload if isinstance(payload, list) else [])
    rule_ids = {str(item.get("rule_id")) for item in rules if isinstance(item, dict) and item.get("rule_id")}
    return rule_ids, {
        "path": str(rule_path),
        "loaded": True,
        "rule_count": len(rule_ids),
        "candidate_count": payload.get("candidate_count") if isinstance(payload, dict) else None,
    }


def _series_by_step(
    pressure: pd.DataFrame,
    source_steps: np.ndarray,
    column: str,
) -> np.ndarray:
    if column not in pressure.columns:
        return np.zeros(len(source_steps), dtype=float)
    table = pressure.drop_duplicates("step").set_index("step").sort_index()
    values = table.reindex(source_steps)[column].to_numpy(dtype=float)
    if np.isnan(values).any():
        known = table[column].astype(float)
        values = (
            known.reindex(source_steps)
            .interpolate(method="index")
            .ffill()
            .bfill()
            .to_numpy(dtype=float)
        )
    return np.nan_to_num(values, nan=0.0)


def _control_series(controls: pd.DataFrame, source_steps: np.ndarray, column: str) -> np.ndarray:
    if column not in controls.columns:
        return np.zeros(len(source_steps), dtype=float)
    grouped = controls.groupby("step", sort=True)[column].mean()
    return np.nan_to_num(grouped.reindex(source_steps).ffill().bfill().to_numpy(dtype=float), nan=0.0)


def infer_knowledge_signals(
    controls: pd.DataFrame,
    pressure: pd.DataFrame,
    calibration_steps: np.ndarray,
    rule_ids: set[str],
) -> dict[str, Any]:
    """Convert the audited expert rules into a small, reproducible signal set.

    The signal extraction uses calibration steps only.  No held-out pressure or
    fiber sample is used to set the prior, preventing validation leakage.
    """

    steps = np.asarray(calibration_steps, dtype=int)
    if steps.size == 0:
        return {"risk_score": 0.0, "rule_matches": [], "sample_count": 0}

    bhp = _series_by_step(pressure, steps, "bottomhole_pressure_mpa")
    sand_ratio = _series_by_step(pressure, steps, "sand_ratio_percent")
    flow = _series_by_step(pressure, steps, "flow_rate_m3_min")
    if not np.any(flow):
        flow = _control_series(controls, steps, "flow_rate_m3_min")

    time_s = _series_by_step(pressure, steps, "time_s")
    if len(time_s) < 2 or not np.all(np.diff(time_s) > 0):
        time_s = steps.astype(float)
    dt_min = np.maximum(np.diff(time_s) / 60.0, 1.0e-6)
    pressure_diff = np.diff(bhp)
    pressure_slopes = pressure_diff / dt_min
    max_pressure_slope = float(np.max(pressure_slopes)) if pressure_slopes.size else 0.0
    pressure_rise = float(max(0.0, np.max(bhp) - bhp[0]))
    high_sand_ratio = float(np.max(sand_ratio)) if sand_ratio.size else 0.0
    flow_falling_fraction = (
        float(np.mean(np.diff(flow) < -1.0e-9)) if flow.size > 1 else 0.0
    )

    pressure_rise_score = float(np.clip((pressure_rise - 8.0) / 30.0, 0.0, 1.0))
    pressure_slope_score = float(np.clip((max_pressure_slope - 0.5) / 1.0, 0.0, 1.0))
    high_sand_score = float(np.clip((high_sand_ratio - 8.0) / 8.0, 0.0, 1.0))
    flow_down_score = float(np.clip(flow_falling_fraction / 0.5, 0.0, 1.0))

    # These weights correspond to the meaning of the existing fused rules:
    # pressure rise is the main signal; high sand and falling rate increase the
    # uncertainty about blockage/near-wellbore and leakoff explanations.
    risk_score = float(
        np.clip(
            0.40 * pressure_rise_score
            + 0.30 * pressure_slope_score
            + 0.20 * high_sand_score
            + 0.10 * flow_down_score,
            0.0,
            1.0,
        )
    )

    matched: list[str] = []
    if "SP-R02" in rule_ids and pressure_rise >= 8.0 and max_pressure_slope >= 1.0:
        matched.append("SP-R02")
    if "SP-R04" in rule_ids and pressure_rise >= 8.0 and flow_falling_fraction >= 0.25:
        matched.append("SP-R04")
    if "SP-R09" in rule_ids and high_sand_ratio >= 13.0 and pressure_rise >= 8.0:
        matched.append("SP-R09")
    if "SP-R06" in rule_ids and np.max(np.diff(flow)) >= 1.0:
        # This is an exclusion signal: a rate increase should not be treated as
        # direct evidence of blockage. It is retained for auditability.
        matched.append("SP-R06_exclusion")

    return {
        "sample_count": int(len(steps)),
        "calibration_start_step": int(steps[0]),
        "calibration_end_step": int(steps[-1]),
        "pressure_rise_mpa": pressure_rise,
        "max_pressure_slope_mpa_min": max_pressure_slope,
        "max_sand_ratio_percent": high_sand_ratio,
        "flow_falling_fraction": flow_falling_fraction,
        "pressure_rise_score": pressure_rise_score,
        "pressure_slope_score": pressure_slope_score,
        "high_sand_score": high_sand_score,
        "flow_down_score": flow_down_score,
        "risk_score": risk_score,
        "rule_matches": matched,
        "rule_match_count": len(matched),
    }


def build_knowledge_guided_distribution(
    controls: pd.DataFrame,
    pressure: pd.DataFrame,
    calibration_steps: np.ndarray,
    base_spread: np.ndarray,
    base_process: np.ndarray,
    config: KnowledgeGuidedPriorConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Backward-compatible wrapper returning spread and process noise only."""

    result = build_knowledge_guided_prior(
        controls=controls,
        pressure=pressure,
        calibration_steps=calibration_steps,
        prior_mean=np.zeros(len(np.asarray(base_spread))),
        base_spread=base_spread,
        base_process=base_process,
        config=config,
    )
    return (
        np.asarray(result["spread"], dtype=float),
        np.asarray(result["process"], dtype=float),
        dict(result["meta"]),
    )


def _nearest_correlation_matrix(matrix: np.ndarray) -> np.ndarray:
    """Project a symmetric correlation proposal to a valid PSD matrix."""

    symmetric = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    projected = (eigenvectors * np.clip(eigenvalues, 1.0e-8, None)) @ eigenvectors.T
    diagonal = np.sqrt(np.maximum(np.diag(projected), 1.0e-12))
    projected = projected / np.outer(diagonal, diagonal)
    np.fill_diagonal(projected, 1.0)
    return projected


def _rule_correlation_matrix(signals: dict[str, Any], scale: float) -> np.ndarray:
    """Build a small, interpretable correlation structure for five PKN states."""

    pressure = float(signals.get("pressure_rise_score", 0.0))
    sand = float(signals.get("high_sand_score", 0.0))
    flow_down = float(signals.get("flow_down_score", 0.0))
    risk = float(signals.get("risk_score", 0.0))
    corr = np.eye(5, dtype=float)
    # Pressure anomalies have two plausible explanations in this reduced
    # model: leakoff and minimum stress.  Keep them correlated rather than
    # forcing the filter to choose one explanation from a single gauge.
    corr[1, 3] = corr[3, 1] = 0.35 * pressure * scale
    # High sand and falling rate increase the coupling between viscosity and
    # leakoff hypotheses, but the coefficient remains deliberately modest.
    corr[1, 2] = corr[2, 1] = 0.18 * max(sand, flow_down) * scale
    corr[0, 3] = corr[3, 0] = 0.10 * risk * scale
    return _nearest_correlation_matrix(corr)


def build_knowledge_guided_prior(
    controls: pd.DataFrame,
    pressure: pd.DataFrame,
    calibration_steps: np.ndarray,
    prior_mean: np.ndarray,
    base_spread: np.ndarray,
    base_process: np.ndarray,
    config: KnowledgeGuidedPriorConfig,
) -> dict[str, Any]:
    """Construct a KG-informed parameter ensemble distribution.

    State order is ``[log E', log C_L, log mu, sigma_min, log K_IC]``.  The
    signed mean shift is intentionally small and only represents a prior
    hypothesis: rising pressure with high sand makes higher leakoff/stress
    explanations slightly more plausible.  Real observations still determine
    the posterior through the standard Kalman update.
    """

    base_spread = np.asarray(base_spread, dtype=float)
    base_process = np.asarray(base_process, dtype=float)
    prior_mean = np.asarray(prior_mean, dtype=float)
    if not (len(prior_mean) == len(base_spread) == len(base_process)):
        raise ValueError("prior_mean, base_spread and base_process must have the same length")
    if len(prior_mean) != 5:
        raise ValueError("KG-EnKF currently expects the five global PKN parameters")
    mode = str(config.mode).strip().lower()
    if mode not in VALID_MODES:
        raise ValueError(f"unsupported KG-EnKF mode: {mode}; expected one of {sorted(VALID_MODES)}")
    rule_ids, rule_meta = _load_rule_ids(config.rules_path)
    signals = infer_knowledge_signals(controls, pressure, calibration_steps, rule_ids)
    active_mode = mode if config.enabled and mode != "off" else "off"
    if active_mode == "off":
        identity = np.eye(len(prior_mean), dtype=float)
        meta = {
            "enabled": False,
            "mode": "off",
            "state_parameter_names": STATE_PARAMETER_NAMES,
            "prior_mean_shift": [0.0] * len(base_spread),
            "prior_mean": prior_mean.tolist(),
            "correlation_matrix": identity.tolist(),
            "initial_spread": base_spread.tolist(),
            "process_noise": base_process.tolist(),
            "observation_noise_multiplier": {"share": 1.0, "pressure": 1.0},
            "max_update_scale": None,
            "rule_source": rule_meta,
            "signals": signals,
        }
        return {
            "mean": prior_mean.copy(),
            "spread": base_spread.copy(),
            "process": base_process.copy(),
            "covariance": np.diag(base_spread**2),
            "process_covariance": np.diag(base_process**2),
            "meta": meta,
        }

    strength = float(np.clip(config.strength, 0.0, 1.0))
    risk = float(signals.get("risk_score", 0.0))
    # Conservative uncertainty-only mapping.  C_L and sigma_min receive the
    # largest expansion because the selected rules primarily describe pressure
    # rise, leakoff/near-wellbore ambiguity, and blockage-like behavior.
    spread_multiplier = 1.0 + strength * risk * np.asarray([0.25, 0.75, 0.40, 0.70, 0.35])
    process_multiplier = 1.0 + 0.50 * strength * risk * np.asarray([0.20, 0.70, 0.35, 0.60, 0.30])
    adjusted_spread = base_spread * spread_multiplier
    adjusted_process = base_process * process_multiplier

    mean_shift = np.zeros_like(prior_mean)
    if active_mode in {"soft_prior", "soft_correlated"}:
        pressure_context = float(signals.get("pressure_rise_score", 0.0))
        sand_context = float(signals.get("high_sand_score", 0.0))
        flow_context = float(signals.get("flow_down_score", 0.0))
        # State-ordered engineering hypothesis, not a direct answer:
        # pressure rise may be explained by leakoff/stress; high sand or rate
        # decline slightly increases viscosity uncertainty.  K_IC is left
        # centered because the current rule set does not identify toughness.
        direction = np.asarray(
            [0.0, 0.10 * pressure_context, 0.04 * max(sand_context, flow_context), 0.80 * pressure_context, 0.0],
            dtype=float,
        )
        mean_shift = strength * risk * float(config.mean_shift_scale) * direction

    correlation = np.eye(len(prior_mean), dtype=float)
    if active_mode == "soft_correlated":
        correlation = _rule_correlation_matrix(signals, float(config.covariance_scale))
    covariance = np.diag(adjusted_spread) @ correlation @ np.diag(adjusted_spread)
    process_covariance = np.diag(adjusted_process) @ correlation @ np.diag(adjusted_process)
    pressure_multiplier = 1.0
    if active_mode == "soft_correlated":
        # A pressure rise can represent unmodelled near-wellbore/friction
        # effects.  Reduce overconfidence in that one channel rather than
        # forcing all parameter updates in the same direction.
        pressure_multiplier += float(config.observation_noise_scale) * strength * risk
    max_update_scale = None
    if active_mode == "soft_correlated":
        max_update_scale = max(0.75, float(config.max_update_scale) * (1.0 - 0.25 * risk))
    meta = {
        "enabled": True,
        "mode": active_mode,
        "strength": strength,
        "state_parameter_names": STATE_PARAMETER_NAMES,
        "prior_mean_shift": mean_shift.tolist(),
        "prior_mean": (prior_mean + mean_shift).tolist(),
        "spread_multiplier": spread_multiplier.tolist(),
        "process_multiplier": process_multiplier.tolist(),
        "initial_spread": adjusted_spread.tolist(),
        "process_noise": adjusted_process.tolist(),
        "correlation_matrix": correlation.tolist(),
        "observation_noise_multiplier": {"share": 1.0, "pressure": pressure_multiplier},
        "max_update_scale": max_update_scale,
        "rule_source": rule_meta,
        "signals": signals,
        "mapping_note": (
            "The graph supplies a soft prior hypothesis and uncertainty structure. "
            "EnKF still performs the observation update and the PKN operator is "
            "re-run after parameter analysis; fracture length is never assigned "
            "from the graph or observation directly."
        ),
    }
    return {
        "mean": prior_mean + mean_shift,
        "spread": adjusted_spread,
        "process": adjusted_process,
        "covariance": covariance,
        "process_covariance": process_covariance,
        "meta": meta,
    }


def apply_knowledge_guided_observation_std(
    observation_std: np.ndarray,
    meta: dict[str, Any],
    n_clusters: int,
) -> np.ndarray:
    """Apply KG confidence only to the observation channel it can explain."""

    result = np.asarray(observation_std, dtype=float).copy()
    multipliers = meta.get("observation_noise_multiplier", {})
    share = float(multipliers.get("share", 1.0))
    pressure = float(multipliers.get("pressure", 1.0))
    if result.size >= 2 * max(n_clusters - 1, 0) + 1:
        result[: 2 * (n_clusters - 1)] *= max(share, 1.0e-6)
        result[-1] *= max(pressure, 1.0e-6)
    return result


def project_knowledge_guided_update(
    updated: np.ndarray,
    reference: np.ndarray,
    spread: np.ndarray,
    meta: dict[str, Any],
) -> np.ndarray:
    """Limit a KG-guided analysis jump without replacing physical clipping."""

    limit = meta.get("max_update_scale")
    if limit is None:
        return np.asarray(updated, dtype=float)
    delta = np.asarray(updated, dtype=float) - np.asarray(reference, dtype=float)
    max_delta = max(float(limit), 0.0) * np.maximum(np.asarray(spread, dtype=float), 1.0e-9)
    return np.asarray(reference, dtype=float) + np.clip(delta, -max_delta, max_delta)
