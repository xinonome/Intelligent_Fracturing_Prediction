from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = ROOT / "scenario_config.yaml"


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    display_name: str
    description: str
    pressure_offset_mpa: float
    pressure_trend_mpa: float
    flow_multiplier: float
    sand_multiplier: float
    posterior_error_base: float
    posterior_error_trend: float
    abnormal_probability_base: float
    abnormal_probability_trend: float
    sand_plug_probability_base: float
    sand_plug_probability_trend: float
    length_growth_multiplier: float
    balance_imbalance: float

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load scenario_config.yaml") from exc
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if "scenarios" not in data:
        raise ValueError(f"Scenario config has no 'scenarios' section: {path}")
    return data


def load_scenarios(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, ScenarioSpec]:
    config_path = Path(path)
    raw = _load_yaml(config_path)["scenarios"]
    scenarios: dict[str, ScenarioSpec] = {}
    for name, values in raw.items():
        scenarios[name] = ScenarioSpec(
            name=name,
            display_name=str(values.get("display_name", name)),
            description=str(values.get("description", "")),
            pressure_offset_mpa=float(values.get("pressure_offset_mpa", 0.0)),
            pressure_trend_mpa=float(values.get("pressure_trend_mpa", 0.0)),
            flow_multiplier=float(values.get("flow_multiplier", 1.0)),
            sand_multiplier=float(values.get("sand_multiplier", 1.0)),
            posterior_error_base=float(values.get("posterior_error_base", 0.10)),
            posterior_error_trend=float(values.get("posterior_error_trend", 0.0)),
            abnormal_probability_base=float(values.get("abnormal_probability_base", 0.05)),
            abnormal_probability_trend=float(values.get("abnormal_probability_trend", 0.0)),
            sand_plug_probability_base=float(values.get("sand_plug_probability_base", 0.02)),
            sand_plug_probability_trend=float(values.get("sand_plug_probability_trend", 0.0)),
            length_growth_multiplier=float(values.get("length_growth_multiplier", 1.0)),
            balance_imbalance=float(values.get("balance_imbalance", 0.10)),
        )
    return scenarios


def available_scenarios(path: str | Path = DEFAULT_CONFIG_PATH) -> list[str]:
    return sorted(load_scenarios(path))


def _progress(size: int) -> np.ndarray:
    if size <= 1:
        return np.zeros(size, dtype=float)
    return np.linspace(0.0, 1.0, size, dtype=float)


def _ensure_context_column(context: pd.DataFrame, name: str, values: np.ndarray) -> None:
    if name in context:
        base = pd.to_numeric(context[name], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(base)
        merged = values.copy()
        merged[finite] = 0.5 * base[finite] + 0.5 * values[finite]
        context[name] = merged
    else:
        context[name] = values


def apply_scenario(
    features: np.ndarray,
    meta: pd.DataFrame,
    context: pd.DataFrame,
    scenario_name: str,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    feature_names: list[str] | None = None,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return scenario-adjusted training arrays without changing row count.

    The current platform has historical data but limited same-time simulator
    feedback. Scenario adjustment creates controlled stress cases for strategy
    training while preserving the original feature shape expected by SB3.
    """

    scenarios = load_scenarios(config_path)
    if scenario_name not in scenarios:
        raise ValueError(f"Unsupported scenario: {scenario_name}. Choices: {sorted(scenarios)}")
    spec = scenarios[scenario_name]
    x = np.asarray(features, dtype=np.float32).copy()
    scenario_meta = meta.reset_index(drop=True).copy()
    scenario_context = context.reset_index(drop=True).copy()
    size = len(scenario_meta)
    p = _progress(size)

    if size == 0:
        return x, scenario_meta, scenario_context, spec.to_dict()

    if "current_pressure" in scenario_meta:
        pressure = pd.to_numeric(scenario_meta["current_pressure"], errors="coerce").to_numpy(dtype=float)
        fill = np.nanmedian(pressure) if np.isfinite(pressure).any() else 70.0
        pressure = np.nan_to_num(pressure, nan=fill)
        scenario_meta["current_pressure"] = pressure + spec.pressure_offset_mpa + spec.pressure_trend_mpa * p
    if "current_flow" in scenario_meta:
        scenario_meta["current_flow"] = np.clip(
            pd.to_numeric(scenario_meta["current_flow"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            * spec.flow_multiplier,
            0.0,
            None,
        )
    if "current_sand_ratio" in scenario_meta:
        scenario_meta["current_sand_ratio"] = np.clip(
            pd.to_numeric(scenario_meta["current_sand_ratio"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            * spec.sand_multiplier,
            0.0,
            None,
        )

    # Keep the 300-second observation window consistent with the scenario.
    # Historical-window columns retain their temporal shape; only engineering
    # magnitudes are transformed. Slope features scale but are not offset.
    if feature_names and len(feature_names) == x.shape[1]:
        progress = p.astype(np.float32)
        for col_idx, name in enumerate(feature_names):
            if name.startswith("SGBY_"):
                x[:, col_idx] *= np.float32(1.0)
                if not name.endswith("_std") and not name.endswith("_slope"):
                    x[:, col_idx] += (spec.pressure_offset_mpa + spec.pressure_trend_mpa * progress).astype(np.float32)
                elif name.endswith("_slope"):
                    x[:, col_idx] += np.float32(spec.pressure_trend_mpa / max(size - 1, 1))
            elif name.startswith("PL_"):
                x[:, col_idx] *= np.float32(spec.flow_multiplier)
            elif name.startswith("SB_"):
                x[:, col_idx] *= np.float32(spec.sand_multiplier)

    base_length = None
    if "posterior_total_half_length_m" in scenario_context:
        base_length = pd.to_numeric(scenario_context["posterior_total_half_length_m"], errors="coerce").to_numpy(dtype=float)
    if base_length is None or not np.isfinite(base_length).any():
        base_length = 100.0 + 80.0 * p
    base_length = pd.Series(base_length).interpolate(limit_direction="both").fillna(100.0).to_numpy(dtype=float)
    start = float(base_length[0])
    scenario_length = start + np.maximum(base_length - start, 0.0) * spec.length_growth_multiplier
    _ensure_context_column(scenario_context, "posterior_total_half_length_m", scenario_length)
    _ensure_context_column(scenario_context, "posterior_error", np.clip(spec.posterior_error_base + spec.posterior_error_trend * p, 0.0, 1.0))
    _ensure_context_column(scenario_context, "abnormal_probability", np.clip(spec.abnormal_probability_base + spec.abnormal_probability_trend * p, 0.0, 1.0))
    _ensure_context_column(scenario_context, "sand_plug_probability", np.clip(spec.sand_plug_probability_base + spec.sand_plug_probability_trend * p, 0.0, 1.0))

    if "bottomhole_pressure_mpa" in scenario_context:
        bhp = pd.to_numeric(scenario_context["bottomhole_pressure_mpa"], errors="coerce").to_numpy(dtype=float)
        fill = np.nanmedian(bhp) if np.isfinite(bhp).any() else np.nanmedian(scenario_meta["current_pressure"])
        bhp = np.nan_to_num(bhp, nan=fill)
    else:
        bhp = pd.to_numeric(scenario_meta.get("current_pressure", pd.Series(np.full(size, 70.0))), errors="coerce").fillna(70.0).to_numpy(dtype=float)
    _ensure_context_column(scenario_context, "bottomhole_pressure_mpa", bhp + spec.pressure_offset_mpa + spec.pressure_trend_mpa * p)
    net = np.maximum(scenario_context["bottomhole_pressure_mpa"].to_numpy(dtype=float) - 70.0, 0.0)
    _ensure_context_column(scenario_context, "net_pressure_mpa", net)
    scenario_context["scenario_name"] = spec.name
    scenario_context["scenario_balance_imbalance"] = spec.balance_imbalance
    scenario_meta["scenario_name"] = spec.name
    return x, scenario_meta, scenario_context, spec.to_dict()
