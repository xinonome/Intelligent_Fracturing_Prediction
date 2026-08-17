from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PressureModelConfig:
    """Parameters for converting surface pressure to bottom-hole/net pressure.

    Defaults are engineering placeholders for making the pipeline runnable.
    They should be replaced by client-confirmed values before quantitative use.
    """

    fcd: float = 1.0e-10
    dwell_m: float = 0.10
    proppant_density_kg_m3: float = 2650.0
    base_fluid_density_kg_m3: float = 1000.0
    jzl: float = 0.5
    perforation_count: int = 48
    perforation_diameter_m: float = 0.010
    perforation_erosion_coeff: float = 0.0
    perforation_flow_coeff: float = 0.85
    perforation_flow_decay_coeff: float = 0.0
    min_horizontal_stress_mpa: float = 60.0
    rolling_window_seconds: int = 30
    step_seconds: float = 1.0
    pressure_bias_mpa: float = 0.0
    calibration_status: str = "engineering_default"
    calibration_id: str | None = None


def load_pressure_model_config(path: str | Path | None = None) -> PressureModelConfig:
    if path is None or not Path(path).exists():
        return PressureModelConfig()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    fields = {key: value for key, value in payload.items() if key in PressureModelConfig.__dataclass_fields__}
    return PressureModelConfig(**fields)


def load_stage_pressure_schedule(
    path: str | Path,
    config: PressureModelConfig | None = None,
    measured_depth_m: float = 5200.0,
    vertical_depth_m: float = 3200.0,
) -> tuple[pd.DataFrame, dict]:
    """Load client stage pressure XLS and compute pressure terms.

    Expected current stage-08 XLS layout by column index:
    0 second index, 1-4 liquid split, 5-8 sand split, 9 surface pressure MPa,
    10 cumulative liquid m3, 11 sand ratio percent, 12 auxiliary stage/displacement value.
    """

    config = config or PressureModelConfig()
    raw = pd.read_excel(path)
    if raw.shape[1] < 13:
        raise ValueError(f"Pressure schedule requires at least 13 columns, got {raw.shape[1]}")

    columns = _resolve_columns(raw)

    out = pd.DataFrame()
    out["source_second"] = pd.to_numeric(raw[columns["source_second"]], errors="coerce").ffill().fillna(1).astype(int)
    out["step"] = (out["source_second"] - 1).clip(lower=0)
    out["time_s"] = out["step"] * float(config.step_seconds)
    out["surface_pressure_mpa"] = pd.to_numeric(raw[columns["surface_pressure_mpa"]], errors="coerce").interpolate().ffill().bfill()
    out["cumulative_liquid_m3"] = pd.to_numeric(raw[columns["cumulative_liquid_m3"]], errors="coerce").interpolate().ffill().bfill()
    out["sand_ratio_percent"] = pd.to_numeric(raw[columns["sand_ratio_percent"]], errors="coerce").interpolate().ffill().bfill()
    out["aux_displacement"] = pd.to_numeric(raw[columns["aux_displacement"]], errors="coerce").fillna(0.0)
    out["liquid_split_sum"] = raw.iloc[:, 1:5].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)
    out["sand_split_sum"] = raw.iloc[:, 5:9].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)

    if columns.get("flow_rate_m3_min") is not None:
        measured_flow = pd.to_numeric(raw[columns["flow_rate_m3_min"]], errors="coerce")
        if measured_flow.notna().sum() >= max(3, len(raw) // 20):
            out["flow_rate_m3_min"] = measured_flow.interpolate().ffill().bfill().clip(lower=0.0)
            flow_source = "measured_flow_rate_column"
        else:
            out["flow_rate_m3_min"], flow_source = _derive_flow(out, config)
    else:
        out["flow_rate_m3_min"], flow_source = _derive_flow(out, config)
    out["flow_rate_m3_s"] = out["flow_rate_m3_min"] / 60.0

    pressure_terms = compute_pressure_terms(
        out,
        config=config,
        measured_depth_m=measured_depth_m,
        vertical_depth_m=vertical_depth_m,
    )
    out = pd.concat([out, pressure_terms], axis=1)

    meta = {
        "source": str(path),
        "rows": int(len(out)),
        "layout_assumption": {
            "source_columns": columns,
            "flow_source": flow_source,
        },
        "measured_depth_m": float(measured_depth_m),
        "vertical_depth_m": float(vertical_depth_m),
        "config": asdict(config),
        "calibration_status": config.calibration_status,
        "calibration_id": config.calibration_id,
        "warning": "Friction, depth and stress parameters are engineering defaults until client calibration is supplied.",
    }
    return out, meta


def compute_pressure_terms(
    schedule: pd.DataFrame,
    config: PressureModelConfig,
    measured_depth_m: float,
    vertical_depth_m: float,
) -> pd.DataFrame:
    sand_fraction = np.clip(schedule["sand_ratio_percent"].to_numpy(dtype=float) / 100.0, 0.0, 0.95)
    slurry_density = config.base_fluid_density_kg_m3 * (1.0 - sand_fraction) + config.proppant_density_kg_m3 * sand_fraction
    hydrostatic_mpa = slurry_density * 9.80665 * float(vertical_depth_m) / 1.0e6

    q_m3_min = schedule["flow_rate_m3_min"].rolling(config.rolling_window_seconds, min_periods=1).mean().to_numpy(dtype=float)
    q_m3_min = np.clip(q_m3_min, 0.0, None)
    mzl = 1.0 - (((config.jzl - 0.5) / (21.0 - 2.0)) * (q_m3_min - 2.0) + 0.5)
    mzl = np.clip(mzl, 0.05, 2.0)
    pipe_friction_mpa = (
        config.fcd
        * 1.385e6
        * max(config.dwell_m, 1e-9) ** (-4.8)
        * np.power(q_m3_min, 1.6)
        * float(measured_depth_m)
        * mzl
        / 1.0e6
    )

    perf_friction_mpa, perforation_diameter_m, perforation_flow_coeff = _perforation_friction(schedule, config)

    surface = schedule["surface_pressure_mpa"].to_numpy(dtype=float)
    bottomhole = surface + hydrostatic_mpa - pipe_friction_mpa - perf_friction_mpa + float(config.pressure_bias_mpa)
    net_raw = bottomhole - config.min_horizontal_stress_mpa

    return pd.DataFrame(
        {
            "slurry_density_kg_m3": slurry_density,
            "hydrostatic_pressure_mpa": hydrostatic_mpa,
            "pipe_friction_mpa": pipe_friction_mpa,
            "perforation_friction_mpa": perf_friction_mpa,
            "bottomhole_pressure_mpa": bottomhole,
            "net_pressure_raw_mpa": net_raw,
            "net_pressure_mpa": np.clip(net_raw, 0.0, None),
            "perforation_diameter_m": perforation_diameter_m,
            "perforation_flow_coeff": perforation_flow_coeff,
        }
    )


def _derive_flow(schedule: pd.DataFrame, config: PressureModelConfig) -> tuple[pd.Series, str]:
    delta_liquid = schedule["cumulative_liquid_m3"].diff()
    positive_delta = delta_liquid.where(delta_liquid > 0)
    fallback_delta = float(positive_delta.median()) if positive_delta.notna().any() else 0.0
    delta_liquid = delta_liquid.fillna(fallback_delta).clip(lower=0.0)
    return delta_liquid / max(float(config.step_seconds), 1e-9) * 60.0, "derived_from_cumulative_liquid"


def _resolve_columns(raw: pd.DataFrame) -> dict[str, object]:
    aliases = {
        "source_second": ["序号", "second", "time_s", "time", "秒"],
        "surface_pressure_mpa": ["泵压(MPa)", "泵压", "井口压力", "surface_pressure_mpa", "pressure_mpa"],
        "cumulative_liquid_m3": ["总液量", "累计液量", "cumulative_liquid_m3", "total_liquid"],
        "sand_ratio_percent": ["砂比", "砂比(%)", "sand_ratio_percent", "sand_ratio"],
        "flow_rate_m3_min": ["排出排量", "排量", "flow_rate_m3_min", "flow_rate"],
        "aux_displacement": ["辅助", "位移", "排量辅助", "aux_displacement"],
    }
    names = {str(name).strip(): name for name in raw.columns}
    result: dict[str, object] = {}
    positional = {"source_second": 0, "surface_pressure_mpa": 9, "cumulative_liquid_m3": 10, "sand_ratio_percent": 11, "aux_displacement": 12}
    for key, options in aliases.items():
        result[key] = next((names[name] for name in options if name in names), None)
        if result[key] is None and key in positional:
            result[key] = raw.columns[positional[key]]
    for key in ("source_second", "surface_pressure_mpa", "cumulative_liquid_m3", "sand_ratio_percent", "aux_displacement"):
        if result.get(key) is None:
            raise ValueError(f"pressure schedule is missing required field: {key}")
    return result


def pressure_for_step(schedule: pd.DataFrame, step: int | float) -> pd.Series:
    if schedule.empty:
        raise ValueError("Pressure schedule is empty")
    idx = (schedule["step"].to_numpy(dtype=float) - float(step)).astype(float)
    row_idx = int(np.argmin(np.abs(idx)))
    return schedule.iloc[row_idx]


def _perforation_friction(schedule: pd.DataFrame, config: PressureModelConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(schedule)
    fric = np.zeros(n, dtype=float)
    diam = np.zeros(n, dtype=float)
    cky = np.zeros(n, dtype=float)
    current_d = float(config.perforation_diameter_m)
    current_c = float(config.perforation_flow_coeff)
    cumulative = schedule["cumulative_liquid_m3"].to_numpy(dtype=float)
    delta_liquid = np.diff(np.r_[cumulative[0], cumulative])
    q_m3_s = np.clip(schedule["flow_rate_m3_s"].to_numpy(dtype=float), 0.0, None)
    sand_ratio = np.clip(schedule["sand_ratio_percent"].to_numpy(dtype=float), 0.0, None)

    for i in range(n):
        c_sand = sand_ratio[i] * 15.0
        denom = max(config.perforation_count**2 * np.pi**2 * current_d**4, 1e-18)
        if config.perforation_erosion_coeff:
            current_d += 16.0 * config.perforation_erosion_coeff * c_sand * q_m3_s[i] * max(delta_liquid[i], 0.0) / denom
        if config.perforation_flow_decay_coeff:
            current_c += (
                config.perforation_flow_decay_coeff
                * c_sand
                * (1.0 - current_c / 0.95)
                * 16.0
                * q_m3_s[i]
                * max(delta_liquid[i], 0.0)
                / denom
            )
        current_c = float(np.clip(current_c, 0.05, 0.95))
        current_d = float(max(current_d, 1e-5))
        fric[i] = (
            0.8081e-6
            * config.proppant_density_kg_m3
            / max(config.perforation_count**2 * current_d**4 * current_c**2, 1e-18)
            * q_m3_s[i] ** 2
        )
        diam[i] = current_d
        cky[i] = current_c
    return fric, diam, cky
