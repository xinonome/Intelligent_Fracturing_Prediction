from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .exceptions import ConfigurationError, DataValidationError
from .schemas import ProjectConfig, StageDefinition, StageMaterial
from .stage_definition import slice_stage_logs
from .trajectory import interpolate_at_md


def build_stress_interpolator(logs: pd.DataFrame):
    if len(logs) < 2:
        raise DataValidationError("at least two log samples are required for stress interpolation")
    md = logs["MD_m"].to_numpy(dtype=float)
    values = logs["sigma_hmin_Pa"].to_numpy(dtype=float)

    def stress_at_md(md_m: float) -> float:
        if md_m < md.min() or md_m > md.max():
            raise DataValidationError(f"stress interpolation would extrapolate at MD={md_m}")
        return float(np.interp(md_m, md, values))

    return stress_at_md


def _source_value(logs: pd.DataFrame, config: dict, column: str, mode_key: str, constant_key: str, label: str) -> float:
    mode = str(config[mode_key])
    if mode == "constant":
        value = config.get(constant_key)
        if value is None or float(value) < 0:
            raise ConfigurationError(f"{label} constant value is required and must be non-negative")
        return float(value)
    if mode == "off":
        return 0.0
    if column not in logs:
        raise ConfigurationError(f"{label} source=log requires column {column}")
    return float(logs[column].median())


def build_stage_material(logs: pd.DataFrame, trajectory: pd.DataFrame, stage: StageDefinition, config: ProjectConfig) -> StageMaterial:
    stage_logs = slice_stage_logs(logs, stage)
    if stage_logs.empty:
        raise DataValidationError(f"stage {stage.stage_id} has no logs")
    e_samples = stage_logs["young_modulus_Pa"].to_numpy(dtype=float) / (1.0 - stage_logs["poisson_ratio"].to_numpy(dtype=float) ** 2)
    e_median = float(np.median(e_samples))
    spread = float((e_samples.max() - e_samples.min()) / max(e_median, 1.0e-30))
    warnings: list[str] = []
    threshold = float(config.material.get("e_prime_warning_relative_spread", 0.2))
    if spread > threshold:
        warnings.append(f"Eprime relative spread {spread:.3f} exceeds warning threshold {threshold:.3f}")
    stress_fn = build_stress_interpolator(logs)
    center_stress = stress_fn(stage.center_md_m)
    kic = _source_value(stage_logs, config.material["toughness"], "KIC_Pa_sqrt_m", "mode", "constant_pa_sqrt_m", "toughness")
    carter = _source_value(stage_logs, config.material["leakoff"], "carter_m_sqrt_s", "mode", "constant_m_sqrt_s", "leakoff")
    return StageMaterial(
        e_prime_pa=e_median,
        e_prime_min_pa=float(e_samples.min()),
        e_prime_max_pa=float(e_samples.max()),
        e_prime_std_pa=float(np.std(e_samples)),
        poisson_ratio_median=float(stage_logs["poisson_ratio"].median()),
        stress_center_pa=center_stress,
        stress_min_pa=float(stage_logs["sigma_hmin_Pa"].min()),
        stress_max_pa=float(stage_logs["sigma_hmin_Pa"].max()),
        toughness_pa_sqrt_m=kic,
        carter_m_sqrt_s=carter,
        warnings=tuple(warnings),
    )


def local_stress_function(logs: pd.DataFrame, stage: StageDefinition, trajectory: pd.DataFrame):
    stress_fn = build_stress_interpolator(logs)
    center_tvd = interpolate_at_md(trajectory, stage.center_md_m)["TVD_m"]
    md = logs["MD_m"].to_numpy(dtype=float)
    tvd = np.interp(md, trajectory["MD_m"].to_numpy(dtype=float), trajectory["TVD_m"].to_numpy(dtype=float))
    order = np.argsort(tvd)
    tvd_sorted = tvd[order]
    md_sorted = md[order]
    if np.any(np.diff(tvd_sorted) <= 0):
        raise DataValidationError("trajectory/log TVD samples are not strictly invertible for local stress mapping")
    tvd_min, tvd_max = float(tvd_sorted.min()), float(tvd_sorted.max())

    def confining_stress_local(_x: float, v_local_m: float) -> float:
        global_tvd = center_tvd - float(v_local_m)
        if global_tvd < tvd_min or global_tvd > tvd_max:
            raise DataValidationError("local stress lookup would extrapolate outside available TVD")
        md_guess = float(np.interp(global_tvd, tvd_sorted, md_sorted))
        return stress_fn(md_guess)

    return confining_stress_local
