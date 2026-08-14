from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .exceptions import ConfigurationError, DataValidationError
from .schemas import ProjectConfig, StageDefinition

LOG_REQUIRED = {"MD_m", "GR_API", "poisson_ratio", "young_modulus_Pa", "sigma_hmin_Pa"}
TRAJECTORY_REQUIRED = {"MD_m", "X_m", "Y_m", "TVD_m"}


def _numeric(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    for column in columns:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise DataValidationError(f"{label}.{column} must be numeric")
        if frame[column].isna().any() or not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise DataValidationError(f"{label}.{column} contains NaN or infinite values")


def validate_logs(frame: pd.DataFrame, stages: tuple[StageDefinition, ...]) -> list[str]:
    missing = LOG_REQUIRED - set(frame.columns)
    if missing:
        raise DataValidationError(f"logs missing required columns: {sorted(missing)}")
    optional = {"KIC_Pa_sqrt_m", "carter_m_sqrt_s"} & set(frame.columns)
    _numeric(frame, LOG_REQUIRED | optional, "logs")
    md = frame["MD_m"].to_numpy(dtype=float)
    if len(md) < 2 or np.any(np.diff(md) <= 0):
        raise DataValidationError("logs MD_m must be strictly increasing; duplicates are not allowed")
    if ((frame["poisson_ratio"] <= 0) | (frame["poisson_ratio"] >= 0.5)).any():
        raise DataValidationError("logs poisson_ratio must satisfy 0 < nu < 0.5")
    if (frame["young_modulus_Pa"] <= 0).any() or (frame["sigma_hmin_Pa"] <= 0).any():
        raise DataValidationError("logs modulus and minimum horizontal stress must be positive")
    for column in optional:
        if (frame[column] < 0).any():
            raise DataValidationError(f"logs {column} must be non-negative")
    warnings: list[str] = []
    for stage in stages:
        if stage.md_start_m < md.min() or stage.md_end_m > md.max():
            raise DataValidationError(
                f"stage {stage.stage_id} [{stage.md_start_m}, {stage.md_end_m}] is outside logs [{md.min()}, {md.max()}]"
            )
        if not ((frame["MD_m"] >= stage.md_start_m) & (frame["MD_m"] <= stage.md_end_m)).any():
            raise DataValidationError(f"stage {stage.stage_id} has no log samples")
    return warnings


def validate_trajectory(frame: pd.DataFrame, stages: tuple[StageDefinition, ...]) -> None:
    missing = TRAJECTORY_REQUIRED - set(frame.columns)
    if missing:
        raise DataValidationError(f"trajectory missing required columns: {sorted(missing)}")
    _numeric(frame, TRAJECTORY_REQUIRED, "trajectory")
    md = frame["MD_m"].to_numpy(dtype=float)
    if len(md) < 2 or np.any(np.diff(md) <= 0):
        raise DataValidationError("trajectory MD_m must be strictly increasing")
    for stage in stages:
        if stage.md_start_m < md.min() or stage.md_end_m > md.max():
            raise DataValidationError(f"stage {stage.stage_id} is outside trajectory range")


def validate_config_values(raw: dict[str, Any]) -> None:
    def required(path: str) -> Any:
        value: Any = raw
        for part in path.split("."):
            if not isinstance(value, dict) or part not in value:
                raise ConfigurationError(f"missing required config field: {path}")
            value = value[part]
        if value is None or (isinstance(value, str) and value.strip().upper() == "REQUIRED"):
            raise ConfigurationError(f"unresolved REQUIRED config field: {path}")
        return value

    for path in (
        "project.output_root", "inputs.logs_csv", "inputs.trajectory_csv",
        "injection.rate_m3_s", "injection.duration_s", "fluid.viscosity_pa_s",
        "fluid.density_kg_m3", "material.toughness.mode", "material.leakoff.mode",
        "simulation.final_time_s", "simulation.save_every_s", "simulation.fracture_height_m",
        "mesh.x_half_extent_m", "mesh.y_half_extent_m", "mesh.nx", "mesh.ny",
        "diagnostics.barrier_detection.tolerance_m", "diagnostics.handover.min_consecutive_snapshots",
        "diagnostics.handover.vertical_growth_epsilon_m_s", "diagnostics.handover.lateral_growth_min_m_s",
        "runtime.max_memory_gb", "validation.max_mass_balance_relative_error",
    ):
        required(path)
    if float(required("injection.rate_m3_s")) <= 0 or float(required("injection.duration_s")) <= 0:
        raise ConfigurationError("injection rate and duration must be positive")
    if float(required("fluid.viscosity_pa_s")) <= 0 or float(required("fluid.density_kg_m3")) <= 0:
        raise ConfigurationError("fluid viscosity and density must be positive")
    if float(required("simulation.final_time_s")) <= 0 or float(required("simulation.save_every_s")) <= 0:
        raise ConfigurationError("simulation times must be positive")
    if float(required("simulation.fracture_height_m")) <= 0:
        raise ConfigurationError("fracture height must be positive")
    for field in ("x_half_extent_m", "y_half_extent_m"):
        if float(required(f"mesh.{field}")) <= 0:
            raise ConfigurationError(f"mesh.{field} must be positive")
    for field in ("nx", "ny"):
        value = int(required(f"mesh.{field}"))
        if value < 3 or value % 2 == 0:
            raise ConfigurationError(f"mesh.{field} must be an odd integer >= 3")
    if raw.get("simulation", {}).get("front_advancing", "predictor-corrector") != "predictor-corrector":
        raise ConfigurationError("V1 requires predictor-corrector front advancement")
    mode = str(raw.get("simulation", {}).get("mode", "snapshot"))
    if mode not in {"snapshot", "native"}:
        raise ConfigurationError("simulation.mode must be snapshot or native")
    stages = raw.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ConfigurationError("stages must be a non-empty list")
    ids: set[str] = set()
    for item in stages:
        if not isinstance(item, dict) or not {"id", "md_start_m", "md_end_m"} <= set(item):
            raise ConfigurationError("each stage requires id, md_start_m and md_end_m")
        if item["id"] in ids:
            raise ConfigurationError(f"duplicate stage id: {item['id']}")
        ids.add(item["id"])
        if float(item["md_end_m"]) <= float(item["md_start_m"]):
            raise ConfigurationError(f"stage {item['id']} must have md_end_m > md_start_m")
    if str(raw["material"]["toughness"]["mode"]) not in {"log", "constant"}:
        raise ConfigurationError("material.toughness.mode must be log or constant")
    if str(raw["material"]["leakoff"]["mode"]) not in {"log", "constant", "off"}:
        raise ConfigurationError("material.leakoff.mode must be log, constant or off")
    if raw["material"]["toughness"]["mode"] == "constant" and float(required("material.toughness.constant_pa_sqrt_m")) < 0:
        raise ConfigurationError("constant toughness must be non-negative")
    if raw["material"]["leakoff"]["mode"] == "constant" and float(required("material.leakoff.constant_m_sqrt_s")) < 0:
        raise ConfigurationError("constant Carter coefficient must be non-negative")
    mapping = raw.get("mapping", {})
    if mapping.get("plane_mode", "transverse_vertical") != "transverse_vertical":
        raise ConfigurationError("V1 mapping.plane_mode must be transverse_vertical")
    if mapping.get("fracture_azimuth_deg") is not None and not -360.0 <= float(mapping["fracture_azimuth_deg"]) <= 360.0:
        raise ConfigurationError("fracture_azimuth_deg must be within [-360, 360]")


def validate_input_files(config: ProjectConfig, logs: pd.DataFrame, trajectory: pd.DataFrame) -> list[str]:
    warnings = validate_logs(logs, config.stages)
    validate_trajectory(trajectory, config.stages)
    if config.injection.duration_s > config.simulation.final_time_s:
        warnings.append("injection.duration_s is greater than simulation.final_time_s; the solver will stop at final_time_s")
    if config.simulation.mode == "native":
        warnings.append("native PyFrac mode is legacy and numerically sensitive; use snapshot for smoke/reproducibility")
    return warnings
