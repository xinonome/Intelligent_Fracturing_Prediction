from __future__ import annotations

from pathlib import Path
from typing import Any

from .exceptions import ConfigurationError
from .io import load_csv, load_yaml
from .schemas import FluidConfig, InjectionConfig, MeshConfig, ProjectConfig, SimulationConfig, StageDefinition
from .validation import validate_config_values, validate_input_files


def _path(root: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()


def load_config(path: str | Path) -> ProjectConfig:
    path = Path(path).resolve()
    raw: dict[str, Any] = load_yaml(path)
    validate_config_values(raw)
    # The reference layout stores configs/ beside data/, outputs/ and src/.
    # Standalone temporary configs continue to resolve relative to their own folder.
    root = path.parent.parent if path.parent.name == "configs" else path.parent
    stages = tuple(
        StageDefinition(str(item["id"]), float(item["md_start_m"]), float(item["md_end_m"]))
        for item in raw["stages"]
    )
    config = ProjectConfig(
        path=path,
        raw=raw,
        project_root=root,
        logs_path=_path(root, raw["inputs"]["logs_csv"]),
        trajectory_path=_path(root, raw["inputs"]["trajectory_csv"]),
        output_root=_path(root, raw["project"]["output_root"]),
        stages=stages,
        mesh=MeshConfig(float(raw["mesh"]["x_half_extent_m"]), float(raw["mesh"]["y_half_extent_m"]), int(raw["mesh"]["nx"]), int(raw["mesh"]["ny"])),
        fluid=FluidConfig(float(raw["fluid"]["viscosity_pa_s"]), float(raw["fluid"].get("compressibility_pa_inv", 0.0)), float(raw["fluid"]["density_kg_m3"])),
        injection=InjectionConfig(float(raw["injection"]["rate_m3_s"]), float(raw["injection"]["duration_s"])),
        simulation=SimulationConfig(float(raw["simulation"]["final_time_s"]), float(raw["simulation"]["save_every_s"]), str(raw["simulation"].get("front_advancing", "predictor-corrector")), str(raw["simulation"].get("mode", "snapshot")), float(raw["simulation"]["fracture_height_m"])),
        mapping=dict(raw.get("mapping", {})),
        diagnostics=dict(raw["diagnostics"]),
        validation=dict(raw["validation"]),
        runtime=dict(raw["runtime"]),
        material=dict(raw["material"]),
    )
    logs = load_csv(config.logs_path)
    trajectory = load_csv(config.trajectory_path)
    warnings = tuple(validate_input_files(config, logs, trajectory))
    return ProjectConfig(**{**config.__dict__, "warnings": warnings})


def load_inputs(config: ProjectConfig):
    return load_csv(config.logs_path), load_csv(config.trajectory_path)
