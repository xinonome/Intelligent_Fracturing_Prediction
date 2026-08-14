from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StageDefinition:
    stage_id: str
    md_start_m: float
    md_end_m: float

    @property
    def center_md_m(self) -> float:
        return (self.md_start_m + self.md_end_m) / 2.0


@dataclass(frozen=True)
class MeshConfig:
    x_half_extent_m: float
    y_half_extent_m: float
    nx: int
    ny: int


@dataclass(frozen=True)
class FluidConfig:
    viscosity_pa_s: float
    compressibility_pa_inv: float
    density_kg_m3: float


@dataclass(frozen=True)
class InjectionConfig:
    rate_m3_s: float
    duration_s: float


@dataclass(frozen=True)
class SimulationConfig:
    final_time_s: float
    save_every_s: float
    front_advancing: str
    mode: str
    fracture_height_m: float


@dataclass(frozen=True)
class StageMaterial:
    e_prime_pa: float
    e_prime_min_pa: float
    e_prime_max_pa: float
    e_prime_std_pa: float
    poisson_ratio_median: float
    stress_center_pa: float
    stress_min_pa: float
    stress_max_pa: float
    toughness_pa_sqrt_m: float
    carter_m_sqrt_s: float
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProjectConfig:
    path: Path
    raw: dict[str, Any]
    project_root: Path
    logs_path: Path
    trajectory_path: Path
    output_root: Path
    stages: tuple[StageDefinition, ...]
    mesh: MeshConfig
    fluid: FluidConfig
    injection: InjectionConfig
    simulation: SimulationConfig
    mapping: dict[str, Any]
    diagnostics: dict[str, Any]
    validation: dict[str, Any]
    runtime: dict[str, Any]
    material: dict[str, Any]
    warnings: tuple[str, ...] = ()


@dataclass
class StageSnapshot:
    stage_id: str
    time_s: float
    mesh_x: list[float]
    mesh_y: list[float]
    width_m: float
    fluid_pressure_pa: float
    net_pressure_pa: float
    front_velocity_m_s: float
    front_coordinates_local: list[list[float]]
    fracture_volume_m3: float
    efficiency: float
    injected_volume_m3: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StageResult:
    stage_id: str
    times_s: list[float]
    snapshots: list[StageSnapshot]
    metrics: Any
    metadata: dict[str, Any]
