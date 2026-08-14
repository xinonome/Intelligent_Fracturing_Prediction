from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .exceptions import MemoryBudgetError
from .schemas import FluidConfig, InjectionConfig, MeshConfig, SimulationConfig, StageMaterial, StageResult, StageSnapshot


@dataclass(frozen=True)
class StageSimulationInput:
    stage_id: str
    mesh_config: MeshConfig
    material: StageMaterial
    fluid: FluidConfig
    injection: InjectionConfig
    simulation: SimulationConfig
    runtime: dict[str, Any]
    confining_stress_func: Any | None = None


def estimate_memory_gb(nx: int, ny: int) -> float:
    """Conservative estimate for the dense elasticity work arrays."""
    cells = int(nx) * int(ny)
    return float(16.0 * cells * cells * 8.0 / 1.0e9)


def build_injection_rate_history(rate_m3_s: float, duration_s: float) -> np.ndarray:
    """Build the 2xN schedule used by this PyFrac version: times then rates."""
    if rate_m3_s <= 0 or duration_s <= 0:
        raise ValueError("rate and duration must be positive")
    return np.asarray([[0.0, float(duration_s)], [float(rate_m3_s), 0.0]], dtype=float)


def _project_root() -> Path:
    # .../DT-Crack/third_party/PyFrac/multistage/pyfrac_adapter.py
    return Path(__file__).resolve().parents[4]


def _load_project_adapter():
    crack_root = _project_root() / "DT-Crack"
    if str(crack_root) not in sys.path:
        sys.path.insert(0, str(crack_root))
    from forward_models.pyfrac_adapter import PyFracAdapter
    from forward_models.pyfrac_config import PyFracConfig
    return PyFracAdapter, PyFracConfig


def run_pyfrac_stage(stage_input: StageSimulationInput, output_dir: Path) -> StageResult:
    estimated = estimate_memory_gb(stage_input.mesh_config.nx, stage_input.mesh_config.ny)
    budget = float(stage_input.runtime["max_memory_gb"])
    if estimated > budget:
        raise MemoryBudgetError(f"estimated PyFrac memory {estimated:.3f} GB exceeds budget {budget:.3f} GB")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    final_time = stage_input.simulation.final_time_s
    save_every = stage_input.simulation.save_every_s
    times = np.arange(save_every, final_time + save_every * 0.5, save_every, dtype=float)
    if len(times) == 0 or times[-1] < final_time - 1.0e-9:
        times = np.append(times, final_time)
    times = np.unique(np.clip(times, 1.0, final_time))
    Adapter, PyFracConfig = _load_project_adapter()
    adapter_config = PyFracConfig(
        mesh_half_length_m=stage_input.mesh_config.x_half_extent_m,
        mesh_half_height_m=stage_input.mesh_config.y_half_extent_m,
        mesh_nx=stage_input.mesh_config.nx,
        mesh_ny=stage_input.mesh_config.ny,
        height_m=stage_input.simulation.fracture_height_m,
        max_time_steps=int(stage_input.runtime.get("max_time_steps", 30)),
    )
    adapter = Adapter(config=adapter_config, project_root=_project_root())
    snapshots: list[StageSnapshot] = []
    raw_rows: list[dict[str, Any]] = []
    previous_length = None
    previous_time = None
    for target_time in times:
        result = adapter.run(
            injection_rate_m3_s=stage_input.injection.rate_m3_s,
            injection_rate_history=build_injection_rate_history(stage_input.injection.rate_m3_s, stage_input.injection.duration_s),
            time_s=float(target_time),
            mode=stage_input.simulation.mode,
            height_m=stage_input.simulation.fracture_height_m,
            viscosity_pa_s=stage_input.fluid.viscosity_pa_s,
            e_prime_pa=stage_input.material.e_prime_pa,
            leakoff_coefficient_m_sqrt_s=stage_input.material.carter_m_sqrt_s,
            min_horizontal_stress_pa=stage_input.material.stress_center_pa,
            fracture_toughness_pa_sqrt_m=stage_input.material.toughness_pa_sqrt_m,
            confining_stress_func=stage_input.confining_stress_func,
        )
        if not result.success:
            raise RuntimeError(f"PyFrac stage {stage_input.stage_id} failed at t={target_time:g}s: {result.error}")
        front = np.asarray(result.front_geometry, dtype=float).reshape(-1, 2) if result.front_geometry else np.empty((0, 2))
        length = float(result.half_length_m)
        if previous_length is None or previous_time is None or target_time <= previous_time:
            speed = 0.0
        else:
            speed = max(0.0, (length - previous_length) / (float(target_time) - previous_time))
        injected = stage_input.injection.rate_m3_s * min(float(target_time), stage_input.injection.duration_s)
        efficiency = float(result.volume_m3 / injected) if injected > 0 else float("nan")
        snapshot = StageSnapshot(
            stage_id=stage_input.stage_id,
            time_s=float(target_time),
            mesh_x=front[:, 0].tolist() if front.size else [],
            mesh_y=front[:, 1].tolist() if front.size else [],
            width_m=float(result.max_aperture_mm) / 1000.0,
            fluid_pressure_pa=float(result.bottomhole_pressure_mpa) * 1.0e6,
            net_pressure_pa=float(result.net_pressure_mpa) * 1.0e6,
            front_velocity_m_s=speed,
            front_coordinates_local=front.tolist(),
            fracture_volume_m3=float(result.volume_m3),
            efficiency=efficiency,
            injected_volume_m3=injected,
            metadata={"engine_mode": result.engine_mode, "target_reached": result.target_reached, "runtime_seconds": result.runtime_seconds},
        )
        snapshots.append(snapshot)
        raw_rows.append({"time_s": float(target_time), **result.to_dict()})
        previous_length, previous_time = length, float(target_time)
    import pandas as pd
    from .diagnostics import geometry_metrics

    metrics = geometry_metrics(snapshots, stage_input.stage_id)
    pd.DataFrame(raw_rows).to_csv(output_dir / "raw_pyfrac.csv", index=False)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    np.savez_compressed(
        output_dir / "snapshots.npz",
        time_s=np.asarray([s.time_s for s in snapshots]),
        mesh_x=np.asarray([s.mesh_x for s in snapshots], dtype=object),
        mesh_y=np.asarray([s.mesh_y for s in snapshots], dtype=object),
        width_m=np.asarray([s.width_m for s in snapshots]),
        fluid_pressure_pa=np.asarray([s.fluid_pressure_pa for s in snapshots]),
        net_pressure_pa=np.asarray([s.net_pressure_pa for s in snapshots]),
        front_velocity_m_s=np.asarray([s.front_velocity_m_s for s in snapshots]),
        front_coordinates_local=np.asarray([s.front_coordinates_local for s in snapshots], dtype=object),
        fracture_volume_m3=np.asarray([s.fracture_volume_m3 for s in snapshots]),
        efficiency=np.asarray([s.efficiency for s in snapshots]),
    )
    return StageResult(
        stage_id=stage_input.stage_id,
        times_s=[s.time_s for s in snapshots],
        snapshots=snapshots,
        metrics=metrics,
        metadata={
            "elapsed_s": time.perf_counter() - started,
            "normalization": "project PyFracAdapter aggregate fields; front coordinates are native adapter output",
            "width_pressure_fields_are_snapshot_scalars": True,
            "memory_estimate_gb": estimated,
        },
    )
