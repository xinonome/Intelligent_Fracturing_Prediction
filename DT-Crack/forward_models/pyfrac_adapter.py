"""Adapter around the vendored PyFrac source.

PyFrac is an external GPLv3 simulator.  This module does not modify its
source files; it only loads the documented classes and converts their output
to the project's common fracture-state schema.

Two modes are exposed:

``snapshot``
    Build a PyFrac Cartesian mesh and initialize a PKN fracture at the
    requested time.  This is quick and useful for model-space comparison and
    teacher-data prototyping, but it is *not* a time-marching PyFrac run.

``native``
    Start from a resolved PyFrac PKN warm state and let ``Controller.run``
    advance the fracture. It is the high-fidelity reference path and can be
    slow or numerically sensitive on old PyFrac versions. The adapter checks
    that the final time actually reaches the requested target; failures are
    returned with diagnostic metadata instead of being silently relabelled as
    PyFrac output.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from .pyfrac_config import PyFracConfig
from .fracture_length_models import LengthForwardModel


@dataclass(frozen=True)
class PyFracRunResult:
    """Normalized result from one PyFrac evaluation."""

    half_length_m: float
    max_aperture_mm: float
    area_m2: float
    volume_m3: float
    net_pressure_mpa: float
    bottomhole_pressure_mpa: float
    front_geometry: list[list[float]]
    runtime_seconds: float
    model_name: str
    engine_mode: str
    success: bool
    error: str | None = None
    final_time_s: float = float("nan")
    successful_time_steps: int = 0
    failed_time_steps: int = 0
    target_reached: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PyFracAdapter:
    """Load PyFrac and run one planar fracture case."""

    def __init__(self, config: PyFracConfig | None = None, project_root: str | Path | None = None) -> None:
        self.config = config or PyFracConfig()
        self.project_root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[2]
        self.pyfrac_root = self.config.resolved_root(self.project_root)
        self.src_root = self.pyfrac_root / "src"
        self._modules: dict[str, Any] | None = None

    def verify_installation(self) -> dict[str, Any]:
        return {
            "root": str(self.pyfrac_root),
            "src": str(self.src_root),
            "source_exists": self.src_root.is_dir(),
            "license_exists": (self.pyfrac_root / "LICENSE.TXT").is_file(),
            "gpl_exists": (self.pyfrac_root / "GPL.txt").is_file(),
            "commit": _git_commit(self.pyfrac_root),
        }

    def run(
        self,
        injection_rate_m3_s: float,
        time_s: float,
        mode: str = "snapshot",
        height_m: float | None = None,
        viscosity_pa_s: float | None = None,
        e_prime_pa: float | None = None,
        leakoff_coefficient_m_sqrt_s: float | None = None,
        min_horizontal_stress_pa: float | None = None,
        fracture_toughness_pa_sqrt_m: float | None = None,
        confining_stress_func: Any | None = None,
        injection_rate_history: np.ndarray | None = None,
    ) -> PyFracRunResult:
        started = time.perf_counter()
        mode = mode.lower().strip()
        if mode not in {"snapshot", "native"}:
            raise ValueError("PyFrac mode must be 'snapshot' or 'native'")
        if not self.src_root.is_dir():
            return self._failed(started, mode, "PyFrac source directory does not exist")

        try:
            modules = self._load_modules()
            result = self._run_case(
                modules,
                injection_rate_m3_s=(
                    np.asarray(injection_rate_history, dtype=float)
                    if injection_rate_history is not None
                    else max(float(injection_rate_m3_s), 1.0e-9)
                ),
                time_s=max(float(time_s), self.config.initial_time_s),
                mode=mode,
                height_m=float(height_m or self.config.height_m),
                viscosity_pa_s=float(viscosity_pa_s or self.config.viscosity_pa_s),
                e_prime_pa=float(e_prime_pa or self.config.e_prime_pa),
                leakoff_coefficient_m_sqrt_s=float(
                    leakoff_coefficient_m_sqrt_s or self.config.leakoff_coefficient_m_sqrt_s
                ),
                min_horizontal_stress_pa=float(min_horizontal_stress_pa or self.config.min_horizontal_stress_pa),
                fracture_toughness_pa_sqrt_m=float(
                    fracture_toughness_pa_sqrt_m or self.config.fracture_toughness_pa_sqrt_m
                ),
                confining_stress_func=confining_stress_func,
            )
            return _replace_runtime(result, time.perf_counter() - started)
        except Exception as exc:  # PyFrac is legacy code; preserve diagnostics for reproducibility.
            error = f"{type(exc).__name__}: {exc}"
            return self._failed(started, mode, error)

    def _load_modules(self) -> dict[str, Any]:
        if self._modules is not None:
            return self._modules
        source = str(self.src_root)
        if source not in sys.path:
            sys.path.insert(0, source)

        # PyFrac 1.1.1 predates NumPy 1.24. These aliases are process-local
        # compatibility shims; the vendored GPL source remains unchanged.
        for name, value in (("int", int), ("float", float), ("bool", bool)):
            if name not in np.__dict__:
                setattr(np, name, value)

        import level_set

        level_set.Eikonal_Res = _scalar_eikonal_residual
        from controller import Controller
        from fracture import Fracture
        from fracture_initialization import Geometry, InitializationParameters
        from mesh import CartesianMesh
        from properties import FluidProperties, InjectionProperties, MaterialProperties, SimulationProperties

        self._modules = {
            "Controller": Controller,
            "Fracture": Fracture,
            "Geometry": Geometry,
            "InitializationParameters": InitializationParameters,
            "CartesianMesh": CartesianMesh,
            "FluidProperties": FluidProperties,
            "InjectionProperties": InjectionProperties,
            "MaterialProperties": MaterialProperties,
            "SimulationProperties": SimulationProperties,
        }
        return self._modules

    def _run_case(self, modules: dict[str, Any], **kwargs: Any) -> PyFracRunResult:
        height = kwargs["height_m"]
        q = kwargs["injection_rate_m3_s"]
        q_scale = float(q[1, 0]) if isinstance(q, np.ndarray) else float(q)
        t = kwargs["time_s"]
        eprime = kwargs["e_prime_pa"]
        viscosity = kwargs["viscosity_pa_s"]
        leakoff = kwargs["leakoff_coefficient_m_sqrt_s"]
        stress = kwargs["min_horizontal_stress_pa"]
        toughness = kwargs["fracture_toughness_pa_sqrt_m"]
        confining_stress_func = kwargs.get("confining_stress_func")
        mode = kwargs["mode"]

        # Domain size is scaled from the requested PKN time so the analytical
        # initialization is resolved without immediately touching boundaries.
        pkn_length = 0.68 * ((q_scale**3 * eprime) / (viscosity * height**4)) ** 0.2 * t**0.8
        # Do not enlarge the horizontal domain merely because the requested
        # height is large: that makes dx so coarse that the resolved warm
        # fracture becomes sub-cell sized.  The configured extent and the
        # analytical PKN scale are sufficient for this adapter path.
        half_length = max(self.config.mesh_half_length_m, 1.8 * pkn_length, height * 1.5)
        # Keep several cells across the fixed PKN height.  A fixed 25 m
        # half-height would make a small smoke-test fracture sub-cell-sized.
        half_height = max(height * 1.5, min(self.config.mesh_half_height_m, height * 4.0))
        mesh = modules["CartesianMesh"](
            half_length,
            half_height,
            max(int(self.config.mesh_nx), 31),
            max(int(self.config.mesh_ny), 17),
        )
        solid_kwargs = {"Carters_coef": leakoff}
        if confining_stress_func is None:
            solid_kwargs["confining_stress"] = stress
        else:
            solid_kwargs["confining_stress_func"] = confining_stress_func
        solid = modules["MaterialProperties"](mesh, eprime, toughness, **solid_kwargs)
        fluid = modules["FluidProperties"](viscosity=viscosity)
        injection = modules["InjectionProperties"](q, mesh)
        sim = modules["SimulationProperties"]()
        sim.finalTime = t
        sim.maxTimeSteps = int(self.config.max_time_steps)
        sim.plotFigure = False
        sim.saveToDisk = False
        sim.log2file = False
        sim.enableRemeshing = False
        sim.meshExtension = [False, False, False, False]
        sim.blockFigure = False
        sim.verbositylevel = "error"
        sim.frontAdvancing = "predictor-corrector"

        geometry = modules["Geometry"]("height contained", fracture_height=height)
        init = modules["InitializationParameters"](geometry, regime="PKN", time=t)
        fracture = modules["Fracture"](mesh, init, solid, fluid, injection, sim)

        engine_mode = "pyfrac_pkn_grid_snapshot"
        final = fracture
        final_time_s = float(t)
        successful_time_steps = 0
        failed_time_steps = 0
        target_reached = True
        if mode == "native" and t > self.config.initial_time_s:
            # Native mode starts from a resolved PKN state and time-marches it.
            # The old solver may fail for particular meshes/versions; the
            # caller receives an explicit failure rather than fake output.
            native_start = max(float(self.config.native_start_time_s), float(self.config.initial_time_s))
            if t <= native_start:
                raise ValueError(
                    f"native target time {t:g}s is not later than the resolved warm-start "
                    f"time {native_start:g}s; use snapshot for early-time output"
                )
            init_geometry = modules["Geometry"]("height contained", fracture_height=height)
            init_param = modules["InitializationParameters"](
                init_geometry, regime="PKN", time=native_start
            )
            initial = modules["Fracture"](mesh, init_param, solid, fluid, injection, sim)
            controller = modules["Controller"](initial, solid, fluid, injection, sim)
            controller.run()
            final = controller.fracture
            final_time_s = float(final.time)
            successful_time_steps = int(controller.successfulTimeSteps)
            failed_time_steps = int(controller.failedTimeSteps)
            target_reached = final_time_s >= 0.999 * float(t)
            if not target_reached:
                raise RuntimeError(
                    "PyFrac native solver stopped before target time: "
                    f"final_time={final_time_s:g}s, target_time={float(t):g}s, "
                    f"successful_steps={successful_time_steps}, failed_steps={failed_time_steps}"
                )
            engine_mode = "pyfrac_native_dynamic"

        crack = np.asarray(final.EltCrack, dtype=int)
        if crack.size == 0:
            raise RuntimeError("PyFrac returned an empty fracture footprint")
        coords = np.asarray(mesh.CenterCoor[crack], dtype=float)
        front = np.asarray(mesh.CenterCoor[np.asarray(final.EltTip, dtype=int)], dtype=float)
        length = float(np.max(np.abs(coords[:, 0])))
        aperture = float(np.nanmax(np.asarray(final.w)[crack])) * 1000.0
        area = float(crack.size * mesh.EltArea)
        volume = float(getattr(final, "FractureVolume", np.nansum(final.w[crack]) * mesh.EltArea))
        net = float(np.nanmean(np.asarray(final.pNet)[crack])) / 1.0e6
        return PyFracRunResult(
            half_length_m=length,
            max_aperture_mm=aperture,
            area_m2=area,
            volume_m3=volume,
            net_pressure_mpa=net,
            bottomhole_pressure_mpa=net + stress / 1.0e6,
            front_geometry=front.tolist(),
            runtime_seconds=0.0,
            model_name="PyFrac",
            engine_mode=engine_mode,
            success=True,
            final_time_s=final_time_s,
            successful_time_steps=successful_time_steps,
            failed_time_steps=failed_time_steps,
            target_reached=target_reached,
        )

    def _failed(self, started: float, mode: str, error: str) -> PyFracRunResult:
        return PyFracRunResult(
            half_length_m=float("nan"),
            max_aperture_mm=float("nan"),
            area_m2=float("nan"),
            volume_m3=float("nan"),
            net_pressure_mpa=float("nan"),
            bottomhole_pressure_mpa=float("nan"),
            front_geometry=[],
            runtime_seconds=time.perf_counter() - started,
            model_name="PyFrac",
            engine_mode=f"pyfrac_{mode}",
            success=False,
            error=error,
        )


class PyFracLengthForwardModel(LengthForwardModel):
    """LengthForwardModel-compatible wrapper for offline PyFrac studies.

    The wrapper is intentionally not used by the online EnKF default.  It is
    available through ``build_length_forward_model('pyfrac')`` for offline
    comparison and uses the explicit ``snapshot`` mode by default.
    """

    model_name = "pyfrac_reference"

    def __init__(self, config: PyFracConfig | None = None, mode: str = "snapshot") -> None:
        self.config = config or PyFracConfig()
        self.mode = mode
        self.adapter = PyFracAdapter(self.config)

    def simulate_lengths(
        self,
        factor_state: np.ndarray,
        cluster_x: np.ndarray,
        q_base: np.ndarray | float,
        viscosity_pa_s: float,
        e_prime_pa: float,
        height_m: float,
        t_seconds: float,
    ):
        import pandas as pd
        from .fracture_length_models import LengthForwardResult

        x = np.asarray(cluster_x, dtype=float)
        n_clusters = len(x)
        factors = np.clip(np.resize(np.asarray(factor_state, dtype=float), n_clusters), 0.65, 1.35)
        base = np.maximum(np.resize(np.asarray(q_base, dtype=float), n_clusters), 1.0e-9)
        capacity = base * factors
        q_cluster = capacity * base.sum() / max(float(capacity.sum()), 1.0e-12)
        rows: list[dict[str, float]] = []
        for index, (cluster_x_value, q_value, factor) in enumerate(zip(x, q_cluster, factors), start=1):
            result = self.adapter.run(
                injection_rate_m3_s=float(q_value),
                time_s=float(t_seconds),
                mode=self.mode,
                height_m=height_m,
                viscosity_pa_s=viscosity_pa_s,
                e_prime_pa=e_prime_pa,
            )
            if not result.success:
                raise RuntimeError(f"PyFrac cluster {index} failed: {result.error}")
            rows.append(
                {
                    "cluster_id": index,
                    "x_center_m": float(cluster_x_value),
                    "Q_cluster_m3s": float(q_value),
                    "cluster_factor": float(factor),
                    "half_length_m": result.half_length_m,
                    "max_aperture_mm": result.max_aperture_mm,
                    "area_m2": result.area_m2,
                    "volume_m3": result.volume_m3,
                    "net_pressure_mpa": result.net_pressure_mpa,
                    "bottomhole_pressure_mpa": result.bottomhole_pressure_mpa,
                    "pyfrac_runtime_seconds": result.runtime_seconds,
                }
            )
        return LengthForwardResult(pd.DataFrame(rows), self.model_name)


def _replace_runtime(result: PyFracRunResult, runtime: float) -> PyFracRunResult:
    payload = result.to_dict()
    payload["runtime_seconds"] = float(runtime)
    return PyFracRunResult(**payload)


def _git_commit(path: Path) -> str | None:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def _scalar_eikonal_residual(Tij: Any, *args: Any) -> float:
    """NumPy-2-compatible form of PyFrac's old scalar Eikonal residual."""

    scalar = lambda value: float(np.asarray(value).reshape(-1)[0])
    value = scalar(Tij)
    left, right, bottom, top, fij, dx, dy = [scalar(value) for value in args]
    return (
        max((value - left) / dx, 0.0) ** 2
        + min((right - value) / dx, 0.0) ** 2
        + max((value - bottom) / dy, 0.0) ** 2
        + min((top - value) / dy, 0.0) ** 2
        - fij**2
    )
