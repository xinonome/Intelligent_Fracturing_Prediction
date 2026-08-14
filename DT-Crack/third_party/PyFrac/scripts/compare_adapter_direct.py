from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    project_root = Path(__file__).resolve().parents[4]
    crack_root = project_root / "DT-Crack"
    sys.path.insert(0, str(crack_root))
    from forward_models.pyfrac_adapter import PyFracAdapter
    from forward_models.pyfrac_config import PyFracConfig

    config = PyFracConfig(mesh_half_length_m=20., mesh_half_height_m=8., mesh_nx=81, mesh_ny=41, height_m=12.)
    adapter = PyFracAdapter(config=config, project_root=project_root)
    values = dict(injection_rate_m3_s=.001, time_s=30., mode="snapshot", height_m=12., viscosity_pa_s=.1, e_prime_pa=3.2e10, leakoff_coefficient_m_sqrt_s=5e-7, min_horizontal_stress_pa=8.0e7, fracture_toughness_pa_sqrt_m=5.0e5)
    wrapped = adapter.run(**values)
    modules = adapter._load_modules()
    # The project adapter expands the half-height to resolve the requested
    # 12 m contained fracture; mirror that actual upstream call here.
    mesh = modules["CartesianMesh"](20., 18., 81, 41)
    solid = modules["MaterialProperties"](mesh, 3.2e10, 5.0e5, Carters_coef=5e-7, confining_stress=8e7)
    fluid = modules["FluidProperties"](viscosity=.1)
    injection = modules["InjectionProperties"](.001, mesh)
    sim = modules["SimulationProperties"](); sim.finalTime = 30.; sim.plotFigure = False; sim.saveToDisk = False; sim.log2file = False; sim.meshExtension = [False] * 4; sim.blockFigure = False; sim.verbositylevel = "error"; sim.frontAdvancing = "predictor-corrector"
    init = modules["InitializationParameters"](modules["Geometry"]("height contained", fracture_height=12.), regime="PKN", time=30.)
    direct = modules["Fracture"](mesh, init, solid, fluid, injection, sim)
    crack = np.asarray(direct.EltCrack, dtype=int)
    direct_values = {"half_length_m": float(np.max(np.abs(mesh.CenterCoor[crack, 0]))), "max_aperture_mm": float(np.nanmax(np.asarray(direct.w)[crack]) * 1000.), "area_m2": float(len(crack) * mesh.EltArea), "volume_m3": float(direct.FractureVolume), "net_pressure_mpa": float(np.nanmean(np.asarray(direct.pNet)[crack]) / 1e6)}
    relative = {key: abs(float(getattr(wrapped, key)) - value) / max(abs(value), 1e-12) for key, value in direct_values.items()}
    summary = {"wrapped_success": wrapped.success, "direct": direct_values, "adapter": {key: getattr(wrapped, key) for key in direct_values}, "relative_difference": relative, "pass": all(value <= .05 for value in relative.values())}
    output = project_root / "DT-Crack" / "third_party" / "PyFrac" / "outputs" / "baseline" / "adapter_direct_comparison.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS" if summary["pass"] else "FAIL", "summary": str(output)}))
    return 0 if summary["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
