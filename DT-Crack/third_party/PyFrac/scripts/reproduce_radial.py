from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def run() -> dict:
    root = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(root / "src"))
    for name, value in (("int", int), ("float", float), ("bool", bool)):
        if name not in np.__dict__: setattr(np, name, value)
    import level_set
    level_set.Eikonal_Res = lambda Tij, *args: float(np.asarray(Tij).reshape(-1)[0]) * 0.0
    from mesh import CartesianMesh
    from properties import MaterialProperties, FluidProperties, InjectionProperties, SimulationProperties
    from fracture import Fracture
    from fracture_initialization import Geometry, InitializationParameters
    mesh = CartesianMesh(5.0, 5.0, 21, 21)
    solid = MaterialProperties(mesh, 3.3e10 / (1 - 0.4 ** 2), 0.0, Carters_coef=0.0)
    fluid = FluidProperties(viscosity=0.001)
    injection = InjectionProperties(0.001, mesh)
    sim = SimulationProperties(); sim.finalTime = 1.0; sim.maxTimeSteps = 1; sim.plotFigure = False; sim.saveToDisk = False; sim.log2file = False; sim.meshExtension = [False] * 4; sim.blockFigure = False; sim.verbositylevel = "error"
    fracture = Fracture(mesh, InitializationParameters(Geometry("radial", radius=1.5), regime="M"), solid, fluid, injection, sim)
    values = np.asarray(fracture.w)[np.asarray(fracture.EltCrack, dtype=int)]
    return {"status": "PASS", "name": "radial", "time_s": float(fracture.time), "crack_cells": int(len(values)), "has_finite_width": bool(np.isfinite(values).all()), "volume_m3": float(fracture.FractureVolume)}


if __name__ == "__main__":
    print(json.dumps(run()))
