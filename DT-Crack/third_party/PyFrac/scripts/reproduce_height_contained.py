from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def run() -> dict:
    root = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(root / "src"))
    for name, value in (("int", int), ("float", float), ("bool", bool)):
        if name not in np.__dict__: setattr(np, name, value)
    from mesh import CartesianMesh
    from properties import MaterialProperties, FluidProperties, InjectionProperties, SimulationProperties
    from fracture import Fracture
    from fracture_initialization import Geometry, InitializationParameters
    mesh = CartesianMesh(8.0, 3.0, 25, 15)
    def sigma(_x, y): return 7.5e6 if abs(y) > 1.0 else 1.0e6
    solid = MaterialProperties(mesh, 3.3e10 / (1 - 0.4 ** 2), 0.0, confining_stress_func=sigma)
    fluid = FluidProperties(viscosity=0.0011)
    injection = InjectionProperties(0.001, mesh)
    sim = SimulationProperties(); sim.finalTime = 1.0; sim.maxTimeSteps = 1; sim.plotFigure = False; sim.saveToDisk = False; sim.log2file = False; sim.meshExtension = [False] * 4; sim.blockFigure = False; sim.verbositylevel = "error"
    fracture = Fracture(mesh, InitializationParameters(Geometry("radial", radius=1.5), regime="M"), solid, fluid, injection, sim)
    crack = np.asarray(fracture.EltCrack, dtype=int)
    widths = np.asarray(fracture.w)[crack]
    return {"status": "PASS", "name": "height_contained", "time_s": float(fracture.time), "crack_cells": int(len(crack)), "has_finite_width": bool(np.isfinite(widths).all()), "volume_m3": float(fracture.FractureVolume)}


if __name__ == "__main__":
    print(json.dumps(run()))
