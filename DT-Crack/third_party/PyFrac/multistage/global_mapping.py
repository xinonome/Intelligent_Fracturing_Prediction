from __future__ import annotations

import numpy as np
import pandas as pd

from .exceptions import GeometryError
from .schemas import StageDefinition
from .trajectory import interpolate_at_md, tangent_at_md


def basis_at_stage(trajectory: pd.DataFrame, stage: StageDefinition, fracture_azimuth_deg: float | None = None, vertical_threshold: float = 0.85):
    center = interpolate_at_md(trajectory, stage.center_md_m)
    center_xyz = np.array([center["X_m"], center["Y_m"], -center["TVD_m"]], dtype=float)
    e_up = np.array([0.0, 0.0, 1.0])
    e_well = tangent_at_md(trajectory, stage.center_md_m)
    e_well_xyz = np.array([e_well[0], e_well[1], -e_well[2]], dtype=float)
    e_well_xyz /= np.linalg.norm(e_well_xyz)
    if abs(float(np.dot(e_well_xyz, e_up))) > vertical_threshold and fracture_azimuth_deg is None:
        raise GeometryError("near-vertical well segment requires explicit fracture_azimuth_deg")
    if fracture_azimuth_deg is None:
        e_frac = np.cross(e_up, e_well_xyz)
    else:
        angle = np.deg2rad(float(fracture_azimuth_deg))
        e_frac = np.array([np.cos(angle), np.sin(angle), 0.0])
    norm = np.linalg.norm(e_frac)
    if norm <= 1.0e-12:
        raise GeometryError("fracture orientation is undefined")
    e_frac = e_frac / norm
    return center_xyz, e_frac, e_up


def local_to_global(front_local: pd.DataFrame, trajectory: pd.DataFrame, stage: StageDefinition, fracture_azimuth_deg: float | None = None) -> pd.DataFrame:
    center, e_frac, e_up = basis_at_stage(trajectory, stage, fracture_azimuth_deg)
    local = front_local.copy()
    uv = local[["u_m", "v_m"]].to_numpy(dtype=float)
    xyz = center[None, :] + uv[:, 0, None] * e_frac[None, :] + uv[:, 1, None] * e_up[None, :]
    output = local.copy()
    output["X_m"], output["Y_m"], output["Z_m"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    output["TVD_m"] = -output["Z_m"]
    return output


def global_to_local(global_xyz: np.ndarray, center: np.ndarray, e_frac: np.ndarray, e_up: np.ndarray) -> np.ndarray:
    delta = np.asarray(global_xyz, dtype=float) - np.asarray(center, dtype=float)
    return np.column_stack([delta @ e_frac, delta @ e_up])
