from __future__ import annotations

import numpy as np
import pandas as pd

from .exceptions import DataValidationError, GeometryError


def interpolate_at_md(trajectory: pd.DataFrame, md_m: float) -> dict[str, float]:
    md = trajectory["MD_m"].to_numpy(dtype=float)
    if md_m < md.min() or md_m > md.max():
        raise DataValidationError(f"trajectory interpolation would extrapolate at MD={md_m}")
    return {column: float(np.interp(md_m, md, trajectory[column].to_numpy(dtype=float))) for column in ("X_m", "Y_m", "TVD_m")}


def tangent_at_md(trajectory: pd.DataFrame, md_m: float, delta_m: float = 1.0) -> np.ndarray:
    md = trajectory["MD_m"].to_numpy(dtype=float)
    if md_m - delta_m < md.min() or md_m + delta_m > md.max():
        raise DataValidationError("trajectory tangent requires samples on both sides of stage center")
    p0 = np.array([interpolate_at_md(trajectory, md_m - delta_m)[c] for c in ("X_m", "Y_m", "TVD_m")])
    p1 = np.array([interpolate_at_md(trajectory, md_m + delta_m)[c] for c in ("X_m", "Y_m", "TVD_m")])
    tangent = p1 - p0
    norm = float(np.linalg.norm(tangent))
    if norm <= 1.0e-12:
        raise GeometryError("well trajectory tangent is zero")
    return tangent / norm
