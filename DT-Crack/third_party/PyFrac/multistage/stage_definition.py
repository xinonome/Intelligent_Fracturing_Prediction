from __future__ import annotations

import pandas as pd

from .schemas import StageDefinition


def slice_stage_logs(logs: pd.DataFrame, stage: StageDefinition) -> pd.DataFrame:
    mask = (logs["MD_m"] >= stage.md_start_m) & (logs["MD_m"] <= stage.md_end_m)
    return logs.loc[mask].copy().reset_index(drop=True)


def stage_boundary_tvd(trajectory: pd.DataFrame, stage: StageDefinition) -> tuple[float, float, float]:
    import numpy as np

    md = trajectory["MD_m"].to_numpy(dtype=float)
    tvd = trajectory["TVD_m"].to_numpy(dtype=float)
    top = float(np.interp(stage.md_start_m, md, tvd))
    bottom = float(np.interp(stage.md_end_m, md, tvd))
    center = float(np.interp(stage.center_md_m, md, tvd))
    return top, bottom, center
