from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_well_trajectory(path: str | Path) -> pd.DataFrame:
    """Load a four-column well trajectory CSV.

    The current 3Dfrac trajectory file has corrupted Chinese headers but stable
    numeric columns. The numeric columns are treated as:
    measured depth, vertical depth, north offset, east offset.
    """

    path = Path(path)
    rows: list[list[float]] = []
    for line in path.read_text(encoding="gb2312", errors="replace").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            rows.append([float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])])
        except ValueError:
            continue

    df = pd.DataFrame(rows, columns=["measured_depth_m", "vertical_depth_m", "north_m", "east_m"])
    if df.empty:
        return df
    return df.drop_duplicates("measured_depth_m").sort_values("measured_depth_m").reset_index(drop=True)


def interpolate_trajectory_at_md(trajectory: pd.DataFrame, measured_depths_m: np.ndarray) -> pd.DataFrame:
    """Interpolate 3D well coordinates at requested measured depths."""

    if trajectory.empty:
        return pd.DataFrame(
            {
                "measured_depth_m": measured_depths_m,
                "vertical_depth_m": np.nan,
                "north_m": np.nan,
                "east_m": np.nan,
            }
        )

    md = trajectory["measured_depth_m"].to_numpy(dtype=float)
    measured_depths_m = np.clip(np.asarray(measured_depths_m, dtype=float), md.min(), md.max())
    return pd.DataFrame(
        {
            "measured_depth_m": measured_depths_m,
            "vertical_depth_m": np.interp(measured_depths_m, md, trajectory["vertical_depth_m"].to_numpy(dtype=float)),
            "north_m": np.interp(measured_depths_m, md, trajectory["north_m"].to_numpy(dtype=float)),
            "east_m": np.interp(measured_depths_m, md, trajectory["east_m"].to_numpy(dtype=float)),
        }
    )


def make_cluster_trajectory_positions(
    trajectory: pd.DataFrame,
    n_clusters: int,
    stage_md_start_m: float | None = None,
    stage_md_end_m: float | None = None,
) -> pd.DataFrame:
    """Build per-cluster trajectory positions for visualization.

    If the exact perforation measured depths are not provided, the last 120 m of
    the trajectory is used as the stage interval. This is only a display mapping;
    the EnKF inversion still targets per-cluster half-length.
    """

    if trajectory.empty:
        return interpolate_trajectory_at_md(trajectory, np.arange(n_clusters, dtype=float))

    md_min = float(trajectory["measured_depth_m"].min())
    md_max = float(trajectory["measured_depth_m"].max())
    if stage_md_end_m is None:
        stage_md_end_m = md_max
    if stage_md_start_m is None:
        stage_md_start_m = max(md_min, stage_md_end_m - 120.0)
    stage_md_start_m = max(md_min, min(float(stage_md_start_m), md_max))
    stage_md_end_m = max(md_min, min(float(stage_md_end_m), md_max))
    if stage_md_end_m <= stage_md_start_m:
        stage_md_start_m = max(md_min, stage_md_end_m - 120.0)

    measured_depths = np.linspace(stage_md_start_m, stage_md_end_m, n_clusters)
    positions = interpolate_trajectory_at_md(trajectory, measured_depths)
    positions.insert(0, "cluster_id", np.arange(1, n_clusters + 1))
    positions["stage_md_start_m"] = stage_md_start_m
    positions["stage_md_end_m"] = stage_md_end_m
    return positions
