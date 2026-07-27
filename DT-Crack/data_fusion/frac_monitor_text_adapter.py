from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .fiber_api_adapter import FiberApiTables, stage_info_to_controls


@dataclass(frozen=True)
class FracMonitorTextTables:
    """Normalized tables parsed from the FracMonitor text export.

    The real 3Dfrac file is not the same JSON shape as the fiber HTTP API, but
    it contains the same stage-level cluster controls:

    well, stage, time, balance, cumulative_balance
    #cluster|liquid|sand|cumulative_liquid|cumulative_sand
    """

    meta: dict[str, object]
    stage_info: pd.DataFrame
    controls: pd.DataFrame


def load_frac_monitor_text(path: str | Path, default_step_seconds: float = 1.0) -> FracMonitorTextTables:
    stage_info = parse_frac_monitor_text(path)
    controls = stage_info_to_controls(stage_info, default_step_seconds=default_step_seconds)
    meta = {
        "source": str(Path(path)),
        "wellName": None if stage_info.empty else stage_info["well_name"].iloc[0],
        "stage": None if stage_info.empty else stage_info["stage"].iloc[0],
        "time_min": None if stage_info.empty else str(stage_info["time"].min()),
        "time_max": None if stage_info.empty else str(stage_info["time"].max()),
        "time_steps": 0 if stage_info.empty else int(stage_info["step"].nunique()),
        "cluster_count": 0 if stage_info.empty else int(stage_info["cluster_id"].nunique()),
    }
    return FracMonitorTextTables(meta=meta, stage_info=stage_info, controls=controls)


def parse_frac_monitor_text(path: str | Path) -> pd.DataFrame:
    """Parse the real FracMonitor text file into stageInfo-compatible rows."""

    path = Path(path)
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        first_line = True
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if first_line and line.startswith("FracMonitor"):
                first_line = False
                continue
            first_line = False

            parts = line.split("#")
            main = parts[0].split(",")
            if len(main) < 5:
                continue
            well_name, stage, time_text, balance_degree, cumulative_balance_degree = main[:5]
            for cluster_part in parts[1:]:
                fields = cluster_part.split("|")
                if len(fields) != 5:
                    continue
                rows.append(
                    {
                        "well_name": well_name,
                        "stage": stage,
                        "time": time_text,
                        "balance_degree": _to_float(balance_degree),
                        "cumulative_balance_degree": _to_float(cumulative_balance_degree),
                        "cluster_id": _to_int(fields[0]),
                        "fracture_id": _to_int(fields[0]),
                        "liquid_volume_m3": _to_float(fields[1]),
                        "sand_mass_t": _to_float(fields[2]),
                        "cumulative_liquid_volume_m3": _to_float(fields[3]),
                        "cumulative_sand_mass_t": _to_float(fields[4]),
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "well_name",
                "stage",
                "step",
                "time",
                "balance_degree",
                "cumulative_balance_degree",
                "cluster_id",
                "fracture_id",
                "liquid_volume_m3",
                "sand_mass_t",
                "cumulative_liquid_volume_m3",
                "cumulative_sand_mass_t",
            ]
        )

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    step_map = {value: idx for idx, value in enumerate(sorted(df["time"].dropna().unique()))}
    df["step"] = df["time"].map(step_map).fillna(0).astype(int)
    return df.sort_values(["step", "cluster_id"]).reset_index(drop=True)


def frac_monitor_to_fiber_api_tables(path: str | Path, default_step_seconds: float = 1.0) -> FiberApiTables:
    """Expose a FracMonitor text file through the same table object as JSON API."""

    parsed = load_frac_monitor_text(path, default_step_seconds=default_step_seconds)
    return FiberApiTables(
        meta=parsed.meta,
        amplitude=pd.DataFrame(columns=["time_index", "depth_index", "time", "depth_m", "amplitude"]),
        stage_info=parsed.stage_info,
        controls=parsed.controls,
    )


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _to_int(value: object, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default
