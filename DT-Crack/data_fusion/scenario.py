"""Data contracts for the two field-observation digital-twin scenarios."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json


@dataclass(frozen=True)
class DigitalTwinScenario:
    scenario_id: str
    display_name: str
    observation_mode: str
    pressure_source: str
    fiber_source: str | None
    trajectory_source: str
    cluster_geometry_source: str | None
    well_id: str
    stage_id: str
    source_start_s: float
    source_end_s: float | None
    calibration_status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_scenarios(path: str | Path) -> dict[str, DigitalTwinScenario]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = payload.get("scenarios", payload) if isinstance(payload, dict) else {}
    result: dict[str, DigitalTwinScenario] = {}
    for scenario_id, value in records.items():
        if not isinstance(value, dict):
            continue
        result[str(scenario_id)] = DigitalTwinScenario(
            scenario_id=str(value.get("scenario_id", scenario_id)),
            display_name=str(value.get("display_name", scenario_id)),
            observation_mode=str(value.get("observation_mode", "pressure_only")),
            pressure_source=str(value.get("pressure_source", "")),
            fiber_source=value.get("fiber_source"),
            trajectory_source=str(value.get("trajectory_source", "")),
            cluster_geometry_source=value.get("cluster_geometry_source"),
            well_id=str(value.get("well_id", "")),
            stage_id=str(value.get("stage_id", "")),
            source_start_s=float(value.get("source_start_s", 1.0)),
            source_end_s=(None if value.get("source_end_s") is None else float(value["source_end_s"])),
            calibration_status=str(value.get("calibration_status", "待校准")),
        )
    return result


def load_cluster_geometry(path: str | Path | None, stage_id: str | None = None):
    """Load measured/configured cluster positions without inventing positions."""
    import pandas as pd

    columns = ["stage_id", "cluster_id", "md_m", "tvd_m", "east_m", "north_m", "fracture_azimuth_deg"]
    if not path or not Path(path).exists():
        return pd.DataFrame(columns=columns), {"status": "not_available", "source": None}
    frame = pd.read_csv(path, encoding="utf-8-sig")
    missing = [column for column in columns[:3] if column not in frame.columns]
    if missing:
        raise ValueError(f"cluster geometry is missing required columns: {missing}")
    if stage_id is not None:
        frame = frame[frame["stage_id"].astype(str) == str(stage_id)].copy()
    for column in columns[2:]:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["cluster_id"] = pd.to_numeric(frame["cluster_id"], errors="coerce").astype("Int64")
    frame = frame.dropna(subset=["stage_id", "cluster_id", "md_m"]).sort_values("cluster_id").reset_index(drop=True)
    return frame, {"status": "available", "source": str(Path(path)), "provenance": "configured"}
