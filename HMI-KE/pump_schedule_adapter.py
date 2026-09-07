"""Read and align static pump schedules with second-point construction data.

The pump schedule workbook is a stage-level reference, not a replacement for
the measured construction curve.  This adapter therefore keeps measured
``PL``/``SB`` untouched and adds namespaced ``SCHEDULE_*`` columns.  A
schedule is only attached when its well/stage identity matches the selected
frame; silently applying one stage's schedule to another stage would make the
training data invalid.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd


SCHEDULE_NUMERIC_COLUMNS = [
    "SCHEDULE_FLOW",
    "SCHEDULE_SAND",
    "SCHEDULE_PROGRESS",
]


def _normalise(value: object) -> str:
    return "".join(ch for ch in str(value).strip().lower() if ch.isalnum())


def _number(value: object) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).strip().replace(",", "")
    if not text or text.lower() in {"nan", "none", "null", "-"}:
        return np.nan
    text = text.replace("%", "")
    return float(pd.to_numeric(text, errors="coerce"))


def _find_column(columns: list[object], aliases: tuple[str, ...]) -> object | None:
    normalised = {_normalise(column): column for column in columns}
    for alias in aliases:
        candidate = normalised.get(_normalise(alias))
        if candidate is not None:
            return candidate
    for column in columns:
        key = _normalise(column)
        if any(_normalise(alias) in key for alias in aliases):
            return column
    return None


@dataclass(frozen=True)
class PumpSchedule:
    source_path: str
    well_name: str | None
    stage_id: str | None
    rows: pd.DataFrame
    status: str


def _identity_from_basic_sheet(path: Path) -> tuple[str | None, str | None]:
    try:
        basic = pd.read_excel(path, sheet_name=0, header=None)
    except Exception:
        return None, None
    values: dict[str, object] = {}
    for row in basic.itertuples(index=False, name=None):
        if len(row) < 2:
            continue
        key = _normalise(row[0])
        if key:
            values[key] = row[1]
    well = values.get("井名") or values.get("井") or values.get("wellname")
    stage = values.get("段号") or values.get("段") or values.get("stage")
    well_text = str(well).strip() if well is not None and str(well).strip() else None
    stage_text = str(stage).strip() if stage is not None and str(stage).strip() else None
    return well_text, stage_text


def _schedule_sheet(path: Path) -> str:
    book = pd.ExcelFile(path)
    for name in book.sheet_names:
        key = _normalise(name)
        if "泵序" in str(name) or "pumpschedule" in key or "pump" in key:
            return name
    raise ValueError(f"未找到施工泵序表工作表: {path}")


def load_pump_schedule(path: str | Path) -> PumpSchedule:
    """Load one static schedule and derive elapsed-time intervals."""

    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"泵序文件不存在: {source}")
    sheet = _schedule_sheet(source)
    raw = pd.read_excel(source, sheet_name=sheet, header=0)
    raw = raw.dropna(how="all").reset_index(drop=True)
    if raw.empty:
        raise ValueError(f"施工泵序表为空: {source}")

    columns = list(raw.columns)
    aliases = {
        "sequence": ("序号", "步骤", "step", "sequence"),
        "fluid_type": ("黏度/类型", "粘度/类型", "液体类型", "类型"),
        "flow_m3_min": ("排量", "排量m3/min", "flow", "rate"),
        "net_fluid_m3": ("净液量", "净液", "阶段液量", "netfluid"),
        "cumulative_fluid_m3": ("累计液量", "累计液", "cumulativefluid"),
        "sand_ratio_percent": ("砂比", "砂比%", "sandratio"),
        "sand_concentration": ("砂浓度", "浓度", "sandconcentration"),
        "stage_sand_volume": ("阶段砂体积", "阶段砂量", "sandvolume"),
        "cumulative_sand_volume": ("累计砂体积", "累计砂量", "cumulativesand"),
        "proppant_type": ("支撑剂类型", "支撑剂", "proppant"),
        "diverter": ("暂堵剂", "分流剂", "diverter"),
        "diverter_ball": ("暂堵球", "分流球", "diverterball"),
    }
    selected = {name: _find_column(columns, names) for name, names in aliases.items()}
    if selected["flow_m3_min"] is None:
        raise ValueError(f"施工泵序表缺少排量列，现有列: {columns}")

    schedule = pd.DataFrame(index=raw.index)
    for name, column in selected.items():
        if column is None:
            schedule[name] = np.nan
        elif name in {
            "sequence",
            "flow_m3_min",
            "net_fluid_m3",
            "cumulative_fluid_m3",
            "sand_ratio_percent",
            "sand_concentration",
            "stage_sand_volume",
            "cumulative_sand_volume",
        }:
            schedule[name] = raw[column].map(_number)
        else:
            schedule[name] = raw[column].fillna("").astype(str).str.strip()
    schedule["sequence"] = schedule["sequence"].fillna(pd.Series(np.arange(1, len(schedule) + 1), index=schedule.index))
    schedule["flow_m3_min"] = pd.to_numeric(schedule["flow_m3_min"], errors="coerce")
    schedule = schedule.loc[schedule["flow_m3_min"].gt(0)].reset_index(drop=True)
    if schedule.empty:
        raise ValueError(f"施工泵序表没有有效排量行: {source}")

    # Prefer the explicit net volume.  If it is absent, derive stage volume
    # from cumulative volume differences; no invented fixed duration is used.
    net = pd.to_numeric(schedule["net_fluid_m3"], errors="coerce")
    cumulative = pd.to_numeric(schedule["cumulative_fluid_m3"], errors="coerce")
    derived_net = cumulative.diff().fillna(cumulative)
    schedule["net_fluid_m3"] = net.where(net.gt(0), derived_net)
    schedule["duration_s"] = schedule["net_fluid_m3"] / schedule["flow_m3_min"] * 60.0
    schedule["duration_s"] = schedule["duration_s"].where(schedule["duration_s"].gt(0))
    schedule["start_s"] = schedule["duration_s"].fillna(0.0).cumsum().shift(fill_value=0.0)
    schedule["end_s"] = schedule["start_s"] + schedule["duration_s"]
    schedule["sand_ratio_percent"] = pd.to_numeric(schedule["sand_ratio_percent"], errors="coerce").fillna(0.0)
    schedule["stage_sand_volume"] = pd.to_numeric(schedule["stage_sand_volume"], errors="coerce").fillna(0.0)
    schedule["schedule_phase"] = np.select(
        [
            schedule["diverter"].astype(str).str.strip().ne("") | schedule["diverter_ball"].astype(str).str.strip().ne(""),
            schedule["sand_ratio_percent"].gt(0) | schedule["stage_sand_volume"].gt(0),
        ],
        ["diverter", "proppant"],
        default="fluid",
    )
    well_name, stage_id = _identity_from_basic_sheet(source)
    status = "usable_for_elapsed_time_alignment" if schedule["end_s"].notna().any() else "missing_duration"
    return PumpSchedule(str(source), well_name, stage_id, schedule, status)


def _frame_identity(frame: pd.DataFrame) -> tuple[set[str], set[str]]:
    well_values = set()
    stage_values = set()
    for column, target in (("JTBH", well_values), ("FDBH", stage_values), ("JTH", well_values)):
        if column not in frame:
            continue
        for value in frame[column].dropna().astype(str).head(1000):
            normalised = _normalise(value)
            if normalised:
                target.add(normalised)
    return well_values, stage_values


def identities_match(frame: pd.DataFrame, schedule: PumpSchedule) -> bool:
    well_values, stage_values = _frame_identity(frame)
    well_match = not schedule.well_name or not well_values or _normalise(schedule.well_name) in well_values
    stage_match = not schedule.stage_id or not stage_values or _normalise(schedule.stage_id) in stage_values
    return bool(well_match and stage_match)


def attach_schedule_to_frame(
    frame: pd.DataFrame,
    schedule: PumpSchedule,
    time_column: str = "SGSJ",
    strict_identity: bool = True,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Attach schedule reference columns without changing measured columns."""

    if strict_identity and not identities_match(frame, schedule):
        well_values, stage_values = _frame_identity(frame)
        raise ValueError(
            "泵序身份与选中井段不一致："
            f"泵序={schedule.well_name or '-'}/段{schedule.stage_id or '-'}，"
            f"井段={sorted(well_values)[:3] or ['-']}/FDBH{sorted(stage_values)[:3] or ['-']}"
        )

    out = frame.copy()
    for column in SCHEDULE_NUMERIC_COLUMNS:
        out[column] = np.nan
    out["SCHEDULE_PHASE"] = ""
    out["SCHEDULE_ROW"] = np.nan
    out["SCHEDULE_ALIGNMENT"] = "not_aligned"
    if time_column not in out or schedule.status != "usable_for_elapsed_time_alignment":
        return out, {"status": "missing_time_or_duration", "method": "none"}

    times = pd.to_datetime(out[time_column], errors="coerce")
    if not times.notna().any():
        return out, {"status": "invalid_frame_time", "method": "none"}
    elapsed = (times - times.dropna().iloc[0]).dt.total_seconds().to_numpy(dtype=float)
    ends = schedule.rows["end_s"].to_numpy(dtype=float)
    valid = np.isfinite(elapsed)
    indices = np.searchsorted(ends, np.where(valid, elapsed, np.inf), side="right")
    in_range = valid & (indices >= 0) & (indices < len(schedule.rows))
    if not in_range.any():
        return out, {"status": "frame_outside_schedule", "method": "elapsed_seconds"}
    for target, source in (
        ("SCHEDULE_FLOW", "flow_m3_min"),
        ("SCHEDULE_SAND", "sand_ratio_percent"),
        ("SCHEDULE_PROGRESS", "end_s"),
        ("SCHEDULE_ROW", "sequence"),
        ("SCHEDULE_PHASE", "schedule_phase"),
    ):
        values = schedule.rows[source].to_numpy()
        out.loc[in_range, target] = values[indices[in_range]]
    duration = float(schedule.rows["end_s"].iloc[-1])
    if duration > 0:
        out.loc[in_range, "SCHEDULE_PROGRESS"] = np.clip(elapsed[in_range] / duration, 0.0, 1.0)
    out.loc[in_range, "SCHEDULE_ALIGNMENT"] = "elapsed_seconds"
    return out, {
        "status": "attached",
        "method": "elapsed_seconds",
        "covered_rows": int(in_range.sum()),
        "total_rows": int(len(out)),
        "schedule_duration_s": duration,
        "well_name": schedule.well_name,
        "stage_id": schedule.stage_id,
    }
