from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .future_risk_data import _summarize_window, _window_feature_names


DEFAULT_SAND_RISK_FEATURES = (
    "SGBY",
    "PL",
    "SB",
    "LJYL",
    "LJSL",
    "BZJD",
    "YTND",
    "JDPL",
    "JDSL",
)

ALIASES = {
    "SGBY": ("SGBY", "施工泵压", "泵压", "压力"),
    "PL": ("PL", "排量", "流量"),
    "SB": ("SB", "砂比", "沙比", "砂比(%)"),
    "LJYL": ("LJYL", "累计液量"),
    "LJSL": ("LJSL", "累计砂量"),
    "BZJD": ("BZJD", "泵注阶段"),
    "YTND": ("YTND", "液体黏度", "粘度"),
    "JDPL": ("JDPL",),
    "JDSL": ("JDSL",),
    "SGRQ": ("SGRQ", "施工日期", "日期"),
    "SGSJ": ("SGSJ", "施工时间", "时间"),
}


@dataclass
class SandRiskDataset:
    features: np.ndarray
    windows: np.ndarray
    rule_scores: np.ndarray
    targets: dict[int, np.ndarray]
    current_targets: np.ndarray
    groups: np.ndarray
    segment_groups: np.ndarray
    metadata: pd.DataFrame
    base_feature_names: list[str]
    feature_names: list[str]
    event_table: pd.DataFrame
    source_file_count: int


def _find_column(frame: pd.DataFrame, canonical: str) -> str | None:
    lookup = {str(column).strip().lower(): str(column) for column in frame.columns}
    for alias in ALIASES[canonical]:
        result = lookup.get(alias.strip().lower())
        if result is not None:
            return result
    return None


def _detect_header(excel: pd.ExcelFile, sheet_name: str, nrows: int = 20) -> int:
    preview = pd.read_excel(excel, sheet_name=sheet_name, header=None, nrows=nrows)
    keywords = {"SGBY", "PL", "SB", "SGRQ", "SGSJ"}
    best_row = 0
    best_count = -1
    for index, row in preview.iterrows():
        cells = {str(value).strip().upper() for value in row if pd.notna(value)}
        count = len(keywords.intersection(cells))
        if count > best_count:
            best_row = int(index)
            best_count = count
    return best_row if best_count >= 2 else 0


def _combine_timestamp(frame: pd.DataFrame) -> pd.Series:
    time_column = _find_column(frame, "SGSJ")
    if time_column is None:
        return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    time_raw = frame[time_column]
    date_column = _find_column(frame, "SGRQ")
    if date_column is None:
        return pd.to_datetime(time_raw, errors="coerce")
    dates = pd.to_datetime(frame[date_column], errors="coerce").dt.normalize()

    def time_delta(value: object) -> pd.Timedelta | pd.NaT:
        if pd.isna(value):
            return pd.NaT
        if hasattr(value, "hour"):
            return pd.Timedelta(
                hours=int(value.hour),
                minutes=int(value.minute),
                seconds=int(value.second),
            )
        text = str(value).strip()
        parsed = pd.to_datetime(text, errors="coerce")
        if pd.isna(parsed):
            return pd.NaT
        return pd.Timedelta(hours=parsed.hour, minutes=parsed.minute, seconds=parsed.second)

    deltas = time_raw.map(time_delta)
    combined = dates + deltas
    if combined.notna().all():
        return combined
    direct = pd.to_datetime(time_raw, errors="coerce")
    return combined.where(combined.notna(), direct)


def _read_warning_events(result_path: Path) -> pd.DataFrame:
    try:
        warnings = pd.read_excel(result_path, sheet_name="砂堵预警")
    except ValueError:
        return pd.DataFrame()
    required = {"段号", "开始时间", "结束时间", "预警类型"}
    if not required.issubset(warnings.columns):
        return pd.DataFrame()
    warnings = warnings.loc[warnings["预警类型"].astype(str).str.strip().eq("砂堵迹象")].copy()
    warnings["event_start"] = pd.to_datetime(warnings["开始时间"], errors="coerce")
    warnings["event_end"] = pd.to_datetime(warnings["结束时间"], errors="coerce")
    warnings["segment"] = warnings["段号"].astype(str).str.strip()
    return warnings.dropna(subset=["event_start", "event_end"])


def _active_event(times: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    if len(starts) == 0:
        return np.zeros(len(times), dtype=np.int64)
    active = np.zeros(len(times), dtype=bool)
    for start, end in zip(starts, ends):
        active |= (times >= start) & (times <= end)
    return active.astype(np.int64)


def _future_event_targets(
    times: np.ndarray,
    event_starts: np.ndarray,
    horizons_seconds: Iterable[int],
) -> dict[int, np.ndarray]:
    if len(event_starts) == 0:
        return {int(horizon): np.zeros(len(times), dtype=np.int64) for horizon in horizons_seconds}
    starts = np.sort(event_starts.astype("datetime64[ns]"))
    left = np.searchsorted(starts, times, side="right")
    result: dict[int, np.ndarray] = {}
    for horizon in horizons_seconds:
        limit = times + np.timedelta64(int(horizon), "s")
        right = np.searchsorted(starts, limit, side="right")
        result[int(horizon)] = (right > left).astype(np.int64)
    return result


def sand_rule_scores(window: np.ndarray, feature_names: list[str]) -> np.ndarray:
    def values(name: str) -> np.ndarray:
        if name not in feature_names:
            return np.asarray([], dtype=float)
        column = window[:, feature_names.index(name)].astype(float)
        return column[np.isfinite(column)]

    pressure = values("SGBY")
    rate = values("PL")
    sand = values("SB")
    pressure_delta = float(pressure[-1] - pressure[0]) if len(pressure) >= 2 else 0.0
    pressure_rise = float(np.clip(max(pressure_delta, 0.0) / 5.0, 0.0, 1.0))
    rate_stability = 0.0 if len(rate) < max(2, len(window) // 2) else float(
        np.clip(1.0 - np.std(rate) / 0.2, 0.0, 1.0)
    )
    sand_stability = 0.0 if len(sand) < max(2, len(window) // 2) else float(
        np.clip(1.0 - np.std(sand) / 1.0, 0.0, 1.0)
    )
    combined = pressure_rise * rate_stability * sand_stability
    return np.asarray(
        [pressure_rise, rate_stability, sand_stability, combined], dtype=np.float32
    )


def _raw_path_for_result(result_path: Path) -> Path:
    suffix = "_小条统计_增强版.xlsx"
    if not result_path.name.endswith(suffix):
        raise ValueError(f"Unexpected enhanced result name: {result_path.name}")
    return result_path.with_name(result_path.name[: -len(suffix)] + ".xlsx")


def build_sand_risk_dataset(
    data_dir: str | Path,
    *,
    horizons_seconds: Iterable[int] = (60, 180, 300),
    feature_columns: Iterable[str] = DEFAULT_SAND_RISK_FEATURES,
    window_size: int = 6,
    stride: int = 3,
    max_gap_seconds: float = 30.0,
    result_glob: str = "*_小条统计_增强版.xlsx",
) -> SandRiskDataset:
    horizons = sorted({int(value) for value in horizons_seconds})
    if not horizons or any(value <= 0 for value in horizons):
        raise ValueError("horizons_seconds must contain positive integers")
    if window_size < 2 or stride < 1:
        raise ValueError("window_size must be >=2 and stride must be >=1")

    directory = Path(data_dir)
    result_files = sorted(directory.glob(result_glob))
    pairs = [(result, _raw_path_for_result(result)) for result in result_files]
    pairs = [(result, raw) for result, raw in pairs if raw.exists()]
    if not pairs:
        raise ValueError(f"No enhanced-result/raw-workbook pairs found in {directory}")

    requested = list(feature_columns)
    features: list[np.ndarray] = []
    windows: list[np.ndarray] = []
    rules: list[np.ndarray] = []
    targets = {horizon: [] for horizon in horizons}
    current_targets: list[int] = []
    groups: list[str] = []
    segment_groups: list[str] = []
    metadata: list[dict[str, object]] = []
    all_events: list[pd.DataFrame] = []

    for result_path, raw_path in pairs:
        warning_events = _read_warning_events(result_path)
        well_name = raw_path.stem.removesuffix("_更新后").removesuffix("_split")
        if not warning_events.empty:
            warning_events = warning_events.copy()
            warning_events["well"] = well_name
            warning_events["source_file"] = raw_path.name
            all_events.append(warning_events)

        with pd.ExcelFile(raw_path) as excel:
            for sheet_name in excel.sheet_names:
                frame = pd.read_excel(excel, sheet_name=sheet_name, header=0)
                if _find_column(frame, "SGBY") is None or _find_column(frame, "PL") is None:
                    header = _detect_header(excel, sheet_name)
                    frame = pd.read_excel(excel, sheet_name=sheet_name, header=header)
                if frame.empty:
                    continue
                time = _combine_timestamp(frame)
                numeric = pd.DataFrame(index=frame.index)
                for canonical in requested:
                    column = _find_column(frame, canonical)
                    numeric[canonical] = (
                        pd.to_numeric(frame[column], errors="coerce")
                        if column is not None
                        else np.nan
                    )
                order = np.argsort(time.to_numpy(dtype="datetime64[ns]"), kind="stable")
                time_values = time.iloc[order].to_numpy(dtype="datetime64[ns]")
                numeric_values = numeric.iloc[order].to_numpy(dtype=np.float64)
                valid_time = ~np.isnat(time_values)
                time_values = time_values[valid_time]
                numeric_values = numeric_values[valid_time]
                if len(time_values) < window_size:
                    continue

                segment = str(sheet_name).strip()
                segment_events = warning_events.loc[
                    warning_events["segment"].eq(segment)
                ] if not warning_events.empty else pd.DataFrame()
                event_starts = (
                    segment_events["event_start"].to_numpy(dtype="datetime64[ns]")
                    if not segment_events.empty
                    else np.asarray([], dtype="datetime64[ns]")
                )
                event_ends = (
                    segment_events["event_end"].to_numpy(dtype="datetime64[ns]")
                    if not segment_events.empty
                    else np.asarray([], dtype="datetime64[ns]")
                )
                future = _future_event_targets(time_values, event_starts, horizons)
                active = _active_event(time_values, event_starts, event_ends)
                segment_group = f"{well_name}::{segment}"

                for end in range(window_size - 1, len(time_values), stride):
                    start = end - window_size + 1
                    relevant_times = time_values[start : end + 1]
                    gaps = np.diff(relevant_times).astype("timedelta64[ms]").astype(float) / 1000.0
                    if np.any(gaps <= 0) or np.any(gaps > max_gap_seconds):
                        continue
                    window = numeric_values[start : end + 1]
                    features.append(_summarize_window(window))
                    windows.append(window.astype(np.float32))
                    rules.append(sand_rule_scores(window, requested))
                    for horizon in horizons:
                        targets[horizon].append(int(future[horizon][end]))
                    current_targets.append(int(active[end]))
                    groups.append(well_name)
                    segment_groups.append(segment_group)
                    metadata.append(
                        {
                            "group": segment_group,
                            "well": well_name,
                            "segment": segment,
                            "source_file": raw_path.name,
                            "window_start": str(pd.Timestamp(time_values[start])),
                            "window_end": str(pd.Timestamp(time_values[end])),
                        }
                    )

    if not features:
        raise ValueError("No valid sand-risk samples were generated")
    event_table = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    return SandRiskDataset(
        features=np.vstack(features).astype(np.float32),
        windows=np.stack(windows).astype(np.float32),
        rule_scores=np.vstack(rules).astype(np.float32),
        targets={key: np.asarray(value, dtype=np.int64) for key, value in targets.items()},
        current_targets=np.asarray(current_targets, dtype=np.int64),
        groups=np.asarray(groups, dtype=object),
        segment_groups=np.asarray(segment_groups, dtype=object),
        metadata=pd.DataFrame(metadata),
        base_feature_names=requested,
        feature_names=_window_feature_names(requested, window_size),
        event_table=event_table,
        source_file_count=len(pairs),
    )
