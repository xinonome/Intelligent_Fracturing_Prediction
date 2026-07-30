from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SUPPORTED_SUFFIXES = {".xlsx", ".xls", ".csv", ".parquet", ".json"}


@dataclass
class DatasetBundle:
    x: np.ndarray
    y: np.ndarray
    meta: pd.DataFrame
    feature_names: list[str]
    target_names: list[str]
    action_bounds: dict[str, dict[str, float]]


def read_table(path: Path, header: int | None = 0, nrows: int | None = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, header=header, nrows=nrows)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, header=header, nrows=nrows)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported data file: {path}")


def load_reference_columns(path: str | None) -> list[str] | None:
    if not path:
        return None
    return read_table(Path(path), header=0, nrows=0).columns.tolist()


def load_frame_with_reference(
    path: Path,
    reference_columns: list[str] | None,
    required_columns: Iterable[str],
    max_rows_per_file: int,
) -> pd.DataFrame | None:
    required = set(required_columns)
    nrows = max_rows_per_file if max_rows_per_file > 0 else None
    if reference_columns is not None:
        try:
            frame = read_table(path, header=None, nrows=nrows)
            if len(frame.columns) == len(reference_columns):
                frame.columns = reference_columns
                if required.issubset(frame.columns):
                    return frame
        except Exception:
            pass
    try:
        frame = read_table(path, header=0, nrows=nrows)
    except Exception:
        return None
    return frame if required.issubset(frame.columns) else None


def discover_segment_frames(
    data_path: str,
    reference_header_path: str | None,
    segment_column: str,
    time_column: str,
    required_columns: list[str],
    exclude_name_patterns: Iterable[str],
    max_files: int,
    max_rows_per_file: int,
) -> dict[str, pd.DataFrame]:
    root = Path(data_path)
    if not root.exists():
        raise FileNotFoundError(f"Data path not found: {root}")
    reference_columns = load_reference_columns(reference_header_path)
    exclude_patterns = tuple(pattern.lower() for pattern in exclude_name_patterns)
    frames: dict[str, pd.DataFrame] = {}

    def add_frame(key: str, frame: pd.DataFrame) -> None:
        candidate = str(key)
        suffix = 2
        while candidate in frames:
            candidate = f"{key}__{suffix}"
            suffix += 1
        frames[candidate] = frame.reset_index(drop=True).copy()

    paths = [root] if root.is_file() else sorted(path for path in root.iterdir() if path.is_file())
    loaded_files = 0
    for path in paths:
        if path.suffix.lower() not in SUPPORTED_SUFFIXES or path.name.startswith("~$"):
            continue
        if any(pattern in path.name.lower() for pattern in exclude_patterns):
            continue
        if max_files and loaded_files >= max_files:
            break
        frame = load_frame_with_reference(
            path,
            reference_columns,
            required_columns + [time_column],
            max_rows_per_file,
        )
        if frame is None:
            continue
        loaded_files += 1
        if segment_column in frame.columns and frame[segment_column].nunique(dropna=False) > 1:
            for segment_id, segment_frame in frame.groupby(segment_column, dropna=False):
                add_frame(str(segment_id), segment_frame)
        else:
            add_frame(path.stem, frame)
    if not frames:
        raise ValueError(f"No usable segment frames found under {root}")
    return frames


def numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return (
        pd.to_numeric(frame[column], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .interpolate(limit_direction="both")
        .fillna(0.0)
    )


def sort_frame(frame: pd.DataFrame, time_column: str) -> pd.DataFrame:
    out = frame.copy()
    if time_column in out.columns:
        out["_sort_time"] = pd.to_datetime(out[time_column], errors="coerce")
        if out["_sort_time"].notna().any():
            out = out.sort_values("_sort_time", kind="stable")
        out = out.drop(columns="_sort_time")
    return out.reset_index(drop=True)


def estimate_sample_interval_seconds(
    frames: dict[str, pd.DataFrame],
    time_column: str,
    default: float,
) -> float:
    deltas: list[float] = []
    for frame in frames.values():
        if time_column not in frame.columns:
            continue
        times = pd.to_datetime(frame[time_column], errors="coerce").dropna().sort_values()
        diff = times.diff().dt.total_seconds().dropna()
        deltas.extend(diff[(diff > 0) & (diff < 3600)].tolist()[:500])
    return float(np.median(deltas)) if deltas else default


def _slope(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.polyfit(np.arange(len(values), dtype=float), values.astype(float), 1)[0])


def _labels(values: np.ndarray) -> str:
    invalid = {"", "nan", "none", "null"}
    return "|".join(sorted({str(value).strip() for value in values if str(value).strip().lower() not in invalid}))


def _label_flags(values: np.ndarray) -> tuple[int, int]:
    labels = _labels(values)
    if not labels:
        return 0, 0
    normalized = labels.replace("正常工况", "正常")
    items = [item.strip() for item in normalized.split("|") if item.strip()]
    abnormal = int(any(item != "正常" for item in items))
    sand_plug = int(any("砂堵" in item for item in items))
    return abnormal, sand_plug


def build_dataset(
    frames: dict[str, pd.DataFrame],
    state_columns: list[str],
    action_columns: list[str],
    time_column: str,
    state_points: int,
    action_points: int,
    label_column: str = "WORKING_TYPE",
) -> DatasetBundle:
    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    meta_rows: list[dict] = []
    feature_names = [
        f"{column}_t-{offset}"
        for column in state_columns
        for offset in range(state_points - 1, -1, -1)
    ]
    feature_names.extend(
        f"{column}_{stat}"
        for column in state_columns
        for stat in ("last", "mean", "std", "slope")
    )

    for segment_id, raw_frame in frames.items():
        frame = sort_frame(raw_frame, time_column)
        if len(frame) < state_points + action_points:
            continue
        values = {
            column: numeric_series(frame, column).to_numpy(dtype=float)
            for column in set(state_columns + action_columns)
        }
        times = frame[time_column].astype(str).to_numpy() if time_column in frame else np.full(len(frame), "")
        labels = frame[label_column].to_numpy(dtype=object) if label_column in frame else np.full(len(frame), "", dtype=object)
        for end in range(state_points - 1, len(frame) - action_points):
            history = slice(end - state_points + 1, end + 1)
            future = slice(end + 1, end + 1 + action_points)
            windows = [values[column][history] for column in state_columns]
            stats = [
                value
                for window in windows
                for value in (window[-1], np.mean(window), np.std(window), _slope(window))
            ]
            x_rows.append(np.concatenate(windows + [np.asarray(stats, dtype=float)]))
            y_rows.append(np.asarray([np.mean(values[column][future]) for column in action_columns]))
            future_abnormal, future_sand_plug = _label_flags(labels[future])
            future_pressure = values[state_columns[0]][future]
            meta_rows.append(
                {
                    "segment_id": segment_id,
                    "time_index": end,
                    "time": times[end],
                    "current_pressure": values[state_columns[0]][end],
                    "current_flow": values["PL"][end] if "PL" in values else np.nan,
                    "current_sand_ratio": values["SB"][end] if "SB" in values else np.nan,
                    "state_working_types": _labels(labels[history]),
                    "future_working_types": _labels(labels[future]),
                    "future_pressure_mean": float(np.mean(future_pressure)),
                    "future_pressure_max": float(np.max(future_pressure)),
                    "future_abnormal": future_abnormal,
                    "future_sand_plug": future_sand_plug,
                }
            )

    if not x_rows:
        raise ValueError("No policy samples generated. Check window sizes and columns.")
    targets = np.vstack(y_rows)
    bounds = {
        column: {
            "p01": float(np.nanpercentile(targets[:, index], 1)),
            "p99": float(np.nanpercentile(targets[:, index], 99)),
            "min": float(np.nanmin(targets[:, index])),
            "max": float(np.nanmax(targets[:, index])),
        }
        for index, column in enumerate(action_columns)
    }
    return DatasetBundle(
        x=np.vstack(x_rows),
        y=targets,
        meta=pd.DataFrame(meta_rows),
        feature_names=feature_names,
        target_names=[f"future_60s_mean_{column}" for column in action_columns],
        action_bounds=bounds,
    )


def segment_split(
    meta: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    segments = meta["segment_id"].drop_duplicates().to_numpy()
    rng.shuffle(segments)
    n_train = max(1, int(round(len(segments) * train_ratio)))
    n_val = max(1, int(round(len(segments) * val_ratio))) if len(segments) >= 3 else 0
    train_segments = set(segments[:n_train])
    val_segments = set(segments[n_train : n_train + n_val])
    test_segments = set(segments[n_train + n_val :])
    if not test_segments and len(segments) > 1:
        moved = next(iter(val_segments or train_segments))
        val_segments.discard(moved)
        train_segments.discard(moved)
        test_segments.add(moved)
    return tuple(
        meta.index[meta["segment_id"].isin(group)].to_numpy()
        for group in (train_segments, val_segments, test_segments)
    )
