from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_FUTURE_RISK_FEATURES = (
    "YTND",
    "SGBY",
    "PL",
    "LJJLTJ",
    "SND",
    "SB",
    "LJSL",
    "ZDCLYL",
    "SY",
    "ZDJ",
    "ZDQ",
    "LJYL",
    "ZDZSL",
    "ZDZYL",
    "JDPL",
    "JDSL",
    "BZJD",
)


@dataclass
class FutureRiskDataset:
    features: np.ndarray
    windows: np.ndarray
    rule_scores: np.ndarray
    targets: np.ndarray
    current_targets: np.ndarray
    groups: np.ndarray
    metadata: pd.DataFrame
    base_feature_names: list[str]
    feature_names: list[str]
    label_counts: dict[str, int]
    blank_label_count: int


def normalize_working_type(
    series: pd.Series,
    normal_label: str,
    blank_label_means_normal: bool,
) -> tuple[pd.Series, int]:
    text = series.fillna("").astype(str).str.strip()
    blank = text.eq("") | text.str.lower().isin({"nan", "none", "null"})
    blank_count = int(blank.sum())
    if blank_count and not blank_label_means_normal:
        raise ValueError(
            f"Found {blank_count} blank working-type labels. "
            "Pass blank_label_means_normal=True only after confirming that blank means normal."
        )
    normalized = text.mask(blank, normal_label)
    return normalized, blank_count


def _read_table(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(source)
    if suffix == ".csv":
        return pd.read_csv(source)
    if suffix == ".parquet":
        return pd.read_parquet(source)
    raise ValueError(f"Unsupported data format: {source}")


def _window_feature_names(columns: Iterable[str], window_size: int) -> list[str]:
    names: list[str] = []
    for column in columns:
        names.extend(f"{column}_t-{window_size - offset - 1}" for offset in range(window_size))
        names.extend(
            [
                f"{column}_last",
                f"{column}_mean",
                f"{column}_std",
                f"{column}_min",
                f"{column}_max",
                f"{column}_delta",
                f"{column}_slope",
            ]
        )
    return names


def _summarize_window(window: np.ndarray) -> np.ndarray:
    valid = np.isfinite(window)
    count = valid.sum(axis=0)
    total = np.where(valid, window, 0.0).sum(axis=0)
    mean = np.divide(total, count, out=np.full(window.shape[1], np.nan), where=count > 0)
    centered = np.where(valid, window - mean, 0.0)
    variance = np.divide(
        np.square(centered).sum(axis=0),
        count,
        out=np.full(window.shape[1], np.nan),
        where=count > 0,
    )
    std = np.sqrt(variance)
    minimum = np.where(valid, window, np.inf).min(axis=0)
    maximum = np.where(valid, window, -np.inf).max(axis=0)
    minimum[count == 0] = np.nan
    maximum[count == 0] = np.nan
    last = window[-1]
    delta = window[-1] - window[0]
    slope = delta / max(len(window) - 1, 1)
    per_feature = np.concatenate(
        [
            window.T,
            last[:, None],
            mean[:, None],
            std[:, None],
            minimum[:, None],
            maximum[:, None],
            delta[:, None],
            slope[:, None],
        ],
        axis=1,
    )
    return per_feature.reshape(-1)


def _rule_scores(window: np.ndarray, feature_names: list[str]) -> np.ndarray:
    def change(column: str) -> float:
        if column not in feature_names:
            return 0.0
        values = window[:, feature_names.index(column)]
        finite = values[np.isfinite(values)]
        if len(finite) < 2:
            return 0.0
        return float(finite[-1] - finite[0])

    pressure_rise = float(np.clip(max(change("SGBY"), 0.0) / 8.0, 0.0, 1.0))
    flow_drop = float(np.clip(max(-change("PL"), 0.0) / 1.0, 0.0, 1.0))
    sand_drop = float(np.clip(max(-change("SB"), 0.0) / 2.0, 0.0, 1.0))
    combined_risk = pressure_rise * max(flow_drop, sand_drop)
    return np.asarray([pressure_rise, flow_drop, sand_drop, combined_risk], dtype=np.float32)


def build_future_risk_dataset(
    data_path: str | Path,
    *,
    well_column: str = "JTBH",
    segment_column: str = "FDBH",
    time_column: str = "SGSJ",
    label_column: str = "WORKING_TYPE",
    normal_label: str = "正常",
    feature_columns: Iterable[str] = DEFAULT_FUTURE_RISK_FEATURES,
    window_size: int = 6,
    horizon: int = 1,
    max_gap_seconds: float = 30.0,
    blank_label_means_normal: bool = False,
) -> FutureRiskDataset:
    if window_size < 1:
        raise ValueError("window_size must be at least 1")
    if horizon < 1:
        raise ValueError("horizon must be at least 1")

    frame = _read_table(data_path)
    required = {well_column, segment_column, time_column, label_column}
    missing_required = sorted(required.difference(frame.columns))
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    requested_features = [column for column in feature_columns if column in frame.columns]
    if not requested_features:
        raise ValueError("None of the requested feature columns exist in the dataset")

    frame = frame.copy()
    frame["_normalized_label"], blank_count = normalize_working_type(
        frame[label_column], normal_label, blank_label_means_normal
    )
    frame["_time"] = pd.to_datetime(frame[time_column], errors="coerce")
    frame["_well"] = frame[well_column].fillna("").astype(str).str.strip()
    frame["_segment"] = frame[segment_column].fillna("").astype(str).str.strip()
    frame["_group"] = frame["_well"] + "::" + frame["_segment"]
    for column in requested_features:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    requested_features = [
        column for column in requested_features if frame[column].notna().any()
    ]
    if not requested_features:
        raise ValueError("All requested feature columns are empty after numeric conversion")

    features: list[np.ndarray] = []
    windows: list[np.ndarray] = []
    rule_scores: list[np.ndarray] = []
    targets: list[int] = []
    current_targets: list[int] = []
    groups: list[str] = []
    metadata: list[dict[str, object]] = []

    for group, segment in frame.groupby("_group", sort=True):
        segment = segment.sort_values("_time", kind="stable").reset_index(drop=True)
        if len(segment) < window_size + horizon:
            continue
        numeric = segment[requested_features].to_numpy(dtype=np.float64)
        labels = segment["_normalized_label"].to_numpy(dtype=object)
        times = segment["_time"].to_numpy(dtype="datetime64[ns]")
        for start in range(0, len(segment) - window_size - horizon + 1):
            end = start + window_size
            target_index = end + horizon - 1
            relevant_times = times[start : target_index + 1]
            if np.isnat(relevant_times).any():
                continue
            gaps = np.diff(relevant_times).astype("timedelta64[ms]").astype(np.float64) / 1000.0
            if np.any(gaps <= 0) or np.any(gaps > max_gap_seconds):
                continue
            window = numeric[start:end]
            target = int(labels[target_index] != normal_label)
            current_target = int(labels[end - 1] != normal_label)
            features.append(_summarize_window(window))
            windows.append(window.astype(np.float32))
            rule_scores.append(_rule_scores(window, requested_features))
            targets.append(target)
            current_targets.append(current_target)
            groups.append(group)
            metadata.append(
                {
                    "group": group,
                    "well": segment.at[end - 1, "_well"],
                    "segment": segment.at[end - 1, "_segment"],
                    "window_start": str(segment.at[start, "_time"]),
                    "window_end": str(segment.at[end - 1, "_time"]),
                    "target_time": str(segment.at[target_index, "_time"]),
                    "current_label": str(labels[end - 1]),
                    "future_label": str(labels[target_index]),
                    "normal_to_abnormal": bool(current_target == 0 and target == 1),
                }
            )

    if not features:
        raise ValueError("No valid future-risk windows were generated")

    label_counts = frame["_normalized_label"].value_counts().astype(int).to_dict()
    return FutureRiskDataset(
        features=np.vstack(features).astype(np.float32),
        windows=np.stack(windows).astype(np.float32),
        rule_scores=np.vstack(rule_scores).astype(np.float32),
        targets=np.asarray(targets, dtype=np.int64),
        current_targets=np.asarray(current_targets, dtype=np.int64),
        groups=np.asarray(groups, dtype=object),
        metadata=pd.DataFrame(metadata),
        base_feature_names=requested_features,
        feature_names=_window_feature_names(requested_features, window_size),
        label_counts={str(key): int(value) for key, value in label_counts.items()},
        blank_label_count=blank_count,
    )


def split_groups(
    groups: np.ndarray,
    *,
    seed: int = 42,
    train_ratio: float = 0.6,
    val_ratio: float = 0.15,
    transfer_ratio: float = 0.1,
    test_ratio: float = 0.15,
) -> dict[str, list[str]]:
    ratios = {
        "train": train_ratio,
        "val": val_ratio,
        "transfer": transfer_ratio,
        "test": test_ratio,
    }
    if not np.isclose(sum(ratios.values()), 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {sum(ratios.values())}")
    unique_groups = np.unique(groups).astype(str)
    if len(unique_groups) < len(ratios):
        raise ValueError("Not enough independent groups for all requested splits")

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_groups)
    raw_counts = {name: ratio * len(shuffled) for name, ratio in ratios.items()}
    counts = {name: int(np.floor(value)) for name, value in raw_counts.items()}
    for name in ratios:
        counts[name] = max(counts[name], 1)
    while sum(counts.values()) > len(shuffled):
        candidate = max((name for name in counts if counts[name] > 1), key=lambda name: counts[name])
        counts[candidate] -= 1
    while sum(counts.values()) < len(shuffled):
        candidate = max(ratios, key=lambda name: raw_counts[name] - counts[name])
        counts[candidate] += 1

    manifest: dict[str, list[str]] = {}
    cursor = 0
    for name in ("train", "val", "transfer", "test"):
        count = counts[name]
        manifest[name] = sorted(shuffled[cursor : cursor + count].tolist())
        cursor += count
    return manifest


def indices_for_manifest(groups: np.ndarray, manifest: dict[str, list[str]]) -> dict[str, np.ndarray]:
    return {
        name: np.flatnonzero(np.isin(groups.astype(str), np.asarray(group_names, dtype=str)))
        for name, group_names in manifest.items()
    }
