from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch_geometric.data import Data


SUPPORTED_SUFFIXES = {".csv", ".xlsx", ".xls", ".parquet", ".json"}
DEFAULT_EXCLUDE_SUBSTRINGS = (
    "mark",
    "remark",
    "comment",
    "success",
    "result",
    "day_time",
    "suggest",
    "chance",
)
DEFAULT_EXCLUDE_COLUMNS = {
    "ID",
    "ID.1",
    "JTBH",
    "SGRQ",
    "PROBABILITY",
    "DATA_ID",
    "JTH",
    "JD",
    "CREATE_BY",
    "CREATE_DATE",
    "UPDATE_BY",
    "UPDATE_DATE",
    "DRAW_TIME",
    "DRAW_TIME_END",
    "QUYU",
}
BLANK_LABEL_VALUES = {"", " ", "nan", "none", "null"}
DEFAULT_DYNAMIC_BASE_COLUMNS = (
    "BZJDH",
    "YTND",
    "SGBY",
    "PL",
    "SB",
    "LJSL",
    "ZDCLYL",
    "ZDJ",
    "LJYL",
    "BZJD",
)


@dataclass
class PreparedData:
    train_graphs: list[Data]
    val_graphs: list[Data]
    test_graphs: list[Data]
    transfer_graphs: list[Data]
    feature_columns: list[str]
    classes: list[str]
    split_manifest: dict[str, list[str]]
    segment_sizes: dict[str, int]


@dataclass
class BasePreparedData:
    segment_frames: dict[str, pd.DataFrame]
    feature_columns: list[str]
    classes: list[str]
    split_manifest: dict[str, list[str]]
    segment_sizes: dict[str, int]
    label_column: str
    label_encoder: LabelEncoder
    imputer: SimpleImputer
    scaler: StandardScaler


def _numeric_or_zero(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def load_tabular_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported file type: {path}")


def load_reference_columns(path: str | None) -> list[str] | None:
    if not path:
        return None
    reference = load_tabular_file(Path(path))
    return reference.columns.tolist()


def maybe_apply_reference_header(
    frame: pd.DataFrame, reference_columns: list[str] | None, label_column: str
) -> pd.DataFrame:
    if label_column in frame.columns:
        return frame
    if not reference_columns:
        return frame
    if len(frame.columns) != len(reference_columns):
        return frame
    renamed = frame.copy()
    renamed.columns = reference_columns
    return renamed


def load_dataset_frame(
    path: Path, reference_columns: list[str] | None, label_column: str
) -> pd.DataFrame:
    frame = load_tabular_file(path)
    if label_column in frame.columns:
        return frame
    if reference_columns is None:
        return frame
    no_header_frame = load_tabular_file_without_header(path)
    return maybe_apply_reference_header(no_header_frame, reference_columns, label_column)


def load_tabular_file_without_header(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, header=None)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, header=None)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported file type: {path}")


def discover_segment_frames(
    data_path: str,
    segment_column: str | None,
    label_column: str,
    reference_header_path: str | None,
    exclude_name_patterns: Iterable[str] | None,
) -> dict[str, pd.DataFrame]:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data path not found: {path}")

    reference_columns = load_reference_columns(reference_header_path)
    exclude_patterns = tuple(pattern.lower() for pattern in (exclude_name_patterns or []))
    segment_frames: dict[str, pd.DataFrame] = {}

    def add_segment_frame(key: str, frame: pd.DataFrame, source_name: str | None = None) -> None:
        candidate = str(key)
        if candidate in segment_frames and source_name:
            candidate = f"{Path(source_name).stem}__{candidate}"
        suffix = 2
        unique_key = candidate
        while unique_key in segment_frames:
            unique_key = f"{candidate}__{suffix}"
            suffix += 1
        segment_frames[unique_key] = frame.reset_index(drop=True).copy()

    if path.is_file():
        frame = load_dataset_frame(path, reference_columns, label_column)
        if segment_column is None or segment_column not in frame.columns:
            raise ValueError(
                "A segment column is required when loading a single total dataset file."
            )
        for segment_id, segment_frame in frame.groupby(segment_column, dropna=False):
            add_segment_frame(str(segment_id), segment_frame, path.name)
        return segment_frames

    for file_path in sorted(path.iterdir()):
        if not file_path.is_file() or file_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        if file_path.name.startswith("~$"):
            continue
        lowered_name = file_path.name.lower()
        if exclude_patterns and any(pattern in lowered_name for pattern in exclude_patterns):
            continue
        frame = load_dataset_frame(file_path, reference_columns, label_column)
        if label_column not in frame.columns:
            continue
        if segment_column and segment_column in frame.columns and frame[segment_column].nunique(dropna=False) > 1:
            for segment_id, segment_frame in frame.groupby(segment_column, dropna=False):
                add_segment_frame(str(segment_id), segment_frame, file_path.name)
            continue
        add_segment_frame(file_path.stem, frame, file_path.name)

    if not segment_frames:
        raise ValueError(f"No supported dataset files found under: {path}")
    return segment_frames


def sort_segment_frames(
    segment_frames: dict[str, pd.DataFrame], time_column: str | None
) -> dict[str, pd.DataFrame]:
    sorted_frames: dict[str, pd.DataFrame] = {}
    for segment_id, frame in segment_frames.items():
        cleaned = frame.copy()
        if time_column and time_column in cleaned.columns:
            cleaned = cleaned.sort_values(time_column, kind="stable").reset_index(drop=True)
        else:
            cleaned = cleaned.reset_index(drop=True)
        sorted_frames[segment_id] = cleaned
    return sorted_frames


def trim_segments_from_first_sand(
    segment_frames: dict[str, pd.DataFrame],
    sand_column: str,
    sand_threshold: float,
) -> dict[str, pd.DataFrame]:
    trimmed: dict[str, pd.DataFrame] = {}
    for segment_id, frame in segment_frames.items():
        if sand_column not in frame.columns:
            trimmed[segment_id] = frame.reset_index(drop=True).copy()
            continue
        sand = _numeric_or_zero(frame[sand_column])
        active_idx = sand[sand > sand_threshold].index
        if len(active_idx) == 0:
            continue
        start = int(active_idx[0])
        updated = frame.iloc[start:].reset_index(drop=True).copy()
        trimmed[segment_id] = updated
    return trimmed


def normalize_label_value(value: object, normal_label: str) -> str:
    if pd.isna(value):
        return normal_label
    text = str(value).strip()
    if text.lower() in BLANK_LABEL_VALUES or text == "":
        return normal_label
    return text


def normalize_labels_in_frames(
    segment_frames: dict[str, pd.DataFrame], label_column: str, normal_label: str
) -> dict[str, pd.DataFrame]:
    normalized: dict[str, pd.DataFrame] = {}
    for segment_id, frame in segment_frames.items():
        updated = frame.copy()
        updated[label_column] = updated[label_column].map(
            lambda value: normalize_label_value(value, normal_label)
        )
        normalized[segment_id] = updated
    return normalized


def binarize_labels_in_frames(
    segment_frames: dict[str, pd.DataFrame],
    label_column: str,
    normal_label: str,
    abnormal_label: str,
) -> dict[str, pd.DataFrame]:
    binarized: dict[str, pd.DataFrame] = {}
    for segment_id, frame in segment_frames.items():
        updated = frame.copy()
        updated[label_column] = updated[label_column].map(
            lambda value: normal_label if str(value) == normal_label else abnormal_label
        )
        binarized[segment_id] = updated
    return binarized


def drop_single_label_segments(
    segment_frames: dict[str, pd.DataFrame],
    label_column: str,
    keep_label: str | None = None,
) -> dict[str, pd.DataFrame]:
    filtered: dict[str, pd.DataFrame] = {}
    for segment_id, frame in segment_frames.items():
        labels = set(frame[label_column].astype(str).unique().tolist())
        if len(labels) == 1 and (keep_label is None or keep_label in labels):
            continue
        filtered[segment_id] = frame
    return filtered


def add_dynamic_features_to_frames(
    segment_frames: dict[str, pd.DataFrame],
    base_columns: Iterable[str],
    sand_column: str,
    rolling_windows: Iterable[int],
) -> dict[str, pd.DataFrame]:
    updated_frames: dict[str, pd.DataFrame] = {}
    windows = [int(window) for window in rolling_windows if int(window) > 1]
    for segment_id, frame in segment_frames.items():
        updated = frame.copy()
        derived: dict[str, pd.Series] = {}
        if sand_column in updated.columns:
            sand = _numeric_or_zero(updated[sand_column])
        else:
            sand = pd.Series(np.zeros(len(updated), dtype=float), index=updated.index)
        sand_active_flag = (sand > 0).astype(float)
        derived["sand_active_flag"] = sand_active_flag
        derived["sand_active_run_length"] = sand_active_flag.groupby(
            (sand_active_flag == 0).cumsum()
        ).cumsum()
        derived["points_since_first_sand"] = pd.Series(
            np.arange(len(updated), dtype=float), index=updated.index
        )
        derived["time_since_first_sand_sec"] = derived["points_since_first_sand"] * 10.0

        for column in base_columns:
            if column not in updated.columns:
                continue
            numeric = _numeric_or_zero(updated[column])
            diff1 = numeric.diff().fillna(0.0)
            derived[f"{column}_diff1"] = diff1
            derived[f"{column}_diff2"] = diff1.diff().fillna(0.0)
            for window in windows:
                rolling = numeric.rolling(window=window, min_periods=1)
                derived[f"{column}_mean_{window}"] = rolling.mean()
                derived[f"{column}_std_{window}"] = rolling.std().fillna(0.0)
                derived[f"{column}_min_{window}"] = rolling.min()
                derived[f"{column}_max_{window}"] = rolling.max()
                derived[f"{column}_slope_{window}"] = (
                    numeric - numeric.shift(window - 1)
                ).fillna(0.0) / float(window)

        for window in windows:
            sand_roll = sand.rolling(window=window, min_periods=1)
            derived[f"{sand_column}_positive_ratio_{window}"] = (
                sand.gt(0).astype(float).rolling(window=window, min_periods=1).mean()
            )
            derived[f"{sand_column}_mean_{window}"] = sand_roll.mean()
            derived[f"{sand_column}_std_{window}"] = sand_roll.std().fillna(0.0)

        derived_frame = pd.DataFrame(derived, index=updated.index)
        updated_frames[segment_id] = pd.concat([updated, derived_frame], axis=1)
    return updated_frames


def select_feature_columns(
    segment_frames: dict[str, pd.DataFrame],
    label_column: str,
    segment_column: str | None,
    time_column: str | None,
    include_columns: Iterable[str] | None,
    exclude_columns: Iterable[str] | None,
) -> list[str]:
    sample = next(iter(segment_frames.values()))
    if include_columns:
        selected = [column for column in include_columns if column in sample.columns]
        if not selected:
            raise ValueError("No requested include columns were found in the dataset.")
        return selected

    excluded_exact = {label_column}
    excluded_exact.update(DEFAULT_EXCLUDE_COLUMNS)
    if segment_column:
        excluded_exact.add(segment_column)
    if time_column:
        excluded_exact.add(time_column)
    if exclude_columns:
        excluded_exact.update(exclude_columns)

    features: list[str] = []
    for column in sample.columns:
        if column in excluded_exact:
            continue
        lowered = column.lower()
        if any(token in lowered for token in DEFAULT_EXCLUDE_SUBSTRINGS):
            continue
        include = False
        for frame in segment_frames.values():
            series = frame[column]
            if pd.api.types.is_numeric_dtype(series):
                include = True
                break
            numeric_series = pd.to_numeric(series, errors="coerce")
            if numeric_series.notna().sum() > 0:
                include = True
                break
        if include:
            features.append(column)
    if not features:
        raise ValueError("No numeric feature columns were selected for training.")
    return features


def coerce_numeric_frame(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    numeric = frame[feature_columns].copy()
    for column in feature_columns:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    return numeric


def compute_split_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    raw = {name: total * ratio for name, ratio in ratios.items()}
    counts = {name: int(np.floor(value)) for name, value in raw.items()}
    remainder = total - sum(counts.values())
    order = sorted(ratios.keys(), key=lambda name: (raw[name] - counts[name]), reverse=True)
    for name in order[:remainder]:
        counts[name] += 1
    return counts


def split_segments(
    segment_ids: list[str], seed: int, ratios: dict[str, float]
) -> dict[str, list[str]]:
    shuffled = list(segment_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(shuffled)
    counts = compute_split_counts(len(shuffled), ratios)

    splits: dict[str, list[str]] = {}
    cursor = 0
    for split_name in ("train", "test", "val", "transfer"):
        count = counts[split_name]
        splits[split_name] = shuffled[cursor : cursor + count]
        cursor += count
    return splits


def segment_contains_any_label(
    frame: pd.DataFrame, label_column: str, required_labels: Iterable[str]
) -> bool:
    labels = set(frame[label_column].astype(str).str.strip().tolist())
    return any(str(label).strip() in labels for label in required_labels)


def split_segments_with_required_test_labels(
    segment_frames: dict[str, pd.DataFrame],
    seed: int,
    ratios: dict[str, float],
    label_column: str,
    required_test_labels: Iterable[str] | None,
) -> dict[str, list[str]]:
    splits = split_segments(sorted(segment_frames.keys()), seed, ratios)
    required = [str(label).strip() for label in (required_test_labels or []) if str(label).strip()]
    if not required:
        return splits

    def split_of(segment_id: str) -> str | None:
        for split_name, ids in splits.items():
            if segment_id in ids:
                return split_name
        return None

    test_ids = set(splits["test"])
    for required_label in required:
        if any(
            segment_contains_any_label(segment_frames[segment_id], label_column, [required_label])
            for segment_id in test_ids
        ):
            continue

        candidates = [
            segment_id
            for segment_id, frame in segment_frames.items()
            if segment_contains_any_label(frame, label_column, [required_label])
        ]
        if not candidates:
            continue

        # Prefer moving the smallest matching segment to reduce distribution disturbance.
        candidates.sort(key=lambda segment_id: len(segment_frames[segment_id]))
        chosen = next((segment_id for segment_id in candidates if segment_id not in test_ids), candidates[0])
        source_split = split_of(chosen)
        if source_split is None or source_split == "test":
            continue

        splits[source_split].remove(chosen)
        splits["test"].append(chosen)
        test_ids.add(chosen)

        # Keep test segment count stable when possible by swapping out a non-required segment.
        swap_out = None
        for segment_id in list(splits["test"]):
            if segment_id == chosen:
                continue
            if not segment_contains_any_label(segment_frames[segment_id], label_column, required):
                swap_out = segment_id
                break
        if swap_out is not None:
            splits["test"].remove(swap_out)
            test_ids.remove(swap_out)
            splits[source_split].append(swap_out)

    return splits


def fit_preprocessors(
    segment_frames: dict[str, pd.DataFrame],
    train_segment_ids: list[str],
    feature_columns: list[str],
) -> tuple[SimpleImputer, StandardScaler]:
    train_matrix = pd.concat(
        [coerce_numeric_frame(segment_frames[segment_id], feature_columns) for segment_id in train_segment_ids],
        axis=0,
        ignore_index=True,
    )
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    transformed = imputer.fit_transform(train_matrix)
    scaler.fit(transformed)
    return imputer, scaler


def fit_label_encoder(
    segment_frames: dict[str, pd.DataFrame], label_column: str
) -> LabelEncoder:
    labels = pd.concat(
        [frame[label_column] for frame in segment_frames.values()],
        axis=0,
        ignore_index=True,
    ).astype(str)
    encoder = LabelEncoder()
    encoder.fit(labels)
    return encoder


def build_edge_index(window_size: int) -> torch.Tensor:
    if window_size <= 1:
        return torch.empty((2, 0), dtype=torch.long)
    edges: list[list[int]] = []
    for node in range(window_size - 1):
        edges.append([node, node + 1])
        edges.append([node + 1, node])
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def build_graphs_for_segments(
    segment_frames: dict[str, pd.DataFrame],
    segment_ids: list[str],
    feature_columns: list[str],
    label_column: str,
    label_encoder: LabelEncoder,
    imputer: SimpleImputer,
    scaler: StandardScaler,
    window_size: int,
) -> list[Data]:
    graphs: list[Data] = []
    edge_index = build_edge_index(window_size)
    for segment_id in segment_ids:
        frame = segment_frames[segment_id]
        if len(frame) < window_size:
            continue
        numeric_frame = coerce_numeric_frame(frame, feature_columns)
        feature_matrix = scaler.transform(imputer.transform(numeric_frame))
        labels = label_encoder.transform(frame[label_column].astype(str))
        for start in range(0, len(frame) - window_size + 1):
            end = start + window_size
            window_x = torch.tensor(feature_matrix[start:end], dtype=torch.float32)
            target = torch.tensor(labels[end - 1], dtype=torch.long)
            graphs.append(
                Data(
                    x=window_x,
                    edge_index=edge_index,
                    y=target,
                    segment_id=segment_id,
                    window_start=start,
                    window_end=end - 1,
                )
            )
    return graphs


def prepare_datasets(
    data_path: str,
    label_column: str,
    segment_column: str | None,
    time_column: str | None,
    include_columns: list[str] | None,
    exclude_columns: list[str] | None,
    reference_header_path: str | None,
    exclude_name_patterns: list[str] | None,
    normal_label: str,
    binary_normal_abnormal: bool,
    abnormal_label: str,
    seed: int,
    window_size: int,
    train_ratio: float,
    test_ratio: float,
    val_ratio: float,
    transfer_ratio: float,
    trim_before_sand: bool = False,
    sand_column: str = "SB",
    sand_threshold: float = 0.0,
    add_dynamic_features: bool = False,
    dynamic_feature_columns: list[str] | None = None,
    rolling_windows: list[int] | None = None,
    drop_pure_normal_segments: bool = False,
    required_test_labels: list[str] | None = None,
) -> PreparedData:
    ratios = {
        "train": train_ratio,
        "test": test_ratio,
        "val": val_ratio,
        "transfer": transfer_ratio,
    }
    total_ratio = sum(ratios.values())
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

    segment_frames = discover_segment_frames(
        data_path,
        segment_column,
        label_column,
        reference_header_path,
        exclude_name_patterns,
    )
    segment_frames = sort_segment_frames(segment_frames, time_column)
    segment_frames = normalize_labels_in_frames(segment_frames, label_column, normal_label)
    if drop_pure_normal_segments:
        segment_frames = drop_single_label_segments(segment_frames, label_column, normal_label)
        if not segment_frames:
            raise ValueError("No segments remained after dropping pure-normal segments.")
    if trim_before_sand:
        segment_frames = trim_segments_from_first_sand(
            segment_frames, sand_column=sand_column, sand_threshold=sand_threshold
        )
        if not segment_frames:
            raise ValueError("No segments remained after trimming from first sand point.")
    if binary_normal_abnormal:
        segment_frames = binarize_labels_in_frames(
            segment_frames, label_column, normal_label, abnormal_label
        )
    if add_dynamic_features:
        segment_frames = add_dynamic_features_to_frames(
            segment_frames,
            base_columns=dynamic_feature_columns or list(DEFAULT_DYNAMIC_BASE_COLUMNS),
            sand_column=sand_column,
            rolling_windows=rolling_windows or [3, 5, 10],
        )
    feature_columns = select_feature_columns(
        segment_frames,
        label_column,
        segment_column,
        time_column,
        include_columns,
        exclude_columns,
    )
    split_manifest = split_segments_with_required_test_labels(
        segment_frames,
        seed,
        ratios,
        label_column,
        required_test_labels,
    )
    if not split_manifest["train"]:
        raise ValueError("No training segments were allocated. Increase available segments.")

    imputer, scaler = fit_preprocessors(segment_frames, split_manifest["train"], feature_columns)
    label_encoder = fit_label_encoder(segment_frames, label_column)
    base = BasePreparedData(
        segment_frames=segment_frames,
        feature_columns=feature_columns,
        classes=label_encoder.classes_.tolist(),
        split_manifest=split_manifest,
        segment_sizes={segment_id: len(frame) for segment_id, frame in segment_frames.items()},
        label_column=label_column,
        label_encoder=label_encoder,
        imputer=imputer,
        scaler=scaler,
    )
    return build_prepared_for_window(base, window_size)


def prepare_base_datasets(
    data_path: str,
    label_column: str,
    segment_column: str | None,
    time_column: str | None,
    include_columns: list[str] | None,
    exclude_columns: list[str] | None,
    reference_header_path: str | None,
    exclude_name_patterns: list[str] | None,
    normal_label: str,
    binary_normal_abnormal: bool,
    abnormal_label: str,
    seed: int,
    train_ratio: float,
    test_ratio: float,
    val_ratio: float,
    transfer_ratio: float,
    trim_before_sand: bool = False,
    sand_column: str = "SB",
    sand_threshold: float = 0.0,
    add_dynamic_features: bool = False,
    dynamic_feature_columns: list[str] | None = None,
    rolling_windows: list[int] | None = None,
    drop_pure_normal_segments: bool = False,
    required_test_labels: list[str] | None = None,
) -> BasePreparedData:
    ratios = {
        "train": train_ratio,
        "test": test_ratio,
        "val": val_ratio,
        "transfer": transfer_ratio,
    }
    total_ratio = sum(ratios.values())
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

    segment_frames = discover_segment_frames(
        data_path,
        segment_column,
        label_column,
        reference_header_path,
        exclude_name_patterns,
    )
    segment_frames = sort_segment_frames(segment_frames, time_column)
    segment_frames = normalize_labels_in_frames(segment_frames, label_column, normal_label)
    if drop_pure_normal_segments:
        segment_frames = drop_single_label_segments(segment_frames, label_column, normal_label)
        if not segment_frames:
            raise ValueError("No segments remained after dropping pure-normal segments.")
    if trim_before_sand:
        segment_frames = trim_segments_from_first_sand(
            segment_frames, sand_column=sand_column, sand_threshold=sand_threshold
        )
        if not segment_frames:
            raise ValueError("No segments remained after trimming from first sand point.")
    if binary_normal_abnormal:
        segment_frames = binarize_labels_in_frames(
            segment_frames, label_column, normal_label, abnormal_label
        )
    if add_dynamic_features:
        segment_frames = add_dynamic_features_to_frames(
            segment_frames,
            base_columns=dynamic_feature_columns or list(DEFAULT_DYNAMIC_BASE_COLUMNS),
            sand_column=sand_column,
            rolling_windows=rolling_windows or [3, 5, 10],
        )
    feature_columns = select_feature_columns(
        segment_frames,
        label_column,
        segment_column,
        time_column,
        include_columns,
        exclude_columns,
    )
    split_manifest = split_segments_with_required_test_labels(
        segment_frames,
        seed,
        ratios,
        label_column,
        required_test_labels,
    )
    if not split_manifest["train"]:
        raise ValueError("No training segments were allocated. Increase available segments.")
    imputer, scaler = fit_preprocessors(segment_frames, split_manifest["train"], feature_columns)
    label_encoder = fit_label_encoder(segment_frames, label_column)
    return BasePreparedData(
        segment_frames=segment_frames,
        feature_columns=feature_columns,
        classes=label_encoder.classes_.tolist(),
        split_manifest=split_manifest,
        segment_sizes={segment_id: len(frame) for segment_id, frame in segment_frames.items()},
        label_column=label_column,
        label_encoder=label_encoder,
        imputer=imputer,
        scaler=scaler,
    )


def build_prepared_for_window(base: BasePreparedData, window_size: int) -> PreparedData:
    return PreparedData(
        train_graphs=build_graphs_for_segments(
            base.segment_frames,
            base.split_manifest["train"],
            base.feature_columns,
            base.label_column,
            base.label_encoder,
            base.imputer,
            base.scaler,
            window_size,
        ),
        test_graphs=build_graphs_for_segments(
            base.segment_frames,
            base.split_manifest["test"],
            base.feature_columns,
            base.label_column,
            base.label_encoder,
            base.imputer,
            base.scaler,
            window_size,
        ),
        val_graphs=build_graphs_for_segments(
            base.segment_frames,
            base.split_manifest["val"],
            base.feature_columns,
            base.label_column,
            base.label_encoder,
            base.imputer,
            base.scaler,
            window_size,
        ),
        transfer_graphs=build_graphs_for_segments(
            base.segment_frames,
            base.split_manifest["transfer"],
            base.feature_columns,
            base.label_column,
            base.label_encoder,
            base.imputer,
            base.scaler,
            window_size,
        ),
        feature_columns=base.feature_columns,
        classes=base.classes,
        split_manifest=base.split_manifest,
        segment_sizes=base.segment_sizes,
    )


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
