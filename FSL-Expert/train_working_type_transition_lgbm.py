from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from frac_gnn.data import prepare_base_datasets, save_json


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train strict next-WORKING_TYPE transition probability model."
    )
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--label-column", default="WORKING_TYPE")
    parser.add_argument("--segment-column", default="FDBH")
    parser.add_argument("--time-column", default="SGSJ")
    parser.add_argument("--include-columns", nargs="*", default=None)
    parser.add_argument("--exclude-columns", nargs="*", default=None)
    parser.add_argument("--reference-header-path", default=None)
    parser.add_argument("--exclude-name-patterns", nargs="*", default=None)
    parser.add_argument("--normal-label", default="正常")
    parser.add_argument("--trim-before-sand", action="store_true")
    parser.add_argument("--sand-column", default="SB")
    parser.add_argument("--sand-threshold", type=float, default=0.0)
    parser.add_argument("--add-dynamic-features", action="store_true")
    parser.add_argument("--dynamic-feature-columns", nargs="*", default=None)
    parser.add_argument("--rolling-windows", nargs="*", type=int, default=[3, 5, 10])
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--transfer-ratio", type=float, default=0.1)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--min-child-samples", type=int, default=20)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--class-weight-max-ratio", type=float, default=10.0)
    parser.add_argument("--disable-class-weights", action="store_true")
    parser.add_argument("--oversample-minority", action="store_true")
    parser.add_argument("--oversample-target-ratio", type=float, default=0.2)
    parser.add_argument("--augment-minority", action="store_true")
    parser.add_argument("--augment-target-ratio", type=float, default=0.2)
    parser.add_argument("--augment-noise-std", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-dir", default="runs/working_type_transition_lgbm")
    return parser


def build_class_weight_summary(
    y: np.ndarray,
    power: float,
    max_ratio: float,
) -> tuple[np.ndarray, dict[int, int], dict[int, float]]:
    classes, counts = np.unique(y, return_counts=True)
    count_map = {int(cls): int(cnt) for cls, cnt in zip(classes, counts)}
    max_count = float(counts.max())
    raw_weights = {}
    for cls, cnt in zip(classes, counts):
        ratio = max_count / float(cnt)
        raw_weights[int(cls)] = min(ratio**power, max_ratio)
    min_weight = min(raw_weights.values())
    normalized = {cls: weight / min_weight for cls, weight in raw_weights.items()}
    sample_weight = np.asarray([normalized[int(label)] for label in y], dtype=np.float32)
    return sample_weight, count_map, normalized


def class_count_map(y: np.ndarray) -> dict[int, int]:
    classes, counts = np.unique(y, return_counts=True)
    return {int(cls): int(cnt) for cls, cnt in zip(classes, counts)}


def augment_minority_samples(
    x: np.ndarray,
    y: np.ndarray,
    numeric_feature_count: int,
    target_ratio: float,
    noise_std: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    if len(y) == 0 or target_ratio <= 0 or noise_std <= 0:
        return x, y, {}

    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    max_count = int(counts.max())
    target_count = max(1, int(round(max_count * target_ratio)))
    added_by_class: dict[int, int] = {}
    synthetic_x = []
    synthetic_y = []

    for cls, count in zip(classes, counts):
        count = int(count)
        if count >= target_count:
            continue
        cls_indices = np.flatnonzero(y == cls)
        add_count = target_count - count
        chosen = rng.choice(cls_indices, size=add_count, replace=True)
        noise = rng.normal(0.0, noise_std, size=x[chosen].shape).astype(x.dtype, copy=False)
        noise[:, numeric_feature_count:] = 0.0
        synthetic_x.append(x[chosen] + noise)
        synthetic_y.append(np.full(add_count, cls, dtype=y.dtype))
        added_by_class[int(cls)] = int(add_count)

    if not synthetic_x:
        return x, y, {}

    return (
        np.concatenate([x, *synthetic_x], axis=0),
        np.concatenate([y, *synthetic_y], axis=0),
        added_by_class,
    )


def oversample_minority_samples(
    x: np.ndarray,
    y: np.ndarray,
    target_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    if len(y) == 0 or target_ratio <= 0:
        return x, y, {}

    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    max_count = int(counts.max())
    target_count = max(1, int(round(max_count * target_ratio)))
    added_by_class: dict[int, int] = {}
    extra_indices = []

    for cls, count in zip(classes, counts):
        count = int(count)
        if count >= target_count:
            continue
        cls_indices = np.flatnonzero(y == cls)
        add_count = target_count - count
        extra_indices.append(rng.choice(cls_indices, size=add_count, replace=True))
        added_by_class[int(cls)] = int(add_count)

    if not extra_indices:
        return x, y, {}

    selected = np.concatenate([np.arange(len(y)), *extra_indices])
    rng.shuffle(selected)
    return x[selected], y[selected], added_by_class


def make_transition_samples(base, segment_ids: list[str], window_size: int) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    rows: list[np.ndarray] = []
    targets: list[int] = []
    meta_rows: list[dict[str, object]] = []
    num_classes = len(base.classes)

    for segment_id in segment_ids:
        frame = base.segment_frames[segment_id]
        if len(frame) <= window_size:
            continue
        numeric_frame = frame[base.feature_columns].copy()
        for column in base.feature_columns:
            numeric_frame[column] = pd.to_numeric(numeric_frame[column], errors="coerce")
        feature_matrix = base.scaler.transform(base.imputer.transform(numeric_frame))
        encoded_labels = base.label_encoder.transform(frame[base.label_column].astype(str))

        for start in range(0, len(frame) - window_size):
            current_index = start + window_size - 1
            next_index = current_index + 1
            window_features = feature_matrix[start : start + window_size].reshape(-1)
            current_label_index = int(encoded_labels[current_index])
            next_label_index = int(encoded_labels[next_index])
            current_label_onehot = np.zeros(num_classes, dtype=np.float32)
            current_label_onehot[current_label_index] = 1.0
            rows.append(np.concatenate([window_features, current_label_onehot]))
            targets.append(next_label_index)
            meta_rows.append(
                {
                    "segment_id": segment_id,
                    "window_start": start,
                    "current_index": current_index,
                    "next_index": next_index,
                    "current_label": base.classes[current_label_index],
                    "next_label": base.classes[next_label_index],
                }
            )

    if not rows:
        return np.empty((0, 0)), np.empty((0,), dtype=np.int64), pd.DataFrame(meta_rows)
    return np.stack(rows, axis=0), np.asarray(targets, dtype=np.int64), pd.DataFrame(meta_rows)


def evaluate(model, x: np.ndarray, y: np.ndarray) -> dict:
    if len(y) == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "sample_count": 0, "report": {}}
    pred = model.predict(x)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "sample_count": int(len(y)),
        "report": classification_report(y, pred, output_dict=True, zero_division=0),
    }


def probability_frame(model, x: np.ndarray, meta: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    probs = model.predict_proba(x)
    if probs.shape[1] != len(classes):
        full = np.zeros((len(x), len(classes)), dtype=float)
        for output_index, class_index in enumerate(model.classes_):
            full[:, int(class_index)] = probs[:, output_index]
        probs = full
    prob_df = pd.DataFrame(probs, columns=[f"prob_next_{label}" for label in classes])
    return pd.concat([meta.reset_index(drop=True), prob_df], axis=1)


def summarize_predicted_transitions(prob_df: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    prob_cols = [f"prob_next_{label}" for label in classes]
    grouped = prob_df.groupby("current_label")[prob_cols].mean()
    grouped.columns = [column.replace("prob_next_", "") for column in grouped.columns]
    return grouped


def main() -> None:
    args = build_arg_parser().parse_args()
    run_root = Path(args.run_dir) / time.strftime("%Y%m%d_%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)
    save_json(run_root / "config.json", vars(args))

    base = prepare_base_datasets(
        data_path=args.data_path,
        label_column=args.label_column,
        segment_column=args.segment_column,
        time_column=args.time_column,
        include_columns=args.include_columns,
        exclude_columns=args.exclude_columns,
        reference_header_path=args.reference_header_path,
        exclude_name_patterns=args.exclude_name_patterns,
        normal_label=args.normal_label,
        binary_normal_abnormal=False,
        abnormal_label="异常",
        seed=args.seed,
        train_ratio=args.train_ratio,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        transfer_ratio=args.transfer_ratio,
        trim_before_sand=args.trim_before_sand,
        sand_column=args.sand_column,
        sand_threshold=args.sand_threshold,
        add_dynamic_features=args.add_dynamic_features,
        dynamic_feature_columns=args.dynamic_feature_columns,
        rolling_windows=args.rolling_windows,
    )

    datasets = {}
    for split_name in ("train", "test", "val", "transfer"):
        x, y, meta = make_transition_samples(base, base.split_manifest[split_name], args.window_size)
        datasets[split_name] = {"x": x, "y": y, "meta": meta}

    x_train_fit = datasets["train"]["x"]
    y_train_fit = datasets["train"]["y"]
    train_class_counts_before = class_count_map(y_train_fit)
    augment_added_by_class = {}
    oversample_added_by_class = {}
    numeric_feature_count = args.window_size * len(base.feature_columns)

    if args.augment_minority:
        x_train_fit, y_train_fit, augment_added_by_class = augment_minority_samples(
            x_train_fit,
            y_train_fit,
            numeric_feature_count=numeric_feature_count,
            target_ratio=args.augment_target_ratio,
            noise_std=args.augment_noise_std,
            seed=args.seed + 17,
        )

    if args.oversample_minority:
        x_train_fit, y_train_fit, oversample_added_by_class = oversample_minority_samples(
            x_train_fit,
            y_train_fit,
            target_ratio=args.oversample_target_ratio,
            seed=args.seed + 29,
        )

    sample_weight = None
    class_counts = {}
    class_weights = {}
    if not args.disable_class_weights:
        sample_weight, class_counts, class_weights = build_class_weight_summary(
            y_train_fit,
            power=args.class_weight_power,
            max_ratio=args.class_weight_max_ratio,
        )

    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(base.classes),
        num_leaves=args.num_leaves,
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_samples=args.min_child_samples,
        random_state=args.seed,
        verbosity=-1,
    )
    model.fit(x_train_fit, y_train_fit, sample_weight=sample_weight)

    metrics = {
        split_name: evaluate(model, payload["x"], payload["y"])
        for split_name, payload in datasets.items()
    }
    result = {
        "classes": base.classes,
        "feature_columns": base.feature_columns,
        "window_size": args.window_size,
        "split_manifest": base.split_manifest,
        "transition_sample_counts": {
            split_name: int(len(payload["y"])) for split_name, payload in datasets.items()
        },
        "train_sample_count_before_resample": int(len(datasets["train"]["y"])),
        "train_sample_count_after_resample": int(len(y_train_fit)),
        "train_class_counts_before_resample": train_class_counts_before,
        "train_class_counts_after_resample": class_count_map(y_train_fit),
        "augment_added_by_class": augment_added_by_class,
        "oversample_added_by_class": oversample_added_by_class,
        "train_class_counts": class_counts,
        "train_class_weights": class_weights,
        "metrics": metrics,
    }
    save_json(run_root / "metrics.json", result)
    with (run_root / "model.pkl").open("wb") as f:
        pickle.dump(model, f)

    for split_name in ("test", "val", "transfer"):
        if len(datasets[split_name]["y"]) == 0:
            continue
        probs = probability_frame(
            model,
            datasets[split_name]["x"],
            datasets[split_name]["meta"],
            base.classes,
        )
        probs.to_csv(run_root / f"{split_name}_next_state_probabilities.csv", index=False, encoding="utf-8-sig")
        transition_probs = summarize_predicted_transitions(probs, base.classes)
        transition_probs.to_csv(
            run_root / f"{split_name}_predicted_transition_probability_matrix.csv",
            encoding="utf-8-sig",
        )

    print(json.dumps({"run_root": str(run_root), "metrics": metrics}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
