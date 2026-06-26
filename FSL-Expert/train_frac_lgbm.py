from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from frac_gnn.data import build_prepared_for_window, prepare_base_datasets, save_json


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LightGBM baseline for frac data.")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--label-column", default="WORKING_TYPE")
    parser.add_argument("--segment-column", default=None)
    parser.add_argument("--time-column", default=None)
    parser.add_argument("--include-columns", nargs="*", default=None)
    parser.add_argument("--exclude-columns", nargs="*", default=None)
    parser.add_argument("--reference-header-path", default=None)
    parser.add_argument("--exclude-name-patterns", nargs="*", default=None)
    parser.add_argument("--normal-label", default="正常")
    parser.add_argument("--binary-normal-abnormal", action="store_true")
    parser.add_argument("--abnormal-label", default="异常")
    parser.add_argument("--drop-pure-normal-segments", action="store_true")
    parser.add_argument("--trim-before-sand", action="store_true")
    parser.add_argument("--sand-column", default="SB")
    parser.add_argument("--sand-threshold", type=float, default=0.0)
    parser.add_argument("--add-dynamic-features", action="store_true")
    parser.add_argument("--dynamic-feature-columns", nargs="*", default=None)
    parser.add_argument("--rolling-windows", nargs="*", type=int, default=[3, 5, 10])
    parser.add_argument("--window-sizes", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6])
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--downsample-normal-ratio", type=float, default=1.0)
    parser.add_argument("--class-weight-power", type=float, default=1.0)
    parser.add_argument("--class-weight-max-ratio", type=float, default=20.0)
    parser.add_argument("--disable-class-weights", action="store_true")
    parser.add_argument("--run-dir", default="runs/frac_lgbm")
    return parser


def flatten_graphs(graphs) -> tuple[np.ndarray, np.ndarray]:
    x = np.stack([graph.x.numpy().reshape(-1) for graph in graphs], axis=0)
    y = np.asarray([int(graph.y) for graph in graphs], dtype=np.int64)
    return x, y


def rebalance_training_data(
    x: np.ndarray,
    y: np.ndarray,
    downsample_normal_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if downsample_normal_ratio >= 1.0:
        return x, y
    unique_classes, counts = np.unique(y, return_counts=True)
    if len(unique_classes) != 2:
        return x, y
    minority_class = unique_classes[np.argmin(counts)]
    majority_class = unique_classes[np.argmax(counts)]
    minority_idx = np.flatnonzero(y == minority_class)
    majority_idx = np.flatnonzero(y == majority_class)
    keep_majority = max(len(minority_idx), int(len(majority_idx) * downsample_normal_ratio))
    keep_majority = min(keep_majority, len(majority_idx))
    rng = np.random.default_rng(seed)
    chosen_majority = rng.choice(majority_idx, size=keep_majority, replace=False)
    selected = np.concatenate([minority_idx, chosen_majority])
    rng.shuffle(selected)
    return x[selected], y[selected]


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


def evaluate(model, x: np.ndarray, y: np.ndarray) -> dict:
    pred = model.predict(x)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "sample_count": int(len(y)),
        "report": classification_report(y, pred, output_dict=True, zero_division=0),
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    run_root = Path(args.run_dir) / time.strftime("%Y%m%d_%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)
    save_json(run_root / "config.json", vars(args))

    base_prepared = prepare_base_datasets(
        data_path=args.data_path,
        label_column=args.label_column,
        segment_column=args.segment_column,
        time_column=args.time_column,
        include_columns=args.include_columns,
        exclude_columns=args.exclude_columns,
        reference_header_path=args.reference_header_path,
        exclude_name_patterns=args.exclude_name_patterns,
        normal_label=args.normal_label,
        binary_normal_abnormal=args.binary_normal_abnormal,
        abnormal_label=args.abnormal_label,
        seed=args.seed,
        train_ratio=args.train_ratio,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        transfer_ratio=args.transfer_ratio,
        drop_pure_normal_segments=args.drop_pure_normal_segments,
        trim_before_sand=args.trim_before_sand,
        sand_column=args.sand_column,
        sand_threshold=args.sand_threshold,
        add_dynamic_features=args.add_dynamic_features,
        dynamic_feature_columns=args.dynamic_feature_columns,
        rolling_windows=args.rolling_windows,
    )

    summary = {}
    for window_size in args.window_sizes:
        prepared = build_prepared_for_window(base_prepared, window_size)
        x_train, y_train = flatten_graphs(prepared.train_graphs)
        x_val, y_val = flatten_graphs(prepared.val_graphs)
        x_test, y_test = flatten_graphs(prepared.test_graphs)
        x_transfer, y_transfer = flatten_graphs(prepared.transfer_graphs)
        x_train_fit, y_train_fit = rebalance_training_data(
            x_train,
            y_train,
            downsample_normal_ratio=args.downsample_normal_ratio,
            seed=args.seed + window_size,
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
            objective="binary" if len(prepared.classes) == 2 else "multiclass",
            num_class=None if len(prepared.classes) == 2 else len(prepared.classes),
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

        result = {
            "window_size": window_size,
            "classes": prepared.classes,
            "feature_columns": prepared.feature_columns,
            "train_sample_count_before_rebalance": int(len(y_train)),
            "train_sample_count_after_rebalance": int(len(y_train_fit)),
            "train_class_counts_after_rebalance": class_counts,
            "train_class_weights_after_rebalance": class_weights,
            "metrics": {
                "train": evaluate(model, x_train, y_train),
                "val": evaluate(model, x_val, y_val),
                "test": evaluate(model, x_test, y_test),
                "transfer": evaluate(model, x_transfer, y_transfer),
            },
        }
        window_dir = run_root / f"window_{window_size}"
        window_dir.mkdir(parents=True, exist_ok=True)
        save_json(window_dir / "metrics.json", result)
        summary[str(window_size)] = {
            "val_macro_f1": result["metrics"]["val"]["macro_f1"],
            "test_macro_f1": result["metrics"]["test"]["macro_f1"],
            "transfer_macro_f1": result["metrics"]["transfer"]["macro_f1"],
        }

    save_json(run_root / "summary.json", {"run_root": str(run_root), "window_results": summary})
    print(json.dumps({"run_root": str(run_root), "window_results": summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
