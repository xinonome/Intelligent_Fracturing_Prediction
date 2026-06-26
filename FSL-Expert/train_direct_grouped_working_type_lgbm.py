from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from frac_gnn.data import build_prepared_for_window, prepare_base_datasets, save_json


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Direct grouped WORKING_TYPE multiclass model.")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--label-column", default="WORKING_TYPE")
    parser.add_argument("--segment-column", default="FDBH")
    parser.add_argument("--time-column", default="SGSJ")
    parser.add_argument("--reference-header-path", default=None)
    parser.add_argument("--exclude-name-patterns", nargs="*", default=["WITHfiltered"])
    parser.add_argument("--normal-label", default="正常")
    parser.add_argument("--major-abnormal-labels", nargs="+", default=["砂堵", "缝口暂堵", "缝内暂堵"])
    parser.add_argument("--other-abnormal-label", default="其他异常")
    parser.add_argument("--trim-before-sand", action="store_true")
    parser.add_argument("--sand-column", default="SB")
    parser.add_argument("--sand-threshold", type=float, default=0.0)
    parser.add_argument("--add-dynamic-features", action="store_true")
    parser.add_argument("--dynamic-feature-columns", nargs="*", default=None)
    parser.add_argument("--rolling-windows", nargs="*", type=int, default=[3, 5, 10])
    parser.add_argument("--window-sizes", nargs="+", type=int, default=[4])
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--transfer-ratio", type=float, default=0.1)
    parser.add_argument(
        "--require-test-labels",
        nargs="*",
        default=None,
        help="Force test split to contain at least one segment with each requested original label.",
    )
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--min-child-samples", type=int, default=20)
    parser.add_argument("--class-weight-power", type=float, default=1.0)
    parser.add_argument("--class-weight-max-ratio", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-dir", default="FSL-Expert/runs/direct_grouped_working_type_lgbm")
    return parser


def flatten_graphs(graphs) -> tuple[np.ndarray, np.ndarray]:
    x = np.stack([graph.x.numpy().reshape(-1) for graph in graphs], axis=0)
    y = np.asarray([int(graph.y) for graph in graphs], dtype=np.int64)
    return x, y


def build_weights(y: np.ndarray, power: float, max_ratio: float) -> np.ndarray:
    classes, counts = np.unique(y, return_counts=True)
    max_count = float(counts.max())
    raw = {int(cls): min((max_count / float(cnt)) ** power, max_ratio) for cls, cnt in zip(classes, counts)}
    min_weight = min(raw.values())
    normalized = {cls: weight / min_weight for cls, weight in raw.items()}
    return np.asarray([normalized[int(label)] for label in y], dtype=np.float32)


def make_lgbm(args, num_class: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="multiclass",
        num_class=num_class,
        num_leaves=args.num_leaves,
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_samples=args.min_child_samples,
        random_state=args.seed,
        verbosity=-1,
    )


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict:
    present_labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    target_names = [labels[index] for index in present_labels]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "sample_count": int(len(y_true)),
        "report": classification_report(
            y_true,
            y_pred,
            labels=present_labels,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        ),
    }


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], out_path: Path, title: str) -> None:
    used = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    matrix = confusion_matrix(y_true, y_pred, labels=used)
    names = [labels[index] for index in used]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(7, len(names) * 0.9), max(5, len(names) * 0.75)))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def build_group_mapping(labels: list[str], normal_label: str, major_labels: list[str], other_label: str) -> tuple[list[str], dict[int, int]]:
    grouped_labels = [normal_label] + [label for label in major_labels if label in labels and label != normal_label] + [other_label]
    grouped_labels = list(dict.fromkeys(grouped_labels))
    mapping = {}
    for index, label in enumerate(labels):
        if label == normal_label:
            grouped = normal_label
        elif label in grouped_labels:
            grouped = label
        else:
            grouped = other_label
        mapping[index] = grouped_labels.index(grouped)
    return grouped_labels, mapping


def map_original_to_grouped(y: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    return np.asarray([mapping[int(label)] for label in y], dtype=np.int64)


def evaluate_direct(model, x: np.ndarray, y_original: np.ndarray, y_grouped: np.ndarray, grouped_labels: list[str]) -> dict:
    pred = model.predict(x).astype(np.int64)
    return {
        "direct_grouped": metrics_dict(y_grouped, pred, grouped_labels),
        "predictions": {
            "y_original": y_original.tolist(),
            "y_grouped_true": y_grouped.tolist(),
            "direct_grouped_pred": pred.tolist(),
        },
    }


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
        include_columns=None,
        exclude_columns=None,
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
        drop_pure_normal_segments=False,
        trim_before_sand=args.trim_before_sand,
        sand_column=args.sand_column,
        sand_threshold=args.sand_threshold,
        add_dynamic_features=args.add_dynamic_features,
        dynamic_feature_columns=args.dynamic_feature_columns,
        rolling_windows=args.rolling_windows,
        required_test_labels=args.require_test_labels,
    )
    labels = base.classes
    if args.normal_label not in labels:
        raise ValueError(f"normal label `{args.normal_label}` not found in labels: {labels}")
    grouped_labels, original_to_grouped = build_group_mapping(
        labels,
        normal_label=args.normal_label,
        major_labels=args.major_abnormal_labels,
        other_label=args.other_abnormal_label,
    )

    summary = {}
    for window_size in args.window_sizes:
        prepared = build_prepared_for_window(base, window_size)
        x_train, y_train_original = flatten_graphs(prepared.train_graphs)
        x_val, y_val_original = flatten_graphs(prepared.val_graphs)
        x_test, y_test_original = flatten_graphs(prepared.test_graphs)
        x_transfer, y_transfer_original = flatten_graphs(prepared.transfer_graphs)

        y_train = map_original_to_grouped(y_train_original, original_to_grouped)
        y_val = map_original_to_grouped(y_val_original, original_to_grouped)
        y_test = map_original_to_grouped(y_test_original, original_to_grouped)
        y_transfer = map_original_to_grouped(y_transfer_original, original_to_grouped)

        model = make_lgbm(args, num_class=len(grouped_labels))
        model.fit(x_train, y_train, sample_weight=build_weights(y_train, args.class_weight_power, args.class_weight_max_ratio))

        result = {
            "window_size": window_size,
            "original_classes": labels,
            "grouped_classes": grouped_labels,
            "normal_label": args.normal_label,
            "major_abnormal_labels": args.major_abnormal_labels,
            "other_abnormal_label": args.other_abnormal_label,
            "original_to_grouped": {labels[k]: grouped_labels[v] for k, v in original_to_grouped.items()},
            "train_original_label_counts": {labels[int(k)]: int(v) for k, v in zip(*np.unique(y_train_original, return_counts=True))},
            "train_grouped_label_counts": {grouped_labels[int(k)]: int(v) for k, v in zip(*np.unique(y_train, return_counts=True))},
            "split_manifest": base.split_manifest,
            "metrics": {
                "val": evaluate_direct(model, x_val, y_val_original, y_val, grouped_labels),
                "test": evaluate_direct(model, x_test, y_test_original, y_test, grouped_labels),
                "transfer": evaluate_direct(model, x_transfer, y_transfer_original, y_transfer, grouped_labels),
            },
        }
        window_dir = run_root / f"window_{window_size}"
        window_dir.mkdir(parents=True, exist_ok=True)
        for split_name in ["val", "test", "transfer"]:
            pred = result["metrics"][split_name]["predictions"]
            plot_confusion(
                np.asarray(pred["y_grouped_true"], dtype=int),
                np.asarray(pred["direct_grouped_pred"], dtype=int),
                grouped_labels,
                window_dir / f"{split_name}_direct_grouped_confusion.png",
                f"{split_name} direct grouped WORKING_TYPE confusion",
            )
            pred_frame = pd.DataFrame(pred)
            pred_frame["y_original_label"] = pred_frame["y_original"].map(lambda value: labels[int(value)])
            pred_frame["y_grouped_true_label"] = pred_frame["y_grouped_true"].map(lambda value: grouped_labels[int(value)])
            pred_frame["direct_grouped_pred_label"] = pred_frame["direct_grouped_pred"].map(lambda value: grouped_labels[int(value)])
            pred_frame.to_csv(window_dir / f"{split_name}_direct_grouped_predictions.csv", index=False, encoding="utf-8-sig")
        save_json(window_dir / "metrics.json", result)
        summary[str(window_size)] = {
            "test_direct_grouped_macro_f1": result["metrics"]["test"]["direct_grouped"]["macro_f1"],
            "test_direct_grouped_accuracy": result["metrics"]["test"]["direct_grouped"]["accuracy"],
            "train_grouped_label_counts": result["train_grouped_label_counts"],
        }

    payload = {"run_root": str(run_root), "window_results": summary}
    save_json(run_root / "summary.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
