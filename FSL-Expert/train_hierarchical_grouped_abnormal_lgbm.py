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
    parser = argparse.ArgumentParser(
        description="Hierarchical WORKING_TYPE model: normal/abnormal + major abnormal classes + other abnormal."
    )
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
    parser.add_argument("--class-weight-power", type=float, default=1.5)
    parser.add_argument("--class-weight-max-ratio", type=float, default=80.0)
    parser.add_argument("--downsample-normal-ratio", type=float, default=0.1)
    parser.add_argument(
        "--abnormal-thresholds",
        nargs="*",
        type=float,
        default=[0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
        help="Probability thresholds for sending a sample to the second-stage abnormal classifier.",
    )
    parser.add_argument(
        "--abnormal-score-multipliers",
        nargs="*",
        type=float,
        default=[0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0],
        help="Multipliers for hierarchical probability fusion abnormal scores.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-dir", default="FSL-Expert/runs/hierarchical_grouped_abnormal_lgbm")
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


def downsample_binary_train(x: np.ndarray, y_binary: np.ndarray, ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if ratio >= 1.0:
        return x, y_binary
    normal_idx = np.flatnonzero(y_binary == 0)
    abnormal_idx = np.flatnonzero(y_binary == 1)
    if len(normal_idx) == 0 or len(abnormal_idx) == 0:
        return x, y_binary
    keep_normal = min(len(normal_idx), max(len(abnormal_idx), int(len(normal_idx) * ratio)))
    rng = np.random.default_rng(seed)
    selected_normal = rng.choice(normal_idx, size=keep_normal, replace=False)
    selected = np.concatenate([selected_normal, abnormal_idx])
    rng.shuffle(selected)
    return x[selected], y_binary[selected]


def make_lgbm(args, objective: str, num_class: int | None = None) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective=objective,
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


def binary_metrics(y_true_binary: np.ndarray, y_pred_binary: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true_binary, y_pred_binary)),
        "macro_f1": float(f1_score(y_true_binary, y_pred_binary, average="macro", zero_division=0)),
        "report": classification_report(
            y_true_binary,
            y_pred_binary,
            labels=[0, 1],
            target_names=["正常", "异常"],
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


def evaluate_grouped_hierarchy(
    binary_model,
    grouped_abnormal_model,
    x: np.ndarray,
    y_original: np.ndarray,
    y_grouped: np.ndarray,
    normal_group_index: int,
    grouped_labels: list[str],
    abnormal_threshold: float = 0.5,
) -> dict:
    y_binary = (y_grouped != normal_group_index).astype(np.int64)
    proba = binary_model.predict_proba(x)
    abnormal_class_position = int(np.where(binary_model.classes_ == 1)[0][0])
    abnormal_proba = proba[:, abnormal_class_position]
    pred_binary = (abnormal_proba >= abnormal_threshold).astype(np.int64)
    final_grouped_pred = np.full_like(y_grouped, fill_value=normal_group_index)
    abnormal_pred_idx = np.flatnonzero(pred_binary == 1)
    if len(abnormal_pred_idx):
        final_grouped_pred[abnormal_pred_idx] = grouped_abnormal_model.predict(x[abnormal_pred_idx]).astype(np.int64)

    true_abnormal_idx = np.flatnonzero(y_binary == 1)
    abnormal_only = {}
    if len(true_abnormal_idx):
        abnormal_only = metrics_dict(
            y_grouped[true_abnormal_idx],
            grouped_abnormal_model.predict(x[true_abnormal_idx]).astype(np.int64),
            grouped_labels,
        )
    return {
        "binary_stage": binary_metrics(y_binary, pred_binary),
        "grouped_abnormal_stage_oracle_true_abnormal_only": abnormal_only,
        "two_stage_grouped_final": metrics_dict(y_grouped, final_grouped_pred, grouped_labels),
        "abnormal_threshold": float(abnormal_threshold),
        "predictions": {
            "y_original": y_original.tolist(),
            "y_grouped_true": y_grouped.tolist(),
            "binary_true": y_binary.tolist(),
            "binary_pred": pred_binary.tolist(),
            "abnormal_proba": abnormal_proba.tolist(),
            "final_grouped_pred": final_grouped_pred.tolist(),
        },
    }


def _binary_abnormal_proba(binary_model, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    proba = binary_model.predict_proba(x)
    abnormal_class_position = int(np.where(binary_model.classes_ == 1)[0][0])
    normal_class_position = int(np.where(binary_model.classes_ == 0)[0][0])
    return proba[:, normal_class_position], proba[:, abnormal_class_position]


def evaluate_probability_fusion(
    binary_model,
    grouped_abnormal_model,
    x: np.ndarray,
    y_original: np.ndarray,
    y_grouped: np.ndarray,
    normal_group_index: int,
    grouped_labels: list[str],
    abnormal_score_multiplier: float = 1.0,
) -> dict:
    y_binary = (y_grouped != normal_group_index).astype(np.int64)
    normal_proba, abnormal_proba = _binary_abnormal_proba(binary_model, x)
    abnormal_detail_proba = grouped_abnormal_model.predict_proba(x)
    scores = np.zeros((len(x), len(grouped_labels)), dtype=np.float64)
    scores[:, normal_group_index] = normal_proba
    for class_position, class_label in enumerate(grouped_abnormal_model.classes_):
        class_index = int(class_label)
        if class_index == normal_group_index or class_index >= len(grouped_labels):
            continue
        scores[:, class_index] = abnormal_score_multiplier * abnormal_proba * abnormal_detail_proba[:, class_position]
    final_grouped_pred = scores.argmax(axis=1).astype(np.int64)
    pred_binary = (final_grouped_pred != normal_group_index).astype(np.int64)
    return {
        "binary_stage_after_fusion": binary_metrics(y_binary, pred_binary),
        "two_stage_probability_fused_final": metrics_dict(y_grouped, final_grouped_pred, grouped_labels),
        "abnormal_score_multiplier": float(abnormal_score_multiplier),
        "predictions": {
            "y_original": y_original.tolist(),
            "y_grouped_true": y_grouped.tolist(),
            "binary_true": y_binary.tolist(),
            "binary_pred_after_fusion": pred_binary.tolist(),
            "normal_proba": normal_proba.tolist(),
            "abnormal_proba": abnormal_proba.tolist(),
            "final_grouped_pred": final_grouped_pred.tolist(),
        },
    }


def select_best_threshold(
    binary_model,
    grouped_abnormal_model,
    x_val: np.ndarray,
    y_val_original: np.ndarray,
    y_val_grouped: np.ndarray,
    normal_group_index: int,
    grouped_labels: list[str],
    thresholds: list[float],
) -> tuple[float, list[dict]]:
    threshold_results = []
    best_threshold = thresholds[0]
    best_macro_f1 = -1.0
    for threshold in thresholds:
        result = evaluate_grouped_hierarchy(
            binary_model,
            grouped_abnormal_model,
            x_val,
            y_val_original,
            y_val_grouped,
            normal_group_index,
            grouped_labels,
            abnormal_threshold=threshold,
        )
        final_metrics = result["two_stage_grouped_final"]
        binary_stage = result["binary_stage"]
        row = {
            "threshold": float(threshold),
            "val_macro_f1": final_metrics["macro_f1"],
            "val_accuracy": final_metrics["accuracy"],
            "val_binary_macro_f1": binary_stage["macro_f1"],
            "val_binary_accuracy": binary_stage["accuracy"],
        }
        threshold_results.append(row)
        if row["val_macro_f1"] > best_macro_f1:
            best_macro_f1 = row["val_macro_f1"]
            best_threshold = threshold
    return float(best_threshold), threshold_results


def select_best_probability_fusion(
    binary_model,
    grouped_abnormal_model,
    x_val: np.ndarray,
    y_val_original: np.ndarray,
    y_val_grouped: np.ndarray,
    normal_group_index: int,
    grouped_labels: list[str],
    multipliers: list[float],
) -> tuple[float, list[dict]]:
    fusion_results = []
    best_multiplier = multipliers[0]
    best_macro_f1 = -1.0
    for multiplier in multipliers:
        result = evaluate_probability_fusion(
            binary_model,
            grouped_abnormal_model,
            x_val,
            y_val_original,
            y_val_grouped,
            normal_group_index,
            grouped_labels,
            abnormal_score_multiplier=multiplier,
        )
        final_metrics = result["two_stage_probability_fused_final"]
        row = {
            "multiplier": float(multiplier),
            "val_macro_f1": final_metrics["macro_f1"],
            "val_accuracy": final_metrics["accuracy"],
            "val_binary_macro_f1_after_fusion": result["binary_stage_after_fusion"]["macro_f1"],
        }
        fusion_results.append(row)
        if row["val_macro_f1"] > best_macro_f1:
            best_macro_f1 = row["val_macro_f1"]
            best_multiplier = multiplier
    return float(best_multiplier), fusion_results


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
    normal_group_index = grouped_labels.index(args.normal_label)

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

        y_train_binary = (y_train != normal_group_index).astype(np.int64)
        x_binary_fit, y_binary_fit = downsample_binary_train(
            x_train,
            y_train_binary,
            ratio=args.downsample_normal_ratio,
            seed=args.seed + window_size,
        )
        binary_model = make_lgbm(args, objective="binary")
        binary_model.fit(
            x_binary_fit,
            y_binary_fit,
            sample_weight=build_weights(y_binary_fit, args.class_weight_power, args.class_weight_max_ratio),
        )

        abnormal_train_idx = np.flatnonzero(y_train != normal_group_index)
        abnormal_classes = np.unique(y_train[abnormal_train_idx])
        if len(abnormal_classes) < 2:
            raise ValueError(f"Need at least two grouped abnormal classes, got {abnormal_classes.tolist()}.")
        grouped_abnormal_model = make_lgbm(args, objective="multiclass", num_class=len(grouped_labels))
        grouped_abnormal_model.fit(
            x_train[abnormal_train_idx],
            y_train[abnormal_train_idx],
            sample_weight=build_weights(y_train[abnormal_train_idx], args.class_weight_power, args.class_weight_max_ratio),
        )
        best_threshold, threshold_results = select_best_threshold(
            binary_model,
            grouped_abnormal_model,
            x_val,
            y_val_original,
            y_val,
            normal_group_index,
            grouped_labels,
            args.abnormal_thresholds,
        )
        best_multiplier, fusion_results = select_best_probability_fusion(
            binary_model,
            grouped_abnormal_model,
            x_val,
            y_val_original,
            y_val,
            normal_group_index,
            grouped_labels,
            args.abnormal_score_multipliers,
        )

        result = {
            "window_size": window_size,
            "original_classes": labels,
            "grouped_classes": grouped_labels,
            "normal_group_index": normal_group_index,
            "normal_label": args.normal_label,
            "major_abnormal_labels": args.major_abnormal_labels,
            "other_abnormal_label": args.other_abnormal_label,
            "selected_abnormal_threshold": best_threshold,
            "threshold_search": threshold_results,
            "selected_abnormal_score_multiplier": best_multiplier,
            "probability_fusion_search": fusion_results,
            "original_to_grouped": {labels[k]: grouped_labels[v] for k, v in original_to_grouped.items()},
            "train_original_label_counts": {labels[int(k)]: int(v) for k, v in zip(*np.unique(y_train_original, return_counts=True))},
            "train_grouped_label_counts": {grouped_labels[int(k)]: int(v) for k, v in zip(*np.unique(y_train, return_counts=True))},
            "split_manifest": base.split_manifest,
            "metrics": {
                "val": evaluate_grouped_hierarchy(
                    binary_model,
                    grouped_abnormal_model,
                    x_val,
                    y_val_original,
                    y_val,
                    normal_group_index,
                    grouped_labels,
                    abnormal_threshold=best_threshold,
                ),
                "test": evaluate_grouped_hierarchy(
                    binary_model,
                    grouped_abnormal_model,
                    x_test,
                    y_test_original,
                    y_test,
                    normal_group_index,
                    grouped_labels,
                    abnormal_threshold=best_threshold,
                ),
                "transfer": evaluate_grouped_hierarchy(
                    binary_model,
                    grouped_abnormal_model,
                    x_transfer,
                    y_transfer_original,
                    y_transfer,
                    normal_group_index,
                    grouped_labels,
                    abnormal_threshold=best_threshold,
                ),
                "val_probability_fusion": evaluate_probability_fusion(
                    binary_model,
                    grouped_abnormal_model,
                    x_val,
                    y_val_original,
                    y_val,
                    normal_group_index,
                    grouped_labels,
                    abnormal_score_multiplier=best_multiplier,
                ),
                "test_probability_fusion": evaluate_probability_fusion(
                    binary_model,
                    grouped_abnormal_model,
                    x_test,
                    y_test_original,
                    y_test,
                    normal_group_index,
                    grouped_labels,
                    abnormal_score_multiplier=best_multiplier,
                ),
                "transfer_probability_fusion": evaluate_probability_fusion(
                    binary_model,
                    grouped_abnormal_model,
                    x_transfer,
                    y_transfer_original,
                    y_transfer,
                    normal_group_index,
                    grouped_labels,
                    abnormal_score_multiplier=best_multiplier,
                ),
            },
        }
        window_dir = run_root / f"window_{window_size}"
        window_dir.mkdir(parents=True, exist_ok=True)
        for split_name in ["val", "test", "transfer"]:
            pred = result["metrics"][split_name]["predictions"]
            plot_confusion(
                np.asarray(pred["y_grouped_true"], dtype=int),
                np.asarray(pred["final_grouped_pred"], dtype=int),
                grouped_labels,
                window_dir / f"{split_name}_grouped_two_stage_confusion.png",
                f"{split_name} grouped two-stage WORKING_TYPE confusion",
            )
            pred_frame = pd.DataFrame(pred)
            pred_frame["y_original_label"] = pred_frame["y_original"].map(lambda value: labels[int(value)])
            pred_frame["y_grouped_true_label"] = pred_frame["y_grouped_true"].map(lambda value: grouped_labels[int(value)])
            pred_frame["final_grouped_pred_label"] = pred_frame["final_grouped_pred"].map(lambda value: grouped_labels[int(value)])
            pred_frame.to_csv(window_dir / f"{split_name}_grouped_predictions.csv", index=False, encoding="utf-8-sig")
        save_json(window_dir / "metrics.json", result)
        summary[str(window_size)] = {
            "test_binary_macro_f1": result["metrics"]["test"]["binary_stage"]["macro_f1"],
            "selected_abnormal_threshold": result["selected_abnormal_threshold"],
            "selected_abnormal_score_multiplier": result["selected_abnormal_score_multiplier"],
            "test_grouped_abnormal_oracle_macro_f1": result["metrics"]["test"][
                "grouped_abnormal_stage_oracle_true_abnormal_only"
            ].get("macro_f1"),
            "test_two_stage_grouped_macro_f1": result["metrics"]["test"]["two_stage_grouped_final"]["macro_f1"],
            "test_two_stage_grouped_accuracy": result["metrics"]["test"]["two_stage_grouped_final"]["accuracy"],
            "test_probability_fusion_macro_f1": result["metrics"]["test_probability_fusion"][
                "two_stage_probability_fused_final"
            ]["macro_f1"],
            "test_probability_fusion_accuracy": result["metrics"]["test_probability_fusion"][
                "two_stage_probability_fused_final"
            ]["accuracy"],
            "train_grouped_label_counts": result["train_grouped_label_counts"],
        }

    payload = {"run_root": str(run_root), "window_results": summary}
    save_json(run_root / "summary.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
