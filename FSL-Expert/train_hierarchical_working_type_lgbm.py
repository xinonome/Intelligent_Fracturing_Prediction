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
from frac_gnn.rule_fusion import evaluate_sand_plug_window


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Two-stage WORKING_TYPE model: normal/abnormal + abnormal subtype.")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--label-column", default="WORKING_TYPE")
    parser.add_argument("--segment-column", default="FDBH")
    parser.add_argument("--time-column", default="SGSJ")
    parser.add_argument("--reference-header-path", default=None)
    parser.add_argument("--exclude-name-patterns", nargs="*", default=["WITHfiltered"])
    parser.add_argument("--normal-label", default="正常")
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
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--min-child-samples", type=int, default=20)
    parser.add_argument("--class-weight-power", type=float, default=1.0)
    parser.add_argument("--class-weight-max-ratio", type=float, default=30.0)
    parser.add_argument("--downsample-normal-ratio", type=float, default=0.3)
    parser.add_argument("--enable-sand-plug-rules", action="store_true")
    parser.add_argument("--sand-plug-label", default="砂堵")
    parser.add_argument("--rule-pressure-column", default="SGBY")
    parser.add_argument("--rule-flow-column", default="PL")
    parser.add_argument("--rule-sand-column", default="SB")
    parser.add_argument("--rule-pressure-delta-threshold", type=float, default=8.0)
    parser.add_argument("--rule-pressure-delta-high", type=float, default=10.0)
    parser.add_argument("--rule-pressure-slope-threshold", type=float, default=1.0)
    parser.add_argument("--rule-pressure-slope-high", type=float, default=1.5)
    parser.add_argument("--rule-flow-drop-eps", type=float, default=0.1)
    parser.add_argument("--rule-sand-drop-eps", type=float, default=0.2)
    parser.add_argument("--rule-recent-zero-sand-points", type=int, default=12)
    parser.add_argument("--rule-max-zero-sand-count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-dir", default="runs/hierarchical_working_type_lgbm")
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
    fig, ax = plt.subplots(figsize=(max(7, len(names) * 0.8), max(5, len(names) * 0.7)))
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


def evaluate_two_stage(
    binary_model,
    abnormal_model,
    x: np.ndarray,
    y: np.ndarray,
    normal_index: int,
    labels: list[str],
    graphs=None,
    segment_frames: dict[str, pd.DataFrame] | None = None,
    sand_plug_index: int | None = None,
    args=None,
) -> dict:
    y_binary = (y != normal_index).astype(np.int64)
    pred_binary = binary_model.predict(x).astype(np.int64)
    final_pred = np.full_like(y, fill_value=normal_index)
    abnormal_pred_idx = np.flatnonzero(pred_binary == 1)
    if len(abnormal_pred_idx):
        final_pred[abnormal_pred_idx] = abnormal_model.predict(x[abnormal_pred_idx]).astype(np.int64)
    model_only_pred = final_pred.copy()

    rule_records = []
    rule_trigger_count = 0
    rule_enabled = bool(args and args.enable_sand_plug_rules and sand_plug_index is not None and graphs is not None and segment_frames)
    if rule_enabled:
        for index, graph in enumerate(graphs):
            segment_id = getattr(graph, "segment_id", None)
            start = getattr(graph, "window_start", None)
            end = getattr(graph, "window_end", None)
            frame = segment_frames.get(segment_id) if segment_id is not None else None
            result = evaluate_sand_plug_window(
                frame,
                int(start) if start is not None else 0,
                int(end) if end is not None else 0,
                pressure_col=args.rule_pressure_column,
                flow_col=args.rule_flow_column,
                sand_col=args.rule_sand_column,
                pressure_delta_threshold=args.rule_pressure_delta_threshold,
                pressure_delta_high=args.rule_pressure_delta_high,
                pressure_slope_threshold=args.rule_pressure_slope_threshold,
                pressure_slope_high=args.rule_pressure_slope_high,
                flow_drop_eps=args.rule_flow_drop_eps,
                sand_drop_eps=args.rule_sand_drop_eps,
                recent_zero_sand_points=args.rule_recent_zero_sand_points,
                max_zero_sand_count=args.rule_max_zero_sand_count,
            )
            if result.triggered:
                final_pred[index] = sand_plug_index
                rule_trigger_count += 1
            if result.triggered or result.score > 0:
                rule_records.append(
                    {
                        "sample_index": int(index),
                        "segment_id": str(segment_id),
                        "window_start": int(start) if start is not None else None,
                        "window_end": int(end) if end is not None else None,
                        "triggered": bool(result.triggered),
                        "score": float(result.score),
                        "reasons": result.reasons,
                        "model_only_pred": labels[int(model_only_pred[index])],
                        "final_pred": labels[int(final_pred[index])],
                        "true_label": labels[int(y[index])],
                    }
                )

    true_abnormal_idx = np.flatnonzero(y_binary == 1)
    abnormal_only = {}
    if len(true_abnormal_idx):
        abnormal_only = metrics_dict(
            y[true_abnormal_idx],
            abnormal_model.predict(x[true_abnormal_idx]).astype(np.int64),
            labels,
        )
    return {
        "binary_stage": binary_metrics(y_binary, pred_binary),
        "abnormal_stage_oracle_true_abnormal_only": abnormal_only,
        "two_stage_model_only": metrics_dict(y, model_only_pred, labels),
        "two_stage_final": metrics_dict(y, final_pred, labels),
        "rule_fusion": {
            "enabled": rule_enabled,
            "sand_plug_label": labels[sand_plug_index] if sand_plug_index is not None else None,
            "trigger_count": int(rule_trigger_count),
            "candidate_count": int(len(rule_records)),
            "records": rule_records,
        },
        "predictions": {
            "y_true": y.tolist(),
            "binary_true": y_binary.tolist(),
            "binary_pred": pred_binary.tolist(),
            "model_only_pred": model_only_pred.tolist(),
            "final_pred": final_pred.tolist(),
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
    )
    labels = base.classes
    if args.normal_label not in labels:
        raise ValueError(f"normal label `{args.normal_label}` not found in labels: {labels}")
    normal_index = labels.index(args.normal_label)
    sand_plug_index = labels.index(args.sand_plug_label) if args.sand_plug_label in labels else None
    if args.enable_sand_plug_rules and sand_plug_index is None:
        print(f"Warning: sand plug label `{args.sand_plug_label}` not found in labels: {labels}. Rule fusion disabled.")

    summary = {}
    for window_size in args.window_sizes:
        prepared = build_prepared_for_window(base, window_size)
        x_train, y_train = flatten_graphs(prepared.train_graphs)
        x_val, y_val = flatten_graphs(prepared.val_graphs)
        x_test, y_test = flatten_graphs(prepared.test_graphs)
        x_transfer, y_transfer = flatten_graphs(prepared.transfer_graphs)

        y_train_binary = (y_train != normal_index).astype(np.int64)
        x_binary_fit, y_binary_fit = downsample_binary_train(
            x_train,
            y_train_binary,
            ratio=args.downsample_normal_ratio,
            seed=args.seed + window_size,
        )
        binary_model = make_lgbm(args, objective="binary")
        binary_model.fit(x_binary_fit, y_binary_fit, sample_weight=build_weights(y_binary_fit, args.class_weight_power, args.class_weight_max_ratio))

        abnormal_train_idx = np.flatnonzero(y_train != normal_index)
        if len(np.unique(y_train[abnormal_train_idx])) < 2:
            raise ValueError("Need at least two abnormal classes in training split for the second-stage model.")
        abnormal_model = make_lgbm(args, objective="multiclass", num_class=len(labels))
        abnormal_model.fit(
            x_train[abnormal_train_idx],
            y_train[abnormal_train_idx],
            sample_weight=build_weights(y_train[abnormal_train_idx], args.class_weight_power, args.class_weight_max_ratio),
        )

        result = {
            "window_size": window_size,
            "classes": labels,
            "normal_index": normal_index,
            "normal_label": args.normal_label,
            "train_label_counts": {labels[int(k)]: int(v) for k, v in zip(*np.unique(y_train, return_counts=True))},
            "train_abnormal_label_counts": {
                labels[int(k)]: int(v)
                for k, v in zip(*np.unique(y_train[abnormal_train_idx], return_counts=True))
            },
            "metrics": {
                "val": evaluate_two_stage(
                    binary_model,
                    abnormal_model,
                    x_val,
                    y_val,
                    normal_index,
                    labels,
                    graphs=prepared.val_graphs,
                    segment_frames=base.segment_frames,
                    sand_plug_index=sand_plug_index,
                    args=args,
                ),
                "test": evaluate_two_stage(
                    binary_model,
                    abnormal_model,
                    x_test,
                    y_test,
                    normal_index,
                    labels,
                    graphs=prepared.test_graphs,
                    segment_frames=base.segment_frames,
                    sand_plug_index=sand_plug_index,
                    args=args,
                ),
                "transfer": evaluate_two_stage(
                    binary_model,
                    abnormal_model,
                    x_transfer,
                    y_transfer,
                    normal_index,
                    labels,
                    graphs=prepared.transfer_graphs,
                    segment_frames=base.segment_frames,
                    sand_plug_index=sand_plug_index,
                    args=args,
                ),
            },
        }
        window_dir = run_root / f"window_{window_size}"
        window_dir.mkdir(parents=True, exist_ok=True)
        for split_name in ["val", "test", "transfer"]:
            pred = result["metrics"][split_name]["predictions"]
            plot_confusion(
                np.asarray(pred["y_true"], dtype=int),
                np.asarray(pred["final_pred"], dtype=int),
                labels,
                window_dir / f"{split_name}_two_stage_confusion.png",
                f"{split_name} two-stage WORKING_TYPE confusion",
            )
            pd.DataFrame(pred).to_csv(window_dir / f"{split_name}_predictions.csv", index=False)
        save_json(window_dir / "metrics.json", result)
        summary[str(window_size)] = {
            "test_binary_macro_f1": result["metrics"]["test"]["binary_stage"]["macro_f1"],
            "test_abnormal_oracle_macro_f1": result["metrics"]["test"]["abnormal_stage_oracle_true_abnormal_only"].get("macro_f1"),
            "test_two_stage_model_only_macro_f1": result["metrics"]["test"]["two_stage_model_only"]["macro_f1"],
            "test_two_stage_rule_fused_macro_f1": result["metrics"]["test"]["two_stage_final"]["macro_f1"],
            "test_two_stage_rule_fused_accuracy": result["metrics"]["test"]["two_stage_final"]["accuracy"],
            "test_rule_trigger_count": result["metrics"]["test"]["rule_fusion"]["trigger_count"],
        }

    payload = {"run_root": str(run_root), "window_results": summary}
    save_json(run_root / "summary.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
