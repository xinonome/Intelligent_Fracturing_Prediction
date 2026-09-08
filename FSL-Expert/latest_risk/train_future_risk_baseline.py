from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import _bootstrap  # noqa: F401

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from frac_gnn.future_risk_data import (
    DEFAULT_FUTURE_RISK_FEATURES,
    build_future_risk_dataset,
    indices_for_manifest,
    split_groups,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Future 10-second abnormal-risk baseline.")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", default="runs/future_risk_baseline")
    parser.add_argument("--well-column", default="JTBH")
    parser.add_argument("--segment-column", default="FDBH")
    parser.add_argument("--time-column", default="SGSJ")
    parser.add_argument("--label-column", default="WORKING_TYPE")
    parser.add_argument("--normal-label", default="正常")
    parser.add_argument("--feature-columns", nargs="*", default=list(DEFAULT_FUTURE_RISK_FEATURES))
    parser.add_argument("--window-size", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--max-gap-seconds", type=float, default=30.0)
    parser.add_argument("--blank-label-means-normal", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model",
        choices=["hist_gradient_boosting", "lightgbm"],
        default="hist_gradient_boosting",
    )
    parser.add_argument(
        "--class-weight",
        choices=["balanced", "none"],
        default="balanced",
    )
    return parser


def metric_dict(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, object]:
    prediction = (probabilities >= threshold).astype(np.int64)
    result: dict[str, object] = {
        "threshold": float(threshold),
        "sample_count": int(len(y_true)),
        "positive_count": int(y_true.sum()),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "macro_f1": float(f1_score(y_true, prediction, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, prediction, labels=[0, 1]).tolist(),
    }
    if len(np.unique(y_true)) > 1:
        result["roc_auc"] = float(roc_auc_score(y_true, probabilities))
        result["average_precision"] = float(average_precision_score(y_true, probabilities))
        result["brier_score"] = float(brier_score_loss(y_true, probabilities))
    return result


def choose_threshold(y_true: np.ndarray, probabilities: np.ndarray, objective: str) -> float:
    candidates = np.linspace(0.05, 0.95, 181)
    scores: list[float] = []
    for threshold in candidates:
        prediction = probabilities >= threshold
        if objective == "accuracy":
            score = accuracy_score(y_true, prediction)
        elif objective == "macro_f1":
            score = f1_score(y_true, prediction, average="macro", zero_division=0)
        else:
            raise ValueError(f"Unknown threshold objective: {objective}")
        scores.append(float(score))
    return float(candidates[int(np.argmax(scores))])


def subset_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    current_targets: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    current_normal = current_targets == 0
    if not current_normal.any():
        return {"sample_count": 0, "note": "No current-normal samples"}
    return metric_dict(y_true[current_normal], probabilities[current_normal], threshold)


def main() -> None:
    args = build_parser().parse_args()
    run_dir = Path(args.output_dir) / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_future_risk_dataset(
        args.data_path,
        well_column=args.well_column,
        segment_column=args.segment_column,
        time_column=args.time_column,
        label_column=args.label_column,
        normal_label=args.normal_label,
        feature_columns=args.feature_columns,
        window_size=args.window_size,
        horizon=args.horizon,
        max_gap_seconds=args.max_gap_seconds,
        blank_label_means_normal=args.blank_label_means_normal,
    )
    manifest = split_groups(dataset.groups, seed=args.seed)
    split_indices = indices_for_manifest(dataset.groups, manifest)

    train_index = split_indices["train"]
    if args.model == "lightgbm":
        from lightgbm import LGBMClassifier, early_stopping, log_evaluation

        model = LGBMClassifier(
            objective="binary",
            n_estimators=1200,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=30,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=1.0,
            class_weight=None if args.class_weight == "none" else "balanced",
            random_state=args.seed,
            n_jobs=-1,
            verbosity=-1,
        )
        val_index = split_indices["val"]
        model.fit(
            dataset.features[train_index],
            dataset.targets[train_index],
            eval_set=[(dataset.features[val_index], dataset.targets[val_index])],
            eval_metric="binary_logloss",
            callbacks=[early_stopping(75, verbose=False), log_evaluation(period=0)],
        )
    else:
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_iter=300,
                        max_leaf_nodes=31,
                        min_samples_leaf=20,
                        l2_regularization=1.0,
                        class_weight=None if args.class_weight == "none" else "balanced",
                        random_state=args.seed,
                    ),
                ),
            ]
        )
        model.fit(dataset.features[train_index], dataset.targets[train_index])

    probabilities = {
        name: model.predict_proba(dataset.features[index])[:, 1]
        for name, index in split_indices.items()
    }
    validation_targets = dataset.targets[split_indices["val"]]
    thresholds = {
        "fixed_0.5": 0.5,
        "validation_accuracy": choose_threshold(
            validation_targets, probabilities["val"], "accuracy"
        ),
        "validation_macro_f1": choose_threshold(
            validation_targets, probabilities["val"], "macro_f1"
        ),
    }

    metrics: dict[str, object] = {}
    for split_name, index in split_indices.items():
        y_true = dataset.targets[index]
        current = dataset.current_targets[index]
        split_result: dict[str, object] = {}
        for threshold_name, threshold in thresholds.items():
            split_result[threshold_name] = {
                "overall": metric_dict(y_true, probabilities[split_name], threshold),
                "current_normal_only": subset_metrics(
                    y_true, probabilities[split_name], current, threshold
                ),
            }
        split_result["baselines"] = {
            "always_normal": metric_dict(y_true, np.zeros_like(y_true, dtype=float), 0.5),
            "current_state_persistence": metric_dict(y_true, current.astype(float), 0.5),
        }
        metrics[split_name] = split_result

        output = dataset.metadata.iloc[index].copy()
        output["target_binary"] = y_true
        output["current_binary"] = current
        output["risk_probability"] = probabilities[split_name]
        output["prediction"] = (
            probabilities[split_name] >= thresholds["validation_macro_f1"]
        ).astype(int)
        output.to_csv(run_dir / f"{split_name}_predictions.csv", index=False, encoding="utf-8-sig")

    result = {
        "task": "past window predicts future normal/abnormal risk",
        "model": args.model,
        "class_weight": args.class_weight,
        "data_path": str(Path(args.data_path).resolve()),
        "window_size": args.window_size,
        "horizon": args.horizon,
        "max_gap_seconds": args.max_gap_seconds,
        "blank_label_means_normal": args.blank_label_means_normal,
        "blank_label_count": dataset.blank_label_count,
        "raw_label_counts": dataset.label_counts,
        "requested_feature_columns": args.feature_columns,
        "used_feature_columns": dataset.base_feature_names,
        "feature_count": len(dataset.feature_names),
        "sample_count": int(len(dataset.targets)),
        "positive_count": int(dataset.targets.sum()),
        "normal_to_abnormal_count": int(
            ((dataset.current_targets == 0) & (dataset.targets == 1)).sum()
        ),
        "group_count": int(len(np.unique(dataset.groups))),
        "split_manifest": manifest,
        "split_sample_counts": {
            name: int(len(index)) for name, index in split_indices.items()
        },
        "thresholds": thresholds,
        "metrics": metrics,
        "limitations": [
            "Blank WORKING_TYPE values are treated as normal by explicit project convention.",
            "The available labeled workbook contains one well, so splits are by construction segment, not by well.",
            "Current-normal-only metrics contain far fewer positive transitions than overall window metrics.",
        ],
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "feature_names.json").write_text(
        json.dumps(dataset.feature_names, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    joblib.dump(model, run_dir / "model.joblib")
    print(json.dumps({"run_dir": str(run_dir), "thresholds": thresholds, "test": metrics["test"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
