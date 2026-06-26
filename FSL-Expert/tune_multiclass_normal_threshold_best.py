from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score

from frac_gnn.data import build_prepared_for_window, prepare_base_datasets, save_json
from train_frac_lgbm import build_class_weight_summary, flatten_graphs


ROOT = Path(__file__).resolve().parent


def predict_with_threshold(model, x: np.ndarray, normal_label: int, threshold: float) -> np.ndarray:
    probs = model.predict_proba(x)
    model_classes = np.asarray(model.classes_, dtype=int)
    normal_col_candidates = np.flatnonzero(model_classes == normal_label)
    if len(normal_col_candidates) != 1:
        raise ValueError(f"Normal label {normal_label} not in model classes {model_classes.tolist()}")
    normal_col = int(normal_col_candidates[0])

    best_cols = np.argmax(probs, axis=1)
    best_labels = model_classes[best_cols]
    non_normal_probs = probs.copy()
    non_normal_probs[:, normal_col] = -1.0
    best_non_normal_labels = model_classes[np.argmax(non_normal_probs, axis=1)]
    return np.where(probs[:, normal_col] >= threshold, normal_label, best_non_normal_labels)


def metrics(y: np.ndarray, pred: np.ndarray, normal_label: int) -> dict:
    report = classification_report(y, pred, output_dict=True, zero_division=0)
    normal = report[str(normal_label)]
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "normal_precision": float(normal["precision"]),
        "normal_recall": float(normal["recall"]),
        "normal_f1": float(normal["f1-score"]),
        "report": report,
    }


def main() -> None:
    run_root = ROOT / "runs" / "frac_lgbm_multiclass_normal_threshold_best" / time.strftime("%Y%m%d_%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)

    base = prepare_base_datasets(
        data_path=str(ROOT),
        label_column="WORKING_TYPE",
        segment_column="FDBH",
        time_column="SGSJ",
        include_columns=None,
        exclude_columns=None,
        reference_header_path=str(ROOT / "FDBH26.xlsx"),
        exclude_name_patterns=["WITHfiltered"],
        normal_label="正常",
        binary_normal_abnormal=False,
        abnormal_label="异常",
        seed=42,
        train_ratio=0.7,
        test_ratio=0.1,
        val_ratio=0.1,
        transfer_ratio=0.1,
        trim_before_sand=True,
        sand_column="SB",
        sand_threshold=0.0,
        add_dynamic_features=True,
        dynamic_feature_columns=["BZJDH", "YTND", "SGBY", "PL", "SB", "LJSL", "ZDCLYL", "ZDJ", "LJYL", "BZJD"],
        rolling_windows=[3, 5, 10],
        drop_pure_normal_segments=False,
    )
    prepared = build_prepared_for_window(base, 4)
    x_train, y_train = flatten_graphs(prepared.train_graphs)
    x_val, y_val = flatten_graphs(prepared.val_graphs)
    x_test, y_test = flatten_graphs(prepared.test_graphs)
    x_transfer, y_transfer = flatten_graphs(prepared.transfer_graphs)

    train_classes, train_counts = np.unique(y_train, return_counts=True)
    normal_label = int(train_classes[np.argmax(train_counts)])
    sample_weight, class_counts, class_weights = build_class_weight_summary(
        y_train, power=0.35, max_ratio=4.0
    )

    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(prepared.classes),
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=120,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_samples=30,
        random_state=42,
        verbosity=-1,
        n_jobs=-1,
    )
    model.fit(x_train, y_train, sample_weight=sample_weight)

    thresholds = np.linspace(0.02, 0.98, 193)
    val_rows = []
    for threshold in thresholds:
        pred = predict_with_threshold(model, x_val, normal_label, float(threshold))
        row = metrics(y_val, pred, normal_label)
        row["threshold"] = float(threshold)
        val_rows.append(row)
    best_by_val = max(val_rows, key=lambda r: (r["normal_f1"], r["macro_f1"]))

    def eval_threshold(split: str, x: np.ndarray, y: np.ndarray, threshold: float) -> dict:
        pred = predict_with_threshold(model, x, normal_label, threshold)
        return metrics(y, pred, normal_label)

    default = {
        "val": metrics(y_val, model.predict(x_val), normal_label),
        "test": metrics(y_test, model.predict(x_test), normal_label),
        "transfer": metrics(y_transfer, model.predict(x_transfer), normal_label),
    }
    tuned = {
        "threshold": best_by_val["threshold"],
        "val": eval_threshold("val", x_val, y_val, best_by_val["threshold"]),
        "test": eval_threshold("test", x_test, y_test, best_by_val["threshold"]),
        "transfer": eval_threshold("transfer", x_transfer, y_transfer, best_by_val["threshold"]),
    }

    # Diagnostic only: this uses test labels and must not be reported as a deployable setting.
    test_rows = []
    for threshold in thresholds:
        row = eval_threshold("test", x_test, y_test, float(threshold))
        row["threshold"] = float(threshold)
        test_rows.append(row)
    best_by_test_diagnostic = max(test_rows, key=lambda r: (r["normal_f1"], r["macro_f1"]))

    result = {
        "classes": prepared.classes,
        "model_classes": [int(v) for v in model.classes_],
        "normal_label": normal_label,
        "class_counts": class_counts,
        "class_weights": class_weights,
        "default": default,
        "tuned_by_val": tuned,
        "best_by_test_diagnostic": {
            "threshold": best_by_test_diagnostic["threshold"],
            "accuracy": best_by_test_diagnostic["accuracy"],
            "macro_f1": best_by_test_diagnostic["macro_f1"],
            "normal_precision": best_by_test_diagnostic["normal_precision"],
            "normal_recall": best_by_test_diagnostic["normal_recall"],
            "normal_f1": best_by_test_diagnostic["normal_f1"],
        },
    }
    save_json(run_root / "normal_threshold_best_metrics.json", result)
    print(json.dumps({
        "run_root": str(run_root),
        "default_test_normal_f1": default["test"]["normal_f1"],
        "val_tuned_threshold": tuned["threshold"],
        "val_tuned_test_normal_f1": tuned["test"]["normal_f1"],
        "best_by_test_diagnostic": result["best_by_test_diagnostic"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
