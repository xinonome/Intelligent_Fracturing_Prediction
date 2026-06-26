from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import classification_report, f1_score

from frac_gnn.data import build_prepared_for_window, prepare_base_datasets, save_json
from train_frac_lgbm import build_class_weight_summary, flatten_graphs


ROOT = Path(__file__).resolve().parent


def predict_with_normal_threshold(
    probabilities: np.ndarray,
    classes: list[str],
    threshold: float,
) -> np.ndarray:
    normal_idx = classes.index("正常") if "正常" in classes else classes.index("姝ｅ父")
    pred = np.argmax(probabilities, axis=1)
    non_normal_probs = probabilities.copy()
    non_normal_probs[:, normal_idx] = -1.0
    best_non_normal = np.argmax(non_normal_probs, axis=1)
    return np.where(probabilities[:, normal_idx] >= threshold, normal_idx, best_non_normal)


def main() -> None:
    run_root = ROOT / "runs" / "frac_lgbm_normal_f1_threshold" / time.strftime("%Y%m%d_%H%M%S")
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

    sample_weight, class_counts, class_weights = build_class_weight_summary(
        y_train,
        power=1.0,
        max_ratio=20.0,
    )

    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(prepared.classes),
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=180,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        random_state=42,
        verbosity=-1,
    )
    model.fit(x_train, y_train, sample_weight=sample_weight)

    normal_idx = prepared.classes.index("正常") if "正常" in prepared.classes else prepared.classes.index("姝ｅ父")
    val_probs = model.predict_proba(x_val)
    thresholds = np.linspace(0.05, 0.95, 181)
    best = None
    for threshold in thresholds:
        pred = predict_with_normal_threshold(val_probs, prepared.classes, float(threshold))
        report = classification_report(y_val, pred, output_dict=True, zero_division=0)
        normal_f1 = report[str(normal_idx)]["f1-score"]
        macro_f1 = f1_score(y_val, pred, average="macro", zero_division=0)
        score = (normal_f1, macro_f1)
        if best is None or score > best["score"]:
            best = {
                "threshold": float(threshold),
                "normal_f1": float(normal_f1),
                "macro_f1": float(macro_f1),
                "score": score,
            }

    def evaluate(name: str, x: np.ndarray, y: np.ndarray) -> dict:
        probs = model.predict_proba(x)
        pred_default = np.argmax(probs, axis=1)
        pred_tuned = predict_with_normal_threshold(probs, prepared.classes, best["threshold"])
        return {
            "default": classification_report(y, pred_default, output_dict=True, zero_division=0),
            "tuned": classification_report(y, pred_tuned, output_dict=True, zero_division=0),
            "default_macro_f1": float(f1_score(y, pred_default, average="macro", zero_division=0)),
            "tuned_macro_f1": float(f1_score(y, pred_tuned, average="macro", zero_division=0)),
            "sample_count": int(len(y)),
        }

    result = {
        "classes": prepared.classes,
        "normal_class_index": normal_idx,
        "class_counts": class_counts,
        "class_weights": class_weights,
        "best_threshold_on_val": {k: v for k, v in best.items() if k != "score"},
        "metrics": {
            "val": evaluate("val", x_val, y_val),
            "test": evaluate("test", x_test, y_test),
            "transfer": evaluate("transfer", x_transfer, y_transfer),
        },
    }
    save_json(run_root / "threshold_metrics.json", result)
    print(json.dumps({"run_root": str(run_root), "best_threshold_on_val": result["best_threshold_on_val"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
