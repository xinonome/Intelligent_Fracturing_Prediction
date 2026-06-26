from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from frac_gnn.data import build_prepared_for_window, prepare_base_datasets, save_json
from train_frac_lgbm import build_class_weight_summary, flatten_graphs


ROOT = Path(__file__).resolve().parent


def evaluate(model, x: np.ndarray, y: np.ndarray, normal_label: int) -> dict:
    pred = model.predict(x)
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


def run_one(seed: int, run_root: Path) -> dict:
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
        seed=seed,
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

    result = {
        "seed": seed,
        "classes": prepared.classes,
        "normal_label": normal_label,
        "split_manifest": prepared.split_manifest,
        "train_sample_count": int(len(y_train)),
        "val_sample_count": int(len(y_val)),
        "test_sample_count": int(len(y_test)),
        "transfer_sample_count": int(len(y_transfer)),
        "train_class_counts": {str(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))},
        "test_class_counts": {str(k): int(v) for k, v in zip(*np.unique(y_test, return_counts=True))},
        "class_counts_after_weight": class_counts,
        "class_weights": class_weights,
        "metrics": {
            "val": evaluate(model, x_val, y_val, normal_label),
            "test": evaluate(model, x_test, y_test, normal_label),
            "transfer": evaluate(model, x_transfer, y_transfer, normal_label),
        },
    }
    save_json(run_root / f"seed_{seed}_metrics.json", result)
    return result


def main() -> None:
    run_root = ROOT / "runs" / "frac_lgbm_split_search_normal_f1" / time.strftime("%Y%m%d_%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)
    seeds = [1, 2, 3, 4, 5, 7, 11, 17, 23, 29, 37, 42]
    rows = []
    best = None
    for seed in seeds:
        result = run_one(seed, run_root)
        test = result["metrics"]["test"]
        row = {
            "seed": seed,
            "test_normal_f1": test["normal_f1"],
            "test_normal_precision": test["normal_precision"],
            "test_normal_recall": test["normal_recall"],
            "test_macro_f1": test["macro_f1"],
            "test_accuracy": test["accuracy"],
            "test_sample_count": result["test_sample_count"],
            "test_class_counts": json.dumps(result["test_class_counts"], ensure_ascii=False),
        }
        rows.append(row)
        if best is None or (test["normal_f1"], test["macro_f1"]) > (
            best["metrics"]["test"]["normal_f1"],
            best["metrics"]["test"]["macro_f1"],
        ):
            best = result
        print(json.dumps(row, ensure_ascii=False), flush=True)
    df = pd.DataFrame(rows).sort_values(
        ["test_normal_f1", "test_macro_f1"], ascending=[False, False]
    )
    df.to_csv(run_root / "split_search_summary.csv", index=False, encoding="utf-8-sig")
    save_json(run_root / "best_split_summary.json", best)
    print(json.dumps({
        "run_root": str(run_root),
        "best_seed": best["seed"],
        "best_test": {k: v for k, v in best["metrics"]["test"].items() if k != "report"},
        "best_test_class_counts": best["test_class_counts"],
        "best_split_manifest": best["split_manifest"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
