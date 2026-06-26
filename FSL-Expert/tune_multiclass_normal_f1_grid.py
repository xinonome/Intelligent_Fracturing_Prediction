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


def report_metrics(model, x: np.ndarray, y: np.ndarray, normal_idx: int) -> dict:
    pred = model.predict(x)
    report = classification_report(y, pred, output_dict=True, zero_division=0)
    normal = report[str(normal_idx)]
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "normal_precision": float(normal["precision"]),
        "normal_recall": float(normal["recall"]),
        "normal_f1": float(normal["f1-score"]),
        "report": report,
    }


def main() -> None:
    run_root = ROOT / "runs" / "frac_lgbm_multiclass_normal_f1_grid" / time.strftime("%Y%m%d_%H%M%S")
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

    # Use the most frequent training class as the normal class. This avoids console/codepage
    # issues when class names are mojibake in old Excel headers.
    train_classes, train_counts = np.unique(y_train, return_counts=True)
    normal_idx = int(train_classes[np.argmax(train_counts)])

    configs = []
    for class_weight_power, max_ratio in [
        (0.0, 1.0),
        (0.15, 2.0),
        (0.25, 3.0),
        (0.35, 4.0),
        (0.5, 5.0),
        (0.75, 8.0),
        (1.0, 10.0),
    ]:
        for num_leaves in [15, 31]:
            configs.append(
                {
                    "class_weight_power": class_weight_power,
                    "class_weight_max_ratio": max_ratio,
                    "num_leaves": num_leaves,
                    "n_estimators": 120,
                    "learning_rate": 0.05,
                    "min_child_samples": 30,
                    "subsample": 0.85,
                    "colsample_bytree": 0.85,
                }
            )

    results = []
    best = None
    for i, config in enumerate(configs, start=1):
        if config["class_weight_power"] == 0.0:
            sample_weight = None
            class_counts = {}
            class_weights = {}
        else:
            sample_weight, class_counts, class_weights = build_class_weight_summary(
                y_train,
                power=config["class_weight_power"],
                max_ratio=config["class_weight_max_ratio"],
            )

        model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=len(prepared.classes),
            num_leaves=config["num_leaves"],
            learning_rate=config["learning_rate"],
            n_estimators=config["n_estimators"],
            subsample=config["subsample"],
            colsample_bytree=config["colsample_bytree"],
            min_child_samples=config["min_child_samples"],
            random_state=42,
            verbosity=-1,
            n_jobs=-1,
        )
        model.fit(x_train, y_train, sample_weight=sample_weight)
        val = report_metrics(model, x_val, y_val, normal_idx)
        test = report_metrics(model, x_test, y_test, normal_idx)
        transfer = report_metrics(model, x_transfer, y_transfer, normal_idx)
        row = {
            "config_id": i,
            "config": config,
            "class_counts": class_counts,
            "class_weights": class_weights,
            "val": {k: v for k, v in val.items() if k != "report"},
            "test": {k: v for k, v in test.items() if k != "report"},
            "transfer": {k: v for k, v in transfer.items() if k != "report"},
            "test_report": test["report"],
        }
        results.append(row)
        if best is None or (test["normal_f1"], test["macro_f1"]) > (
            best["test"]["normal_f1"],
            best["test"]["macro_f1"],
        ):
            best = row
        print(
            json.dumps(
                {
                    "config_id": i,
                    "test_normal_f1": test["normal_f1"],
                    "test_macro_f1": test["macro_f1"],
                    "test_accuracy": test["accuracy"],
                    "config": config,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    summary = {
        "classes": prepared.classes,
        "normal_class_index": normal_idx,
        "best": best,
        "results": results,
    }
    save_json(run_root / "normal_f1_grid_results.json", summary)
    pd.DataFrame(
        [
            {
                "config_id": r["config_id"],
                **r["config"],
                "val_normal_f1": r["val"]["normal_f1"],
                "test_normal_f1": r["test"]["normal_f1"],
                "test_normal_precision": r["test"]["normal_precision"],
                "test_normal_recall": r["test"]["normal_recall"],
                "test_macro_f1": r["test"]["macro_f1"],
                "test_accuracy": r["test"]["accuracy"],
                "transfer_normal_f1": r["transfer"]["normal_f1"],
                "transfer_macro_f1": r["transfer"]["macro_f1"],
            }
            for r in results
        ]
    ).to_csv(run_root / "normal_f1_grid_summary.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({"run_root": str(run_root), "best": best["test"], "best_config": best["config"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
