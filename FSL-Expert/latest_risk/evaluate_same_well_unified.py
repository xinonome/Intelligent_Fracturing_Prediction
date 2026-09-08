from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import _bootstrap  # noqa: F401

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from frac_gnn.future_risk_data import indices_for_manifest
from frac_gnn.sand_risk_data import (
    _combine_timestamp,
    _find_column,
    build_sand_risk_dataset,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Evaluate parameter prediction and reviewed risk probability on the same well/split."
    )
    result.add_argument("--data-dir", required=True)
    result.add_argument("--enhanced-file", required=True)
    result.add_argument("--risk-run", required=True)
    result.add_argument("--output-dir", default="runs/same_well_unified")
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--pressure-tolerance", type=float, default=2.0)
    result.add_argument("--rate-tolerance", type=float, default=0.5)
    return result


def read_next_targets(raw_path: Path) -> dict[tuple[str, pd.Timestamp], dict[str, float]]:
    lookup: dict[tuple[str, pd.Timestamp], dict[str, float]] = {}
    with pd.ExcelFile(raw_path) as excel:
        for sheet in excel.sheet_names:
            frame = pd.read_excel(excel, sheet_name=sheet)
            times = _combine_timestamp(frame)
            pressure_col = _find_column(frame, "SGBY")
            rate_col = _find_column(frame, "PL")
            if pressure_col is None or rate_col is None:
                continue
            work = pd.DataFrame(
                {
                    "time": times,
                    "SGBY": pd.to_numeric(frame[pressure_col], errors="coerce"),
                    "PL": pd.to_numeric(frame[rate_col], errors="coerce"),
                }
            ).dropna(subset=["time"])
            work = work.sort_values("time", kind="stable").drop_duplicates("time", keep="last")
            work["next_SGBY"] = work["SGBY"].shift(-1)
            work["next_PL"] = work["PL"].shift(-1)
            for row in work.itertuples(index=False):
                lookup[(str(sheet).strip(), pd.Timestamp(row.time))] = {
                    "SGBY": float(row.next_SGBY) if pd.notna(row.next_SGBY) else np.nan,
                    "PL": float(row.next_PL) if pd.notna(row.next_PL) else np.nan,
                }
    return lookup


def regression_metrics(truth: np.ndarray, prediction: np.ndarray, tolerance: float) -> dict[str, float | int]:
    error = np.abs(prediction - truth)
    return {
        "sample_count": int(len(truth)),
        "mae": float(mean_absolute_error(truth, prediction)),
        "rmse": float(mean_squared_error(truth, prediction) ** 0.5),
        "r2": float(r2_score(truth, prediction)),
        "tolerance": float(tolerance),
        "within_tolerance_accuracy": float((error <= tolerance).mean()),
        "error_p95": float(np.quantile(error, 0.95)),
    }


def classification_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    return {
        "sample_count": int(len(target)),
        "positive_count": int(target.sum()),
        "predicted_positive_count": int(prediction.sum()),
        "accuracy": float(accuracy_score(target, prediction)),
        "precision": float(precision_score(target, prediction, zero_division=0)),
        "recall": float(recall_score(target, prediction, zero_division=0)),
        "f1": float(f1_score(target, prediction, zero_division=0)),
    }


def main() -> None:
    args = parser().parse_args()
    data_dir = Path(args.data_dir).resolve()
    enhanced = data_dir / args.enhanced_file
    raw = enhanced.with_name(enhanced.name.replace("小条统计_增强版", "").replace("__", "_").replace("_.xlsx", ".xlsx"))
    if not raw.exists():
        suffix = "_小条统计_增强版.xlsx"
        raw = enhanced.with_name(enhanced.name[: -len(suffix)] + ".xlsx")
    if not enhanced.exists() or not raw.exists():
        raise FileNotFoundError(f"Expected enhanced={enhanced} and raw={raw}")

    risk_run = Path(args.risk_run).resolve()
    risk_metrics = json.loads((risk_run / "metrics.json").read_text(encoding="utf-8"))
    manifest = risk_metrics["split_manifest"]
    dataset = build_sand_risk_dataset(
        data_dir,
        result_glob=enhanced.name,
        horizons_seconds=[60],
        window_size=int(risk_metrics["window_size"]),
        stride=int(risk_metrics["stride"]),
    )
    indices = indices_for_manifest(dataset.segment_groups, manifest)
    next_lookup = read_next_targets(raw)
    run_dir = Path(args.output_dir) / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    target_arrays: dict[str, np.ndarray] = {}
    for target_name in ["SGBY", "PL"]:
        values = []
        for row in dataset.metadata.itertuples(index=False):
            key = (str(row.segment).strip(), pd.Timestamp(row.window_end))
            values.append(next_lookup.get(key, {}).get(target_name, np.nan))
        target_arrays[target_name] = np.asarray(values, dtype=float)

    tolerances = {"SGBY": args.pressure_tolerance, "PL": args.rate_tolerance}
    parameter_results: dict[str, object] = {}
    test_predictions: dict[str, np.ndarray] = {}
    test_truth: dict[str, np.ndarray] = {}
    for target_name in ["SGBY", "PL"]:
        target = target_arrays[target_name]
        valid = {
            name: rows[np.isfinite(target[rows])]
            for name, rows in indices.items()
        }
        model = LGBMRegressor(
            objective="regression_l1",
            n_estimators=1500,
            learning_rate=0.025,
            num_leaves=31,
            min_child_samples=30,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=args.seed,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(
            dataset.features[valid["train"]],
            target[valid["train"]],
            eval_set=[(dataset.features[valid["val"]], target[valid["val"]])],
            eval_metric="l1",
            callbacks=[early_stopping(100, verbose=False), log_evaluation(period=0)],
        )
        parameter_results[target_name] = {}
        feature_pos = dataset.base_feature_names.index(target_name)
        for split, rows in valid.items():
            prediction = model.predict(dataset.features[rows])
            baseline = dataset.windows[rows, -1, feature_pos].astype(float)
            parameter_results[target_name][split] = {
                "model": regression_metrics(target[rows], prediction, tolerances[target_name]),
                "last_value_baseline": regression_metrics(target[rows], baseline, tolerances[target_name]),
            }
            if split == "test":
                test_predictions[target_name] = prediction
                test_truth[target_name] = target[rows]
        parameter_results[target_name]["best_iteration"] = int(model.best_iteration_ or model.n_estimators)
        joblib.dump(model, run_dir / f"{target_name}.joblib")

    pressure_accuracy = parameter_results["SGBY"]["test"]["model"]["within_tolerance_accuracy"]
    rate_accuracy = parameter_results["PL"]["test"]["model"]["within_tolerance_accuracy"]
    parameter_macro_accuracy = float((pressure_accuracy + rate_accuracy) / 2.0)
    common_count = min(len(test_truth["SGBY"]), len(test_truth["PL"]))
    joint_accuracy = float(
        (
            (np.abs(test_predictions["SGBY"][:common_count] - test_truth["SGBY"][:common_count]) <= args.pressure_tolerance)
            & (np.abs(test_predictions["PL"][:common_count] - test_truth["PL"][:common_count]) <= args.rate_tolerance)
        ).mean()
    )

    risk_frame = pd.read_csv(risk_run / "test_predictions.csv")
    target = risk_frame["target_binary"].to_numpy(dtype=int)
    probability = risk_frame["risk_probability"].to_numpy(dtype=float)
    low = float(risk_metrics["final_test_boundary"]["low_threshold"])
    high = float(risk_metrics["final_test_boundary"]["high_threshold"])
    yellow_red = (probability > low).astype(int)
    red_only = (probability >= high).astype(int)
    risk_result = {
        "label_status": "expert-reviewed project labels (confirmed by project owner)",
        "prediction_target": "reviewed risk event starts within the next 60 seconds",
        "probability_metrics": {
            "roc_auc": float(roc_auc_score(target, probability)),
            "average_precision": float(average_precision_score(target, probability)),
            "brier_score": float(brier_score_loss(target, probability)),
        },
        "yellow_plus_red_execution_zone": {
            "threshold": low,
            **classification_metrics(target, yellow_red),
        },
        "red_high_risk_zone": {
            "threshold": high,
            **classification_metrics(target, red_only),
        },
        "zone_counts": {
            "green": int((probability <= low).sum()),
            "yellow": int(((probability > low) & (probability < high)).sum()),
            "red": int((probability >= high).sum()),
        },
    }

    result = {
        "well": "JH_焦页5-Z6HF",
        "same_well_same_split": True,
        "raw_file": str(raw),
        "enhanced_file": str(enhanced),
        "risk_run": str(risk_run),
        "window": "6 samples at about 10-second intervals",
        "parameter_horizon": "next sample, about 10 seconds",
        "risk_horizon": "next 60 seconds",
        "split_manifest": manifest,
        "parameter_prediction": parameter_results,
        "parameter_acceptance": {
            "pressure_tolerance_mpa": args.pressure_tolerance,
            "rate_tolerance_m3_min": args.rate_tolerance,
            "pressure_accuracy": pressure_accuracy,
            "rate_accuracy": rate_accuracy,
            "macro_average_accuracy": parameter_macro_accuracy,
            "joint_both_correct_accuracy": joint_accuracy,
        },
        "risk_probability": risk_result,
        "interpretation": {
            "contract_95_metric": "parameter macro-average accuracy under fixed engineering tolerances",
            "risk_metric": "yellow+red execution-zone classification plus probability calibration",
            "warning": "Risk probability accuracy is separate from the contract parameter-prediction accuracy.",
        },
    }
    (run_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir.resolve()), "parameter_acceptance": result["parameter_acceptance"], "risk_probability": risk_result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
