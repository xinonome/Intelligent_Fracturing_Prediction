from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


KEY_COLUMNS = ["group", "window_end", "target_time"]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Calibrate green/yellow/red future-risk boundaries.")
    result.add_argument("--prediction-run", required=True)
    result.add_argument("--output-dir", default="runs/future_risk_boundaries")
    result.add_argument(
        "--validation-accuracy-targets",
        nargs="*",
        type=float,
        default=[0.95, 0.97, 0.98, 0.99],
    )
    return result


def boundary_metrics(frame: pd.DataFrame, low: float, high: float) -> dict[str, object]:
    target = frame["target_binary"].to_numpy(dtype=int)
    probability = frame["risk_probability"].to_numpy(dtype=float)
    green = probability <= low
    red = probability >= high
    decided = green | red
    prediction = red.astype(int)
    correct = prediction == target
    decided_count = int(decided.sum())
    true_positive = int((red & (target == 1)).sum())
    false_positive = int((red & (target == 0)).sum())
    positive_count = int(target.sum())
    return {
        "low_threshold": float(low),
        "high_threshold": float(high),
        "sample_count": int(len(target)),
        "decided_count": decided_count,
        "yellow_count": int((~decided).sum()),
        "coverage": float(decided.mean()),
        "yellow_rate": float((~decided).mean()),
        "decided_accuracy": float(correct[decided].mean()) if decided_count else 0.0,
        "green_count": int(green.sum()),
        "green_accuracy": float((target[green] == 0).mean()) if green.any() else 0.0,
        "red_count": int(red.sum()),
        "red_precision": float(true_positive / max(true_positive + false_positive, 1)),
        "red_recall_over_all_positive": float(true_positive / max(positive_count, 1)),
    }


def choose_boundaries(frame: pd.DataFrame, accuracy_target: float) -> dict[str, object] | None:
    best: dict[str, object] | None = None
    for low in np.linspace(0.02, 0.48, 47):
        for high in np.linspace(0.52, 0.98, 47):
            candidate = boundary_metrics(frame, float(low), float(high))
            if candidate["decided_accuracy"] < accuracy_target:
                continue
            if best is None:
                best = candidate
                continue
            ranking = (
                candidate["coverage"],
                candidate["red_recall_over_all_positive"],
                candidate["decided_accuracy"],
            )
            best_ranking = (
                best["coverage"],
                best["red_recall_over_all_positive"],
                best["decided_accuracy"],
            )
            if ranking > best_ranking:
                best = candidate
    return best


def main() -> None:
    args = parser().parse_args()
    source = Path(args.prediction_run)
    validation = pd.read_csv(source / "val_predictions.csv")
    test = pd.read_csv(source / "test_predictions.csv")
    run_dir = Path(args.output_dir) / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, object] = {}
    for target in args.validation_accuracy_targets:
        selected = choose_boundaries(validation, target)
        key = f"validation_target_{target:.3f}"
        if selected is None:
            results[key] = {"status": "no_feasible_boundaries"}
            continue
        results[key] = {
            "validation": selected,
            "test": boundary_metrics(
                test,
                float(selected["low_threshold"]),
                float(selected["high_threshold"]),
            ),
        }

    output = {
        "prediction_run": str(source.resolve()),
        "method": "validation-calibrated selective classification",
        "decision_rule": "green if p<=low, red if p>=high, otherwise yellow/review",
        "results": results,
        "acceptance_note": "Accuracy above 95% applies only to green/red decided samples; coverage and yellow rate must be reported.",
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"run_dir": str(run_dir), "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
