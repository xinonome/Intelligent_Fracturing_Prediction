from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline import build_dataset, discover_segment_frames, estimate_sample_interval_seconds, segment_split
from response_surrogate import ActionResponseSurrogate


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a real-segment action-response surrogate for a configurable horizon.")
    parser.add_argument("--data-path", default=str(PROJECT_ROOT / "Data" / "raw_frac"))
    parser.add_argument("--reference-header-path", default=str(PROJECT_ROOT / "Data" / "raw_frac" / "WITHfiltered2ASSELECTDISTINCTJTHJDFROMhagHAGMARKPOINTWHEREWORKIN_202511211131.xlsx"))
    parser.add_argument("--segment-column", default="FDBH")
    parser.add_argument("--time-column", default="SGSJ")
    parser.add_argument("--label-column", default="WORKING_TYPE")
    parser.add_argument("--sample-interval-seconds", type=float, default=10.0)
    parser.add_argument("--state-seconds", type=float, default=300.0)
    parser.add_argument("--action-seconds", type=float, default=60.0)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--max-rows-per-file", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--run-dir", default=str(PROJECT_ROOT / "outputs" / "hmi" / "response_surrogate"))
    args = parser.parse_args()

    frames = discover_segment_frames(
        args.data_path, args.reference_header_path, args.segment_column, args.time_column,
        ["SGBY", "PL", "SB", args.label_column], ["WITHfiltered"],
        args.max_files, args.max_rows_per_file,
    )
    interval = estimate_sample_interval_seconds(frames, args.time_column, args.sample_interval_seconds)
    bundle = build_dataset(
        frames, ["SGBY", "PL", "SB"], ["PL", "SB"], args.time_column,
        max(2, int(round(args.state_seconds / interval))),
        max(1, int(round(args.action_seconds / interval))), args.label_column,
    )
    if args.max_samples and len(bundle.x) > args.max_samples:
        chosen = np.linspace(0, len(bundle.x) - 1, args.max_samples, dtype=int)
        bundle.x = bundle.x[chosen]
        bundle.y = bundle.y[chosen]
        bundle.meta = bundle.meta.iloc[chosen].reset_index(drop=True)

    train_idx, val_idx, test_idx = segment_split(bundle.meta, 0.70, 0.10, args.seed)
    evaluation_idx = test_idx if len(test_idx) else val_idx
    model = ActionResponseSurrogate(bundle.feature_names, bundle.action_bounds, args.seed)
    model.fit(bundle.x[train_idx], bundle.meta.iloc[train_idx], bundle.y[train_idx])
    predictions = model.predict_batch(bundle.x[evaluation_idx], bundle.meta.iloc[evaluation_idx], bundle.y[evaluation_idx])
    metrics = model.evaluate(bundle.x[evaluation_idx], bundle.meta.iloc[evaluation_idx], bundle.y[evaluation_idx])

    out = Path(args.run_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / "response_surrogate.joblib")
    result = bundle.meta.iloc[evaluation_idx][[
        "segment_id", "time", "current_pressure", "current_flow", "current_sand_ratio",
        "future_pressure_mean", "future_pressure_max", "future_abnormal", "future_sand_plug",
    ]].reset_index(drop=True)
    result["action_flow"] = bundle.y[evaluation_idx, 0]
    result["action_sand_ratio"] = bundle.y[evaluation_idx, 1]
    for name, values in predictions.items():
        result[f"predicted_{name}"] = values
    result.to_csv(out / "predictions.csv", index=False, encoding="utf-8-sig")
    summary = {
        "model": ActionResponseSurrogate.version,
        "risk_label_policy": ActionResponseSurrogate.risk_label_policy,
        "purpose": f"Predict future {args.action_seconds:g}s pressure and condition risk under candidate flow/sand action",
        "physics_boundary": "Fracture geometry and physical parameters are still computed by PKN-EnKF",
        "sample_interval_seconds": interval,
        "history_seconds": args.state_seconds,
        "response_seconds": args.action_seconds,
        "segments": len(frames),
        "train_samples": len(train_idx),
        "test_samples": len(evaluation_idx),
        "action_bounds": bundle.action_bounds,
        "metrics": metrics,
        "outputs": {"model": str(out / "response_surrogate.joblib"), "predictions": str(out / "predictions.csv")},
    }
    (out / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"run_dir": str(out), "metrics": metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
