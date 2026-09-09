from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from causal_warning_model import CausalWarningModel300s
from condition_taxonomy import has_abnormal_label, has_sand_plug_label, split_labels
from data_pipeline import build_dataset, discover_segment_frames, estimate_sample_interval_seconds


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent


def _segment_bucket(frame: pd.DataFrame) -> str:
    if int(frame["future_sand_plug"].max()) > 0:
        return "sand_plug"
    if int(frame["future_abnormal"].max()) > 0:
        return "other_abnormal"
    return "normal"


def stratified_segment_split(
    meta: pd.DataFrame,
    train_ratio: float,
    validation_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, list[str]]]:
    """Split whole segments while spreading available event segments across splits."""

    rng = np.random.default_rng(seed)
    groups: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    segment_table = pd.DataFrame(
        [
            {"segment_id": str(segment_id), "bucket": _segment_bucket(frame)}
            for segment_id, frame in meta.groupby("segment_id", sort=False)
        ]
    )
    for _, bucket_rows in segment_table.groupby("bucket", sort=False):
        segments = bucket_rows["segment_id"].astype(str).to_numpy()
        rng.shuffle(segments)
        count = len(segments)
        if count == 1:
            n_train, n_validation = 1, 0
        elif count == 2:
            n_train, n_validation = 1, 0
        else:
            n_train = min(count - 2, max(1, int(round(count * train_ratio))))
            n_validation = min(count - n_train - 1, max(1, int(round(count * validation_ratio))))
        groups["train"].extend(segments[:n_train].tolist())
        groups["validation"].extend(segments[n_train : n_train + n_validation].tolist())
        groups["test"].extend(segments[n_train + n_validation :].tolist())

    # Very small corpora can leave validation empty. Move a normal segment,
    # without ever splitting one segment between development and test data.
    if not groups["validation"] and len(groups["train"]) > 1:
        groups["validation"].append(groups["train"].pop())
    if not groups["test"] and len(groups["train"]) > 1:
        groups["test"].append(groups["train"].pop())

    indices = tuple(
        meta.index[meta["segment_id"].astype(str).isin(groups[name])].to_numpy()
        for name in ("train", "validation", "test")
    )
    return indices[0], indices[1], indices[2], groups


def _balanced_cap(indices: np.ndarray, target: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if limit <= 0 or len(indices) <= limit:
        return np.asarray(indices, dtype=int)
    rng = np.random.default_rng(seed)
    positive = indices[target[indices] == 1]
    negative = indices[target[indices] == 0]
    if len(positive) >= limit:
        return np.sort(rng.choice(positive, size=limit, replace=False))
    remaining = limit - len(positive)
    selected_negative = rng.choice(negative, size=min(remaining, len(negative)), replace=False)
    return np.sort(np.concatenate([positive, selected_negative]))


def _cap_negative_ratio(
    indices: np.ndarray,
    target: np.ndarray,
    maximum_negative_to_positive: float,
    seed: int,
) -> np.ndarray:
    if maximum_negative_to_positive <= 0:
        return indices
    positive = indices[target[indices] == 1]
    negative = indices[target[indices] == 0]
    if not len(positive):
        return indices
    negative_limit = int(round(len(positive) * maximum_negative_to_positive))
    if len(negative) <= negative_limit:
        return indices
    rng = np.random.default_rng(seed)
    selected_negative = rng.choice(negative, size=negative_limit, replace=False)
    return np.sort(np.concatenate([positive, selected_negative]))


def binary_metrics(truth: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, object]:
    truth = np.asarray(truth, dtype=int)
    score = np.asarray(score, dtype=float)
    finite = np.isfinite(score)
    truth = truth[finite]
    score = score[finite]
    predicted = score >= threshold
    positive = truth == 1
    negative = ~positive
    tp = int(np.sum(predicted & positive))
    fn = int(np.sum(~predicted & positive))
    fp = int(np.sum(predicted & negative))
    tn = int(np.sum(~predicted & negative))
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    false_alarm = fp / max(fp + tn, 1)
    auc = None
    if len(np.unique(truth)) == 2:
        auc = float(roc_auc_score(truth, score))
    return {
        "samples": int(len(truth)),
        "positive_samples": int(np.sum(positive)),
        "negative_samples": int(np.sum(negative)),
        "threshold": float(threshold),
        "true_positive": tp,
        "false_negative": fn,
        "false_positive": fp,
        "true_negative": tn,
        "recall": float(recall),
        "precision": float(precision),
        "false_alarm_rate": float(false_alarm),
        "roc_auc": auc,
    }


def select_threshold(truth: np.ndarray, score: np.ndarray, target_recall: float) -> tuple[float, dict[str, object]]:
    """Freeze the highest validation threshold that reaches the recall target."""

    truth = np.asarray(truth, dtype=int)
    score = np.asarray(score, dtype=float)
    finite = np.isfinite(score)
    truth = truth[finite]
    score = score[finite]
    if not len(score) or not np.any(truth == 1):
        threshold = 0.5
        return threshold, binary_metrics(truth, score, threshold)
    candidates = np.unique(np.concatenate([[0.0, 1.0], score[truth == 1]]))
    candidates.sort()
    accepted: list[tuple[float, dict[str, object]]] = []
    for threshold in candidates:
        metrics = binary_metrics(truth, score, float(threshold))
        if float(metrics["recall"]) >= target_recall:
            accepted.append((float(threshold), metrics))
    if not accepted:
        threshold = 0.0
        return threshold, binary_metrics(truth, score, threshold)
    return max(accepted, key=lambda item: item[0])


def strict_event_metrics(
    meta: pd.DataFrame,
    score: np.ndarray,
    threshold: float,
    horizon_seconds: float,
) -> dict[str, object]:
    """Score the first positive boundary of each future-event interval.

    With a 300-second future target, the 0->1 boundary is the sample located
    one full horizon before the labelled event begins.  Counting that boundary
    once avoids treating the following 299 seconds as 299 separate events.
    """

    scored = meta.reset_index(drop=True).copy()
    scored["_score"] = np.asarray(score, dtype=float)
    event_scores: list[float] = []
    for _, segment in scored.groupby("segment_id", sort=False):
        truth = segment["future_abnormal"].to_numpy(dtype=int)
        starts = np.flatnonzero((truth == 1) & np.r_[True, truth[:-1] == 0])
        event_scores.extend(segment["_score"].to_numpy(dtype=float)[starts].tolist())
    values = np.asarray(event_scores, dtype=float)
    detected = values >= threshold
    events = int(len(values))
    hits = int(np.sum(detected))
    recall = hits / max(events, 1)
    return {
        "definition": "warning_at_future_event_0_to_1_boundary",
        "horizon_seconds": float(horizon_seconds),
        "event_windows": events,
        "detected_event_windows": hits,
        "missed_event_windows": int(events - hits),
        "recall": float(recall),
        "threshold": float(threshold),
    }


def _history_is_normal(value: object) -> bool:
    labels = split_labels(value)
    return not has_abnormal_label(labels) and not has_sand_plug_label(labels)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a leakage-free 300-second warning model on segment-disjoint data."
    )
    parser.add_argument("--data-path", default=str(PROJECT_ROOT / "Data" / "raw_frac"))
    parser.add_argument("--reference-header-path", default=None)
    parser.add_argument("--segment-column", default="FDBH")
    parser.add_argument("--time-column", default="SGSJ")
    parser.add_argument("--label-column", default="WORKING_TYPE")
    parser.add_argument("--sample-interval-seconds", type=float, default=10.0)
    parser.add_argument("--history-seconds", type=float, default=300.0)
    parser.add_argument("--warning-seconds", type=float, default=300.0)
    parser.add_argument("--target-recall", type=float, default=0.90)
    parser.add_argument("--normal-history-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--max-rows-per-file", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=50000)
    parser.add_argument("--maximum-train-negative-ratio", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--run-dir", default=str(PROJECT_ROOT / "outputs" / "hmi" / "causal_warning_300s"))
    args = parser.parse_args()

    frames = discover_segment_frames(
        args.data_path,
        args.reference_header_path,
        args.segment_column,
        args.time_column,
        ["SGBY", "PL", "SB", args.label_column],
        ["WITHfiltered", "便签数据", "综合", "aggregate", "combined"],
        args.max_files,
        args.max_rows_per_file,
    )
    interval = estimate_sample_interval_seconds(frames, args.time_column, args.sample_interval_seconds)
    history_points = max(2, int(round(args.history_seconds / interval)))
    warning_points = max(1, int(round(args.warning_seconds / interval)))
    bundle = build_dataset(
        frames,
        ["SGBY", "PL", "SB"],
        ["PL", "SB"],
        args.time_column,
        history_points,
        warning_points,
        args.label_column,
    )

    source_samples = len(bundle.meta)
    if args.normal_history_only:
        keep = bundle.meta["state_working_types"].map(_history_is_normal).to_numpy(dtype=bool)
        bundle.x = bundle.x[keep]
        bundle.y = bundle.y[keep]
        bundle.meta = bundle.meta.loc[keep].reset_index(drop=True)
    if bundle.meta.empty:
        raise ValueError("No causal warning samples remain after history filtering.")

    train_idx, validation_idx, test_idx, split_segments = stratified_segment_split(
        bundle.meta, 0.70, 0.15, args.seed
    )
    target = bundle.meta["future_abnormal"].to_numpy(dtype=int)
    if args.max_samples > 0:
        split_limit = max(1, args.max_samples // 3)
        train_idx = _balanced_cap(train_idx, target, args.max_samples - 2 * split_limit, args.seed)
        validation_idx = _balanced_cap(validation_idx, target, split_limit, args.seed + 1)
        test_idx = _balanced_cap(test_idx, target, split_limit, args.seed + 2)
    train_idx = _cap_negative_ratio(
        train_idx,
        target,
        args.maximum_train_negative_ratio,
        args.seed + 3,
    )
    if not len(train_idx) or not len(test_idx):
        raise ValueError("Segment-disjoint split did not produce both training and test samples.")

    model = CausalWarningModel300s(bundle.feature_names, args.seed)
    model.fit(bundle.x[train_idx], bundle.meta.iloc[train_idx])
    if model.abnormal_model is None:
        raise ValueError("Training segments do not contain both abnormal and normal future labels.")

    calibration_idx = validation_idx
    calibration_source = "validation"
    if not len(calibration_idx) or not np.any(target[calibration_idx] == 1):
        calibration_idx = train_idx
        calibration_source = "training_fallback_no_positive_validation_window"
    validation_predictions = model.predict_batch(
        bundle.x[calibration_idx], bundle.meta.iloc[calibration_idx]
    )
    abnormal_threshold, abnormal_validation = select_threshold(
        target[calibration_idx],
        validation_predictions["abnormal_probability"],
        args.target_recall,
    )

    sand_target = bundle.meta["future_sand_plug"].to_numpy(dtype=int)
    sand_threshold = 0.5
    sand_validation: dict[str, object] = {
        "samples": int(len(calibration_idx)),
        "positive_samples": int(np.sum(sand_target[calibration_idx] == 1)),
        "threshold": sand_threshold,
        "available": False,
        "reason": "classifier_or_positive_validation_labels_unavailable",
    }
    if model.sand_plug_model is not None and np.any(sand_target[calibration_idx] == 1):
        sand_threshold, sand_validation = select_threshold(
            sand_target[calibration_idx],
            validation_predictions["sand_plug_probability"],
            args.target_recall,
        )
        sand_validation["available"] = True

    test_meta = bundle.meta.iloc[test_idx].reset_index(drop=True)
    test_predictions = model.predict_batch(bundle.x[test_idx], test_meta)
    abnormal_test = binary_metrics(
        test_meta["future_abnormal"].to_numpy(dtype=int),
        test_predictions["abnormal_probability"],
        abnormal_threshold,
    )
    strict_event_test = strict_event_metrics(
        test_meta,
        test_predictions["abnormal_probability"],
        abnormal_threshold,
        args.warning_seconds,
    )
    abnormal_test["target_recall"] = float(args.target_recall)
    abnormal_test["sample_window_recall"] = abnormal_test["recall"]
    abnormal_test["strict_event_300s"] = strict_event_test
    abnormal_test["pass_5min_warning_recall"] = bool(
        strict_event_test["event_windows"] > 0
        and strict_event_test["recall"] >= args.target_recall
    )
    sand_test = binary_metrics(
        test_meta["future_sand_plug"].to_numpy(dtype=int),
        test_predictions["sand_plug_probability"],
        sand_threshold,
    )

    output_root = Path(args.run_dir)
    out = output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / "causal_warning_300s.joblib"
    prediction_path = out / "predictions.csv"
    model.save(model_path)
    result_columns = [
        "segment_id",
        "time",
        "current_pressure",
        "current_flow",
        "current_sand_ratio",
        "state_working_types",
        "future_working_types",
        "future_abnormal",
        "future_sand_plug",
    ]
    result = test_meta[result_columns].copy()
    result["predicted_abnormal_probability"] = test_predictions["abnormal_probability"]
    result["predicted_sand_plug_probability"] = test_predictions["sand_plug_probability"]
    result["decision_threshold"] = abnormal_threshold
    result["sand_plug_decision_threshold"] = sand_threshold
    result.to_csv(prediction_path, index=False, encoding="utf-8-sig")

    summary = {
        "model": model.version,
        "purpose": f"Predict explicit abnormal labels occurring in the next {args.warning_seconds:g} seconds",
        "input_contract": model.input_contract,
        "future_measured_actions_used": False,
        "sample_interval_seconds": float(interval),
        "history_seconds": float(args.history_seconds),
        "warning_seconds": float(args.warning_seconds),
        "normal_history_only": bool(args.normal_history_only),
        "maximum_train_negative_ratio": float(args.maximum_train_negative_ratio),
        "source_segments": int(len(frames)),
        "source_samples": int(source_samples),
        "retained_samples": int(len(bundle.meta)),
        "split_segments": split_segments,
        "split_samples": {
            "train": int(len(train_idx)),
            "validation": int(len(validation_idx)),
            "test": int(len(test_idx)),
        },
        "label_statistics": model.label_statistics,
        "threshold_calibration": {
            "source": calibration_source,
            "target_recall": float(args.target_recall),
            "abnormal": abnormal_validation,
            "sand_plug": sand_validation,
        },
        "held_out_test": {
            "abnormal": abnormal_test,
            "sand_plug": sand_test,
        },
        "scientific_status": (
            "offline_acceptance_candidate"
            if abnormal_test["pass_5min_warning_recall"]
            else "development_only"
        ),
        "outputs": {"model": str(model_path), "predictions": str(prediction_path)},
    }
    (out / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"run_dir": str(out), "held_out_test": summary["held_out_test"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
