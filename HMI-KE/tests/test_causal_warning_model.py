from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from causal_warning_model import CausalWarningModel300s
from train_causal_warning_300s import (
    binary_metrics,
    select_threshold,
    stratified_segment_split,
    strict_event_metrics,
)


def _sample_data(size: int = 80):
    rng = np.random.default_rng(7)
    feature_names = [
        "SGBY_t-1",
        "PL_t-1",
        "SB_t-1",
        "SGBY_last",
        "SGBY_mean",
        "SGBY_std",
        "SGBY_slope",
        "PL_last",
        "PL_mean",
        "PL_std",
        "PL_slope",
        "SB_last",
        "SB_mean",
        "SB_std",
        "SB_slope",
    ]
    x = rng.normal(size=(size, len(feature_names)))
    meta = pd.DataFrame(
        {
            "current_pressure": 70.0 + 5.0 * x[:, 3],
            "current_flow": 10.0 + x[:, 7],
            "current_sand_ratio": 4.0 + x[:, 11],
            "future_abnormal": (x[:, 3] + x[:, 6] > 0.2).astype(int),
            "future_sand_plug": (x[:, 11] > 1.0).astype(int),
        }
    )
    return x, meta, feature_names


def test_warning_model_uses_only_history_and_current_values(tmp_path: Path) -> None:
    x, meta, feature_names = _sample_data()
    model = CausalWarningModel300s(feature_names, seed=11).fit(x, meta)
    before = model.predict_batch(x, meta)["abnormal_probability"]
    changed_targets = meta.copy()
    changed_targets["future_abnormal"] = 1 - changed_targets["future_abnormal"]
    changed_targets["future_sand_plug"] = 1 - changed_targets["future_sand_plug"]
    after = model.predict_batch(x, changed_targets)["abnormal_probability"]
    assert np.allclose(before, after)

    artifact = tmp_path / "warning.joblib"
    model.save(artifact)
    restored = CausalWarningModel300s.load(artifact)
    assert np.allclose(before, restored.predict_batch(x, meta)["abnormal_probability"])


def test_threshold_selection_meets_requested_recall() -> None:
    truth = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    score = np.array([0.90, 0.80, 0.70, 0.60, 0.65, 0.40, 0.20, 0.10])
    threshold, metrics = select_threshold(truth, score, target_recall=0.75)
    assert threshold == 0.70
    assert metrics["recall"] >= 0.75
    assert binary_metrics(truth, score, threshold)["false_positive"] == 0


def test_segment_split_keeps_segments_disjoint() -> None:
    rows = []
    for index in range(12):
        bucket = index % 3
        for _ in range(3):
            rows.append(
                {
                    "segment_id": f"S{index}",
                    "future_abnormal": int(bucket > 0),
                    "future_sand_plug": int(bucket == 2),
                }
            )
    meta = pd.DataFrame(rows)
    train, validation, test, groups = stratified_segment_split(meta, 0.70, 0.15, 7)
    assert len(train) and len(validation) and len(test)
    assert set(groups["train"]).isdisjoint(groups["validation"])
    assert set(groups["train"]).isdisjoint(groups["test"])
    assert set(groups["validation"]).isdisjoint(groups["test"])


def test_strict_event_metric_counts_each_event_once_at_horizon_boundary() -> None:
    meta = pd.DataFrame(
        {
            "segment_id": ["A"] * 7,
            "future_abnormal": [0, 1, 1, 1, 0, 1, 1],
        }
    )
    scores = np.array([0.1, 0.8, 0.2, 0.1, 0.1, 0.3, 0.9])
    metrics = strict_event_metrics(meta, scores, threshold=0.5, horizon_seconds=300.0)
    assert metrics["event_windows"] == 2
    assert metrics["detected_event_windows"] == 1
    assert metrics["recall"] == 0.5
