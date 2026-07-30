from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from response_surrogate import ActionResponseSurrogate


def test_action_response_surrogate_trains_and_predicts_candidate_action() -> None:
    rng = np.random.default_rng(7)
    size = 80
    x = rng.normal(size=(size, 42))
    flow = rng.uniform(8.0, 18.0, size)
    sand = rng.uniform(0.0, 12.0, size)
    pressure = rng.uniform(55.0, 75.0, size)
    meta = pd.DataFrame({
        "current_flow": flow,
        "current_sand_ratio": sand,
        "current_pressure": pressure,
        "future_pressure_mean": pressure + 0.5 * (flow - 12.0) + 0.15 * sand,
        "future_pressure_max": pressure + 0.7 * (flow - 12.0) + 0.25 * sand,
        "future_abnormal": (sand > 8.0).astype(int),
        "future_sand_plug": ((sand > 9.5) & (flow > 14.0)).astype(int),
    })
    feature_names = [f"raw_{i}" for i in range(30)] + [
        f"{column}_{stat}" for column in ("SGBY", "PL", "SB") for stat in ("last", "mean", "std", "slope")
    ]
    bounds = {"PL": {"p01": 8.0, "p99": 18.0}, "SB": {"p01": 0.0, "p99": 12.0}}
    model = ActionResponseSurrogate(feature_names, bounds, seed=7).fit(x, meta, np.column_stack([flow, sand]))
    result = model.predict_one(x[0], meta.iloc[0], 16.0, 10.0)
    assert set(result) == {"pressure_mean", "pressure_max", "abnormal_probability", "sand_plug_probability", "ood_score"}
    assert all(np.isfinite(value) for value in result.values())
    assert 0.0 <= result["abnormal_probability"] <= 1.0

