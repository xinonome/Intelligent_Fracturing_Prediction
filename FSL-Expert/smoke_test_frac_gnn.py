from __future__ import annotations

import tempfile
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

from frac_gnn.train import run_experiment


def build_synthetic_segment(segment_id: str, points: int) -> pd.DataFrame:
    rng = np.random.default_rng(abs(hash(segment_id)) % (2**32))
    rows = []
    working_types = ["正常", "砂堵", "正常", "其他"]
    for index in range(points):
        rows.append(
            {
                "segment_id": segment_id,
                "timestamp": index * 10,
                "sand_ratio": 0.05 + rng.normal(0.0, 0.005),
                "pump_pressure": 100 + index * 0.5 + rng.normal(0.0, 1.0),
                "displacement": 7.5 + rng.normal(0.0, 0.1),
                "viscosity": 50 + rng.normal(0.0, 2.0),
                "MARKS": f"note-{index}",
                "DAY_TIME": "day",
                "WORKING_TYPE": working_types[index % len(working_types)],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        for idx in range(10):
            frame = build_synthetic_segment(f"segment_{idx:02d}", 12)
            frame.to_csv(root / f"segment_{idx:02d}.csv", index=False)

        args = Namespace(
            data_path=str(root),
            label_column="WORKING_TYPE",
            segment_column=None,
            time_column="timestamp",
            include_columns=None,
            exclude_columns=None,
            reference_header_path=None,
            exclude_name_patterns=None,
            normal_label="正常",
            binary_normal_abnormal=True,
            abnormal_label="异常",
            window_sizes=[1, 2],
            train_ratio=0.7,
            test_ratio=0.1,
            val_ratio=0.1,
            transfer_ratio=0.1,
            hidden_dim=16,
            num_layers=2,
            dropout=0.1,
            epochs=2,
            batch_size=16,
            learning_rate=1e-3,
            weight_decay=1e-4,
            disable_class_weights=False,
            class_weight_exponent=0.5,
            class_weight_max_ratio=3.0,
            oversample_minority=True,
            patience=2,
            seed=42,
            run_dir=str(root / "runs"),
            device="cpu",
        )
        overview = run_experiment(args)
        assert "1" in overview["window_results"]
        assert "2" in overview["window_results"]
        print("Smoke test passed")


if __name__ == "__main__":
    main()
