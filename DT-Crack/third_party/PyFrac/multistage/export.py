from __future__ import annotations

from pathlib import Path

import pandas as pd


def export_combined(results: dict, output_dir: Path) -> None:
    combined = output_dir / "combined"
    combined.mkdir(exist_ok=True)
    metrics = []
    fronts = []
    handovers = []
    for stage_id, result in results.items():
        metrics.append(result.metrics)
        front = pd.read_csv(output_dir / stage_id / "final_front_global.csv")
        fronts.append(front)
        handovers.append({"stage_id": stage_id, "handover_time_s": result.metadata.get("handover_time_s")})
    if metrics:
        pd.concat(metrics, ignore_index=True).to_csv(combined / "all_stages_metrics.csv", index=False)
        pd.concat(fronts, ignore_index=True).to_csv(combined / "all_stages_final_front_global.csv", index=False)
        pd.DataFrame(handovers).to_csv(combined / "handover_points.csv", index=False)
