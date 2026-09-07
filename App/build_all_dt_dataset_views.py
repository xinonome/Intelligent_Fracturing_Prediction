"""Build no-DAS cache and 3D HTML for every registered single-stage dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "App"))

from App.build_dt_3d_realtime_html import build_html
from App.build_dt_realtime_cache import build_cache
from App.data.dt_dataset_registry import list_datasets


def build_all(frame_count: int = 120) -> list[dict]:
    results = []
    for dataset in list_datasets():
        if dataset.get("adapter") != "raw_frac_construction":
            continue
        dataset_id = dataset["dataset_id"]
        if dataset.get("data_scope") == "composite":
            results.append({"dataset_id": dataset_id, "status": "composite_skipped"})
            continue
        directory = ROOT / "outputs" / "app" / "datasets" / dataset_id
        cache = directory / "dt_realtime_cache.json"
        html = directory / "dt_realtime_3d_no_das.html"
        try:
            payload = build_cache(cache, dataset_id=dataset_id)
            build_html(cache, html, frame_count=frame_count, scenario_id="no_das_pressure_only")
            results.append({
                "dataset_id": dataset_id,
                "status": "ok",
                "rows": len(payload.get("timeline_s", [])),
                "source_end_s": payload.get("meta", {}).get("source_end_s"),
                "cache": str(cache.relative_to(ROOT)),
                "html": str(html.relative_to(ROOT)),
            })
        except Exception as exc:
            results.append({
                "dataset_id": dataset_id,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            })
    report = ROOT / "outputs" / "app" / "datasets" / "raw_dataset_view_build.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build all single-stage no-DAS DT views")
    parser.add_argument("--frame-count", type=int, default=120)
    args = parser.parse_args()
    print(json.dumps(build_all(args.frame_count), ensure_ascii=False, indent=2))
