"""Build one registered pressure-only DT cache and its matching 3D view."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from App.build_dt_3d_realtime_html import build_html
from App.build_dt_realtime_cache import build_cache
from App.core.paths import PATHS
from App.data.dt_dataset_registry import get_dataset


def build_selected(dataset_id: str, frame_count: int = 120) -> dict:
    dataset = get_dataset(dataset_id)
    if dataset.get("adapter") != "raw_frac_construction":
        raise ValueError("当前入口仅用于无 DAS 独立施工井段；有 DAS 参考段使用已登记校正结果。")
    cache = PATHS.root / str(dataset["cache_source"])
    html = PATHS.root / str((dataset.get("html_by_scenario") or {})["no_das_pressure_only"])
    payload = build_cache(cache, dataset_id=dataset_id)
    build_html(cache, html, frame_count=frame_count, scenario_id="no_das_pressure_only")
    result = {
        "status": "completed",
        "dataset_id": dataset_id,
        "source": dataset.get("pressure_source"),
        "points": len(payload.get("timeline_s", [])),
        "cache": str(cache),
        "html": str(html),
    }
    manifest = cache.parent / "view_manifest.json"
    manifest.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--frame-count", type=int, default=120)
    args = parser.parse_args()
    try:
        print(json.dumps(build_selected(args.dataset_id, args.frame_count), ensure_ascii=False))
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        raise
