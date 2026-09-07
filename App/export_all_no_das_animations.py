"""Export one no-DAS fracture-evolution GIF for every registered single stage."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from App.data.dt_dataset_registry import list_datasets
from App.export_no_das_animation import export_gif


def _safe_name(value: str) -> str:
    text = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff._-]+", "_", str(value)).strip("._")
    return text or "single_stage"


def export_all(output_dir: Path, frame_count: int = 48, fps: int = 8, force: bool = False) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for dataset in list_datasets():
        if dataset.get("data_scope") == "composite":
            continue
        dataset_id = str(dataset["dataset_id"])
        cache_path = ROOT / str(dataset["cache_source"])
        if not cache_path.exists():
            results.append({"dataset_id": dataset_id, "status": "missing_cache"})
            continue
        if dataset_id == "jy84_z1_stage08":
            label = "焦页84-Z1_Stage08"
        else:
            label = str(dataset.get("stage_id") or dataset_id)
        output_path = (output_dir / f"无DAS_裂缝演变_{_safe_name(label)}.gif").resolve()
        try:
            if force or not output_path.exists() or output_path.stat().st_size < 1024:
                export_gif(cache_path, output_path, frame_count=frame_count, fps=fps)
            results.append({
                "dataset_id": dataset_id,
                "stage_id": label,
                "status": "ok",
                "output": str(output_path.relative_to(ROOT.resolve())),
                "size_bytes": output_path.stat().st_size,
            })
        except Exception as exc:  # pragma: no cover - data-dependent batch path
            results.append({
                "dataset_id": dataset_id,
                "stage_id": label,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            })
    report = output_dir / "no_das_animation_manifest.json"
    report.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export all single-stage no-DAS fracture evolution GIFs")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "app" / "no_das_ppt_delivery" / "all_stages"))
    parser.add_argument("--frame-count", type=int, default=48)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--force", action="store_true", help="重新生成已存在的动画")
    args = parser.parse_args()
    result = export_all(Path(args.output_dir), frame_count=args.frame_count, fps=args.fps, force=args.force)
    print(json.dumps(result, ensure_ascii=False, indent=2))
