"""Discover and validate DT well/stage datasets.

The registry deliberately separates source registration from the numerical
model.  A new stage can therefore be added by registering its files and field
mapping without changing the APP or PKN/pressure-only calculation code.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = ROOT / "App" / "config" / "dt_dataset_registry.json"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _relative(value: str | Path | None) -> str | None:
    if value in (None, ""):
        return None
    path = Path(str(value).replace("/", "\\"))
    if not path.is_absolute():
        path = ROOT / path
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _safe_id(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z_-]+", "_", value).strip("_").lower()
    return value or "dataset"


def _discover_raw_frac(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    discovery = config.get("raw_frac_discovery", {}) or {}
    if not discovery.get("enabled", False):
        return {}
    directory = ROOT / str(discovery.get("directory", "Data/raw_frac"))
    extensions = {str(item).lower() for item in discovery.get("include_extensions", [])}
    excluded = [str(item).lower() for item in discovery.get("exclude_name_contains", [])]
    # Raw exports are already supported by the positional fallback in the
    # pressure adapter.  Do not make every dataset load the 96 MB historical
    # workbook merely to borrow its column names; that made dataset discovery
    # and cache generation unnecessarily slow.  A small header workbook can
    # still be registered explicitly for a new source when needed.
    header_source = (
        _relative(discovery.get("reference_header_source"))
        if discovery.get("use_reference_header", False)
        else None
    )
    composite_markers = [
        str(item).lower()
        for item in discovery.get("composite_name_contains", ["便签数据"])
    ]
    result: dict[str, dict[str, Any]] = {}
    if not directory.exists():
        return result
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        lowered = path.name.lower()
        if any(item in lowered for item in excluded):
            continue
        dataset_id = f"raw_{_safe_id(path.stem)}"
        provenance = _read_json(path.with_suffix(".source.json")) if path.suffix.lower() == ".csv" else {}
        # A workbook such as 便签数据1(已自动还原).xlsx contains many
        # FDBH/井段 groups in one aggregate export.  It must not become a
        # selectable DT dataset: doing so makes a multi-stage aggregate look
        # like one physical stage in the APP.  Keep it out of the catalog
        # entirely instead of registering it and disabling it in the UI.
        if any(item and item in lowered for item in composite_markers):
            continue
        recorded_well = str(provenance.get("well_name") or "").strip() or None
        recorded_stage = str(provenance.get("stage_id") or "").strip() or path.stem
        display_name = (
            f"{recorded_well} · {recorded_stage}（单井段·无 DAS）"
            if recorded_well else f"{path.stem}（单井段·无 DAS 施工曲线）"
        )
        source_note = (
            "单个 FDBH 井段施工曲线；无 DAS 和轨迹时，仅生成压力校正驱动的阶段级 PKN/"
            "假设簇演变，簇级结果为模型估计。"
        )
        dataset_header_source = header_source
        result[dataset_id] = {
            "display_name": display_name,
            "adapter": "raw_frac_construction",
            "well_id": recorded_well,
            "stage_id": recorded_stage,
            "identity_source": provenance.get("identity_source") or "source_filename",
            "source_provenance": _relative(path.with_suffix(".source.json")) if provenance else None,
            "data_scope": "single_stage",
            "supported_scenarios": ["no_das_pressure_only"],
            "evolution_supported": True,
            "pressure_source": _relative(path),
            "reference_header_source": dataset_header_source,
            "fiber_source": None,
            "trajectory_source": None,
            "cluster_geometry_source": None,
            "pressure_calibration_source": "App/config/pressure_calibration.json",
            "cache_source": f"outputs/app/datasets/{dataset_id}/dt_realtime_cache.json",
            "hmi_recommendations_source": f"outputs/app/datasets/{dataset_id}/hmi_recommendations.csv",
            "html_by_scenario": {
                "no_das_pressure_only": f"outputs/app/datasets/{dataset_id}/dt_realtime_3d_no_das.html"
            },
            "source_start_s": 1,
            "source_end_s": None,
            "assumed_cluster_count": 6,
            "hmi_available": False,
            "calibration_status": "待校准",
            "source_note": source_note,
        }
    return result


@lru_cache(maxsize=8)
def _load_dataset_catalog_cached(path_value: str) -> dict[str, Any]:
    """Load a catalog once per process.

    Replay frame construction asks for the selected dataset at every replay
    node.  The previous implementation rediscovered every file in
    ``Data/raw_frac`` and resolved every path for every node, which turned a
    240-frame scenario switch into a several-second operation.  The registry
    is configuration, not per-frame state, so keeping the parsed catalog in a
    small process-local cache is safe.  ``clear_dataset_catalog_cache`` is
    provided for tooling that edits the registry while the app is running.
    """

    config = _read_json(Path(path_value))
    configured = config.get("datasets", {}) or {}
    datasets = {str(key): dict(value) for key, value in configured.items() if isinstance(value, dict)}
    datasets.update(_discover_raw_frac(config))
    return {
        "schema_version": config.get("schema_version", 1),
        "default_dataset_id": config.get("default_dataset_id", "jy84_z1_stage08"),
        "datasets": datasets,
    }


def clear_dataset_catalog_cache() -> None:
    _load_dataset_catalog_cached.cache_clear()


def load_dataset_catalog(path: Path | None = None) -> dict[str, Any]:
    selected_path = (path or REGISTRY_PATH).resolve()
    return _load_dataset_catalog_cached(str(selected_path))


def get_dataset(dataset_id: str | None = None, path: Path | None = None) -> dict[str, Any]:
    catalog = load_dataset_catalog(path)
    selected = dataset_id or catalog.get("default_dataset_id")
    datasets = catalog.get("datasets", {})
    if selected not in datasets:
        raise KeyError(f"unknown DT dataset: {selected}")
    result = dict(datasets[selected])
    result["dataset_id"] = selected
    return result


def list_datasets(path: Path | None = None) -> list[dict[str, Any]]:
    catalog = load_dataset_catalog(path)
    return [dict(value, dataset_id=key) for key, value in sorted(catalog.get("datasets", {}).items())]
