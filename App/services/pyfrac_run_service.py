"""Prepare isolated, restart-from-initial native PyFrac runs for the APP."""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from ..core.paths import PATHS
from ..data.pyfrac_runtime_loader import REFERENCE_RUN, PyFracRuntime, load_pyfrac_runtime


WORKER_SCRIPT = PATHS.root / "DT-Crack" / "inversion" / "run_pyfrac_sigma_restart_inversion.py"
REFERENCE_SPEC = REFERENCE_RUN / "workers" / "candidate_00" / "spec.json"
ALGORITHM_PYTHON = Path(
    os.environ.get("FRACTURING_ALGORITHM_PYTHON", r"C:\Users\xinonome\anaconda3\python.exe")
)
REFERENCE_DATASET_ID = "jy84_z1_stage08"
REFERENCE_PRESSURE_SOURCE = "Data/3Dfrac/JY84-Z1-stage08-f1.xls"


def native_context_reason(dataset: dict[str, Any] | None) -> str:
    """Fail closed: the verified injection schedule belongs to one stage only."""
    dataset = dataset or {}
    if (dataset.get("dataset_id") != REFERENCE_DATASET_ID
            or dataset.get("well_id") != "JY84-Z1"
            or str(dataset.get("stage_id")) != "08"
            or dataset.get("adapter") != "stage_3dfrac"):
        return "当前井段暂无已登记的 PyFrac 输入方案。"
    source = PATHS.root / str(dataset.get("pressure_source", ""))
    if source.resolve() != (PATHS.root / REFERENCE_PRESSURE_SOURCE).resolve():
        return "当前井段与已登记的 PyFrac 输入方案不匹配。"
    try:
        spec = json.loads(REFERENCE_SPEC.read_text(encoding="utf-8"))
        summary = json.loads((REFERENCE_RUN / "summary.json").read_text(encoding="utf-8"))
        schedule = Path(spec["schedule_path"])
        declared = Path(summary["outputs"]["injection_schedule"])
        pressure = PATHS.root / summary["pressure_meta"]["source"]
        if schedule.resolve() != declared.resolve() or pressure.resolve() != source.resolve():
            return "PyFrac模板、注入历史与来源摘要不一致。"
        if not source.is_file() or not schedule.is_file():
            return "已验证压力源或注入历史文件缺失。"
    except (OSError, ValueError, KeyError, TypeError):
        return "无法核验PyFrac模板的井段来源与注入历史。"
    return ""


def native_run_reason(dataset: dict[str, Any] | None) -> str:
    reason = native_context_reason(dataset)
    if reason:
        return reason
    if not WORKER_SCRIPT.is_file():
        return "原生PyFrac运行脚本缺失。"
    if not ALGORITHM_PYTHON.is_file():
        return "已配置的算法Python不存在；不自动改用未经验证的运行环境。"
    return ""


def load_context_runtime(dataset: dict[str, Any] | None) -> PyFracRuntime:
    reason = native_context_reason(dataset)
    if reason:
        return PyFracRuntime(note=reason)
    # Never use the loader's global 'latest run' fallback across datasets.
    directory = PATHS.app_outputs / "pyfrac_runs"
    if directory.is_dir():
        for candidate in sorted(directory.iterdir(), reverse=True):
            try:
                manifest = json.loads((candidate / "manifest.json").read_text(encoding="utf-8"))
                manifest_dataset_id = manifest.get("dataset_id")
                if manifest_dataset_id != dataset["dataset_id"] and not _legacy_reference_manifest(manifest, dataset):
                    continue
                # The registered application task is the full 4435 s
                # trajectory. Short diagnostics remain visible in the
                # current session but must not replace the default run after
                # reopening the workbench.
                if not math.isclose(float(manifest.get("target_time_s", 0.0)), 4435.0, abs_tol=1e-3):
                    continue
                value = load_pyfrac_runtime(candidate)
                if native_run_completed(value, float(manifest["target_time_s"])):
                    return value
            except (OSError, ValueError, KeyError, TypeError):
                continue
    return load_pyfrac_runtime(REFERENCE_RUN)


def _legacy_reference_manifest(manifest: dict[str, Any], dataset: dict[str, Any]) -> bool:
    """Accept the first APP run written before dataset metadata was added.

    This exception is deliberately limited to the verified JY84-Z1 / Stage 08
    context and the original 4435 s template.  It lets the APP reuse the
    already completed detailed run (including its persisted front geometry)
    without borrowing a run from another well or stage.
    """

    if dataset.get("dataset_id") != REFERENCE_DATASET_ID:
        return False
    if manifest.get("dataset_id") is not None:
        return False
    try:
        target_time_s = float(manifest.get("target_time_s"))
    except (TypeError, ValueError):
        return False
    if not math.isclose(target_time_s, 4435.0, abs_tol=1e-3):
        return False
    template = manifest.get("source_template")
    if not template:
        return False
    try:
        return Path(template).resolve() == REFERENCE_SPEC.resolve()
    except OSError:
        return False


def native_run_completed(runtime: PyFracRuntime, target_time_s: float) -> bool:
    return bool(runtime.completed and runtime.frames and
                math.isclose(runtime.frames[-1].time_s, target_time_s, abs_tol=1e-3))


def prepare_native_run(sigma_min_mpa: float, target_time_s: float, *,
                       dataset: dict[str, Any] | None = None) -> tuple[Path, list[str], dict[str, str]]:
    """Create a new run directory without mutating any previous PyFrac state."""

    reason = native_run_reason(dataset)
    if reason:
        raise ValueError(reason)
    if not math.isfinite(sigma_min_mpa) or not 20.0 <= sigma_min_mpa <= 120.0:
        raise ValueError("最小水平应力须为20–120 MPa的有限数值。")
    if not math.isfinite(target_time_s) or not 1.0 < target_time_s <= 4435.0:
        raise ValueError("目标时间须大于原生初始时间1 s且不超过已验证注入历史4435 s。")
    spec = json.loads(REFERENCE_SPEC.read_text(encoding="utf-8"))
    run_root = PATHS.app_outputs / "pyfrac_runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_root.mkdir(parents=True, exist_ok=False)
    result_path = run_root / "result.json"
    progress_path = run_root / "progress.json"
    history_path = run_root / "progress_history.jsonl"
    schedule_path = run_root / "injection_schedule.npy"
    shutil.copy2(spec["schedule_path"], schedule_path)
    spec.update(
        {
            "candidate_id": "app_native_run",
            "result_path": str(result_path),
            "sigma_min_mpa": float(sigma_min_mpa),
            "target_time_s": float(target_time_s),
            "dataset_id": dataset["dataset_id"],
            "well_id": dataset["well_id"],
            "stage_id": dataset["stage_id"],
            "pressure_source": dataset["pressure_source"],
            "schedule_path": str(schedule_path),
            "project_root": str(PATHS.root),
        }
    )
    if isinstance(spec.get("pyfrac_config"), dict):
        spec["pyfrac_config"]["min_horizontal_stress_pa"] = float(sigma_min_mpa) * 1.0e6
    spec_path = run_root / "spec.json"
    spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    interpreter = ALGORITHM_PYTHON
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "native_restart_from_initial",
        "dataset_id": dataset["dataset_id"],
        "well_id": dataset["well_id"],
        "stage_id": dataset["stage_id"],
        "pressure_source": dataset["pressure_source"],
        "schedule_path": str(schedule_path),
        "source_template": str(REFERENCE_SPEC),
        "worker_script": str(WORKER_SCRIPT),
        "algorithm_python": str(interpreter),
        "sigma_min_mpa": float(sigma_min_mpa),
        "target_time_s": float(target_time_s),
        "state_reuse": False,
        "result_path": str(result_path),
        "progress_path": str(progress_path),
        "progress_history_path": str(history_path),
    }
    (run_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    arguments = [str(WORKER_SCRIPT), "--worker-spec", str(spec_path)]
    environment = {
        "PYFRAC_PROGRESS_FILE": str(progress_path),
        "PYFRAC_PROGRESS_HISTORY_FILE": str(history_path),
        "PYTHONPATH": str(PATHS.root / "DT-Crack"),
    }
    return run_root, [str(interpreter), *arguments], environment


def save_parameter_scheme(path: str | Path, values: dict[str, Any]) -> None:
    payload = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "PyFrac native restart from initial state",
        "parameters": values,
    }
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


__all__ = [
    "prepare_native_run",
    "save_parameter_scheme",
    "WORKER_SCRIPT",
    "REFERENCE_SPEC",
    "ALGORITHM_PYTHON",
    "native_context_reason", "native_run_reason", "load_context_runtime", "native_run_completed",
]
