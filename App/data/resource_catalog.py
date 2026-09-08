"""Build the operator-visible inventory of real models and run artifacts.

The resource center used to expose only a few logical model labels.  This
module keeps the inventory registry-first, but also discovers model files and
the small set of application run directories that are actually present on
disk.  No readiness or performance claim is inferred from a filename.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..core.paths import PATHS, relative
from .hmi_loader import discover_agent_models


_MODEL_SUFFIXES = {".pt", ".pth", ".pkl", ".joblib", ".npz", ".zip"}
_MODEL_NAME_MARKERS = (
    "model",
    "policy",
    "surrogate",
    "fracturing_policy",
    "transfer_package",
)
_KIND_ORDER = {"模型": 0, "运行结果": 1, "缓存": 2}


def _project_path(path: Path) -> str:
    value = relative(path)
    return str(value or path)


def _size_text(path: Path) -> str:
    try:
        size = path.stat().st_size
    except OSError:
        return "--"
    if size >= 1024 * 1024:
        return f"{size / 1024 / 1024:.1f} MB"
    if size >= 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size} B"


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _same_path(left: Path | None, right: Path | None) -> bool:
    if left is None or right is None:
        return False
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return left == right


def _module_paths(module: dict[str, Any]) -> list[Path]:
    """Resolve registered files without requiring every field to exist."""

    values: list[str] = []

    def add(value: Any) -> None:
        if isinstance(value, (str, Path)) and str(value).strip():
            values.append(str(value))

    for key in ("summary", "frame_source", "policy_model", "aggregate_comparison", "html"):
        add(module.get(key))
    for value in (module.get("html_by_scenario", {}) or {}).values():
        add(value)
    for key in ("figures", "tables"):
        for value in module.get(key, []) or []:
            add(value.get("path") if isinstance(value, dict) else value)
    for key in ("figures", "tables"):
        for value in (module.get("files", {}) or {}).get(key, []) or []:
            add(value.get("path") if isinstance(value, dict) else value)
    for value in (module.get("supporting_summaries", {}) or {}).values():
        add(value)
    paths: list[Path] = []
    seen: set[str] = set()
    for value in values:
        path = PATHS.root / value if not Path(value).is_absolute() else Path(value)
        key = str(path).lower()
        if key in seen:
            continue
        seen.add(key)
        if path.exists():
            paths.append(path)
    return paths


def _module_label(module_name: str) -> tuple[str, str]:
    if module_name == "fsl":
        return "模型", "工况识别、风险预测与跨井迁移"
    if module_name == "dt":
        return "模型", "PKN / KG-EnKF 压力与裂缝状态演化"
    return "模型", "智能调控离线回放与建议"


def _add_row(
    rows: list[dict[str, Any]],
    seen: set[str],
    *,
    kind: str,
    name: str,
    status: str,
    purpose: str,
    context: str,
    path: Path,
    registered_path: Path | None = None,
) -> None:
    if not path.exists():
        return
    try:
        key = str(path.resolve()).lower()
    except OSError:
        key = str(path).lower()
    if key in seen:
        return
    seen.add(key)
    final_status = status
    if registered_path is not None and _same_path(path, registered_path):
        final_status = "当前注册"
    rows.append(
        {
            "kind": kind,
            "name": name,
            "status": final_status,
            "purpose": purpose,
            "context": context,
            "size": _size_text(path),
            "path": str(path),
            "display_path": _project_path(path),
            "mtime": _mtime(path),
        }
    )


def _classify_model(path: Path) -> tuple[str, str, str]:
    text = str(path).lower().replace("\\", "/")
    name = path.name
    if "transfer" in text:
        return "预测模型", "跨井迁移模型", "迁移训练 / 目标井应用"
    if "/artifacts/fsl/" in text or "/outputs/fsl/" in text:
        return "预测模型", "工况识别与风险预测", "多规则标注 / 逐点预测"
    if "/outputs/dt/" in text or "/artifacts/dt/" in text or "/dt-crack/" in text:
        return "数字孪生模型", "PKN / KG-EnKF / PyFrac", "压力换算、参数更新与裂缝演化"
    if "/hmi- ke/" in text or "/hmi-ke/" in text or "/hmi/" in text:
        algorithm = next((item.upper() for item in ("sac", "td3", "ppo") if item in text), "代理")
        return "智能体模型", algorithm, "智能决策离线回放"
    return "模型", name, "项目模型资源"


def _scan_model_files() -> list[tuple[Path, tuple[str, str, str]]]:
    roots = [
        PATHS.artifacts,
        PATHS.app_outputs / "transfer_runs",
        PATHS.outputs / "dt",
        PATHS.outputs / "hmi",
        PATHS.root / "DT-Crack" / "forward_models" / "cache",
        PATHS.root / "HMI-KE" / "outputs" / "hmi",
        PATHS.root / "HMI-KE" / "runs",
    ]
    found: list[tuple[Path, tuple[str, str, str]]] = []
    seen: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        try:
            iterator = root.rglob("*")
            for path in iterator:
                if not path.is_file() or path.suffix.lower() not in _MODEL_SUFFIXES:
                    continue
                lower_name = path.name.lower()
                lower_path = str(path).lower().replace("\\", "/")
                if "checkpoints" in path.parts and path.suffix.lower() == ".zip":
                    continue
                if lower_name == "shared_frame_cache.pkl" or lower_name == "frames.pkl":
                    continue
                if path.suffix.lower() == ".zip" and "fracturing_policy" not in lower_name:
                    continue
                if path.suffix.lower() != ".zip" and not any(marker in lower_name or marker in lower_path for marker in _MODEL_NAME_MARKERS):
                    continue
                key = str(path.resolve()).lower()
                if key in seen:
                    continue
                seen.add(key)
                found.append((path, _classify_model(path)))
        except OSError:
            continue
    return found


def _add_registered_module_rows(rows: list[dict[str, Any]], seen: set[str], registry) -> None:
    for module_name in ("fsl", "dt", "hmi"):
        module = registry.module(module_name)
        kind, purpose = _module_label(module_name)
        registered_model = registry.path(module.get("policy_model")) if hasattr(registry, "path") else None
        if module_name == "fsl":
            name = "FSL · 多规则融合工况识别与风险预测"
            context = "注册模块 / 结果与规则"
        elif module_name == "dt":
            name = "DT · PKN / KG-EnKF"
            context = "有 DAS / 无 DAS 双场景"
        else:
            name = "HMI · 当前智能体回放"
            context = str(module.get("active_policy") or "运行时选择")
        summary = registry.path(module.get("summary")) if hasattr(registry, "path") else None
        anchor = registered_model or summary
        if anchor and anchor.exists():
            _add_row(
                rows,
                seen,
                kind=kind,
                name=name,
                status="已登记·结果可用",
                purpose=purpose,
                context=context,
                path=anchor,
                registered_path=registered_model,
            )
        for path in _module_paths(module):
            if path == anchor:
                continue
            suffix = path.suffix.lower()
            result_kind = "缓存" if "cache" in path.name.lower() else "运行结果"
            _add_row(
                rows,
                seen,
                kind=result_kind,
                name=f"{module_name.upper()} · {path.name}",
                status="已生成",
                purpose=purpose,
                context=context,
                path=path,
                registered_path=registered_model if suffix in _MODEL_SUFFIXES else None,
            )


def _add_application_runs(rows: list[dict[str, Any]], seen: set[str]) -> None:
    # A run is represented by its result file where possible.  Progress files
    # are intentionally not shown as separate inventory items.
    for root, label, purpose in (
        (PATHS.app_outputs / "pyfrac_runs", "PyFrac", "PyFrac 原生推演与内部计算点"),
        (PATHS.app_outputs / "transfer_runs", "迁移", "跨井迁移训练、应用与结果"),
    ):
        if not root.exists():
            continue
        for run in sorted((item for item in root.iterdir() if item.is_dir()), key=_mtime, reverse=True):
            result = run / ("result.json" if label == "PyFrac" else "transfer_result.json")
            manifest = run / "manifest.json"
            anchor = result if result.exists() else manifest if manifest.exists() else run
            if not anchor.exists():
                continue
            status = "结果可用" if result.exists() else "运行记录"
            _add_row(
                rows,
                seen,
                kind="运行结果",
                name=f"{label} · {run.name}",
                status=status,
                purpose=purpose,
                context=_project_path(run),
                path=anchor,
            )


def _add_application_caches(rows: list[dict[str, Any]], seen: set[str]) -> None:
    direct = (
        (PATHS.app_outputs / "fsl_timeline_cache.json", "FSL · 施工曲线与事件缓存", "施工曲线、逐点预测与事件边界"),
        (PATHS.app_outputs / "dt_realtime_cache.json", "DT · 实时数字孪生缓存", "压力、参数与裂缝演化回放"),
        (PATHS.app_outputs / "dt_realtime_cache_registry_check.json", "DT · 缓存注册检查", "缓存来源与完整性检查"),
        (PATHS.app_outputs / "no_das_agent_cache_manifest.json", "HMI · 无 DAS 建议缓存清单", "无 DAS 场景的离线建议缓存"),
    )
    for path, name, purpose in direct:
        _add_row(rows, seen, kind="缓存", name=name, status="可读取", purpose=purpose, context="App/outputs", path=path)
    datasets_root = PATHS.app_outputs / "datasets"
    if datasets_root.exists():
        for path in sorted(datasets_root.glob("*/dt_realtime_cache.json"), key=_mtime, reverse=True):
            _add_row(
                rows,
                seen,
                kind="缓存",
                name=f"DT · {path.parent.name} 缓存",
                status="可读取",
                purpose="独立井段数字孪生回放",
                context=_project_path(path.parent),
                path=path,
            )


def collect_resource_inventory(registry) -> list[dict[str, Any]]:
    """Return real model, result and cache rows for the resource center."""

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    _add_registered_module_rows(rows, seen, registry)

    runtime = registry.runtime_selection() if hasattr(registry, "runtime_selection") else {}
    registered_policy = registry.path(runtime.get("policy_model_path")) if hasattr(registry, "path") else None
    try:
        agent_models = discover_agent_models(registry)
    except Exception:
        agent_models = []
    for item in agent_models:
        policy = registry.path(item.get("policy_path")) if hasattr(registry, "path") else None
        evaluation = registry.path(item.get("evaluation_path")) if hasattr(registry, "path") else None
        if policy and policy.exists():
            _add_row(
                rows,
                seen,
                kind="模型",
                name=f"{item.get('display_name') or item.get('model_id') or '--'} · 策略包",
                status="可回放",
                purpose="智能调控离线回放",
                context=f"seed {item.get('seed') or '--'} / {item.get('run_dir') or '--'}",
                path=policy,
                registered_path=registered_policy,
            )
        if evaluation and evaluation.exists():
            _add_row(
                rows,
                seen,
                kind="运行结果",
                name=f"{item.get('display_name') or item.get('model_id') or '--'} · 回放结果",
                status="可读取",
                purpose="智能调控离线回放结果",
                context=f"seed {item.get('seed') or '--'} / 评价表",
                path=evaluation,
            )

    for path, (kind, name, purpose) in _scan_model_files():
        _add_row(
            rows,
            seen,
            kind="模型",
            name=f"{name} · {path.name}",
            status="已生成",
            purpose=purpose,
            context=_project_path(path.parent),
            path=path,
            registered_path=registered_policy,
        )

    _add_application_runs(rows, seen)
    _add_application_caches(rows, seen)
    rows.sort(key=lambda item: (_KIND_ORDER.get(item["kind"], 9), -float(item.get("mtime", 0)), item["name"]))
    return rows


__all__ = ["collect_resource_inventory"]
