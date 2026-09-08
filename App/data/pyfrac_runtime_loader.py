"""Read real native-PyFrac progress histories for the desktop workbench."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.paths import PATHS


REFERENCE_RUN = (
    PATHS.root
    / "outputs"
    / "dt"
    / "pyfrac_sigma_4435_volume_projection_refine1125_20260830_r1"
)


@dataclass
class PyFracRuntimeFrame:
    time_s: float
    successful_step: int
    time_step_s: float | None = None
    half_length_m: float | None = None
    fracture_height_m: float | None = None
    max_aperture_mm: float | None = None
    mean_net_pressure_mpa: float | None = None
    max_net_pressure_mpa: float | None = None
    injected_volume_m3: float | None = None
    fracture_volume_m3: float | None = None
    leakoff_volume_m3: float | None = None
    crack_cells: int | None = None
    tip_cells: int | None = None
    front_geometry: list[list[float]] = field(default_factory=list)
    pressure_field: list[dict[str, float]] = field(default_factory=list)


@dataclass
class PyFracRuntime:
    available: bool = False
    run_root: Path | None = None
    frames: list[PyFracRuntimeFrame] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
    note: str = ""

    @property
    def point_count(self) -> int:
        return len(self.frames)

    @property
    def target_time_s(self) -> float | None:
        if self.frames:
            return self.frames[-1].time_s
        return _number(self.parameters.get("target_time_s"))

    @property
    def completed(self) -> bool:
        return bool(self.result.get("success") and self.result.get("target_reached"))


def load_pyfrac_runtime(run_root: str | Path | None = None) -> PyFracRuntime:
    root = Path(run_root).resolve() if run_root else (_latest_completed_run() or REFERENCE_RUN.resolve())
    if not root.exists():
        return PyFracRuntime(run_root=root, note=f"原生运行目录不存在：{root}")

    history_path = _first_existing(
        root / "progress_history.jsonl",
        root / "workers" / "candidate_00" / "progress_history.jsonl",
        root / "workers" / "posterior_validation" / "progress_history.jsonl",
    )
    spec_path = _first_existing(
        root / "spec.json",
        root / "workers" / "candidate_00" / "spec.json",
        root / "workers" / "posterior_validation" / "spec.json",
    )
    result_path = _first_existing(
        root / "result.json",
        root / "workers" / "candidate_00" / "result.json",
        root / "workers" / "posterior_validation" / "result.json",
    )
    progress_path = _first_existing(
        root / "progress.json",
        root / "workers" / "candidate_00" / "progress.json",
        root / "workers" / "posterior_validation" / "progress.json",
    )
    parameters = _read_json(spec_path) if spec_path else {}
    result_payload = _read_json(result_path) if result_path else {}
    result = result_payload.get("result", result_payload) if isinstance(result_payload, dict) else {}
    progress = _read_json(progress_path) if progress_path else {}
    if history_path is None:
        # Very short native runs can complete before the worker writes a
        # heartbeat history.  Their final result is still a genuine PyFrac
        # state; expose that single persisted point instead of reporting an
        # empty run.  Do not infer any intermediate points.
        final_time = _number(result.get("final_time_s")) if isinstance(result, dict) else None
        if final_time is not None and result.get("success"):
            frame = PyFracRuntimeFrame(
                time_s=final_time,
                successful_step=_integer(result.get("successful_time_steps")) or 1,
                half_length_m=_number(result.get("half_length_m")),
                fracture_height_m=_number(result.get("fracture_height_m")),
                max_aperture_mm=_number(result.get("max_aperture_mm")),
                mean_net_pressure_mpa=_number(result.get("net_pressure_mpa")),
                max_net_pressure_mpa=_number(result.get("net_pressure_mpa")),
                injected_volume_m3=_number(result.get("injected_volume_m3")),
                fracture_volume_m3=_number(result.get("fracture_volume_m3")),
                leakoff_volume_m3=_number(result.get("leakoff_volume_m3")),
                front_geometry=_points(result.get("front_geometry")),
                pressure_field=_field_points(result.get("pressure_field")),
            )
            return PyFracRuntime(
                available=True,
                run_root=root,
                frames=[frame],
                parameters=parameters,
                result=result,
                note="本次短时原生运行仅保存了真实终点；未生成可回放的中间点。",
            )
        return PyFracRuntime(run_root=root, parameters=parameters, result=result, note="运行未保存内部计算点。")

    frames: list[PyFracRuntimeFrame] = []
    last_step = 0
    try:
        lines = history_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return PyFracRuntime(run_root=root, note=f"进度历史读取失败：{exc}")
    for line in lines:
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        step = _integer(row.get("successful_time_steps")) or 0
        # One accepted native point is followed by a second heartbeat for the
        # next attempt. Keep only the first record for each newly accepted step.
        if row.get("phase") != "time_marching" or step <= last_step:
            continue
        time_s = _number(row.get("time_s"))
        if time_s is None:
            continue
        ledger = row.get("volume_ledger") if isinstance(row.get("volume_ledger"), dict) else {}
        frames.append(
            PyFracRuntimeFrame(
                time_s=time_s,
                successful_step=step,
                time_step_s=_number(row.get("last_accepted_delta_time_s")),
                half_length_m=_number(row.get("half_length_m")),
                fracture_height_m=_number(row.get("fracture_height_m")),
                max_aperture_mm=_number(row.get("max_aperture_mm")),
                mean_net_pressure_mpa=_number(row.get("mean_net_pressure_mpa")),
                max_net_pressure_mpa=_number(row.get("max_net_pressure_mpa")),
                injected_volume_m3=_number(ledger.get("injected_volume_m3")),
                fracture_volume_m3=_number(ledger.get("fracture_volume_m3")),
                leakoff_volume_m3=_number(ledger.get("leakoff_volume_m3")),
                crack_cells=_integer(row.get("crack_cells")),
                tip_cells=_integer(row.get("tip_cells")),
                front_geometry=_points(row.get("front_geometry")),
                pressure_field=_field_points(row.get("pressure_field")),
            )
        )
        last_step = step

    # Older runs did not include field/front data in every heartbeat.  The
    # final progress/result payload is still a real native output, so merge it
    # only into the final internal point.  Intermediate points remain empty
    # when PyFrac did not persist them; never interpolate or invent geometry.
    if frames and isinstance(result, dict):
        final = frames[-1]
        final.half_length_m = final.half_length_m or _number(result.get("half_length_m"))
        final.fracture_height_m = final.fracture_height_m or _number(result.get("fracture_height_m"))
        final.max_aperture_mm = final.max_aperture_mm or _number(result.get("max_aperture_mm"))
        final.mean_net_pressure_mpa = final.mean_net_pressure_mpa or _number(result.get("net_pressure_mpa"))
        final.max_net_pressure_mpa = final.max_net_pressure_mpa or _number(result.get("net_pressure_mpa"))
        final.front_geometry = (
            final.front_geometry
            or _points(progress.get("front_geometry"))
            or _points(result.get("front_geometry"))
        )
        final.pressure_field = (
            final.pressure_field
            or _field_points(progress.get("pressure_field"))
            or _field_points(result.get("pressure_field"))
        )

    if frames and isinstance(progress, dict):
        final = frames[-1]
        final.injected_volume_m3 = final.injected_volume_m3 or _number(
            (progress.get("volume_ledger") or {}).get("injected_volume_m3")
        )
        final.fracture_volume_m3 = final.fracture_volume_m3 or _number(
            (progress.get("volume_ledger") or {}).get("fracture_volume_m3")
        )
        final.leakoff_volume_m3 = final.leakoff_volume_m3 or _number(
            (progress.get("volume_ledger") or {}).get("leakoff_volume_m3")
        )

    if not frames:
        return PyFracRuntime(
            run_root=root,
            parameters=parameters,
            result=result if isinstance(result, dict) else {},
            note="进度历史中没有成功接受的原生内部计算点。",
        )
    detailed = sum(bool(frame.pressure_field) and bool(frame.front_geometry) for frame in frames)
    front_points = sum(bool(frame.front_geometry) for frame in frames)
    field_points = sum(bool(frame.pressure_field) for frame in frames)
    note = f"真实原生进度历史：{len(frames)} 个成功内部计算点"
    if detailed:
        note += f"；{detailed} 个点包含压力场与裂缝前缘"
    elif front_points:
        note += f"；{front_points} 个点包含裂缝前缘，{field_points} 个点包含压力场"
    else:
        note += "；历史运行仅在终点保存几何，重新推演后将逐点保存压力场"
    return PyFracRuntime(
        available=True,
        run_root=root,
        frames=frames,
        parameters=parameters,
        result=result if isinstance(result, dict) else {},
        note=note,
    )


def _first_existing(*paths: Path) -> Path | None:
    return next((path for path in paths if path.is_file()), None)


def _latest_completed_run() -> Path | None:
    run_root = PATHS.app_outputs / "pyfrac_runs"
    if not run_root.is_dir():
        return None
    for candidate in sorted((path for path in run_root.iterdir() if path.is_dir()), reverse=True):
        payload = _read_json(candidate / "result.json")
        result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
        # The default workbench represents the registered 4435 s task. Short
        # diagnostic reruns remain loadable by explicit path but must not
        # silently replace that full reference trajectory.
        if (result.get("success") and result.get("target_reached")
                and (_number(result.get("final_time_s")) or 0.0) >= 4435.0):
            return candidate.resolve()
    return None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _integer(value: Any) -> int | None:
    number = _number(value)
    return int(number) if number is not None else None


def _points(value: Any) -> list[list[float]]:
    if not isinstance(value, list):
        return []
    output = []
    for item in value:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            x, y = _number(item[0]), _number(item[1])
            if x is not None and y is not None:
                output.append([x, y])
    return output


def _field_points(value: Any) -> list[dict[str, float]]:
    if not isinstance(value, list):
        return []
    output = []
    for item in value:
        if not isinstance(item, dict):
            continue
        x, y = _number(item.get("x_m")), _number(item.get("y_m"))
        pressure = _number(item.get("net_pressure_mpa"))
        width = _number(item.get("width_mm"))
        if x is None or y is None:
            continue
        output.append({"x_m": x, "y_m": y, "net_pressure_mpa": pressure or 0.0, "width_mm": width or 0.0})
    return output


__all__ = ["PyFracRuntime", "PyFracRuntimeFrame", "REFERENCE_RUN", "load_pyfrac_runtime"]
