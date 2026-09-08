"""Load the registered, offline native-PyFrac comparison artifact.

The acceptance APP is a frozen/replay application.  It must not launch a
multi-minute PyFrac solve while a user is looking at the online page.  This
loader therefore reads a checked-in comparison CSV produced by the offline
runner and exposes only successful native dynamic checkpoints.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PyFracComparisonPoint:
    time_s: float
    pkn_half_length_m: float | None
    pyfrac_half_length_m: float | None
    pkn_max_aperture_mm: float | None
    pyfrac_max_aperture_mm: float | None
    pkn_net_pressure_mpa: float | None
    pyfrac_net_pressure_mpa: float | None
    pkn_bottomhole_pressure_mpa: float | None
    pyfrac_bottomhole_pressure_mpa: float | None
    pyfrac_volume_m3: float | None
    pyfrac_runtime_s: float | None
    successful_time_steps: int | None
    final_time_s: float | None


@dataclass
class PyFracComparison:
    available: bool = False
    source: str = ""
    engine_mode: str = ""
    stage: str = "08"
    cluster_id: int = 1
    points: list[PyFracComparisonPoint] = field(default_factory=list)
    note: str = ""

    @property
    def latest(self) -> PyFracComparisonPoint | None:
        return self.points[-1] if self.points else None


def load_pyfrac_comparison(registry: Any) -> PyFracComparison:
    """Read the registry's native-PyFrac comparison CSV.

    Failed native checkpoints are intentionally excluded from plotted lines,
    but the note reports how many failed rows were present in the artifact.
    """

    module = registry.module("dt")
    configured = module.get("pyfrac_comparison")
    path = registry.path(configured) if configured else None
    if path is None:
        path = Path(__file__).resolve().parents[2] / "outputs" / "dt" / "pyfrac_native_single_cluster_native_ok" / "single_cluster_comparison.csv"
    if not path.exists():
        return PyFracComparison(note=f"对比文件未找到：{path}")

    rows: list[PyFracComparisonPoint] = []
    failed_count = 0
    raw_rows: list[dict[str, str]] = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            raw_rows = list(csv.DictReader(handle))
    except (OSError, csv.Error) as exc:
        return PyFracComparison(note=f"对比文件读取失败：{type(exc).__name__}: {exc}")

    for row in raw_rows:
        if _boolean(row.get("pyfrac_success")) is not True or row.get("pyfrac_engine_mode") != "pyfrac_native_dynamic":
            failed_count += 1
            continue
        time_s = _number(row.get("time_s"))
        if time_s is None:
            continue
        rows.append(
            PyFracComparisonPoint(
                time_s=time_s,
                pkn_half_length_m=_number(row.get("pkn_half_length_m")),
                pyfrac_half_length_m=_number(row.get("pyfrac_half_length_m")),
                pkn_max_aperture_mm=_number(row.get("pkn_max_aperture_mm")),
                pyfrac_max_aperture_mm=_number(row.get("pyfrac_max_aperture_mm")),
                pkn_net_pressure_mpa=_number(row.get("net_pressure_mpa")),
                pyfrac_net_pressure_mpa=_number(row.get("pyfrac_net_pressure_mpa")),
                pkn_bottomhole_pressure_mpa=_number(row.get("bottomhole_pressure_mpa")),
                pyfrac_bottomhole_pressure_mpa=_number(row.get("pyfrac_bottomhole_pressure_mpa")),
                pyfrac_volume_m3=_number(row.get("pyfrac_volume_m3")),
                pyfrac_runtime_s=_number(row.get("pyfrac_runtime_seconds")),
                successful_time_steps=_integer(row.get("pyfrac_successful_time_steps")),
                final_time_s=_number(row.get("pyfrac_final_time_s")),
            )
        )

    rows.sort(key=lambda item: item.time_s)
    deduplicated: dict[float, PyFracComparisonPoint] = {item.time_s: item for item in rows}
    points = list(sorted(deduplicated.values(), key=lambda item: item.time_s))
    if not points:
        return PyFracComparison(
            source=str(path),
            note=f"没有成功的原生动态检查点；失败/未完成 {failed_count} 个。",
        )
    note = "原生 Controller.run() 离线动态检查点"
    if failed_count:
        note += f"；另有 {failed_count} 个未完成点未绘制"
    return PyFracComparison(
        available=True,
        source=str(path),
        engine_mode="pyfrac_native_dynamic",
        points=points,
        note=note,
    )


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _integer(value: Any) -> int | None:
    number = _number(value)
    return int(number) if number is not None else None


def _boolean(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None
