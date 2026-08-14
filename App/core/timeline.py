"""Common relative-second timeline and missing-data provenance helpers."""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class AlignedValue:
    value: Any = None
    valid: bool = False
    source: str = "missing"
    interpolation: str = "none"

    def as_dict(self) -> dict[str, Any]:
        return {"value": self.value, "valid": self.valid, "source": self.source, "interpolation": self.interpolation}


def relative_seconds(timestamps: Iterable[float | int]) -> list[float]:
    values = sorted({float(value) for value in timestamps if value is not None and math.isfinite(float(value))})
    if not values:
        return []
    start = values[0]
    return [value - start + 1.0 for value in values]


def build_time_axis(*series: Iterable[float | int], step_s: float = 1.0) -> list[float]:
    points: list[float] = []
    for timestamps in series:
        points.extend(float(value) for value in timestamps if value is not None)
    relative = relative_seconds(points)
    if not relative:
        return []
    end = max(relative)
    count = max(int(math.floor((end - 1.0) / step_s)) + 1, 1)
    return [1.0 + i * step_s for i in range(count)]


def _numeric(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def align_numeric(source_times: list[float], source_values: list[Any], target_times: list[float], source_name: str = "source") -> list[AlignedValue]:
    points = [(float(t), _numeric(v)) for t, v in zip(source_times, source_values)]
    points = sorted((t, v) for t, v in points if v is not None and math.isfinite(t))
    if not points:
        return [AlignedValue() for _ in target_times]
    times = [item[0] for item in points]
    values = [item[1] for item in points]
    result: list[AlignedValue] = []
    for target in target_times:
        if target < times[0] or target > times[-1]:
            result.append(AlignedValue(None, False, "missing", "none"))
            continue
        pos = bisect.bisect_left(times, target)
        if pos < len(times) and math.isclose(times[pos], target):
            result.append(AlignedValue(values[pos], True, source_name, "none"))
            continue
        left, right = max(pos - 1, 0), min(pos, len(times) - 1)
        if left == right:
            result.append(AlignedValue(values[left], True, source_name, "hold"))
            continue
        fraction = (target - times[left]) / max(times[right] - times[left], 1.0e-12)
        value = values[left] + (values[right] - values[left]) * fraction
        result.append(AlignedValue(value, True, source_name, "linear_interpolation"))
    return result


def align_categorical(source_times: list[float], source_values: list[Any], target_times: list[float], source_name: str = "source") -> list[AlignedValue]:
    points = sorted((float(t), value) for t, value in zip(source_times, source_values) if value not in (None, ""))
    if not points:
        return [AlignedValue() for _ in target_times]
    result: list[AlignedValue] = []
    for target in target_times:
        eligible = [item for item in points if item[0] <= target]
        if eligible:
            result.append(AlignedValue(eligible[-1][1], True, source_name, "forward_fill"))
        else:
            nearest = min(points, key=lambda item: abs(item[0] - target))
            result.append(AlignedValue(nearest[1], True, source_name, "nearest"))
    return result


class CallbackSignal:
    """Tiny signal implementation used by the non-Qt timeline tests."""

    def __init__(self) -> None:
        self._callbacks: list[Callable[[Any], None]] = []

    def connect(self, callback: Callable[[Any], None]) -> None:
        self._callbacks.append(callback)

    def emit(self, value: Any) -> None:
        for callback in list(self._callbacks):
            callback(value)


class TimelineController:
    """Framework-neutral main clock shared by every page.

    It intentionally exposes a ``frameChanged`` signal with the Qt spelling so
    widgets and headless tests use the same API.
    """

    def __init__(self, frames: list[Any] | None = None) -> None:
        self.frames = list(frames or [])
        self.index = 0
        self.frameChanged = CallbackSignal()

    @property
    def current(self) -> Any | None:
        return self.frames[self.index] if self.frames else None

    def set_frames(self, frames: list[Any]) -> None:
        self.frames = list(frames)
        self.set_index(0, emit=True)

    def set_index(self, index: int, emit: bool = True) -> Any | None:
        self.index = max(0, min(int(index), max(len(self.frames) - 1, 0)))
        current = self.current
        if emit and current is not None:
            self.frameChanged.emit(current)
        return current

    def set_time(self, time_s: float) -> Any | None:
        if not self.frames:
            return None
        def frame_time(frame: Any) -> float:
            value = getattr(frame, "time_s", None)
            if value is None and isinstance(frame, dict):
                value = frame.get("time_s", 0)
            return float(value or 0.0)

        index = min(range(len(self.frames)), key=lambda i: abs(frame_time(self.frames[i]) - time_s))
        return self.set_index(index)

    def step(self, delta: int) -> Any | None:
        return self.set_index(self.index + int(delta))
