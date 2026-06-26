from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SandPlugRuleResult:
    triggered: bool
    label: str | None
    reasons: list[str]
    score: float


def _numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").interpolate(limit_direction="both").fillna(0.0).to_numpy(dtype=float)


def _has_continuous_decrease(values: np.ndarray, min_length: int = 3, eps: float = 0.1) -> bool:
    if len(values) < min_length:
        return False
    diffs = np.diff(values)
    run = 0
    for diff in diffs:
        if diff < -eps:
            run += 1
            if run >= min_length - 1:
                return True
        else:
            run = 0
    return False


def _has_continuous_increase(values: np.ndarray, min_length: int = 3, eps: float = 0.05) -> tuple[bool, float]:
    if len(values) < min_length:
        return False, 0.0
    diffs = np.diff(values)
    run_start = None
    max_rise = 0.0
    for index, diff in enumerate(diffs):
        if diff > eps:
            if run_start is None:
                run_start = index
        else:
            if run_start is not None and index - run_start >= min_length - 1:
                max_rise = max(max_rise, float(values[index] - values[run_start]))
            run_start = None
    if run_start is not None and len(diffs) - run_start >= min_length - 1:
        max_rise = max(max_rise, float(values[-1] - values[run_start]))
    return max_rise > 0, max_rise


def evaluate_sand_plug_window(
    frame: pd.DataFrame,
    start: int,
    end: int,
    pressure_col: str = "SGBY",
    flow_col: str = "PL",
    sand_col: str = "SB",
    sample_interval_seconds: float = 10.0,
    pressure_delta_threshold: float = 8.0,
    pressure_delta_high: float = 10.0,
    pressure_slope_threshold: float = 1.0,
    pressure_slope_high: float = 1.5,
    flow_drop_eps: float = 0.1,
    sand_drop_eps: float = 0.2,
    flow_rise_eps: float = 0.05,
    flow_rise_total_threshold: float = 1.0,
    recent_zero_sand_points: int = 12,
    max_zero_sand_count: int = 3,
) -> SandPlugRuleResult:
    missing = [col for col in [pressure_col, flow_col, sand_col] if col not in frame.columns]
    if missing:
        return SandPlugRuleResult(False, None, [f"missing_columns:{','.join(missing)}"], 0.0)

    start = max(0, int(start))
    end = min(len(frame) - 1, int(end))
    if end <= start:
        return SandPlugRuleResult(False, None, ["empty_window"], 0.0)

    tail_end = min(len(frame), end + 1 + max(1, (end - start + 1) // 5))
    window = frame.iloc[start:tail_end]
    pressure = _numeric(window[pressure_col])
    flow = _numeric(window[flow_col])
    sand = _numeric(window[sand_col])
    if len(pressure) < 4:
        return SandPlugRuleResult(False, None, ["window_too_short"], 0.0)

    pressure_delta = float(np.nanmax(pressure) - pressure[0])
    duration_minutes = max((len(pressure) - 1) * sample_interval_seconds / 60.0, 1e-6)
    pressure_slope = pressure_delta / duration_minutes
    main_pressure = pressure_delta >= pressure_delta_threshold and pressure_slope >= pressure_slope_threshold
    high_pressure = pressure_delta >= pressure_delta_high and pressure_slope >= pressure_slope_high
    if not main_pressure:
        return SandPlugRuleResult(False, None, [f"pressure_not_triggered:delta={pressure_delta:.2f},slope={pressure_slope:.2f}"], 0.0)

    flow_has_increase, max_flow_rise = _has_continuous_increase(flow, eps=flow_rise_eps)
    sand_has_decrease = _has_continuous_decrease(sand, eps=sand_drop_eps)
    if flow_has_increase and max_flow_rise >= flow_rise_total_threshold and not sand_has_decrease:
        return SandPlugRuleResult(False, None, [f"excluded_by_flow_rise:delta_q={max_flow_rise:.2f}"], 0.2)

    recent_start = max(0, start - recent_zero_sand_points)
    recent_sand = _numeric(frame.iloc[recent_start : start + 1][sand_col])
    if len(recent_sand) and np.any(recent_sand == 0):
        return SandPlugRuleResult(False, None, ["excluded_by_recent_zero_sand"], 0.25)

    flow_has_decrease = _has_continuous_decrease(flow, eps=flow_drop_eps)
    zero_sand_count = int(np.sum(sand == 0))
    reasons = [f"pressure_delta={pressure_delta:.2f}", f"pressure_slope={pressure_slope:.2f}"]
    score = 0.6

    if high_pressure and zero_sand_count <= max_zero_sand_count:
        reasons.append("high_pressure_trigger")
        return SandPlugRuleResult(True, "砂堵", reasons, 0.95)
    if zero_sand_count > max_zero_sand_count:
        reasons.append(f"zero_sand_count={zero_sand_count}")
        return SandPlugRuleResult(False, None, reasons, 0.35)
    if flow_has_decrease:
        reasons.append("flow_decrease")
        score += 0.15
    if sand_has_decrease:
        reasons.append("sand_ratio_decrease")
        score += 0.15
    if flow_has_decrease or sand_has_decrease:
        return SandPlugRuleResult(True, "砂堵", reasons, min(score, 0.9))
    reasons.append("pressure_only_abnormal")
    return SandPlugRuleResult(False, None, reasons, 0.45)
