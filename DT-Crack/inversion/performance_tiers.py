"""Common throughput-tier classification for native PyFrac experiments.

The ratio is defined as wall-clock compute time divided by native simulated
time.  Lower is better.  A run may therefore be reported as a 10%, 20%, or
30% throughput result.  This module intentionally does not decide whether a
run is physically acceptable: conservation, target-time completion, solver
health, and pressure error remain independent gates.
"""

from __future__ import annotations

from typing import Any, Iterable

import math


DEFAULT_THROUGHPUT_TIERS = (0.10, 0.20, 0.30)


def _finite_ratio(value: object) -> float | None:
    try:
        ratio = float(value)
    except (TypeError, ValueError):
        return None
    return ratio if math.isfinite(ratio) and ratio >= 0.0 else None


def classify_throughput(
    runtime_s: object,
    simulated_time_s: object,
    thresholds: Iterable[float] = DEFAULT_THROUGHPUT_TIERS,
) -> dict[str, Any]:
    """Classify one run against ordered throughput thresholds.

    ``runtime_s / simulated_time_s`` is the primary ratio.  The returned
    ``tier`` is the best threshold met, not a statement that the model is
    scientifically validated.  Missing or invalid timing is reported as
    ``not_measured`` rather than being treated as failure of the solver.
    """

    ratio = None
    try:
        runtime = float(runtime_s)
        simulated = float(simulated_time_s)
        if math.isfinite(runtime) and math.isfinite(simulated) and runtime >= 0.0 and simulated > 0.0:
            ratio = runtime / simulated
    except (TypeError, ValueError):
        ratio = None

    cleaned: list[float] = []
    for threshold in thresholds:
        try:
            value = float(threshold)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0.0 and value not in cleaned:
            cleaned.append(value)
    cleaned.sort()
    if not cleaned:
        cleaned = list(DEFAULT_THROUGHPUT_TIERS)

    passed = {f"{int(round(value * 100))}%": bool(ratio is not None and ratio <= value) for value in cleaned}
    if ratio is None:
        tier = "not_measured"
        best_threshold = None
    else:
        best_threshold = next((value for value in cleaned if ratio <= value), None)
        tier = f"{int(round(best_threshold * 100))}%" if best_threshold is not None else "not_met"

    return {
        "ratio": ratio,
        "runtime_s": float(runtime_s) if _finite_ratio(runtime_s) is not None else None,
        "simulated_time_s": float(simulated_time_s) if _finite_ratio(simulated_time_s) is not None else None,
        "tier": tier,
        "best_threshold": best_threshold,
        "thresholds": cleaned,
        "passed": passed,
        "interpretation": (
            "throughput only; independent physical acceptance gates still required"
            if ratio is not None
            else "timing unavailable"
        ),
    }


def evaluate_throughput_requirement(
    runtime_s: object,
    simulated_time_s: object,
    required_ratio: object | None = None,
) -> dict[str, Any]:
    """Evaluate an optional formal runtime requirement.

    The 10/20/30% values are reporting tiers.  They are deliberately not a
    mandatory scientific gate: a project may accept the 20% or 30% tier while
    the physical acceptance gates remain unchanged.  ``None`` therefore means
    "report only; no formal throughput requirement was selected".
    """

    if required_ratio is None:
        return {
            "required_ratio": None,
            "passed": None,
            "status": "not_required",
            "ratio": classify_throughput(runtime_s, simulated_time_s)["ratio"],
        }
    try:
        required = float(required_ratio)
    except (TypeError, ValueError):
        return {
            "required_ratio": None,
            "passed": False,
            "status": "invalid_requirement",
            "ratio": classify_throughput(runtime_s, simulated_time_s)["ratio"],
        }
    if not math.isfinite(required) or required <= 0.0:
        return {
            "required_ratio": None,
            "passed": False,
            "status": "invalid_requirement",
            "ratio": classify_throughput(runtime_s, simulated_time_s)["ratio"],
        }
    ratio = classify_throughput(runtime_s, simulated_time_s)["ratio"]
    return {
        "required_ratio": required,
        "passed": bool(ratio is not None and ratio <= required),
        "status": "passed" if ratio is not None and ratio <= required else "not_met",
        "ratio": ratio,
    }
