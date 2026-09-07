"""Small, dependency-light guards for legacy PyFrac front advancement.

The original controller decides that a time step was too large only after
the front has already been reconstructed.  This module provides two pieces
of defensive logic that are safe to apply before that reconstruction:

* a front-traversal CFL-like time-step cap; and
* a deterministic audit/canonicalisation of the discrete front cell lists.

These are numerical guards, not changes to the hydraulic-fracture equations.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def front_cfl_time_step(
    time_step: float,
    velocities: Any,
    hx: float,
    hy: float,
    cfl: float = 0.8,
) -> tuple[float, bool]:
    """Limit a proposed step so the fastest tip moves less than one cell.

    ``cfl`` is expressed as a fraction of the smallest cell dimension.  A
    non-positive or non-finite velocity does not provide a useful cap, so the
    proposed step is returned unchanged in that case.
    """

    proposed = float(time_step)
    if not np.isfinite(proposed) or proposed <= 0.0:
        return proposed, False
    try:
        velocity = np.asarray(velocities, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return proposed, False
    positive = velocity[np.isfinite(velocity) & (velocity > 0.0)]
    cell = min(float(hx), float(hy))
    fraction = float(cfl)
    if positive.size == 0 or not np.isfinite(cell) or cell <= 0.0:
        return proposed, False
    if not np.isfinite(fraction) or fraction <= 0.0:
        return proposed, False
    limit = fraction * cell / float(np.max(positive))
    if not np.isfinite(limit) or limit <= 0.0 or limit >= proposed:
        return proposed, False
    return limit, True


def _stable_unique(values: Any) -> tuple[np.ndarray, bool]:
    array = np.asarray(values, dtype=int).reshape(-1)
    if array.size == 0:
        return array, False
    _, first = np.unique(array, return_index=True)
    first = np.sort(first)
    result = array[first]
    return result, result.size != array.size


def audit_front_state(fracture: Any) -> list[str]:
    """Return structural front-state errors without mutating the fracture."""

    errors: list[str] = []
    arrays: dict[str, np.ndarray] = {}
    for name in ("EltChannel", "EltTip", "EltCrack", "EltRibbon", "fully_traversed"):
        if not hasattr(fracture, name):
            continue
        values = np.asarray(getattr(fracture, name), dtype=int).reshape(-1)
        arrays[name] = values
        if np.unique(values).size != values.size:
            errors.append(f"duplicate cells in {name}")

    # ``EltRibbon`` is a geometric search band in this PyFrac version and may
    # legitimately include cells that are already in ``EltCrack``.  Only
    # semantic front-region conflicts are rejected here.
    for left, right in (("EltChannel", "EltTip"), ("EltTip", "fully_traversed")):
        if left in arrays and right in arrays:
            overlap = np.intersect1d(arrays[left], arrays[right])
            if overlap.size:
                errors.append(f"overlap between {left} and {right}: {overlap[:8].tolist()}")

    if "EltTip" in arrays:
        tip_count = arrays["EltTip"].size
        for name in ("l", "alpha", "v", "FillF", "ZeroVertex", "Ffront"):
            if hasattr(fracture, name):
                shape = np.asarray(getattr(fracture, name)).shape
                if not shape or shape[0] != tip_count:
                    errors.append(f"tip metadata size mismatch: {name}={shape} EltTip={tip_count}")
    return errors


def canonicalize_front_state(fracture: Any) -> list[str]:
    """Remove duplicate discrete cell ids and aligned duplicate tip metadata.

    The operation preserves first occurrence order.  It is intentionally
    limited to exact duplicates; overlaps between different semantic regions
    are reported by :func:`audit_front_state` and are not silently repaired.
    """

    repairs: list[str] = []
    tip_old = np.asarray(getattr(fracture, "EltTip", []), dtype=int).reshape(-1)
    tip_new, tip_changed = _stable_unique(tip_old)
    keep = None
    if tip_changed:
        _, first = np.unique(tip_old, return_index=True)
        keep = np.sort(first)
        fracture.EltTip = tip_new
        repairs.append(f"deduplicated EltTip {tip_old.size}->{tip_new.size}")

    for name in ("EltChannel", "EltCrack", "EltRibbon", "fully_traversed"):
        if not hasattr(fracture, name):
            continue
        old = np.asarray(getattr(fracture, name), dtype=int).reshape(-1)
        new, changed = _stable_unique(old)
        if changed:
            setattr(fracture, name, new)
            repairs.append(f"deduplicated {name} {old.size}->{new.size}")

    if keep is not None:
        for name in ("l", "alpha", "v", "FillF", "ZeroVertex", "Ffront"):
            if not hasattr(fracture, name):
                continue
            values = np.asarray(getattr(fracture, name))
            if values.ndim and values.shape[0] == tip_old.size:
                setattr(fracture, name, values[keep].copy())
                repairs.append(f"aligned {name} with deduplicated EltTip")
    return repairs
