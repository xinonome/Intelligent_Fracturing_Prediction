"""Numerical-robustness helpers for the legacy PyFrac adapter.

The vendored PyFrac solver owns the hydraulic-fracture equations.  This
module owns the orchestration around it: deterministic mesh sizing, in-memory
checkpoints, relaxed parameter assimilation and convergence acceptance.  It
is deliberately dependency-light so these policies can be unit tested
without importing the legacy solver.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import math
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MeshDecision:
    """The mesh selected for one PyFrac state/restart window."""

    level: int
    half_length_m: float
    half_height_m: float
    nx: int
    ny: int
    dx_m: float
    dy_m: float
    estimated_half_length_m: float
    cells_across_front: float
    boundary_margin_m: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def choose_mesh(
    *,
    estimated_half_length_m: float,
    height_m: float,
    base_half_length_m: float,
    base_half_height_m: float,
    base_nx: int,
    base_ny: int,
    min_front_cells: int = 8,
    boundary_margin_cells: int = 6,
    refinement_factor: float = 1.35,
    max_levels: int = 3,
    max_nx: int = 241,
    max_ny: int = 121,
    minimum_half_length_m: float | None = None,
) -> MeshDecision:
    """Choose a resolved domain and cell count deterministically.

    The old adapter fixed ``nx``/``ny`` regardless of the expected front
    scale.  Here the domain is first given a boundary margin, then the cell
    count is raised until the front has the requested number of cells.  The
    returned decision is also the audit record used by convergence reports.
    """

    length = max(float(estimated_half_length_m), 1.0e-6)
    height = max(float(height_m), 1.0e-6)
    physical_minimum = 1.5 * height if minimum_half_length_m is None else float(minimum_half_length_m)
    half_length = max(float(base_half_length_m), 1.8 * length, physical_minimum)
    half_height = max(float(base_half_height_m), 1.5 * height)
    base_nx = max(int(base_nx), 31)
    base_ny = max(int(base_ny), 17)
    min_front_cells = max(int(min_front_cells), 2)
    boundary_margin_cells = max(int(boundary_margin_cells), 1)
    refinement_factor = max(float(refinement_factor), 1.05)
    max_levels = max(int(max_levels), 0)

    level = 0
    reason = "base_mesh_resolved"
    while level < max_levels:
        dx = 2.0 * half_length / max(base_nx - 1, 1)
        cells = length / max(dx, 1.0e-12)
        margin = (half_length - length) / max(dx, 1.0e-12)
        if cells >= min_front_cells and margin >= boundary_margin_cells:
            break
        level += 1
        base_nx = min(max_nx, max(base_nx + 2, int(math.ceil(base_nx * refinement_factor)) | 1))
        base_ny = min(max_ny, max(base_ny + 2, int(math.ceil(base_ny * refinement_factor)) | 1))
        # If the front is too close to the boundary, extend the domain before
        # refining it. This avoids making a very fine mesh on an undersized
        # domain and lets PyFrac's own remeshing take over later if required.
        if half_length - length < boundary_margin_cells * dx:
            half_length = max(half_length * refinement_factor, length * 2.0)
            reason = "domain_extended_and_mesh_refined"
        else:
            reason = "mesh_refined_for_front_resolution"

    dx = 2.0 * half_length / max(base_nx - 1, 1)
    dy = 2.0 * half_height / max(base_ny - 1, 1)
    final_cells = length / max(dx, 1.0e-12)
    final_margin_cells = (half_length - length) / max(dx, 1.0e-12)
    if final_cells < min_front_cells or final_margin_cells < boundary_margin_cells:
        reason = f"{reason}_limit_reached"
    return MeshDecision(
        level=level,
        half_length_m=float(half_length),
        half_height_m=float(half_height),
        nx=int(base_nx),
        ny=int(base_ny),
        dx_m=float(dx),
        dy_m=float(dy),
        estimated_half_length_m=float(length),
        cells_across_front=float(final_cells),
        boundary_margin_m=float(half_length - length),
        reason=reason,
    )


@dataclass
class Checkpoint:
    """A rollback point for one continuous native session."""

    checkpoint_id: str
    time_s: float
    fracture: Any
    last_parameters: dict[str, float]
    successful_steps: int
    failed_steps: int
    active: bool


class CheckpointManager:
    """Keep exact in-memory fracture checkpoints.

    PyFrac mutates the retained ``Fracture`` object during a controller run,
    so saving only its timestamp is insufficient. ``deepcopy`` preserves the
    complete numerical state and lets a failed attempt be discarded safely.
    """

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self._counter = 0

    def save(
        self,
        *,
        time_s: float,
        fracture: Any,
        last_parameters: dict[str, float],
        successful_steps: int,
        failed_steps: int,
        active: bool,
    ) -> Checkpoint:
        self._counter += 1
        checkpoint = Checkpoint(
            checkpoint_id=f"cp-{self._counter:05d}",
            time_s=float(time_s),
            fracture=deepcopy(fracture),
            last_parameters=dict(last_parameters),
            successful_steps=int(successful_steps),
            failed_steps=int(failed_steps),
            active=bool(active),
        )
        self.events.append({"checkpoint_id": checkpoint.checkpoint_id, "time_s": checkpoint.time_s, "event": "saved"})
        return checkpoint

    def restore(self, checkpoint: Checkpoint, *, reason: str, retry_count: int) -> Any:
        self.events.append(
            {
                "checkpoint_id": checkpoint.checkpoint_id,
                "time_s": checkpoint.time_s,
                "event": "restored",
                "reason": reason,
                "retry_count": int(retry_count),
            }
        )
        return deepcopy(checkpoint.fracture)


@dataclass(frozen=True)
class RelaxationResult:
    state: np.ndarray
    raw_update_norm: float
    applied_update_norm: float
    relaxation: float
    clipped_components: int


def relaxed_update(
    prior: np.ndarray,
    analysis: np.ndarray,
    *,
    relaxation: float = 0.35,
    max_step: np.ndarray | list[float] | tuple[float, ...] | None = None,
) -> RelaxationResult:
    """Blend an EnKF analysis into the prior with per-parameter step limits."""

    prior_array = np.asarray(prior, dtype=float)
    analysis_array = np.asarray(analysis, dtype=float)
    if prior_array.shape != analysis_array.shape:
        raise ValueError("prior and analysis must have the same shape")
    alpha = float(np.clip(relaxation, 0.0, 1.0))
    delta = analysis_array - prior_array
    proposed = alpha * delta
    clipped = 0
    if max_step is not None:
        limits = np.asarray(max_step, dtype=float)
        if limits.shape != prior_array.shape and limits.size == prior_array.shape[-1]:
            limits = np.broadcast_to(limits, prior_array.shape)
        if limits.shape != prior_array.shape:
            raise ValueError("max_step must match state shape or its final dimension")
        safe_limits = np.maximum(np.abs(limits), 0.0)
        bounded = np.clip(proposed, -safe_limits, safe_limits)
        clipped = int(np.count_nonzero(~np.isclose(bounded, proposed)))
        proposed = bounded
    state = prior_array + proposed
    return RelaxationResult(
        state=state,
        raw_update_norm=float(np.linalg.norm(delta)),
        applied_update_norm=float(np.linalg.norm(proposed)),
        relaxation=alpha,
        clipped_components=clipped,
    )


def relative_difference(left: float, right: float, epsilon: float = 1.0e-12) -> float:
    left_value = float(left)
    right_value = float(right)
    if not np.isfinite(left_value) or not np.isfinite(right_value):
        return float("nan")
    return abs(left_value - right_value) / max(abs(right_value), abs(left_value), epsilon)


def evaluate_convergence(
    records: list[dict[str, Any]],
    *,
    reference_name: str,
    comparison_name: str,
    metrics: tuple[str, ...],
    tolerance: float = 0.05,
    mass_balance_tolerance: float = 0.10,
) -> dict[str, Any]:
    """Evaluate grid/time-step records without turning failures into passes."""

    by_name = {str(row.get("name")): row for row in records}
    reference = by_name.get(reference_name, {})
    comparison = by_name.get(comparison_name, {})
    differences: dict[str, float | None] = {}
    metric_pass: dict[str, bool] = {}
    for metric in metrics:
        difference = relative_difference(comparison.get(metric, np.nan), reference.get(metric, np.nan))
        differences[metric] = None if not np.isfinite(difference) else float(difference)
        metric_pass[metric] = bool(np.isfinite(difference) and difference <= tolerance)
    valid = bool(
        reference.get("success", False)
        and comparison.get("success", False)
        and reference.get("target_reached", False)
        and comparison.get("target_reached", False)
        and int(reference.get("failed_time_steps", 0)) == 0
        and int(comparison.get("failed_time_steps", 0)) == 0
        and np.isfinite(float(reference.get("mass_balance_relative_error", np.nan)))
        and np.isfinite(float(comparison.get("mass_balance_relative_error", np.nan)))
        and float(reference.get("mass_balance_relative_error", np.inf)) <= mass_balance_tolerance
        and float(comparison.get("mass_balance_relative_error", np.inf)) <= mass_balance_tolerance
    )
    return {
        "reference": reference_name,
        "comparison": comparison_name,
        "tolerance": float(tolerance),
        "mass_balance_tolerance": float(mass_balance_tolerance),
        "differences": differences,
        "metric_pass": metric_pass,
        "valid_runs": valid,
        "passed": bool(valid and all(metric_pass.values())),
    }
