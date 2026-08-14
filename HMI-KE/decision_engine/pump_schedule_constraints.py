from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class PumpScheduleConstraint:
    """Soft design reference plus hard operational bounds for one schedule family.

    ``max_sand_increase_percent`` is deliberately separate from the absolute
    maximum.  The former is the advisory step size; it must not be interpreted
    as evidence that the absolute maximum is suitable for the current stage.
    """

    name: str
    description: str
    max_flow_m3_min: float
    max_sand_ratio_percent: float
    max_flow_step_m3_min: float
    max_sand_increase_percent: float
    allow_sand_pause: bool
    max_sand_decrease_percent: float = 3.0

    def to_dict(self) -> dict:
        return asdict(self)


SCHEDULES = {
    # Extracted conservatively from the two supplied design schedules. The design
    # values are references, while the maxima and one-step increase are safety bounds.
    "continuous": PumpScheduleConstraint(
        name="continuous",
        description="保守连续建议：以当前观测砂比为基准小幅微调，不自动猜测目标砂比。",
        max_flow_m3_min=18.0,
        max_sand_ratio_percent=14.0,
        max_flow_step_m3_min=4.0,
        # This is an advisory default for the HMI demo, not a field-approved
        # target.  The action must remain close to the measured/current value.
        max_sand_increase_percent=0.5,
        allow_sand_pause=True,
        max_sand_decrease_percent=3.0,
    ),
    "staged_with_pause": PumpScheduleConstraint(
        name="staged_with_pause",
        description="分段加砂：允许阶段间暂停或降低砂比，再重新逐级提升。",
        max_flow_m3_min=18.0,
        max_sand_ratio_percent=15.0,
        max_flow_step_m3_min=4.0,
        max_sand_increase_percent=4.0,
        allow_sand_pause=True,
        max_sand_decrease_percent=4.0,
    ),
}

ALIASES = {
    "no_interval": "continuous",
    "continuous_sanding": "continuous",
    "interval": "staged_with_pause",
    "staged": "staged_with_pause",
}


def get_schedule_constraint(name: str) -> PumpScheduleConstraint:
    key = ALIASES.get(name, name)
    if key not in SCHEDULES:
        raise ValueError(f"Unsupported pump schedule type: {name}. Choices: {sorted(SCHEDULES)}")
    return SCHEDULES[key]


def constrain_actions(
    flow: np.ndarray,
    sand_ratio: np.ndarray,
    current_flow: np.ndarray,
    current_sand_ratio: np.ndarray,
    constraint: PumpScheduleConstraint,
    reference_sand_ratio: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Project suggested actions into schedule-derived operational bounds.

    For conservative advisory control, the sand envelope is anchored to the
    measured/current sand ratio at this decision point.  It is intentionally
    not anchored to the previous recommendation, which prevents repeated
    small increases from accumulating into an unjustified high-sand plateau.
    """

    flow = np.asarray(flow, dtype=float)
    sand_ratio = np.asarray(sand_ratio, dtype=float)
    current_flow = np.nan_to_num(np.asarray(current_flow, dtype=float), nan=0.0)
    current_sand_ratio = np.nan_to_num(np.asarray(current_sand_ratio, dtype=float), nan=0.0)
    if reference_sand_ratio is None:
        reference_sand = current_sand_ratio.copy()
    else:
        reference_sand = np.nan_to_num(np.asarray(reference_sand_ratio, dtype=float), nan=0.0)
    reference_sand = np.clip(reference_sand, 0.0, constraint.max_sand_ratio_percent)

    safe_flow = np.clip(
        flow,
        np.maximum(0.0, current_flow - constraint.max_flow_step_m3_min),
        np.minimum(constraint.max_flow_m3_min, current_flow + constraint.max_flow_step_m3_min),
    )
    if constraint.allow_sand_pause:
        sand_lower = np.maximum(reference_sand - constraint.max_sand_decrease_percent, 0.0)
    else:
        # Legacy continuous-sanding behavior: do not spontaneously fall once
        # sanding starts.  Conservative HMI mode does not use this branch.
        sand_lower = np.where(current_sand_ratio > 0.0, current_sand_ratio, 0.0)
    sand_upper = np.minimum(
        constraint.max_sand_ratio_percent,
        reference_sand + constraint.max_sand_increase_percent,
    )
    # Never let an advisory action cross into the absolute high-sand limit.
    # Reaching the limit from below requires an approved schedule/engineer
    # decision; holding an already measured value is still allowed.
    approaching_limit = (
        constraint.allow_sand_pause
        & (reference_sand < constraint.max_sand_ratio_percent - 1e-9)
        & (sand_upper >= constraint.max_sand_ratio_percent - 1e-9)
    )
    sand_upper = np.where(approaching_limit, reference_sand, sand_upper)
    sand_upper = np.maximum(sand_upper, sand_lower)
    safe_sand = np.clip(
        sand_ratio,
        sand_lower,
        sand_upper,
    )
    diagnostics = {
        "flow_was_clipped": ~np.isclose(safe_flow, flow),
        "sand_was_clipped": ~np.isclose(safe_sand, sand_ratio),
        "flow_delta": safe_flow - current_flow,
        "sand_delta": safe_sand - current_sand_ratio,
        "sand_reference_ratio": reference_sand,
        "sand_lower_bound": sand_lower,
        "sand_upper_bound": sand_upper,
        "sand_at_absolute_limit": np.isclose(
            safe_sand, constraint.max_sand_ratio_percent, atol=1e-6
        ),
    }
    return safe_flow, safe_sand, diagnostics


def schedule_reward(
    proposed_flow: np.ndarray,
    proposed_sand_ratio: np.ndarray,
    current_flow: np.ndarray,
    current_sand_ratio: np.ndarray,
    constraint: PumpScheduleConstraint,
    reference_flow: np.ndarray | None = None,
    reference_sand_ratio: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Return a bounded reward where safety dominates soft design similarity.

    This is a transition-level reward. It intentionally does not take the maximum
    over a 60-row window, because that would hide unsafe actions at other steps.
    """

    q = np.asarray(proposed_flow, dtype=float)
    s = np.asarray(proposed_sand_ratio, dtype=float)
    q0 = np.nan_to_num(np.asarray(current_flow, dtype=float), nan=0.0)
    s0 = np.nan_to_num(np.asarray(current_sand_ratio, dtype=float), nan=0.0)

    q_over = np.maximum(q - constraint.max_flow_m3_min, 0.0) / max(constraint.max_flow_m3_min, 1e-6)
    s_over = np.maximum(s - constraint.max_sand_ratio_percent, 0.0) / max(constraint.max_sand_ratio_percent, 1e-6)
    q_step_over = np.maximum(np.abs(q - q0) - constraint.max_flow_step_m3_min, 0.0) / max(constraint.max_flow_step_m3_min, 1e-6)
    s_step_over = np.maximum((s - s0) - constraint.max_sand_increase_percent, 0.0) / max(constraint.max_sand_increase_percent, 1e-6)
    negative_penalty = (q < 0.0).astype(float) + (s < 0.0).astype(float)
    pause_penalty = np.zeros_like(s)
    if not constraint.allow_sand_pause:
        pause_penalty = ((s0 > 0.0) & (s < s0)).astype(float)

    safety_penalty = q_over + s_over + q_step_over + s_step_over + negative_penalty + pause_penalty
    smoothness_reward = np.exp(-0.5 * (np.abs(q - q0) / max(constraint.max_flow_step_m3_min, 1e-6) + np.abs(s - s0) / max(constraint.max_sand_increase_percent, 1e-6)))

    design_reward = np.zeros_like(q)
    if reference_flow is not None and reference_sand_ratio is not None:
        q_ref = np.asarray(reference_flow, dtype=float)
        s_ref = np.asarray(reference_sand_ratio, dtype=float)
        design_error = 0.5 * (
            np.abs(q - q_ref) / max(constraint.max_flow_m3_min, 1e-6)
            + np.abs(s - s_ref) / max(constraint.max_sand_ratio_percent, 1e-6)
        )
        design_reward = np.exp(-design_error)

    total = 1.0 + 0.5 * smoothness_reward + 0.25 * design_reward - 3.0 * safety_penalty
    return {
        "total_reward": total,
        "safety_penalty": safety_penalty,
        "smoothness_reward": smoothness_reward,
        "design_reference_reward": design_reward,
    }
