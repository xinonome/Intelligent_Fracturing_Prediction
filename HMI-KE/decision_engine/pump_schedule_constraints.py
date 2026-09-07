from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class PumpScheduleConstraint:
    """Schedule references and optional field-approved operational bounds.

    A historical maximum is not automatically a safety limit.  The sand ratio
    scale is used only for normalization in the learning environment.  A hard
    sand limit is optional and must come from an explicit field configuration;
    it is deliberately not inferred from the largest observed value.
    """

    name: str
    description: str
    max_flow_m3_min: float
    sand_ratio_scale_percent: float
    max_flow_step_m3_min: float
    max_sand_increase_percent: float
    allow_sand_pause: bool
    hard_sand_ratio_limit_percent: float | None = None
    max_sand_decrease_percent: float = 3.0
    high_sand_warning_percent: float = 10.0

    def to_dict(self) -> dict:
        return asdict(self)


SCHEDULES = {
    # The 14% value appeared in historical data, but is not promoted to a hard
    # engineering limit without an explicit field configuration.
    "continuous": PumpScheduleConstraint(
        name="continuous",
        description="保守连续建议：以当前观测砂比为基准小幅微调，不自动猜测目标砂比。",
        max_flow_m3_min=18.0,
        sand_ratio_scale_percent=20.0,
        max_flow_step_m3_min=4.0,
        # This is a provisional action-step setting for the HMI demo, not a
        # field-approved sand-ratio limit or target.
        max_sand_increase_percent=0.5,
        allow_sand_pause=True,
        hard_sand_ratio_limit_percent=None,
        max_sand_decrease_percent=3.0,
    ),
    "staged_with_pause": PumpScheduleConstraint(
        name="staged_with_pause",
        description="分段加砂：仍以当前观测砂比为基准小幅调整，高砂比需要人工确认。",
        max_flow_m3_min=18.0,
        sand_ratio_scale_percent=20.0,
        max_flow_step_m3_min=4.0,
        max_sand_increase_percent=0.5,
        allow_sand_pause=True,
        hard_sand_ratio_limit_percent=None,
        max_sand_decrease_percent=3.0,
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
    raw_current_sand = np.nan_to_num(np.asarray(current_sand_ratio, dtype=float), nan=0.0)
    current_sand_ratio = np.maximum(raw_current_sand, 0.0)
    if reference_sand_ratio is None:
        reference_sand = current_sand_ratio.copy()
    else:
        reference_sand = np.nan_to_num(np.asarray(reference_sand_ratio, dtype=float), nan=0.0)
    reference_sand = np.maximum(reference_sand, 0.0)
    if constraint.hard_sand_ratio_limit_percent is not None:
        reference_sand = np.minimum(reference_sand, constraint.hard_sand_ratio_limit_percent)

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
    sand_upper = reference_sand + constraint.max_sand_increase_percent
    # The warning threshold is a conservative review boundary, not an
    # engineering maximum.  Once the measured value is already in that range,
    # the default advisory policy may hold or reduce it, but not increase it.
    already_high = reference_sand >= constraint.high_sand_warning_percent
    sand_upper = np.where(already_high, reference_sand, sand_upper)
    if constraint.hard_sand_ratio_limit_percent is not None:
        sand_upper = np.minimum(constraint.hard_sand_ratio_limit_percent, sand_upper)
    sand_upper = np.maximum(sand_upper, sand_lower)
    # If a field-approved limit is configured, an already over-limit measured
    # value is retained as a data/safety exception and cannot be increased.
    over_limit_input = (
        np.zeros_like(raw_current_sand, dtype=bool)
        if constraint.hard_sand_ratio_limit_percent is None
        else raw_current_sand > constraint.hard_sand_ratio_limit_percent + 1e-9
    )
    if constraint.hard_sand_ratio_limit_percent is not None:
        sand_upper = np.where(over_limit_input, reference_sand, sand_upper)
    sand_lower = np.minimum(sand_lower, sand_upper)
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
        "sand_at_absolute_limit": (
            np.zeros_like(safe_sand, dtype=bool)
            if constraint.hard_sand_ratio_limit_percent is None
            else np.isclose(safe_sand, constraint.hard_sand_ratio_limit_percent, atol=1e-6)
        ),
        "current_sand_above_absolute_limit": over_limit_input,
        "high_sand_ratio": safe_sand >= constraint.high_sand_warning_percent,
        "sand_requires_confirmation": safe_sand >= constraint.high_sand_warning_percent,
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
    s0 = np.maximum(np.nan_to_num(np.asarray(current_sand_ratio, dtype=float), nan=0.0), 0.0)

    q_over = np.maximum(q - constraint.max_flow_m3_min, 0.0) / max(constraint.max_flow_m3_min, 1e-6)
    if constraint.hard_sand_ratio_limit_percent is None:
        s_over = np.zeros_like(s)
    else:
        s_over = np.maximum(s - constraint.hard_sand_ratio_limit_percent, 0.0) / max(
            constraint.hard_sand_ratio_limit_percent, 1e-6
        )
    q_step_over = np.maximum(np.abs(q - q0) - constraint.max_flow_step_m3_min, 0.0) / max(constraint.max_flow_step_m3_min, 1e-6)
    s_step_over = np.maximum((s - s0) - constraint.max_sand_increase_percent, 0.0) / max(constraint.max_sand_increase_percent, 1e-6)
    negative_penalty = (q < 0.0).astype(float) + (s < 0.0).astype(float)
    pause_penalty = np.zeros_like(s)
    if not constraint.allow_sand_pause:
        pause_penalty = ((s0 > 0.0) & (s < s0)).astype(float)

    high_sand_penalty = np.maximum(s - constraint.high_sand_warning_percent, 0.0) / max(
        constraint.sand_ratio_scale_percent - constraint.high_sand_warning_percent, 1e-6
    )
    safety_penalty = (
        q_over + s_over + q_step_over + s_step_over + negative_penalty + pause_penalty
        + 2.0 * high_sand_penalty
    )
    smoothness_reward = np.exp(-0.5 * (np.abs(q - q0) / max(constraint.max_flow_step_m3_min, 1e-6) + np.abs(s - s0) / max(constraint.max_sand_increase_percent, 1e-6)))

    design_reward = np.zeros_like(q)
    if reference_flow is not None and reference_sand_ratio is not None:
        q_ref = np.asarray(reference_flow, dtype=float)
        s_ref = np.asarray(reference_sand_ratio, dtype=float)
        design_error = 0.5 * (
            np.abs(q - q_ref) / max(constraint.max_flow_m3_min, 1e-6)
            + np.abs(s - s_ref) / max(constraint.sand_ratio_scale_percent, 1e-6)
        )
        design_reward = np.exp(-design_error)

    total = 1.0 + 0.5 * smoothness_reward + 0.25 * design_reward - 3.0 * safety_penalty
    return {
        "total_reward": total,
        "safety_penalty": safety_penalty,
        "high_sand_penalty": high_sand_penalty,
        "smoothness_reward": smoothness_reward,
        "design_reference_reward": design_reward,
    }
