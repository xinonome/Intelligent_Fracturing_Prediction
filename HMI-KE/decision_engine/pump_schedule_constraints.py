from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class PumpScheduleConstraint:
    """Soft design reference plus hard operational bounds for one schedule family."""

    name: str
    description: str
    max_flow_m3_min: float
    max_sand_ratio_percent: float
    max_flow_step_m3_min: float
    max_sand_increase_percent: float
    allow_sand_pause: bool

    def to_dict(self) -> dict:
        return asdict(self)


SCHEDULES = {
    # Extracted conservatively from the two supplied design schedules. The design
    # values are references, while the maxima and one-step increase are safety bounds.
    "continuous": PumpScheduleConstraint(
        name="continuous",
        description="连续/无间隔加砂：砂比逐级提升，不主动回到零砂比。",
        max_flow_m3_min=18.0,
        max_sand_ratio_percent=14.0,
        max_flow_step_m3_min=4.0,
        max_sand_increase_percent=3.0,
        allow_sand_pause=False,
    ),
    "staged_with_pause": PumpScheduleConstraint(
        name="staged_with_pause",
        description="分段加砂：允许阶段间暂停或降低砂比，再重新逐级提升。",
        max_flow_m3_min=18.0,
        max_sand_ratio_percent=15.0,
        max_flow_step_m3_min=4.0,
        max_sand_increase_percent=4.0,
        allow_sand_pause=True,
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
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Project suggested actions into schedule-derived operational bounds."""

    flow = np.asarray(flow, dtype=float)
    sand_ratio = np.asarray(sand_ratio, dtype=float)
    current_flow = np.nan_to_num(np.asarray(current_flow, dtype=float), nan=0.0)
    current_sand_ratio = np.nan_to_num(np.asarray(current_sand_ratio, dtype=float), nan=0.0)

    safe_flow = np.clip(
        flow,
        np.maximum(0.0, current_flow - constraint.max_flow_step_m3_min),
        np.minimum(constraint.max_flow_m3_min, current_flow + constraint.max_flow_step_m3_min),
    )
    sand_lower = np.zeros_like(current_sand_ratio)
    if not constraint.allow_sand_pause:
        # Continuous sanding should not spontaneously fall to zero once sanding starts.
        sand_lower = np.where(current_sand_ratio > 0.0, current_sand_ratio, 0.0)
    safe_sand = np.clip(
        sand_ratio,
        sand_lower,
        np.minimum(
            constraint.max_sand_ratio_percent,
            current_sand_ratio + constraint.max_sand_increase_percent,
        ),
    )
    diagnostics = {
        "flow_was_clipped": ~np.isclose(safe_flow, flow),
        "sand_was_clipped": ~np.isclose(safe_sand, sand_ratio),
        "flow_delta": safe_flow - current_flow,
        "sand_delta": safe_sand - current_sand_ratio,
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
