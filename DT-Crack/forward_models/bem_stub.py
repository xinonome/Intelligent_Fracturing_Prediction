from __future__ import annotations

import numpy as np

from .base import ForwardModel, ForwardResult
from .pkn_model import OBSERVATION_NAMES, STATE_NAMES


class BoundaryElementForwardModel(ForwardModel):
    """Boundary-element placeholder with the same interface as PKN.

    This class proves the framework is model-pluggable. It is not a real BEM
    solver yet; replace the internals with a boundary-element kernel when the
    engineering solver is available.
    """

    def simulate(
        self,
        state: np.ndarray,
        controls: dict[str, np.ndarray | float],
        reservoir: dict[str, float],
        dt: float,
    ) -> ForwardResult:
        state = np.asarray(state, dtype=float)
        if state.ndim == 1:
            state = state.reshape(1, -1)
        q = np.asarray(controls.get("flow_rate_m3_min", 4.0), dtype=float)
        q = np.resize(q.reshape(-1), len(state))
        interaction = 1.0 + 0.04 * np.sin(np.arange(len(state)))
        next_state = state.copy()
        next_state[:, 0] += interaction * (1.6 + 0.55 * np.sqrt(np.maximum(q, 0.0)))
        next_state[:, 2] *= 0.995 + 0.01 * interaction
        obs = np.column_stack(
            [
                38.0 + next_state[:, 3] + 0.004 * next_state[:, 0],
                0.012 * next_state[:, 0] + 0.2 * next_state[:, 2],
                0.006 * next_state[:, 0] + 0.03 * next_state[:, 2],
                next_state[:, 0] * 0.95,
            ]
        )
        return ForwardResult(next_state, obs, list(STATE_NAMES), list(OBSERVATION_NAMES))
