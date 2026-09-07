from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inversion import PhysicalEnKFConfig, parameterized_allocation_state_size
from inversion.validate_direct_observations import (
    PARAMETER_CLASS_NAMES,
    clip_augmented_state,
    ensemble_parameter_statistics,
)


def test_unbounded_control_preserves_out_of_domain_state_without_delta_clipping() -> None:
    state = np.zeros(parameterized_allocation_state_size(6), dtype=float)
    state[0] = np.log(10.0)
    state[2] = np.log(12.0)
    state[3] = 140.0
    state[5] = 9.0

    constrained = clip_augmented_state(state, 6, parameter_bound_mode="constrained")
    unbounded = clip_augmented_state(state, 6, parameter_bound_mode="unbounded_control")

    assert constrained[0] < state[0]
    assert constrained[2] < state[2]
    assert constrained[3] < state[3]
    assert np.array_equal(unbounded, state)


def test_parameter_statistics_cover_three_classes_and_fourteen_parameters() -> None:
    ensemble = np.zeros((4, parameterized_allocation_state_size(6)), dtype=float)
    ensemble[:, 0] = np.linspace(-0.1, 0.1, 4)
    ensemble[:, 3] = np.linspace(55.0, 65.0, 4)
    stats = ensemble_parameter_statistics(ensemble, PhysicalEnKFConfig(), 6, "posterior_ensemble")

    parameter_names = [name for names in PARAMETER_CLASS_NAMES.values() for name in names]
    assert len(parameter_names) == 14
    for name in parameter_names:
        assert f"posterior_ensemble_{name}_std" in stats
        assert np.isfinite(stats[f"posterior_ensemble_{name}_std"])
