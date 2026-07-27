from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import ForwardModel, ForwardResult


STATE_NAMES = [
    "fracture_length_m",
    "fracture_height_m",
    "fracture_width_mm",
    "net_pressure_mpa",
    "conductivity_index",
]

OBSERVATION_NAMES = [
    "pressure_mpa",
    "das_strain_energy",
    "dts_temperature_anomaly",
    "microseismic_extent_m",
]


@dataclass(frozen=True)
class PKNParameters:
    """Default engineering parameters for a lightweight PKN-style model."""

    young_modulus_gpa: float = 28.0
    poisson_ratio: float = 0.24
    viscosity_cp: float = 18.0
    leakoff_coefficient: float = 0.015
    min_horizontal_stress_mpa: float = 38.0
    height_relaxation: float = 0.08

    @property
    def plane_strain_modulus_mpa(self) -> float:
        return self.young_modulus_gpa * 1000.0 / (1.0 - self.poisson_ratio**2)


class PKNForwardModel(ForwardModel):
    """Fast PKN-style analytical forward model.

    This is a lightweight engineering approximation for online inversion demos.
    It is intentionally fast and differentiates itself from the old toy forward
    model by using per-fracture flow/liquid/sand allocations as controls.
    """

    state_names = STATE_NAMES
    observation_names = OBSERVATION_NAMES

    def __init__(self, params: PKNParameters | None = None) -> None:
        self.params = params or PKNParameters()

    def simulate(
        self,
        state: np.ndarray,
        controls: dict[str, np.ndarray | float],
        reservoir: dict[str, float] | None = None,
        dt: float = 10.0,
    ) -> ForwardResult:
        reservoir = reservoir or {}
        state = np.asarray(state, dtype=float)
        if state.ndim == 1:
            state = state.reshape(1, -1)

        q = _as_column(controls.get("flow_rate_m3_min", 4.0), len(state))
        liquid = _as_column(controls.get("liquid_volume_m3", 20.0), len(state))
        sand = _as_column(controls.get("sand_mass_t", 1.0), len(state))
        dt_min = max(float(dt) / 60.0, 1e-6)

        e_prime = float(reservoir.get("plane_strain_modulus_mpa", self.params.plane_strain_modulus_mpa))
        stress = float(reservoir.get("min_horizontal_stress_mpa", self.params.min_horizontal_stress_mpa))
        leakoff = float(reservoir.get("leakoff_coefficient", self.params.leakoff_coefficient))
        viscosity = float(reservoir.get("viscosity_cp", self.params.viscosity_cp))

        length = np.maximum(state[:, 0], 10.0)
        height = np.maximum(state[:, 1], 5.0)
        width = np.maximum(state[:, 2], 0.2)
        net_pressure = np.maximum(state[:, 3], 0.2)
        conductivity = np.maximum(state[:, 4], 0.1)

        # PKN-style scaling: length grows with injected volume and rate, but is
        # reduced by leakoff and increasing height. Width grows with net pressure.
        effective_volume = np.maximum(liquid * (1.0 - np.clip(leakoff, 0.0, 0.5)), 0.1)
        rate_factor = np.power(np.maximum(q, 0.05), 0.35)
        viscosity_factor = np.power(max(viscosity, 1.0) / 10.0, 0.08)
        stiffness_factor = np.power(30000.0 / max(e_prime, 1.0), 0.18)
        length_increment = 2.8 * rate_factor * np.power(effective_volume * dt_min, 0.42) * viscosity_factor * stiffness_factor

        next_length = length + length_increment / np.power(height / 45.0, 0.15)
        target_width = 0.45 + 0.052 * net_pressure * np.power(height / 45.0, 0.25) * stiffness_factor
        next_width = 0.72 * width + 0.28 * target_width

        sand_effect = np.log1p(np.maximum(sand, 0.0))
        net_pressure_increment = 0.025 * q + 0.18 * sand_effect - 0.010 * (next_length - length)
        next_net_pressure = np.maximum(0.2, 0.82 * net_pressure + 0.18 * (stress * 0.18 + net_pressure_increment))

        height_target = height + self.params.height_relaxation * np.maximum(next_net_pressure - 6.0, 0.0)
        next_height = np.clip(0.96 * height + 0.04 * height_target, 10.0, 120.0)

        next_conductivity = np.maximum(
            0.1,
            0.86 * conductivity + 0.14 * (next_width**2 * np.log1p(np.maximum(sand, 0.0)) + 0.15 * q),
        )

        next_state = np.column_stack([next_length, next_height, next_width, next_net_pressure, next_conductivity])

        observations = self.observe_state(
            next_state,
            {
                "flow_rate_m3_min": q,
                "liquid_volume_m3": liquid,
                "sand_mass_t": sand,
            },
            reservoir,
        )

        return ForwardResult(
            state=next_state,
            observations=observations,
            state_names=list(self.state_names),
            observation_names=list(self.observation_names),
        )

    def observe_state(
        self,
        state: np.ndarray,
        controls: dict[str, np.ndarray | float],
        reservoir: dict[str, float] | None = None,
    ) -> np.ndarray:
        """Map current fracture state to pressure/DAS/DTS/microseismic observables."""

        reservoir = reservoir or {}
        state = np.asarray(state, dtype=float)
        if state.ndim == 1:
            state = state.reshape(1, -1)
        q = _as_column(controls.get("flow_rate_m3_min", 4.0), len(state))
        liquid = _as_column(controls.get("liquid_volume_m3", 20.0), len(state))
        stress = float(reservoir.get("min_horizontal_stress_mpa", self.params.min_horizontal_stress_mpa))
        length = np.maximum(state[:, 0], 10.0)
        height = np.maximum(state[:, 1], 5.0)
        width = np.maximum(state[:, 2], 0.2)
        net_pressure = np.maximum(state[:, 3], 0.2)
        pressure_obs = stress + net_pressure + 0.004 * length + 0.015 * height
        das_obs = 0.012 * length + 0.20 * width + 0.35 * np.sqrt(np.maximum(q, 0.0))
        dts_obs = 0.028 * liquid / np.maximum(height, 1.0) + 0.025 * width + 0.006 * length
        ms_extent = length * (0.92 + 0.05 * np.tanh((net_pressure - 5.0) / 4.0))
        return np.column_stack([pressure_obs, das_obs, dts_obs, ms_extent])


def _as_column(value: np.ndarray | float | object, rows: int) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(rows, float(arr))
    return np.resize(arr.reshape(-1), rows)
