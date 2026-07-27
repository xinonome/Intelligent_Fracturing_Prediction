from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

from rl.fracturing_env import FracturingControlEnv, FracturingEnvConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DT_ROOT = PROJECT_ROOT / "DT-Crack"
if str(DT_ROOT) not in sys.path:
    sys.path.insert(0, str(DT_ROOT))

from inversion import (  # noqa: E402
    PhysicalEnKFConfig,
    clip_state,
    enkf_update,
    pkn_with_carter_leakoff,
    physical_values,
)


@dataclass(frozen=True)
class DigitalTwinEnvConfig(FracturingEnvConfig):
    n_clusters: int = 6
    ensemble_size: int = 48
    action_seconds: float = 60.0
    initial_elapsed_seconds: float = 300.0
    length_observation_std_m: float = 12.0
    pressure_observation_std_mpa: float = 3.0
    fiber_imbalance_strength: float = 0.12


class ConditionRiskAdapter:
    """Blend available working-condition prior with DT physics indicators.

    The historical FSL transition model has 761 anonymous Column_N features,
    so it cannot be called safely from the compact RL state. Until its feature
    schema is exported, the available probability timeline is used as the model
    prior and physics/action indicators provide online conditioning.
    """

    source = "condition_probability_prior_plus_digital_twin_indicators"

    @staticmethod
    def predict(
        base_abnormal: float,
        base_sand_plug: float,
        bottomhole_mpa: float,
        pressure_limit_mpa: float,
        posterior_error: float,
        sand_delta: float,
        max_sand_step: float,
        cluster_spread: float,
    ) -> tuple[float, float]:
        pressure_risk = np.clip((bottomhole_mpa - 0.82 * pressure_limit_mpa) / max(0.18 * pressure_limit_mpa, 1e-6), 0.0, 1.0)
        jump_risk = np.clip(max(sand_delta, 0.0) / max(max_sand_step, 1e-6), 0.0, 1.0)
        residual_risk = np.clip(posterior_error / 0.15, 0.0, 1.0)
        imbalance_risk = np.clip(cluster_spread / 0.25, 0.0, 1.0)
        abnormal = np.clip(0.45 * base_abnormal + 0.25 * pressure_risk + 0.18 * residual_risk + 0.12 * imbalance_risk, 0.0, 1.0)
        sand_plug = np.clip(0.45 * base_sand_plug + 0.25 * pressure_risk + 0.22 * jump_risk + 0.08 * imbalance_risk, 0.0, 1.0)
        return float(abnormal), float(sand_plug)


class DigitalTwinFracturingControlEnv(FracturingControlEnv):
    """Action-conditioned PKN -> observation -> EnKF -> PKN environment."""

    def __init__(
        self,
        features,
        meta,
        context,
        schedule,
        reward_config,
        env_config: DigitalTwinEnvConfig | None = None,
        seed: int = 2026,
        random_start: bool = True,
    ) -> None:
        cfg = env_config or DigitalTwinEnvConfig()
        super().__init__(
            features,
            meta,
            context,
            schedule,
            reward_config,
            env_config=cfg,
            seed=seed,
            random_start=random_start,
        )
        self.dt_config = cfg
        self.physics_config = PhysicalEnKFConfig()
        self._ensemble = np.empty((0, 0))
        self._truth_state = np.empty(0)
        self._last_dt_diagnostics: dict[str, float | str] = {}

    def _initialize_twin(self) -> None:
        n = self.dt_config.n_clusters
        mean = np.r_[0.0, 0.0, 0.0, 60.0, np.ones(n)]
        spread = np.r_[0.12, 0.30, 0.20, 4.0, np.full(n, 0.06)]
        self._ensemble = clip_state(
            mean + self.np_random.normal(0.0, spread, size=(self.dt_config.ensemble_size, len(mean))), n
        )
        imbalance = float(self.context.iloc[self._cursor].get("scenario_balance_imbalance", 0.08))
        pattern = np.linspace(-1.0, 1.0, n)
        truth_factors = np.clip(1.0 + imbalance * pattern, 0.70, 1.30)
        self._truth_state = clip_state(np.r_[0.08, -0.12, 0.10, 64.0, truth_factors], n)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        observation, info = super().reset(seed=seed, options=options)
        self._initialize_twin()
        return self._observation(), {**info, "response_model": "pkn_enkf_digital_twin"}

    def _simulate_response(self, flow: float, sand: float) -> dict[str, float]:
        n = self.dt_config.n_clusters
        elapsed = self.dt_config.initial_elapsed_seconds + (self._steps + 1) * self.dt_config.action_seconds
        q_total = max(flow, 0.05) / 60.0
        q_base = np.full(n, q_total / n, dtype=float)
        sand_delta = sand - self._current_sand
        process = np.r_[0.008, 0.015, 0.010, 0.15, np.full(n, 0.008)]
        self._ensemble = clip_state(
            self._ensemble + self.np_random.normal(0.0, process, size=self._ensemble.shape), n
        )

        prior_state = self._ensemble.mean(axis=0)
        prior = pkn_with_carter_leakoff(prior_state, q_base, elapsed, self.physics_config)
        ensemble_predictions = [pkn_with_carter_leakoff(x, q_base, elapsed, self.physics_config) for x in self._ensemble]
        predicted_obs = np.column_stack([
            np.asarray([p["half_length_m"] for p in ensemble_predictions]),
            np.asarray([p["bottomhole_pressure_mpa"] for p in ensemble_predictions]),
        ])

        # Synthetic field response for scenario training. In deployment these
        # two observations are replaced by DAS observation-operator output and
        # measured bottom-hole pressure.
        truth = pkn_with_carter_leakoff(self._truth_state, q_base, elapsed, self.physics_config)
        observed_lengths = np.asarray(truth["half_length_m"]) + self.np_random.normal(0.0, self.dt_config.length_observation_std_m, n)
        historical_pressure = float(self.meta.iloc[self._cursor]["current_pressure"])
        observed_bhp = 0.65 * float(truth["bottomhole_pressure_mpa"]) + 0.35 * historical_pressure
        observed_bhp += self.np_random.normal(0.0, self.dt_config.pressure_observation_std_mpa)
        observed_obs = np.r_[observed_lengths, observed_bhp]
        obs_std = np.r_[np.full(n, self.dt_config.length_observation_std_m), self.dt_config.pressure_observation_std_mpa]

        self._ensemble, gain = enkf_update(
            self._ensemble, predicted_obs, observed_obs, obs_std, self.np_random
        )
        self._ensemble = clip_state(self._ensemble, n)
        posterior_state = self._ensemble.mean(axis=0)
        posterior = pkn_with_carter_leakoff(posterior_state, q_base, elapsed, self.physics_config)
        posterior_lengths = np.asarray(posterior["half_length_m"])
        posterior_error = float(np.mean(np.abs(posterior_lengths - observed_lengths) / np.maximum(np.abs(observed_lengths), 1.0)))

        base = self._base_context(self._cursor)
        base_abnormal = float(np.nan_to_num(base.get("abnormal_probability", 0.04), nan=0.04))
        base_sand_plug = float(np.nan_to_num(base.get("sand_plug_probability", 0.02), nan=0.02))
        factors = np.asarray(physical_values(posterior_state, self.physics_config, n)["cluster_factors"])
        abnormal, sand_plug = ConditionRiskAdapter.predict(
            base_abnormal, base_sand_plug, float(posterior["bottomhole_pressure_mpa"]),
            self.reward_config.bottomhole_pressure_max_mpa, posterior_error, sand_delta,
            self.schedule.max_sand_increase_percent, float(np.ptp(factors)),
        )

        self._last_dt_diagnostics = {
            "response_model": "pkn_enkf_digital_twin",
            "condition_probability_source": ConditionRiskAdapter.source,
            "prior_total_half_length_m": float(np.sum(prior["half_length_m"])),
            "observed_total_half_length_m": float(np.sum(observed_lengths)),
            "posterior_total_half_length_m": float(np.sum(posterior_lengths)),
            "mean_abs_kalman_gain": float(np.mean(np.abs(gain))),
            "posterior_eprime_gpa": float(physical_values(posterior_state, self.physics_config, n)["eprime_pa"]) / 1e9,
            "posterior_leakoff_m_sqrt_s": float(physical_values(posterior_state, self.physics_config, n)["leakoff_m_sqrt_s"]),
            "posterior_viscosity_pa_s": float(physical_values(posterior_state, self.physics_config, n)["viscosity_pa_s"]),
            "posterior_min_stress_mpa": float(physical_values(posterior_state, self.physics_config, n)["min_horizontal_stress_mpa"]),
        }
        return {
            "pressure": float(posterior["bottomhole_pressure_mpa"]),
            "length": float(np.sum(posterior_lengths)),
            "posterior_error": posterior_error,
            "bottomhole_pressure_mpa": float(posterior["bottomhole_pressure_mpa"]),
            "net_pressure_mpa": float(posterior["net_pressure_mpa"]),
            "abnormal_probability": abnormal,
            "sand_plug_probability": sand_plug,
        }

    def step(self, action: np.ndarray):
        observation, reward, terminated, truncated, info = super().step(action)
        info.update(self._last_dt_diagnostics)
        return observation, reward, terminated, truncated, info
