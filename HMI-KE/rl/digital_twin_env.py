from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
from gymnasium import spaces

from rl.fracturing_env import (
    FracturingControlEnv,
    FracturingEnvConfig,
    HierarchicalFracturingControlEnv,
    HierarchicalFracturingEnvConfig,
)


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
    minimum_pkn_flow_m3_min: float = 1.0
    low_flow_pressure_relaxation: float = 0.25
    low_flow_abnormal_decay: float = 0.72
    low_flow_sand_plug_decay: float = 0.62


@dataclass(frozen=True)
class HierarchicalDigitalTwinEnvConfig(DigitalTwinEnvConfig, HierarchicalFracturingEnvConfig):
    """Combined configuration for option-level control and PKN-EnKF response."""


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
        flow_delta: float,
        max_flow_step: float,
        sand_delta: float,
        max_sand_step: float,
        cluster_spread: float,
    ) -> tuple[float, float]:
        pressure_risk = np.clip((bottomhole_mpa - 0.82 * pressure_limit_mpa) / max(0.18 * pressure_limit_mpa, 1e-6), 0.0, 1.0)
        jump_risk = np.clip(max(sand_delta, 0.0) / max(max_sand_step, 1e-6), 0.0, 1.0)
        residual_risk = np.clip(posterior_error / 0.15, 0.0, 1.0)
        imbalance_risk = np.clip(cluster_spread / 0.25, 0.0, 1.0)
        flow_relief = np.clip(max(-flow_delta, 0.0) / max(max_flow_step, 1e-6), 0.0, 1.0)
        sand_relief = np.clip(max(-sand_delta, 0.0) / max(max_sand_step, 1e-6), 0.0, 1.0)
        relief = np.clip(0.35 * flow_relief + 0.65 * sand_relief, 0.0, 1.0)
        abnormal = np.clip(0.45 * base_abnormal + 0.25 * pressure_risk + 0.18 * residual_risk + 0.12 * imbalance_risk, 0.0, 1.0)
        sand_plug = np.clip(0.45 * base_sand_plug + 0.25 * pressure_risk + 0.22 * jump_risk + 0.08 * imbalance_risk, 0.0, 1.0)
        abnormal *= 1.0 - 0.45 * relief
        sand_plug *= 1.0 - 0.65 * relief
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
        response_surrogate=None,
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
        self._cumulative_injected_volume_m3 = 0.0
        self._abnormal_memory = 0.0
        self._sand_plug_memory = 0.0
        self._previous_cluster_lengths = np.zeros(cfg.n_clusters, dtype=float)
        self.response_surrogate = response_surrogate

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
        self._cumulative_injected_volume_m3 = (
            max(self._current_flow, 0.0) / 60.0 * self.dt_config.initial_elapsed_seconds
        )
        base = self._base_context(self._cursor)
        self._abnormal_memory = float(np.nan_to_num(base.get("abnormal_probability", 0.04), nan=0.04))
        self._sand_plug_memory = float(np.nan_to_num(base.get("sand_plug_probability", 0.02), nan=0.02))
        self._previous_cluster_lengths = np.full(
            self.dt_config.n_clusters,
            max(float(self._previous_length), 0.0) / max(self.dt_config.n_clusters, 1),
            dtype=float,
        )
        return self._observation(), {**info, "response_model": "pkn_enkf_digital_twin"}

    def _simulate_low_flow_response(self, flow: float, sand: float) -> dict[str, float]:
        """Handle shut-in/very-low-rate windows outside the PKN propagation regime."""
        base = self._base_context(self._cursor)
        target_pressure = float(self.meta.iloc[self._cursor]["current_pressure"])
        pressure = self._current_pressure + self.dt_config.low_flow_pressure_relaxation * (
            target_pressure - self._current_pressure
        )
        bottomhole = float(base.get("bottomhole_pressure_mpa", pressure))
        if not np.isfinite(bottomhole):
            bottomhole = pressure
        bottomhole += pressure - self._current_pressure
        net_pressure = float(base.get("net_pressure_mpa", max(bottomhole - 70.0, 0.0)))
        if not np.isfinite(net_pressure):
            net_pressure = max(bottomhole - 70.0, 0.0)
        net_pressure = max(net_pressure + pressure - self._current_pressure, 0.0)
        sand_delta = sand - self._current_sand
        abnormal = self.dt_config.low_flow_abnormal_decay * self._abnormal_memory
        sand_plug = self.dt_config.low_flow_sand_plug_decay * self._sand_plug_memory
        if sand_delta > 0.0:
            sand_plug += 0.15 * sand_delta / max(self.schedule.max_sand_increase_percent, 1e-6)
        self._abnormal_memory = float(np.clip(abnormal, 0.0, 1.0))
        self._sand_plug_memory = float(np.clip(sand_plug, 0.0, 1.0))
        posterior_error = float(base.get("posterior_error", self.reward_config.target_posterior_error))
        if not np.isfinite(posterior_error):
            posterior_error = self.reward_config.target_posterior_error
        physical = physical_values(self._ensemble.mean(axis=0), self.physics_config, self.dt_config.n_clusters)
        self._last_dt_diagnostics = {
            "response_model": "low_flow_pressure_relaxation",
            "condition_probability_source": "risk_memory_decay_without_pkn_propagation",
            "prior_total_half_length_m": float(self._previous_length),
            "observed_total_half_length_m": float(self._previous_length),
            "posterior_total_half_length_m": float(self._previous_length),
            "mean_abs_kalman_gain": 0.0,
            "posterior_eprime_gpa": float(physical["eprime_pa"]) / 1e9,
            "posterior_leakoff_m_sqrt_s": float(physical["leakoff_m_sqrt_s"]),
            "posterior_viscosity_pa_s": float(physical["viscosity_pa_s"]),
            "posterior_min_stress_mpa": float(physical["min_horizontal_stress_mpa"]),
            "cumulative_injected_volume_m3": float(self._cumulative_injected_volume_m3),
            "current_total_rate_m3_s": max(float(flow), 0.0) / 60.0,
            "pkn_update_skipped": True,
        }
        return {
            "pressure": float(pressure),
            "length": float(self._previous_length),
            "posterior_error": float(np.clip(posterior_error, 0.0, 1.0)),
            "bottomhole_pressure_mpa": float(bottomhole),
            "net_pressure_mpa": float(net_pressure),
            "abnormal_probability": self._abnormal_memory,
            "sand_plug_probability": self._sand_plug_memory,
        }

    def _simulate_response(self, flow: float, sand: float) -> dict[str, float]:
        if flow < self.dt_config.minimum_pkn_flow_m3_min:
            return self._simulate_low_flow_response(flow, sand)
        n = self.dt_config.n_clusters
        elapsed = self.dt_config.initial_elapsed_seconds + (self._steps + 1) * self.dt_config.action_seconds
        q_current_total = max(flow, 0.05) / 60.0
        self._cumulative_injected_volume_m3 += q_current_total * self.dt_config.action_seconds
        q_history_total = self._cumulative_injected_volume_m3 / max(elapsed, 1.0)
        q_base = np.full(n, q_history_total / n, dtype=float)
        q_current = np.full(n, q_current_total / n, dtype=float)
        sand_delta = sand - self._current_sand
        flow_delta = flow - self._current_flow
        process = np.r_[0.008, 0.015, 0.010, 0.15, np.full(n, 0.008)]
        self._ensemble = clip_state(
            self._ensemble + self.np_random.normal(0.0, process, size=self._ensemble.shape), n
        )

        prior_state = self._ensemble.mean(axis=0)
        prior = pkn_with_carter_leakoff(prior_state, q_base, elapsed, self.physics_config, q_current)
        ensemble_predictions = [
            pkn_with_carter_leakoff(x, q_base, elapsed, self.physics_config, q_current)
            for x in self._ensemble
        ]
        predicted_obs = np.column_stack([
            np.asarray([p["half_length_m"] for p in ensemble_predictions]),
            np.asarray([p["bottomhole_pressure_mpa"] for p in ensemble_predictions]),
        ])

        # Synthetic field response for scenario training. In deployment these
        # two observations are replaced by DAS observation-operator output and
        # measured bottom-hole pressure.
        truth = pkn_with_carter_leakoff(self._truth_state, q_base, elapsed, self.physics_config, q_current)
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
        posterior = pkn_with_carter_leakoff(posterior_state, q_base, elapsed, self.physics_config, q_current)
        posterior_lengths = np.maximum(
            np.asarray(posterior["half_length_m"], dtype=float),
            self._previous_cluster_lengths,
        )
        self._previous_cluster_lengths = posterior_lengths.copy()
        posterior_error = float(np.mean(np.abs(posterior_lengths - observed_lengths) / np.maximum(np.abs(observed_lengths), 1.0)))

        base = self._base_context(self._cursor)
        base_abnormal = float(np.nan_to_num(base.get("abnormal_probability", 0.04), nan=0.04))
        base_sand_plug = float(np.nan_to_num(base.get("sand_plug_probability", 0.02), nan=0.02))
        factors = np.asarray(physical_values(posterior_state, self.physics_config, n)["cluster_factors"])
        abnormal, sand_plug = ConditionRiskAdapter.predict(
            base_abnormal, base_sand_plug, float(posterior["bottomhole_pressure_mpa"]),
            self.reward_config.bottomhole_pressure_max_mpa, posterior_error,
            flow_delta, self.schedule.max_flow_step_m3_min, sand_delta,
            self.schedule.max_sand_increase_percent, float(np.ptp(factors)),
        )
        surrogate_result = None
        posterior_bhp = float(posterior["bottomhole_pressure_mpa"])
        posterior_net = float(posterior["net_pressure_mpa"])
        if self.response_surrogate is not None:
            surrogate_result = self.response_surrogate.predict_one(
                self.features[self._cursor], self.meta.iloc[self._cursor], flow, sand
            )
            # The real-data surrogate learns surface-pressure and condition
            # residuals. Geometry and physical parameters remain PKN-EnKF outputs.
            surface_delta = surrogate_result["pressure_mean"] - float(self.meta.iloc[self._cursor]["current_pressure"])
            posterior_bhp += surface_delta
            posterior_net = max(posterior_net + surface_delta, 0.0)
            abnormal = 0.65 * surrogate_result["abnormal_probability"] + 0.35 * abnormal
            sand_plug = 0.65 * surrogate_result["sand_plug_probability"] + 0.35 * sand_plug
        # Risk persists over the following decision windows. A conservative
        # action can dissipate it gradually, but one safe point cannot erase a
        # sand-plug warning immediately.
        relief = np.clip(
            0.35 * max(-flow_delta, 0.0) / max(self.schedule.max_flow_step_m3_min, 1e-6)
            + 0.65 * max(-sand_delta, 0.0) / max(self.schedule.max_sand_increase_percent, 1e-6),
            0.0,
            1.0,
        )
        abnormal = max(abnormal, (0.82 - 0.30 * relief) * self._abnormal_memory)
        sand_plug = max(sand_plug, (0.88 - 0.38 * relief) * self._sand_plug_memory)
        self._abnormal_memory = float(np.clip(abnormal, 0.0, 1.0))
        self._sand_plug_memory = float(np.clip(sand_plug, 0.0, 1.0))

        self._last_dt_diagnostics = {
            "response_model": "learned_action_response_plus_pkn_enkf" if surrogate_result else "pkn_enkf_digital_twin",
            "condition_probability_source": "real_segment_action_response_surrogate" if surrogate_result else ConditionRiskAdapter.source,
            "prior_total_half_length_m": float(np.sum(prior["half_length_m"])),
            "observed_total_half_length_m": float(np.sum(observed_lengths)),
            "posterior_total_half_length_m": float(np.sum(posterior_lengths)),
            "mean_abs_kalman_gain": float(np.mean(np.abs(gain))),
            "posterior_eprime_gpa": float(physical_values(posterior_state, self.physics_config, n)["eprime_pa"]) / 1e9,
            "posterior_leakoff_m_sqrt_s": float(physical_values(posterior_state, self.physics_config, n)["leakoff_m_sqrt_s"]),
            "posterior_viscosity_pa_s": float(physical_values(posterior_state, self.physics_config, n)["viscosity_pa_s"]),
            "posterior_min_stress_mpa": float(physical_values(posterior_state, self.physics_config, n)["min_horizontal_stress_mpa"]),
            "cumulative_injected_volume_m3": float(self._cumulative_injected_volume_m3),
            "current_total_rate_m3_s": float(q_current_total),
            "pkn_update_skipped": False,
            "surrogate_ood_score": float(surrogate_result["ood_score"]) if surrogate_result else 0.0,
            "surrogate_future_surface_pressure_mpa": float(surrogate_result["pressure_mean"]) if surrogate_result else np.nan,
            "surrogate_future_surface_pressure_max_mpa": float(surrogate_result["pressure_max"]) if surrogate_result else np.nan,
        }
        return {
            "pressure": posterior_bhp,
            "length": float(np.sum(posterior_lengths)),
            "posterior_error": posterior_error,
            "bottomhole_pressure_mpa": posterior_bhp,
            "net_pressure_mpa": posterior_net,
            "abnormal_probability": abnormal,
            "sand_plug_probability": sand_plug,
        }

    def step(self, action: np.ndarray):
        observation, reward, terminated, truncated, info = super().step(action)
        info.update(self._last_dt_diagnostics)
        return observation, reward, terminated, truncated, info


class HierarchicalDigitalTwinFracturingControlEnv(
    HierarchicalFracturingControlEnv,
    DigitalTwinFracturingControlEnv,
):
    """Rule-selected high-level option plus PPO/SAC low-level DT controller."""

    def __init__(
        self,
        features,
        meta,
        context,
        schedule,
        reward_config,
        env_config: HierarchicalDigitalTwinEnvConfig | None = None,
        seed: int = 2026,
        random_start: bool = True,
        response_surrogate=None,
    ) -> None:
        DigitalTwinFracturingControlEnv.__init__(
            self,
            features,
            meta,
            context,
            schedule,
            reward_config,
            env_config or HierarchicalDigitalTwinEnvConfig(),
            seed,
            random_start,
            response_surrogate=response_surrogate,
        )
        self.hierarchical_config = self.config
        self._current_option = 0
        self._option_age = 0
        base_size = int(self.observation_space.shape[0])
        self.observation_space = spaces.Box(
            -10.0, 10.0, shape=(base_size + len(self.OPTIONS) + 1,), dtype=np.float32
        )
