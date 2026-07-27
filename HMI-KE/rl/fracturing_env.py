from __future__ import annotations

from dataclasses import asdict, dataclass

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from decision_engine.integrated_reward import IntegratedRewardConfig, calculate_integrated_reward
from decision_engine.pump_schedule_constraints import (
    PumpScheduleConstraint,
    constrain_actions,
    schedule_reward,
)


@dataclass(frozen=True)
class FracturingEnvConfig:
    episode_steps: int = 60
    pressure_action_gain: float = 0.35
    pressure_sand_gain: float = 0.18
    pressure_relaxation: float = 0.12
    fracture_flow_exponent: float = 0.62
    fracture_sand_gain: float = 0.35
    abnormal_pressure_gain: float = 0.35
    abnormal_sand_gain: float = 0.25
    unsafe_termination_penalty: float = 8.0
    terminate_on_unsafe: bool = False
    schedule_reward_weight: float = 0.25
    reward_clip: float = 20.0
    severe_pressure_multiplier: float = 2.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class HierarchicalFracturingEnvConfig(FracturingEnvConfig):
    high_level_interval_steps: int = 6
    high_pressure_ratio: float = 0.92
    medium_abnormal_probability: float = 0.25
    high_abnormal_probability: float = 0.45
    high_posterior_error: float = 0.15


class FracturingControlEnv(gym.Env):
    """Offline action-conditioned simulator for 60-second advisory actions.

    The environment replays historical 300-second state windows. A candidate
    action changes the simulated pressure, fracture increment and abnormal risk;
    these simulated responses feed the integrated reward. This is a training
    surrogate, not a replacement for a calibrated field simulator.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        features: np.ndarray,
        meta: pd.DataFrame,
        context: pd.DataFrame,
        schedule: PumpScheduleConstraint,
        reward_config: IntegratedRewardConfig,
        env_config: FracturingEnvConfig | None = None,
        seed: int = 2026,
        random_start: bool = True,
    ) -> None:
        super().__init__()
        if len(features) != len(meta) or len(features) != len(context):
            raise ValueError("features, meta and reward context must have equal lengths")
        if len(features) < 2:
            raise ValueError("At least two samples are required")
        self.features = np.nan_to_num(np.asarray(features, dtype=np.float32))
        self.meta = meta.reset_index(drop=True).copy()
        self.context = context.reset_index(drop=True).copy()
        self.schedule = schedule
        self.reward_config = reward_config
        self.config = env_config or FracturingEnvConfig()
        self.random_start = random_start
        self._rng = np.random.default_rng(seed)
        self._cursor = 0
        self._steps = 0
        self._current_flow = 0.0
        self._current_sand = 0.0
        self._current_pressure = 0.0
        self._previous_length = 1.0
        self._pressure_reference = float(np.nanmedian(self.meta["current_pressure"]))
        self._pressure_scale = max(float(np.nanstd(self.meta["current_pressure"])), 1.0)

        context_columns = [
            "posterior_total_half_length_m",
            "posterior_error",
            "bottomhole_pressure_mpa",
            "net_pressure_mpa",
            "abnormal_probability",
            "sand_plug_probability",
        ]
        self.context_columns = [col for col in context_columns if col in self.context]
        observation_size = self.features.shape[1] + len(self.context_columns) + 3
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(observation_size,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self._obs_mean = self.features.mean(axis=0)
        self._obs_std = np.maximum(self.features.std(axis=0), 1e-5)

    def _base_context(self, index: int) -> dict[str, float]:
        row = self.context.iloc[index]
        return {
            col: float(row[col]) if pd.notna(row[col]) else np.nan
            for col in self.context_columns
        }

    def _observation(self) -> np.ndarray:
        state = np.clip((self.features[self._cursor] - self._obs_mean) / self._obs_std, -10.0, 10.0)
        base = self._base_context(self._cursor)
        context_values = []
        for col in self.context_columns:
            value = base[col]
            if not np.isfinite(value):
                value = 0.0
            if "pressure" in col:
                value = (value - self._pressure_reference) / self._pressure_scale
            elif "length" in col:
                value = np.log1p(max(value, 0.0)) / 10.0
            context_values.append(value)
        controls = [
            self._current_pressure / max(self._pressure_reference, 1.0),
            self._current_flow / max(self.schedule.max_flow_m3_min, 1.0),
            self._current_sand / max(self.schedule.max_sand_ratio_percent, 1.0),
        ]
        return np.asarray([*state, *context_values, *controls], dtype=np.float32)

    def _decode_action(self, action: np.ndarray) -> tuple[float, float, bool]:
        raw = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        flow_low = max(0.0, self._current_flow - self.schedule.max_flow_step_m3_min)
        flow_high = min(self.schedule.max_flow_m3_min, self._current_flow + self.schedule.max_flow_step_m3_min)
        proposed_flow = flow_low + (raw[0] + 1.0) * 0.5 * (flow_high - flow_low)
        if self.schedule.allow_sand_pause:
            sand_low = max(0.0, self._current_sand - self.schedule.max_sand_increase_percent)
        else:
            sand_low = self._current_sand
        sand_high = min(
            self.schedule.max_sand_ratio_percent,
            self._current_sand + self.schedule.max_sand_increase_percent,
        )
        proposed_sand = sand_low + (raw[1] + 1.0) * 0.5 * (sand_high - sand_low)
        safe_flow, safe_sand, diagnostics = constrain_actions(
            np.array([proposed_flow]),
            np.array([proposed_sand]),
            np.array([self._current_flow]),
            np.array([self._current_sand]),
            self.schedule,
        )
        clipped = bool(diagnostics["flow_was_clipped"][0] or diagnostics["sand_was_clipped"][0])
        return float(safe_flow[0]), float(safe_sand[0]), clipped

    def encode_engineering_action(self, flow: float, sand: float) -> np.ndarray:
        """Convert an engineering-unit action to the environment action coordinates."""
        flow_low = max(0.0, self._current_flow - self.schedule.max_flow_step_m3_min)
        flow_high = min(self.schedule.max_flow_m3_min, self._current_flow + self.schedule.max_flow_step_m3_min)
        flow_action = 2.0 * (float(flow) - flow_low) / max(flow_high - flow_low, 1e-6) - 1.0
        sand_low = max(0.0, self._current_sand - self.schedule.max_sand_increase_percent) if self.schedule.allow_sand_pause else self._current_sand
        sand_high = min(self.schedule.max_sand_ratio_percent, self._current_sand + self.schedule.max_sand_increase_percent)
        sand_action = 2.0 * (float(sand) - sand_low) / max(sand_high - sand_low, 1e-6) - 1.0
        return np.clip(np.array([flow_action, sand_action], dtype=np.float32), -1.0, 1.0)

    def _simulate_response(self, flow: float, sand: float) -> dict[str, float]:
        base = self._base_context(self._cursor)
        flow_delta = flow - self._current_flow
        sand_delta = sand - self._current_sand
        historical_pressure = float(self.meta.iloc[self._cursor]["current_pressure"])
        simulated_pressure = (
            self._current_pressure
            + self.config.pressure_action_gain * flow_delta
            + self.config.pressure_sand_gain * sand_delta
            + self.config.pressure_relaxation * (historical_pressure - self._current_pressure)
        )
        base_length = base.get("posterior_total_half_length_m", self._previous_length)
        if not np.isfinite(base_length):
            base_length = self._previous_length
        normalized_flow = max(flow, 0.0) / max(self.schedule.max_flow_m3_min, 1e-6)
        normalized_sand = max(sand, 0.0) / max(self.schedule.max_sand_ratio_percent, 1e-6)
        base_increment = max(base_length - self._previous_length, 0.0)
        if base_increment <= 1e-6:
            base_increment = max(base_length, 1.0) * 0.002
        fracture_increment = base_increment * (
            0.35
            + normalized_flow ** self.config.fracture_flow_exponent
            + self.config.fracture_sand_gain * normalized_sand
        )
        simulated_length = self._previous_length + max(fracture_increment, 0.0)

        base_abnormal = base.get("abnormal_probability", 0.0)
        base_sand_plug = base.get("sand_plug_probability", 0.0)
        pressure_rise = max(simulated_pressure - historical_pressure, 0.0) / self._pressure_scale
        aggressive_sand = max(sand_delta, 0.0) / max(self.schedule.max_sand_increase_percent, 1e-6)
        abnormal = np.clip(
            base_abnormal
            + self.config.abnormal_pressure_gain * pressure_rise
            + self.config.abnormal_sand_gain * aggressive_sand,
            0.0,
            1.0,
        )
        sand_plug = np.clip(base_sand_plug + 0.5 * pressure_rise + 0.3 * aggressive_sand, 0.0, 1.0)
        posterior_error = base.get("posterior_error", self.reward_config.target_posterior_error)
        if not np.isfinite(posterior_error):
            posterior_error = self.reward_config.target_posterior_error
        posterior_error = float(np.clip(posterior_error + 0.04 * pressure_rise - 0.02 * normalized_flow, 0.0, 1.0))
        bottomhole = base.get("bottomhole_pressure_mpa", simulated_pressure)
        if not np.isfinite(bottomhole):
            bottomhole = simulated_pressure
        bottomhole += simulated_pressure - historical_pressure
        net_pressure = base.get("net_pressure_mpa", max(bottomhole - 70.0, 0.0))
        if not np.isfinite(net_pressure):
            net_pressure = max(bottomhole - 70.0, 0.0)
        net_pressure += simulated_pressure - historical_pressure
        return {
            "pressure": float(simulated_pressure),
            "length": float(simulated_length),
            "posterior_error": posterior_error,
            "bottomhole_pressure_mpa": float(bottomhole),
            "net_pressure_mpa": float(net_pressure),
            "abnormal_probability": float(abnormal),
            "sand_plug_probability": float(sand_plug),
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        max_start = max(0, len(self.features) - self.config.episode_steps - 1)
        if options and "start_index" in options:
            self._cursor = int(np.clip(options["start_index"], 0, max_start))
        elif self.random_start and max_start > 0:
            self._cursor = int(self.np_random.integers(0, max_start + 1))
        else:
            self._cursor = 0
        self._steps = 0
        row = self.meta.iloc[self._cursor]
        self._current_pressure = float(row["current_pressure"])
        self._current_flow = float(np.clip(row["current_flow"], 0.0, self.schedule.max_flow_m3_min))
        self._current_sand = float(np.clip(row["current_sand_ratio"], 0.0, self.schedule.max_sand_ratio_percent))
        base_length = self._base_context(self._cursor).get("posterior_total_half_length_m", 1.0)
        self._previous_length = float(base_length) if np.isfinite(base_length) else 1.0
        return self._observation(), {"start_index": self._cursor}

    def step(self, action: np.ndarray):
        flow, sand, action_clipped = self._decode_action(action)
        response = self._simulate_response(flow, sand)
        two_step_context = pd.DataFrame(
            {
                "posterior_total_half_length_m": [self._previous_length, response["length"]],
                "posterior_error": [response["posterior_error"], response["posterior_error"]],
                "bottomhole_pressure_mpa": [response["bottomhole_pressure_mpa"]] * 2,
                "net_pressure_mpa": [response["net_pressure_mpa"]] * 2,
                "abnormal_probability": [response["abnormal_probability"]] * 2,
                "sand_plug_probability": [response["sand_plug_probability"]] * 2,
            }
        )
        integrated = calculate_integrated_reward(
            np.array([self._current_flow, flow]),
            np.array([self._current_sand, sand]),
            np.array([self._current_flow, self._current_flow]),
            np.array([self._current_sand, self._current_sand]),
            two_step_context,
            self.schedule.max_flow_m3_min,
            self.schedule.max_sand_ratio_percent,
            self.reward_config,
        )
        schedule = schedule_reward(
            np.array([flow]),
            np.array([sand]),
            np.array([self._current_flow]),
            np.array([self._current_sand]),
            self.schedule,
        )
        reward_components = {
            name: float(values[-1])
            for name, values in integrated.items()
            if name.endswith("reward") or name.endswith("penalty")
        }
        reward = reward_components["integrated_reward"] + self.config.schedule_reward_weight * float(schedule["total_reward"][0])
        unsafe = (
            response["bottomhole_pressure_mpa"]
            > self.reward_config.bottomhole_pressure_max_mpa * self.config.severe_pressure_multiplier
            or response["net_pressure_mpa"]
            > self.reward_config.net_pressure_max_mpa * self.config.severe_pressure_multiplier
        )
        if unsafe and self.config.terminate_on_unsafe:
            reward -= self.config.unsafe_termination_penalty
        reward = float(np.clip(reward, -self.config.reward_clip, self.config.reward_clip))

        self._current_flow = flow
        self._current_sand = sand
        self._current_pressure = response["pressure"]
        self._previous_length = response["length"]
        self._steps += 1
        self._cursor += 1
        terminated = bool((unsafe and self.config.terminate_on_unsafe) or self._cursor >= len(self.features) - 1)
        truncated = bool(self._steps >= self.config.episode_steps)
        info = {
            "flow_m3_min": flow,
            "sand_ratio_percent": sand,
            "action_clipped": action_clipped,
            "simulated_pressure_mpa": response["pressure"],
            "simulated_half_length_m": response["length"],
            "bottomhole_pressure_mpa": response["bottomhole_pressure_mpa"],
            "net_pressure_mpa": response["net_pressure_mpa"],
            "posterior_error": response["posterior_error"],
            "abnormal_probability": response["abnormal_probability"],
            "sand_plug_probability": response["sand_plug_probability"],
            "unsafe": unsafe,
            **reward_components,
        }
        return self._observation(), reward, terminated, truncated, info


class HierarchicalFracturingControlEnv(FracturingControlEnv):
    """Lightweight option-based HRL prototype on top of the continuous env.

    The high level is a rule/knowledge option policy that selects an operational
    intent every few steps. PPO/SAC still learns the low-level continuous action
    inside the intent-specific action envelope. This keeps the implementation on
    Gymnasium + Stable-Baselines3 while exposing a hierarchical control surface.
    """

    OPTIONS = ("hold", "grow", "divert", "safe")

    def __init__(
        self,
        features: np.ndarray,
        meta: pd.DataFrame,
        context: pd.DataFrame,
        schedule: PumpScheduleConstraint,
        reward_config: IntegratedRewardConfig,
        env_config: HierarchicalFracturingEnvConfig | None = None,
        seed: int = 2026,
        random_start: bool = True,
    ) -> None:
        super().__init__(
            features,
            meta,
            context,
            schedule,
            reward_config,
            env_config or HierarchicalFracturingEnvConfig(),
            seed,
            random_start,
        )
        self.hierarchical_config = self.config
        self._current_option = 0
        self._option_age = 0
        base_size = int(self.observation_space.shape[0])
        self.observation_space = spaces.Box(
            -10.0,
            10.0,
            shape=(base_size + len(self.OPTIONS) + 1,),
            dtype=np.float32,
        )

    def _select_high_level_option(self) -> int:
        base = self._base_context(self._cursor)
        pressure_limit = self.reward_config.bottomhole_pressure_max_mpa
        bottomhole = base.get("bottomhole_pressure_mpa", self._current_pressure)
        abnormal = base.get("abnormal_probability", 0.0)
        sand_plug = base.get("sand_plug_probability", 0.0)
        posterior_error = base.get("posterior_error", 0.0)
        if not np.isfinite(bottomhole):
            bottomhole = self._current_pressure
        if not np.isfinite(abnormal):
            abnormal = 0.0
        if not np.isfinite(sand_plug):
            sand_plug = 0.0
        if not np.isfinite(posterior_error):
            posterior_error = 0.0

        risk = max(float(abnormal), float(sand_plug))
        if bottomhole > pressure_limit * self.hierarchical_config.high_pressure_ratio or risk >= self.hierarchical_config.high_abnormal_probability:
            return self.OPTIONS.index("safe")
        if risk >= self.hierarchical_config.medium_abnormal_probability:
            return self.OPTIONS.index("divert")
        if posterior_error > self.hierarchical_config.high_posterior_error:
            return self.OPTIONS.index("hold")
        return self.OPTIONS.index("grow")

    def _observation(self) -> np.ndarray:
        base_obs = super()._observation()
        one_hot = np.zeros(len(self.OPTIONS), dtype=np.float32)
        one_hot[self._current_option] = 1.0
        option_progress = min(
            self._option_age / max(float(self.hierarchical_config.high_level_interval_steps), 1.0),
            1.0,
        )
        return np.asarray([*base_obs, *one_hot, option_progress], dtype=np.float32)

    def _decode_action(self, action: np.ndarray) -> tuple[float, float, bool]:
        raw = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        option = self.OPTIONS[self._current_option]
        flow_step = self.schedule.max_flow_step_m3_min
        sand_step = self.schedule.max_sand_increase_percent

        if option == "grow":
            flow_low = self._current_flow
            flow_high = min(self.schedule.max_flow_m3_min, self._current_flow + flow_step)
            sand_low = self._current_sand
            sand_high = min(self.schedule.max_sand_ratio_percent, self._current_sand + sand_step)
        elif option == "safe":
            flow_low = max(0.0, self._current_flow - flow_step)
            flow_high = self._current_flow
            sand_low = max(0.0, self._current_sand - sand_step) if self.schedule.allow_sand_pause else self._current_sand
            sand_high = self._current_sand
        elif option == "divert":
            flow_low = max(0.0, self._current_flow - 0.5 * flow_step)
            flow_high = min(self.schedule.max_flow_m3_min, self._current_flow + 0.5 * flow_step)
            sand_low = self._current_sand
            sand_high = min(self.schedule.max_sand_ratio_percent, self._current_sand + 0.5 * sand_step)
        else:
            flow_low = max(0.0, self._current_flow - 0.25 * flow_step)
            flow_high = min(self.schedule.max_flow_m3_min, self._current_flow + 0.25 * flow_step)
            sand_low = self._current_sand
            sand_high = min(self.schedule.max_sand_ratio_percent, self._current_sand + 0.25 * sand_step)

        proposed_flow = flow_low + (raw[0] + 1.0) * 0.5 * max(flow_high - flow_low, 0.0)
        proposed_sand = sand_low + (raw[1] + 1.0) * 0.5 * max(sand_high - sand_low, 0.0)
        safe_flow, safe_sand, diagnostics = constrain_actions(
            np.array([proposed_flow]),
            np.array([proposed_sand]),
            np.array([self._current_flow]),
            np.array([self._current_sand]),
            self.schedule,
        )
        clipped = bool(diagnostics["flow_was_clipped"][0] or diagnostics["sand_was_clipped"][0])
        return float(safe_flow[0]), float(safe_sand[0]), clipped

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        observation, info = super().reset(seed=seed, options=options)
        self._current_option = self._select_high_level_option()
        self._option_age = 0
        return self._observation(), {**info, "high_level_option": self.OPTIONS[self._current_option]}

    def step(self, action: np.ndarray):
        if self._option_age <= 0 or self._option_age >= self.hierarchical_config.high_level_interval_steps:
            self._current_option = self._select_high_level_option()
            self._option_age = 0
        option_name = self.OPTIONS[self._current_option]
        observation, reward, terminated, truncated, info = super().step(action)
        self._option_age += 1
        info["high_level_option"] = option_name
        info["high_level_option_id"] = int(self._current_option)
        info["option_age"] = int(self._option_age)
        return observation, reward, terminated, truncated, info
