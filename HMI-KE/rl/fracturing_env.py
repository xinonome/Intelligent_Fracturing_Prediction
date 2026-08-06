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
    abnormal_probability_max: float = 0.45
    sand_plug_probability_max: float = 0.35
    posterior_error_max: float = 0.30

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class HierarchicalFracturingEnvConfig(FracturingEnvConfig):
    high_level_interval_steps: int = 6
    high_pressure_ratio: float = 0.92
    medium_abnormal_probability: float = 0.25
    high_abnormal_probability: float = 0.45
    high_posterior_error: float = 0.15
    safety_activation_ratio: float = 0.80
    safe_min_flow_reduction_ratio: float = 0.50
    safe_min_sand_reduction_ratio: float = 0.75


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
        self._episode_end = len(self.features)
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
        segment = self.meta.get("segment_id", pd.Series(np.zeros(len(self.meta), dtype=int))).astype(str)
        scenario = self.meta.get("scenario_name", pd.Series(["default"] * len(self.meta))).astype(str)
        replica = self.meta.get("scenario_replica", pd.Series(np.zeros(len(self.meta), dtype=int))).astype(str)
        self._episode_keys = (segment + "::" + scenario + "::" + replica).to_numpy()
        self._group_end = np.empty(len(self.meta), dtype=int)
        start = 0
        while start < len(self.meta):
            end = start + 1
            while end < len(self.meta) and self._episode_keys[end] == self._episode_keys[start]:
                end += 1
            self._group_end[start:end] = end
            start = end
        minimum = min(self.config.episode_steps + 1, max(len(self.meta), 2))
        self._valid_starts = np.asarray(
            [idx for idx, end in enumerate(self._group_end) if end - idx >= minimum], dtype=int
        )
        if not len(self._valid_starts):
            self._valid_starts = np.asarray(
                [idx for idx, end in enumerate(self._group_end) if end - idx >= 2], dtype=int
            )
        if not len(self._valid_starts):
            self._valid_starts = np.asarray([0], dtype=int)

    def evaluation_starts(self, episodes: int, scenario_name: str | None = None) -> np.ndarray:
        """Return deterministic starts that never cross segment/scenario boundaries."""
        count = max(int(episodes), 1)
        candidates = self._valid_starts
        if scenario_name is not None and "scenario_name" in self.meta:
            mask = self.meta.iloc[candidates]["scenario_name"].astype(str).to_numpy() == str(scenario_name)
            candidates = candidates[mask]
        if not len(candidates):
            return np.asarray([], dtype=int)
        positions = np.linspace(0, len(candidates) - 1, count).round().astype(int)
        return candidates[positions]

    def _emergency_active(self) -> bool:
        response = getattr(self, "_latest_response", {})
        return bool(
            float(response.get("bottomhole_pressure_mpa", 0.0)) > self.reward_config.bottomhole_pressure_max_mpa
            or float(response.get("net_pressure_mpa", 0.0)) > self.reward_config.net_pressure_max_mpa
            or float(response.get("abnormal_probability", 0.0)) > self.config.abnormal_probability_max
            or float(response.get("sand_plug_probability", 0.0)) > self.config.sand_plug_probability_max
        )

    def _resolve_start(self, requested: int) -> int:
        requested = int(np.clip(requested, 0, len(self.features) - 1))
        return int(self._valid_starts[np.argmin(np.abs(self._valid_starts - requested))])

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
        emergency = self._emergency_active()
        if self.schedule.allow_sand_pause or emergency:
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
        if emergency:
            # Safety intervention takes precedence over a normal continuous-sanding schedule.
            safe_sand[0] = np.clip(proposed_sand, 0.0, self._current_sand)
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
        if options and "start_index" in options:
            self._cursor = self._resolve_start(int(options["start_index"]))
        elif self.random_start and len(self._valid_starts) > 1:
            self._cursor = int(self.np_random.choice(self._valid_starts))
        else:
            self._cursor = int(self._valid_starts[0])
        self._episode_end = int(self._group_end[self._cursor])
        self._steps = 0
        row = self.meta.iloc[self._cursor]
        self._current_pressure = float(row["current_pressure"])
        self._current_flow = float(np.clip(row["current_flow"], 0.0, self.schedule.max_flow_m3_min))
        self._current_sand = float(np.clip(row["current_sand_ratio"], 0.0, self.schedule.max_sand_ratio_percent))
        base_length = self._base_context(self._cursor).get("posterior_total_half_length_m", 1.0)
        self._previous_length = float(base_length) if np.isfinite(base_length) else 1.0
        base = self._base_context(self._cursor)
        self._latest_response = {
            "bottomhole_pressure_mpa": base.get("bottomhole_pressure_mpa", self._current_pressure),
            "net_pressure_mpa": base.get("net_pressure_mpa", 0.0),
            "abnormal_probability": base.get("abnormal_probability", 0.0),
            "sand_plug_probability": base.get("sand_plug_probability", 0.0),
            "posterior_error": base.get("posterior_error", 0.0),
        }
        return self._observation(), {"start_index": self._cursor}

    def step(self, action: np.ndarray):
        pre_action_context = self._base_context(self._cursor)
        # Preserve the measured/current control state before decoding the
        # proposed action.  The action is a future 60-second mean setting and
        # must not be presented as the current field input.
        pre_action_flow = float(self._current_flow)
        pre_action_sand = float(self._current_sand)
        pre_action_bottomhole = float(
            pre_action_context.get("bottomhole_pressure_mpa", self._current_pressure)
        )
        pre_action_net = float(pre_action_context.get("net_pressure_mpa", 0.0))
        pre_action_abnormal = float(pre_action_context.get("abnormal_probability", 0.0))
        pre_action_sand_plug = float(pre_action_context.get("sand_plug_probability", 0.0))
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
        unsafe_reasons = []
        if response["bottomhole_pressure_mpa"] > self.reward_config.bottomhole_pressure_max_mpa:
            unsafe_reasons.append("bottomhole_pressure")
        if response["net_pressure_mpa"] > self.reward_config.net_pressure_max_mpa:
            unsafe_reasons.append("net_pressure")
        if response["abnormal_probability"] > self.config.abnormal_probability_max:
            unsafe_reasons.append("abnormal_probability")
        if response["sand_plug_probability"] > self.config.sand_plug_probability_max:
            unsafe_reasons.append("sand_plug_probability")
        unsafe = bool(unsafe_reasons)
        uncertainty_reasons = []
        if response["posterior_error"] > self.config.posterior_error_max:
            uncertainty_reasons.append("posterior_error")
        uncertain = bool(uncertainty_reasons)
        severe_pressure = bool(
            response["bottomhole_pressure_mpa"]
            > self.reward_config.bottomhole_pressure_max_mpa * self.config.severe_pressure_multiplier
            or response["net_pressure_mpa"]
            > self.reward_config.net_pressure_max_mpa * self.config.severe_pressure_multiplier
        )
        if unsafe:
            reward -= self.config.unsafe_termination_penalty
        reward = float(np.clip(reward, -self.config.reward_clip, self.config.reward_clip))

        self._current_flow = flow
        self._current_sand = sand
        self._current_pressure = response["pressure"]
        self._previous_length = response["length"]
        self._latest_response = dict(response)
        self._steps += 1
        next_cursor = self._cursor + 1
        boundary_reached = bool(next_cursor >= self._episode_end or next_cursor >= len(self.features))
        if not boundary_reached:
            self._cursor = next_cursor
        # Ordinary risk remains in the trajectory so the agent can learn a recovery action.
        # Only an extreme pressure breach triggers the configurable hard stop.
        terminated = bool((severe_pressure and self.config.terminate_on_unsafe) or boundary_reached)
        truncated = bool(self._steps >= self.config.episode_steps)
        info = {
            "pre_action_flow_m3_min": pre_action_flow,
            "pre_action_sand_ratio_percent": pre_action_sand,
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
            "pre_action_bottomhole_pressure_mpa": pre_action_bottomhole,
            "pre_action_net_pressure_mpa": pre_action_net,
            "pre_action_abnormal_probability": pre_action_abnormal,
            "pre_action_sand_plug_probability": pre_action_sand_plug,
            "unsafe": unsafe,
            "unsafe_reasons": "|".join(unsafe_reasons),
            "uncertain": uncertain,
            "uncertainty_reasons": "|".join(uncertainty_reasons),
            "severe_pressure_violation": severe_pressure,
            "segment_id": str(self.meta.iloc[self._cursor].get("segment_id", "")),
            "scenario_name": str(self.meta.iloc[self._cursor].get("scenario_name", "default")),
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
        latest = getattr(self, "_latest_response", {})
        pressure_limit = self.reward_config.bottomhole_pressure_max_mpa
        bottomhole = latest.get("bottomhole_pressure_mpa", base.get("bottomhole_pressure_mpa", self._current_pressure))
        abnormal = latest.get("abnormal_probability", base.get("abnormal_probability", 0.0))
        sand_plug = latest.get("sand_plug_probability", base.get("sand_plug_probability", 0.0))
        posterior_error = latest.get("posterior_error", base.get("posterior_error", 0.0))
        if not np.isfinite(bottomhole):
            bottomhole = self._current_pressure
        if not np.isfinite(abnormal):
            abnormal = 0.0
        if not np.isfinite(sand_plug):
            sand_plug = 0.0
        if not np.isfinite(posterior_error):
            posterior_error = 0.0

        risk = max(float(abnormal), float(sand_plug))
        abnormal_guard = self.config.abnormal_probability_max * self.hierarchical_config.safety_activation_ratio
        sand_plug_guard = self.config.sand_plug_probability_max * self.hierarchical_config.safety_activation_ratio
        if (
            bottomhole > pressure_limit * self.hierarchical_config.high_pressure_ratio
            or abnormal >= abnormal_guard
            or sand_plug >= sand_plug_guard
        ):
            return self.OPTIONS.index("safe")
        if posterior_error > self.hierarchical_config.high_posterior_error:
            return self.OPTIONS.index("hold")
        if risk >= self.hierarchical_config.medium_abnormal_probability:
            return self.OPTIONS.index("divert")
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
            flow_high = max(
                flow_low,
                self._current_flow - self.hierarchical_config.safe_min_flow_reduction_ratio * flow_step,
            )
            sand_low = max(0.0, self._current_sand - sand_step)
            sand_high = max(
                sand_low,
                self._current_sand - self.hierarchical_config.safe_min_sand_reduction_ratio * sand_step,
            )
        elif option == "divert":
            flow_low = max(0.0, self._current_flow - 0.5 * flow_step)
            flow_high = min(self.schedule.max_flow_m3_min, self._current_flow + 0.5 * flow_step)
            # Diverting a medium-risk state may redistribute flow, but must not
            # add proppant while an abnormal trend is already developing.
            sand_low = max(0.0, self._current_sand - 0.5 * sand_step)
            sand_high = self._current_sand
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
        if option == "safe":
            safe_sand[0] = np.clip(proposed_sand, 0.0, self._current_sand)
        clipped = bool(diagnostics["flow_was_clipped"][0] or diagnostics["sand_was_clipped"][0])
        return float(safe_flow[0]), float(safe_sand[0]), clipped

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        observation, info = super().reset(seed=seed, options=options)
        self._current_option = self._select_high_level_option()
        self._option_age = 0
        return self._observation(), {**info, "high_level_option": self.OPTIONS[self._current_option]}

    def step(self, action: np.ndarray):
        selected_option = self._select_high_level_option()
        selected_name = self.OPTIONS[selected_option]
        current_name = self.OPTIONS[self._current_option]
        risk_preemption = (
            selected_name == "safe" and current_name != "safe"
            or selected_name == "divert" and current_name not in {"safe", "divert"}
        )
        if self._option_age <= 0 or self._option_age >= self.hierarchical_config.high_level_interval_steps or risk_preemption:
            self._current_option = selected_option
            self._option_age = 0
        option_name = self.OPTIONS[self._current_option]
        observation, reward, terminated, truncated, info = super().step(action)
        self._option_age += 1
        info["high_level_option"] = option_name
        info["high_level_option_id"] = int(self._current_option)
        info["option_age"] = int(self._option_age)
        return observation, reward, terminated, truncated, info
