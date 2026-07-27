from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decision_engine.integrated_reward import IntegratedRewardConfig
from decision_engine.pump_schedule_constraints import get_schedule_constraint
from rl.fracturing_env import (
    FracturingControlEnv,
    FracturingEnvConfig,
    HierarchicalFracturingControlEnv,
    HierarchicalFracturingEnvConfig,
)
from simulator.scenario_generator import apply_scenario
from simulator.fsl_scenario_library import annotate_real_scenarios, select_real_scenario
from simulator.validation_180s import validate_180s


def build_env() -> FracturingControlEnv:
    size = 12
    features = np.zeros((size, 9), dtype=np.float32)
    meta = pd.DataFrame(
        {
            "current_pressure": np.full(size, 75.0),
            "current_flow": np.full(size, 10.0),
            "current_sand_ratio": np.full(size, 4.0),
        }
    )
    context = pd.DataFrame(
        {
            "posterior_total_half_length_m": np.linspace(100.0, 111.0, size),
            "posterior_error": np.full(size, 0.10),
            "bottomhole_pressure_mpa": np.full(size, 85.0),
            "net_pressure_mpa": np.full(size, 20.0),
            "abnormal_probability": np.full(size, 0.05),
            "sand_plug_probability": np.full(size, 0.02),
        }
    )
    return FracturingControlEnv(
        features,
        meta,
        context,
        get_schedule_constraint("continuous"),
        IntegratedRewardConfig(),
        FracturingEnvConfig(episode_steps=5),
        random_start=False,
    )


def test_environment_step_is_action_conditioned() -> None:
    env = build_env()
    observation, _ = env.reset()
    assert observation.shape == env.observation_space.shape
    _, conservative_reward, _, _, conservative = env.step(np.array([-0.5, -1.0], dtype=np.float32))

    env.reset()
    _, aggressive_reward, _, _, aggressive = env.step(np.array([1.0, 1.0], dtype=np.float32))
    assert aggressive["simulated_pressure_mpa"] > conservative["simulated_pressure_mpa"]
    assert aggressive["abnormal_probability"] >= conservative["abnormal_probability"]
    assert np.isfinite(conservative_reward)
    assert np.isfinite(aggressive_reward)


def test_engineering_action_round_trip_stays_within_bounds() -> None:
    env = build_env()
    env.reset()
    encoded = env.encode_engineering_action(12.0, 6.0)
    assert env.action_space.contains(encoded)
    _, _, _, _, info = env.step(encoded)
    assert 0.0 <= info["flow_m3_min"] <= env.schedule.max_flow_m3_min
    assert 0.0 <= info["sand_ratio_percent"] <= env.schedule.max_sand_ratio_percent


def test_hierarchical_environment_exposes_option_and_safe_low_level_action() -> None:
    base = build_env()
    env = HierarchicalFracturingControlEnv(
        base.features,
        base.meta,
        base.context,
        base.schedule,
        base.reward_config,
        HierarchicalFracturingEnvConfig(episode_steps=5, high_level_interval_steps=2),
        random_start=False,
    )
    observation, info = env.reset()
    assert observation.shape == env.observation_space.shape
    assert info["high_level_option"] in env.OPTIONS
    _, reward, _, _, step_info = env.step(np.array([1.0, 1.0], dtype=np.float32))
    assert step_info["high_level_option"] in env.OPTIONS
    assert 0.0 <= step_info["flow_m3_min"] <= env.schedule.max_flow_m3_min
    assert 0.0 <= step_info["sand_ratio_percent"] <= env.schedule.max_sand_ratio_percent
    assert np.isfinite(reward)


def test_scenario_generation_and_180s_validation() -> None:
    env = build_env()
    features, meta, context, spec = apply_scenario(
        env.features,
        env.meta,
        env.context,
        "sand_plug_risk",
    )
    assert features.shape == env.features.shape
    assert spec["name"] == "sand_plug_risk"
    assert context["sand_plug_probability"].iloc[-1] > context["sand_plug_probability"].iloc[0]

    evaluation = pd.DataFrame(
        {
            "episode": [0, 0, 0, 0],
            "step": [0, 1, 2, 3],
            "bottomhole_pressure_mpa": [80.0, 86.0, 90.0, 92.0],
            "net_pressure_mpa": [12.0, 14.0, 15.0, 16.0],
            "abnormal_probability": [0.1, 0.2, 0.5, 0.2],
            "sand_plug_probability": [0.1, 0.2, 0.2, 0.2],
            "unsafe": [False, False, False, False],
        }
    )
    rows, summary = validate_180s(evaluation)
    assert summary["validation_available"]
    assert rows["unsafe_within_180s"].any()


def test_fsl_real_scenario_selection_uses_working_type_without_perturbing_features() -> None:
    features = np.arange(24, dtype=np.float32).reshape(4, 6)
    targets = np.arange(8, dtype=np.float32).reshape(4, 2)
    meta = pd.DataFrame(
        {
            "segment_id": ["A", "A", "B", "B"],
            "current_pressure": [60.0, 62.0, 64.0, 100.0],
            "state_working_types": ["", "砂堵", "缝口暂堵", ""],
            "future_working_types": ["", "", "", ""],
        }
    )
    context = pd.DataFrame(index=range(4))
    annotated = annotate_real_scenarios(meta)
    assert annotated.loc[1, "real_scenario_class"] == "sand_plug_risk"
    assert annotated.loc[2, "real_scenario_class"] == "diversion_stage"
    assert annotated.loc[3, "real_scenario_class"] == "pressure_limit"

    selected_x, selected_y, selected_meta, selected_context, spec = select_real_scenario(
        features, targets, meta, context, "sand_plug_risk"
    )
    assert np.array_equal(selected_x, features[[1]])
    assert np.array_equal(selected_y, targets[[1]])
    assert selected_meta["segment_id"].tolist() == ["A"]
    assert selected_context["sand_plug_probability"].tolist() == [1.0]
    assert spec["source"] == "FSL-Expert/Data/raw_frac"
