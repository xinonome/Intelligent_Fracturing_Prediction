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
from rl.digital_twin_env import (
    HierarchicalDigitalTwinEnvConfig,
    HierarchicalDigitalTwinFracturingControlEnv,
)
from simulator.scenario_generator import apply_scenario
from simulator.fsl_scenario_library import annotate_real_scenarios, select_real_scenario
from simulator.validation_180s import validate_180s
from simulator.contract_acceptance import (
    annotate_5min_warnings,
    evaluate_direct_5min_warning,
    summarize_decision_latency,
)


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
            "posterior_error": [0.1, 0.4, 0.2, 0.1],
            "uncertain": [False, True, False, False],
            "unsafe": [False, False, False, False],
        }
    )
    rows, summary = validate_180s(evaluation)
    assert summary["validation_available"]
    assert rows["unsafe_within_180s"].any()
    assert rows["uncertain_within_180s"].any()
    assert summary["eligible_complete_windows"] == 2
    assert summary["incomplete_windows"] == 2
    assert "preventive_safe_within_180s_rate" in summary
    assert "recovered_within_180s_rate" in summary


def test_five_minute_warning_and_latency_gate() -> None:
    rows = []
    for step in range(8):
        rows.append(
            {
                "episode": 0,
                "step": step,
                "bottomhole_pressure_mpa": 80.0,
                "net_pressure_mpa": 15.0,
                "abnormal_probability": 0.10 if step < 4 else 0.55,
                "sand_plug_probability": 0.05,
                "unsafe": step >= 4,
                "decision_compute_seconds": 0.02,
            }
        )
    warnings, summary = annotate_5min_warnings(pd.DataFrame(rows))
    assert summary["available"]
    assert summary["event_windows"] > 0
    assert warnings["warning_5min"].any()
    latency = summarize_decision_latency(pd.DataFrame(rows))
    assert latency["pass_15s"]
    assert latency["p95_seconds"] < 15.0
    direct = evaluate_direct_5min_warning(
        pd.DataFrame(
            {
                "future_abnormal": [0, 0, 1, 1],
                "predicted_abnormal_probability": [0.05, 0.2, 0.8, 0.9],
            }
        )
    )
    assert direct["pass_5min_warning_recall"]


def test_episode_does_not_cross_segment_boundary() -> None:
    env = build_env()
    env.meta["segment_id"] = ["A"] * 6 + ["B"] * 6
    env = FracturingControlEnv(
        env.features,
        env.meta,
        env.context,
        env.schedule,
        env.reward_config,
        FracturingEnvConfig(episode_steps=10),
        random_start=False,
    )
    env.reset(options={"start_index": 0})
    steps = 0
    done = False
    while not done:
        _, _, terminated, truncated, _ = env.step(np.zeros(2, dtype=np.float32))
        done = terminated or truncated
        steps += 1
    assert steps == 6


def test_hierarchical_digital_twin_environment_runs() -> None:
    base = build_env()
    env = HierarchicalDigitalTwinFracturingControlEnv(
        base.features,
        base.meta,
        base.context,
        base.schedule,
        base.reward_config,
        HierarchicalDigitalTwinEnvConfig(episode_steps=4, ensemble_size=12),
        random_start=False,
    )
    observation, info = env.reset(seed=7)
    assert observation.shape == env.observation_space.shape
    assert info["high_level_option"] in env.OPTIONS
    _, reward, _, _, step_info = env.step(np.zeros(2, dtype=np.float32))
    assert np.isfinite(reward)
    assert step_info["response_model"] == "pkn_enkf_digital_twin"


def test_risk_preempts_to_safe_and_allows_emergency_sand_reduction() -> None:
    base = build_env()
    base.context["sand_plug_probability"] = 0.80
    env = HierarchicalFracturingControlEnv(
        base.features,
        base.meta,
        base.context,
        base.schedule,
        base.reward_config,
        HierarchicalFracturingEnvConfig(episode_steps=5, high_level_interval_steps=6),
        random_start=False,
    )
    env.reset()
    initial_sand = env._current_sand
    _, _, terminated, _, info = env.step(np.array([-1.0, -1.0], dtype=np.float32))
    assert info["high_level_option"] == "safe"
    assert info["sand_ratio_percent"] < initial_sand
    assert not terminated

    env.reset()
    initial_flow = env._current_flow
    initial_sand = env._current_sand
    _, _, _, _, conservative = env.step(np.array([1.0, 1.0], dtype=np.float32))
    assert conservative["flow_m3_min"] <= initial_flow - 0.5 * env.schedule.max_flow_step_m3_min + 1e-6
    assert conservative["sand_ratio_percent"] <= initial_sand - 0.75 * env.schedule.max_sand_increase_percent + 1e-6


def test_low_flow_skips_pkn_and_preserves_fracture_length() -> None:
    base = build_env()
    base.meta["current_flow"] = 0.2
    base.meta["current_sand_ratio"] = 0.0
    env = HierarchicalDigitalTwinFracturingControlEnv(
        base.features,
        base.meta,
        base.context,
        base.schedule,
        base.reward_config,
        HierarchicalDigitalTwinEnvConfig(episode_steps=4, ensemble_size=12, minimum_pkn_flow_m3_min=1.0),
        random_start=False,
    )
    env.reset(seed=11)
    initial_length = env._previous_length
    _, reward, _, _, info = env.step(np.array([-1.0, -1.0], dtype=np.float32))
    assert np.isfinite(reward)
    assert info["pkn_update_skipped"]
    assert info["simulated_half_length_m"] == initial_length
    assert np.isfinite(info["posterior_error"])


def test_posterior_error_is_uncertainty_not_operational_unsafe() -> None:
    env = build_env()
    env.context["posterior_error"] = 0.8
    env = HierarchicalFracturingControlEnv(
        env.features,
        env.meta,
        env.context,
        env.schedule,
        env.reward_config,
        HierarchicalFracturingEnvConfig(episode_steps=4),
        random_start=False,
    )
    _, reset_info = env.reset()
    assert reset_info["high_level_option"] == "hold"
    _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
    assert info["uncertain"]
    assert not info["unsafe"]
    assert info["uncertainty_reasons"] == "posterior_error"


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
