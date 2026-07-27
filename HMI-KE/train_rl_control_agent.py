from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from decision_engine.integrated_reward import IntegratedRewardConfig, load_reward_context
from decision_engine.pump_schedule_constraints import SCHEDULES, get_schedule_constraint
from rl.fracturing_env import (
    FracturingControlEnv,
    FracturingEnvConfig,
    HierarchicalFracturingControlEnv,
    HierarchicalFracturingEnvConfig,
)
from rl.digital_twin_env import DigitalTwinEnvConfig, DigitalTwinFracturingControlEnv
from simulator.scenario_generator import DEFAULT_CONFIG_PATH, apply_scenario, available_scenarios
from simulator.fsl_scenario_library import REAL_SCENARIOS, select_real_scenario
from simulator.validation_180s import Validation180sConfig, validate_180s
from data_pipeline import (
    DatasetBundle,
    build_dataset,
    discover_segment_frames,
    estimate_sample_interval_seconds,
    segment_split,
)


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent


def configure_plot_fonts() -> None:
    for font_path in [Path(r"C:\Windows\Fonts\msyh.ttc"), Path(r"C:\Windows\Fonts\simhei.ttf")]:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
            plt.rcParams["axes.unicode_minus"] = False
            break


def evaluate(model, env: FracturingControlEnv, episodes: int, deterministic: bool = True) -> pd.DataFrame:
    rows: list[dict] = []
    max_start = max(0, len(env.features) - env.config.episode_steps - 1)
    starts = np.linspace(0, max_start, episodes, dtype=int) if episodes > 1 else np.array([0])
    for episode, start in enumerate(starts):
        obs, _ = env.reset(options={"start_index": int(start)})
        done = False
        step = 0
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            rows.append({"episode": episode, "step": step, "reward": reward, **info})
            done = terminated or truncated
            step += 1
    return pd.DataFrame(rows)


def evaluate_historical_baseline(env: FracturingControlEnv, actions: np.ndarray, episodes: int) -> pd.DataFrame:
    rows: list[dict] = []
    max_start = max(0, len(env.features) - env.config.episode_steps - 1)
    starts = np.linspace(0, max_start, episodes, dtype=int) if episodes > 1 else np.array([0])
    for episode, start in enumerate(starts):
        _, _ = env.reset(options={"start_index": int(start)})
        done = False
        step = 0
        while not done:
            idx = min(int(start) + step, len(actions) - 1)
            normalized_action = env.encode_engineering_action(actions[idx, 0], actions[idx, 1])
            _, reward, terminated, truncated, info = env.step(normalized_action)
            rows.append({"episode": episode, "step": step, "reward": reward, **info})
            done = terminated or truncated
            step += 1
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame) -> dict:
    episode_reward = frame.groupby("episode")["reward"].sum()
    fields = [
        "integrated_reward",
        "effectiveness_reward",
        "pressure_safety_penalty",
        "abnormal_risk_penalty",
        "construction_cost_penalty",
    ]
    return {
        "episode_reward_mean": float(episode_reward.mean()),
        "episode_reward_std": float(episode_reward.std(ddof=0)),
        "step_reward_mean": float(frame["reward"].mean()),
        "unsafe_rate": float(frame["unsafe"].mean()),
        "action_clipped_rate": float(frame["action_clipped"].mean()),
        **{f"mean_{field}": float(frame[field].mean()) for field in fields if field in frame},
    }


def plot_comparison(rl: pd.DataFrame, baseline: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    reward_data = [baseline.groupby("episode")["reward"].sum(), rl.groupby("episode")["reward"].sum()]
    axes[0].boxplot(reward_data, tick_labels=["历史动作", "强化学习策略"])
    axes[0].set_title("整段累计奖励对比")
    axes[0].set_ylabel("累计奖励")
    axes[0].grid(axis="y", alpha=0.25)
    component_cols = ["effectiveness_reward", "pressure_safety_penalty", "abnormal_risk_penalty", "construction_cost_penalty"]
    x = np.arange(len(component_cols))
    width = 0.36
    axes[1].bar(x - width / 2, [baseline[c].mean() for c in component_cols], width, label="历史动作")
    axes[1].bar(x + width / 2, [rl[c].mean() for c in component_cols], width, label="强化学习")
    axes[1].set_xticks(x, ["改造效果", "压力惩罚", "异常惩罚", "施工成本"], rotation=15)
    axes[1].set_title("单步奖励分量对比")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    configure_plot_fonts()
    parser = argparse.ArgumentParser(description="Train PPO/SAC on the action-conditioned fracturing control environment.")
    parser.add_argument("--algorithm", choices=["ppo", "sac"], default="ppo")
    parser.add_argument("--data-path", default=str(PROJECT_ROOT / "Data" / "raw_frac"))
    parser.add_argument("--reference-header-path", default=str(PROJECT_ROOT / "Data" / "raw_frac" / "WITHfiltered2ASSELECTDISTINCTJTHJDFROMhagHAGMARKPOINTWHEREWORKIN_202511211131.xlsx"))
    parser.add_argument("--dt-context-csv", default=None)
    parser.add_argument("--abnormal-probability-csv", default=None)
    parser.add_argument("--reward-alignment-mode", choices=["normalized_progress", "same_stage_time"], default="normalized_progress")
    parser.add_argument("--scenario", default="baseline", help=f"Training scenario. Choices are loaded from {DEFAULT_CONFIG_PATH}.")
    parser.add_argument("--scenario-source", choices=["synthetic", "fsl_real"], default="synthetic")
    parser.add_argument("--scenario-config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--label-column", default="WORKING_TYPE")
    parser.add_argument("--pump-schedule-type", choices=sorted(SCHEDULES), default="continuous")
    parser.add_argument("--time-column", default="SGSJ")
    parser.add_argument("--segment-column", default="FDBH")
    parser.add_argument("--sample-interval-seconds", type=float, default=10.0)
    parser.add_argument("--state-seconds", type=float, default=300.0)
    parser.add_argument("--action-seconds", type=float, default=60.0)
    parser.add_argument("--total-timesteps", type=int, default=100000)
    parser.add_argument("--episode-steps", type=int, default=60)
    parser.add_argument("--hierarchical", action="store_true", help="Use lightweight option-based HRL prototype.")
    parser.add_argument("--response-model", choices=["empirical", "digital_twin"], default="empirical")
    parser.add_argument("--high-level-interval-steps", type=int, default=6, help="How often the high-level option is refreshed.")
    parser.add_argument("--terminate-on-unsafe", action="store_true", help="End an episode on severe pressure violation; default is penalty only.")
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=30000)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--max-rows-per-file", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--run-dir", default=str(ROOT / "runs" / "rl_control_agent"))
    args = parser.parse_args()

    frames = discover_segment_frames(
        args.data_path,
        args.reference_header_path,
        args.segment_column,
        args.time_column,
        ["SGBY", "PL", "SB", args.label_column],
        ["WITHfiltered"],
        args.max_files,
        args.max_rows_per_file,
    )
    interval = estimate_sample_interval_seconds(frames, args.time_column, args.sample_interval_seconds)
    bundle = build_dataset(
        frames,
        ["SGBY", "PL", "SB"],
        ["PL", "SB"],
        args.time_column,
        max(2, int(round(args.state_seconds / interval))),
        max(1, int(round(args.action_seconds / interval))),
        args.label_column,
    )
    if args.scenario_source == "synthetic" and args.max_samples and len(bundle.x) > args.max_samples:
        chosen = np.linspace(0, len(bundle.x) - 1, args.max_samples, dtype=int)
        bundle = DatasetBundle(
            bundle.x[chosen], bundle.y[chosen], bundle.meta.iloc[chosen].reset_index(drop=True),
            bundle.feature_names, bundle.target_names, bundle.action_bounds,
        )
    full_context, provenance = load_reward_context(
        len(bundle.x), args.dt_context_csv, args.abnormal_probability_csv, args.reward_alignment_mode
    )
    if args.scenario_source == "fsl_real":
        if args.scenario not in REAL_SCENARIOS:
            raise ValueError(f"Unsupported --scenario {args.scenario} for fsl_real. Choices: {list(REAL_SCENARIOS)}")
        features, targets, meta, full_context, scenario_spec = select_real_scenario(
            bundle.x, bundle.y, bundle.meta, full_context, args.scenario
        )
    else:
        scenario_choices = available_scenarios(args.scenario_config)
        if args.scenario not in scenario_choices:
            raise ValueError(f"Unsupported --scenario {args.scenario}. Choices: {scenario_choices}")
        features, meta, full_context, scenario_spec = apply_scenario(
            bundle.x, bundle.meta, full_context, args.scenario, args.scenario_config, bundle.feature_names
        )
        targets = bundle.y
    bundle = DatasetBundle(features, targets, meta, bundle.feature_names, bundle.target_names, bundle.action_bounds)
    if args.scenario_source == "fsl_real" and args.max_samples and len(bundle.x) > args.max_samples:
        chosen = np.linspace(0, len(bundle.x) - 1, args.max_samples, dtype=int)
        bundle = DatasetBundle(
            bundle.x[chosen], bundle.y[chosen], bundle.meta.iloc[chosen].reset_index(drop=True),
            bundle.feature_names, bundle.target_names, bundle.action_bounds,
        )
        full_context = full_context.iloc[chosen].reset_index(drop=True)
        scenario_spec["samples_after_cap"] = int(len(chosen))
    provenance["scenario"] = scenario_spec
    provenance["scenario_source"] = args.scenario_source
    provenance["available_components"].append("multi_condition_scenario")
    train_idx, _, test_idx = segment_split(bundle.meta, 0.75, 0.1, args.seed)
    if not len(test_idx):
        test_idx = train_idx[-min(1000, len(train_idx)):]
    reward_config = IntegratedRewardConfig()
    if args.response_model == "digital_twin":
        if args.hierarchical:
            raise ValueError("--response-model digital_twin currently supports the flat PPO/SAC controller; hierarchical composition is the next step.")
        env_config = DigitalTwinEnvConfig(
            episode_steps=args.episode_steps,
            terminate_on_unsafe=args.terminate_on_unsafe,
            action_seconds=args.action_seconds,
        )
        env_class = DigitalTwinFracturingControlEnv
    elif args.hierarchical:
        env_config = HierarchicalFracturingEnvConfig(
            episode_steps=args.episode_steps,
            terminate_on_unsafe=args.terminate_on_unsafe,
            high_level_interval_steps=args.high_level_interval_steps,
        )
        env_class = HierarchicalFracturingControlEnv
    else:
        env_config = FracturingEnvConfig(episode_steps=args.episode_steps, terminate_on_unsafe=args.terminate_on_unsafe)
        env_class = FracturingControlEnv
    schedule = get_schedule_constraint(args.pump_schedule_type)

    def make_env(indices: np.ndarray, random_start: bool):
        return env_class(
            bundle.x[indices], bundle.meta.iloc[indices].reset_index(drop=True),
            full_context.iloc[indices].reset_index(drop=True), schedule, reward_config,
            env_config, args.seed, random_start,
        )

    vec_env = DummyVecEnv([lambda: Monitor(make_env(train_idx, True))])
    if args.algorithm == "ppo":
        model = PPO("MlpPolicy", vec_env, learning_rate=3e-4, n_steps=512, batch_size=128, gamma=0.98, gae_lambda=0.95, verbose=1, seed=args.seed, device="cpu")
    else:
        model = SAC("MlpPolicy", vec_env, learning_rate=3e-4, buffer_size=100000, batch_size=256, gamma=0.98, learning_starts=1000, verbose=1, seed=args.seed, device="cpu")
    model.learn(total_timesteps=args.total_timesteps, progress_bar=False)

    test_env = make_env(test_idx, False)
    rl_eval = evaluate(model, test_env, args.eval_episodes)
    baseline_eval = evaluate_historical_baseline(test_env, bundle.y[test_idx], args.eval_episodes)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.run_dir) / f"{args.algorithm}_{timestamp}"
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / f"{args.algorithm}_fracturing_policy")
    rl_eval.to_csv(out / "rl_evaluation.csv", index=False, encoding="utf-8-sig")
    baseline_eval.to_csv(out / "historical_baseline_evaluation.csv", index=False, encoding="utf-8-sig")
    validation_config = Validation180sConfig(action_seconds=args.action_seconds)
    rl_180s, rl_180s_summary = validate_180s(rl_eval, validation_config)
    baseline_180s, baseline_180s_summary = validate_180s(baseline_eval, validation_config)
    rl_180s.to_csv(out / "rl_180s_validation.csv", index=False, encoding="utf-8-sig")
    baseline_180s.to_csv(out / "historical_baseline_180s_validation.csv", index=False, encoding="utf-8-sig")
    plot_comparison(rl_eval, baseline_eval, out / "rl_vs_historical_reward.png")
    summary = {
        "module": "Gymnasium lightweight hierarchical fracturing control agent" if args.hierarchical else "Gymnasium continuous-action fracturing control agent",
        "algorithm": args.algorithm.upper(),
        "hierarchical": bool(args.hierarchical),
        "hierarchical_design": {
            "high_level_policy": "rule/knowledge option selector",
            "high_level_options": list(HierarchicalFracturingControlEnv.OPTIONS) if args.hierarchical else [],
            "low_level_policy": f"Stable-Baselines3 {args.algorithm.upper()} continuous controller",
            "refresh_interval_steps": args.high_level_interval_steps if args.hierarchical else None,
        },
        "scientific_status": provenance["scientific_status"],
        "warning": "Action-conditioned response is an offline surrogate. Field use requires same-well synchronization and simulator calibration.",
        "state": "current 300s pressure/rate/sand ratio plus available DT/risk context",
        "action": "next 60s mean rate and sand ratio",
        "reward": "fracture effectiveness - pressure risk - abnormal risk - construction cost",
        "total_timesteps": args.total_timesteps,
        "scenario": scenario_spec,
        "response_model": args.response_model,
        "train_samples": int(len(train_idx)),
        "test_samples": int(len(test_idx)),
        "environment": env_config.to_dict(),
        "reward_config": reward_config.to_dict(),
        "pump_schedule_constraint": schedule.to_dict(),
        "context_provenance": provenance,
        "rl_policy": summarize(rl_eval),
        "historical_baseline": summarize(baseline_eval),
        "validation_180s": {
            "rl_policy": rl_180s_summary,
            "historical_baseline": baseline_180s_summary,
        },
        "outputs": {
            "model": str(out / f"{args.algorithm}_fracturing_policy.zip"),
            "rl_evaluation": str(out / "rl_evaluation.csv"),
            "historical_evaluation": str(out / "historical_baseline_evaluation.csv"),
            "rl_180s_validation": str(out / "rl_180s_validation.csv"),
            "historical_baseline_180s_validation": str(out / "historical_baseline_180s_validation.csv"),
            "comparison_plot": str(out / "rl_vs_historical_reward.png"),
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"run_dir": str(out), "algorithm": args.algorithm, "rl_policy": summary["rl_policy"], "historical_baseline": summary["historical_baseline"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
