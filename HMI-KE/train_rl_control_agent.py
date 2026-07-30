from __future__ import annotations

import argparse
import hashlib
import json
import sys
from time import perf_counter
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback
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
from rl.digital_twin_env import (
    DigitalTwinEnvConfig,
    DigitalTwinFracturingControlEnv,
    HierarchicalDigitalTwinEnvConfig,
    HierarchicalDigitalTwinFracturingControlEnv,
)
from simulator.scenario_generator import DEFAULT_CONFIG_PATH, apply_scenario, available_scenarios
from simulator.fsl_scenario_library import REAL_SCENARIOS, select_real_scenario
from simulator.validation_180s import Validation180sConfig, validate_180s
from simulator.contract_acceptance import (
    Warning5MinConfig,
    annotate_5min_warnings,
    summarize_decision_latency,
)
from data_pipeline import (
    DatasetBundle,
    build_dataset,
    discover_segment_frames,
    estimate_sample_interval_seconds,
    segment_split,
)
from response_surrogate import ActionResponseSurrogate


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DEFAULT_SCENARIO_WEIGHTS = {
    "baseline": 1,
    "normal_growth": 1,
    "sand_plug_risk": 3,
    "cluster_imbalance": 2,
    "pressure_limit": 2,
    "diversion_stage": 2,
}


def parse_scenario_weights(value: str | None) -> dict[str, int]:
    weights = DEFAULT_SCENARIO_WEIGHTS.copy()
    if not value:
        return weights
    for item in value.split(","):
        name, separator, raw_weight = item.strip().partition("=")
        if not separator or not name:
            raise ValueError(f"Invalid scenario weight: {item}. Expected name=positive_integer")
        weight = int(raw_weight)
        if weight < 1:
            raise ValueError(f"Scenario weight must be >= 1: {item}")
        weights[name] = weight
    return weights


def source_fingerprint() -> dict:
    paths = [
        Path(__file__),
        ROOT / "rl" / "fracturing_env.py",
        ROOT / "rl" / "digital_twin_env.py",
        ROOT / "decision_engine" / "integrated_reward.py",
        ROOT / "simulator" / "scenario_generator.py",
        ROOT / "simulator" / "validation_180s.py",
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return {"sha256": digest.hexdigest(), "files": [str(path) for path in paths]}


def configure_plot_fonts() -> None:
    for font_path in [Path(r"C:\Windows\Fonts\msyh.ttc"), Path(r"C:\Windows\Fonts\simhei.ttf")]:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
            plt.rcParams["axes.unicode_minus"] = False
            break


def evaluate(model, env: FracturingControlEnv, episodes: int, deterministic: bool = True, scenario_name: str | None = None, episode_offset: int = 0) -> pd.DataFrame:
    rows: list[dict] = []
    starts = env.evaluation_starts(episodes, scenario_name)
    for episode, start in enumerate(starts):
        obs, _ = env.reset(options={"start_index": int(start)})
        done = False
        step = 0
        while not done:
            started = perf_counter()
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            elapsed = perf_counter() - started
            rows.append({
                "episode": episode + episode_offset,
                "step": step,
                "reward": reward,
                "decision_compute_seconds": elapsed,
                **info,
            })
            done = terminated or truncated
            step += 1
    return pd.DataFrame(rows)


def evaluate_historical_baseline(env: FracturingControlEnv, actions: np.ndarray, episodes: int, scenario_name: str | None = None, episode_offset: int = 0) -> pd.DataFrame:
    rows: list[dict] = []
    starts = env.evaluation_starts(episodes, scenario_name)
    for episode, start in enumerate(starts):
        _, _ = env.reset(options={"start_index": int(start)})
        done = False
        step = 0
        while not done:
            idx = min(int(start) + step, len(actions) - 1)
            normalized_action = env.encode_engineering_action(actions[idx, 0], actions[idx, 1])
            _, reward, terminated, truncated, info = env.step(normalized_action)
            rows.append({"episode": episode + episode_offset, "step": step, "reward": reward, **info})
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
        "uncertain_rate": float(frame["uncertain"].mean()) if "uncertain" in frame else 0.0,
        "action_clipped_rate": float(frame["action_clipped"].mean()),
        **{f"mean_{field}": float(frame[field].mean()) for field in fields if field in frame},
    }


def summarize_by_scenario(frame: pd.DataFrame) -> dict[str, dict]:
    if "scenario_name" not in frame or frame.empty:
        return {}
    return {
        str(name): summarize(group.reset_index(drop=True))
        for name, group in frame.groupby("scenario_name", sort=True)
    }


def validate_by_scenario(frame: pd.DataFrame, config: Validation180sConfig) -> dict[str, dict]:
    if "scenario_name" not in frame or frame.empty:
        return {}
    result = {}
    for name, group in frame.groupby("scenario_name", sort=True):
        _, summary = validate_180s(group.reset_index(drop=True), config)
        result[str(name)] = summary
    return result


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


def plot_safety_by_scenario(rl: pd.DataFrame, baseline: pd.DataFrame, config: Validation180sConfig, path: Path) -> None:
    rl_summary = validate_by_scenario(rl, config)
    baseline_summary = validate_by_scenario(baseline, config)
    names = sorted(set(rl_summary) | set(baseline_summary))
    x = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(x - width / 2, [baseline_summary[name]["safe_within_180s_rate"] for name in names], width, label="历史动作")
    ax.bar(x + width / 2, [rl_summary[name]["safe_within_180s_rate"] for name in names], width, label="强化学习")
    ax.axhline(1.0, color="#b42318", linestyle="--", linewidth=1.2, label="验收门槛 100%")
    ax.set_xticks(x, names, rotation=18)
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("连续180秒安全率")
    ax.set_title("六类场景严格180秒安全验证")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_quality_gate(
    rl_summary: dict,
    baseline_summary: dict,
    rl_180s_summary: dict,
    scenario_validation: dict[str, dict],
    latency_summary: dict | None = None,
) -> dict:
    preventive_pass = rl_180s_summary.get(
        "pass_preventive_180s_safety", rl_180s_summary["pass_180s_safety"]
    )
    checks = {
        "reward_not_worse_than_baseline": rl_summary["episode_reward_mean"] >= baseline_summary["episode_reward_mean"],
        "unsafe_rate_not_worse_than_baseline": rl_summary["unsafe_rate"] <= baseline_summary["unsafe_rate"],
        "all_preventive_180s_windows_safe": bool(preventive_pass),
        "all_scenarios_have_complete_windows": all(
            values.get("eligible_complete_windows", 0) > 0 for values in scenario_validation.values()
        ),
        "all_scenarios_pass_180s": bool(scenario_validation) and all(
            values.get("pass_preventive_180s_safety", values.get("pass_180s_safety", False))
            for values in scenario_validation.values()
        ),
        "decision_effect_computed_within_15s": bool(
            latency_summary and latency_summary.get("pass_15s", False)
        ),
    }
    return {
        "checks": checks,
        "passed": bool(all(checks.values())),
        "status": "acceptance_candidate" if all(checks.values()) else "development_only",
    }


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
    parser.add_argument("--scenario", default="all", help=f"Training scenario or 'all'. Choices are loaded from {DEFAULT_CONFIG_PATH}.")
    parser.add_argument("--scenario-source", choices=["historical", "synthetic", "fsl_real"], default="historical")
    parser.add_argument("--scenario-config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument(
        "--scenario-weights",
        default=None,
        help="Comma-separated replay weights used with --scenario all, for example sand_plug_risk=3,pressure_limit=2.",
    )
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
    parser.add_argument("--response-model", choices=["empirical", "digital_twin", "learned_hybrid"], default="empirical")
    parser.add_argument("--response-surrogate-path", default=None, help="Trained response_surrogate.joblib required by learned_hybrid.")
    parser.add_argument("--high-level-interval-steps", type=int, default=6, help="How often the high-level option is refreshed.")
    parser.add_argument("--terminate-on-unsafe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-episodes", type=int, default=12)
    parser.add_argument("--eval-episodes-per-scenario", type=int, default=4)
    parser.add_argument("--n-envs", type=int, default=2)
    parser.add_argument("--checkpoint-freq", type=int, default=10000)
    parser.add_argument("--policy-hidden-dim", type=int, default=128)
    parser.add_argument("--resume-model", default=None, help="Existing SB3 .zip model to continue training.")
    parser.add_argument("--eval-only", action="store_true", help="Load --resume-model and only run strict evaluation.")
    parser.add_argument("--max-samples", type=int, default=30000)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--max-rows-per-file", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--run-dir", default=str(ROOT / "runs" / "rl_control_agent"))
    args = parser.parse_args()
    if args.eval_only and not args.resume_model:
        parser.error("--eval-only requires --resume-model")

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
    scenario_weights = None
    if args.scenario_source == "historical":
        features, targets, meta = bundle.x, bundle.y, bundle.meta.copy()
        full_context = full_context.copy()
        meta["scenario_name"] = "historical_real"
        full_context["scenario_name"] = "historical_real"
        scenario_spec = {
            "name": "historical_real",
            "display_name": "真实分段历史状态与动作响应",
            "note": "Fixed synthetic scenarios are reserved for stress testing, not duplicated during main training.",
        }
    elif args.scenario_source == "fsl_real":
        if args.scenario not in REAL_SCENARIOS:
            raise ValueError(f"Unsupported --scenario {args.scenario} for fsl_real. Choices: {list(REAL_SCENARIOS)}")
        features, targets, meta, full_context, scenario_spec = select_real_scenario(
            bundle.x, bundle.y, bundle.meta, full_context, args.scenario
        )
    else:
        scenario_choices = available_scenarios(args.scenario_config)
        scenario_weights = parse_scenario_weights(args.scenario_weights)
        if args.scenario == "all":
            feature_blocks = []
            target_blocks = []
            meta_blocks = []
            context_blocks = []
            scenario_specs = []
            for scenario_name in scenario_choices:
                scenario_x, scenario_meta, scenario_context, spec = apply_scenario(
                    bundle.x,
                    bundle.meta,
                    full_context,
                    scenario_name,
                    args.scenario_config,
                    bundle.feature_names,
                )
                weight = scenario_weights.get(scenario_name, 1)
                for replica in range(weight):
                    replica_meta = scenario_meta.copy()
                    replica_meta["scenario_replica"] = replica
                    replica_context = scenario_context.copy()
                    replica_context["scenario_replica"] = replica
                    feature_blocks.append(scenario_x.copy())
                    target_blocks.append(bundle.y.copy())
                    meta_blocks.append(replica_meta)
                    context_blocks.append(replica_context)
                scenario_specs.append({**spec, "training_weight": weight})
            features = np.concatenate(feature_blocks, axis=0)
            targets = np.concatenate(target_blocks, axis=0)
            meta = pd.concat(meta_blocks, ignore_index=True)
            full_context = pd.concat(context_blocks, ignore_index=True)
            scenario_spec = {
                "name": "all",
                "display_name": "六类工况联合训练",
                "members": scenario_specs,
                "training_weights": {name: scenario_weights.get(name, 1) for name in scenario_choices},
            }
        else:
            if args.scenario not in scenario_choices:
                raise ValueError(f"Unsupported --scenario {args.scenario}. Choices: {scenario_choices} or all")
            features, meta, full_context, scenario_spec = apply_scenario(
                bundle.x, bundle.meta, full_context, args.scenario, args.scenario_config, bundle.feature_names
            )
            targets = bundle.y
            scenario_weights = {args.scenario: 1}
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
    response_surrogate = None
    if args.response_model == "learned_hybrid":
        if not args.response_surrogate_path:
            parser.error("--response-model learned_hybrid requires --response-surrogate-path")
        response_surrogate = ActionResponseSurrogate.load(args.response_surrogate_path)
    if args.response_model in {"digital_twin", "learned_hybrid"}:
        if args.hierarchical:
            env_config = HierarchicalDigitalTwinEnvConfig(
                episode_steps=args.episode_steps,
                terminate_on_unsafe=args.terminate_on_unsafe,
                action_seconds=args.action_seconds,
                high_level_interval_steps=args.high_level_interval_steps,
            )
            env_class = HierarchicalDigitalTwinFracturingControlEnv
        else:
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

    def make_env(indices: np.ndarray, random_start: bool, seed_offset: int = 0):
        kwargs = {"response_surrogate": response_surrogate} if args.response_model == "learned_hybrid" else {}
        return env_class(
            bundle.x[indices], bundle.meta.iloc[indices].reset_index(drop=True),
            full_context.iloc[indices].reset_index(drop=True), schedule, reward_config,
            env_config, args.seed + seed_offset, random_start, **kwargs,
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.run_dir) / f"{args.algorithm}_{timestamp}"
    out.mkdir(parents=True, exist_ok=True)

    def env_factory(rank: int):
        return lambda: Monitor(make_env(train_idx, True, rank))

    vec_env = DummyVecEnv([env_factory(rank) for rank in range(max(args.n_envs, 1))])
    policy_kwargs = {"net_arch": [args.policy_hidden_dim, args.policy_hidden_dim]}
    if args.resume_model:
        model_class = PPO if args.algorithm == "ppo" else SAC
        model = model_class.load(args.resume_model, env=vec_env, device="cpu")
    elif args.algorithm == "ppo":
        model = PPO("MlpPolicy", vec_env, learning_rate=3e-4, n_steps=512, batch_size=256, gamma=0.99, gae_lambda=0.95, ent_coef=0.01, policy_kwargs=policy_kwargs, verbose=1, seed=args.seed, device="cpu")
    else:
        model = SAC("MlpPolicy", vec_env, learning_rate=3e-4, buffer_size=200000, batch_size=256, gamma=0.99, learning_starts=2000, policy_kwargs=policy_kwargs, verbose=1, seed=args.seed, device="cpu")
    checkpoint = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // max(args.n_envs, 1), 1),
        save_path=str(out / "checkpoints"),
        name_prefix=f"{args.algorithm}_policy",
    )
    if not args.eval_only:
        model.learn(total_timesteps=args.total_timesteps, callback=checkpoint, progress_bar=False, reset_num_timesteps=not bool(args.resume_model))

    test_env = make_env(test_idx, False)
    scenario_names = sorted(test_env.meta.get("scenario_name", pd.Series(["default"])).astype(str).unique())
    if args.scenario == "all":
        rl_parts = []
        baseline_parts = []
        offset = 0
        for scenario_name in scenario_names:
            rl_parts.append(evaluate(model, test_env, args.eval_episodes_per_scenario, scenario_name=scenario_name, episode_offset=offset))
            baseline_parts.append(evaluate_historical_baseline(test_env, bundle.y[test_idx], args.eval_episodes_per_scenario, scenario_name=scenario_name, episode_offset=offset))
            offset += args.eval_episodes_per_scenario
        rl_eval = pd.concat(rl_parts, ignore_index=True)
        baseline_eval = pd.concat(baseline_parts, ignore_index=True)
    else:
        rl_eval = evaluate(model, test_env, args.eval_episodes)
        baseline_eval = evaluate_historical_baseline(test_env, bundle.y[test_idx], args.eval_episodes)
    model.save(out / f"{args.algorithm}_fracturing_policy")
    rl_eval.to_csv(out / "rl_evaluation.csv", index=False, encoding="utf-8-sig")
    baseline_eval.to_csv(out / "historical_baseline_evaluation.csv", index=False, encoding="utf-8-sig")
    validation_config = Validation180sConfig(action_seconds=args.action_seconds)
    warning_config = Warning5MinConfig(action_seconds=args.action_seconds)
    warning_5min, warning_5min_summary = annotate_5min_warnings(rl_eval, warning_config)
    warning_5min.to_csv(out / "warning_5min_validation.csv", index=False, encoding="utf-8-sig")
    latency_summary = summarize_decision_latency(rl_eval)
    rl_180s, rl_180s_summary = validate_180s(rl_eval, validation_config)
    baseline_180s, baseline_180s_summary = validate_180s(baseline_eval, validation_config)
    rl_180s.to_csv(out / "rl_180s_validation.csv", index=False, encoding="utf-8-sig")
    baseline_180s.to_csv(out / "historical_baseline_180s_validation.csv", index=False, encoding="utf-8-sig")
    rl_180s.loc[rl_180s["unsafe_within_180s"]].to_csv(out / "rl_180s_failures.csv", index=False, encoding="utf-8-sig")
    rl_180s.loc[rl_180s["uncertain_within_180s"]].to_csv(out / "rl_180s_uncertain_windows.csv", index=False, encoding="utf-8-sig")
    plot_comparison(rl_eval, baseline_eval, out / "rl_vs_historical_reward.png")
    plot_safety_by_scenario(rl_eval, baseline_eval, validation_config, out / "scenario_180s_safety.png")
    rl_summary = summarize(rl_eval)
    baseline_summary = summarize(baseline_eval)
    rl_validation_by_scenario = validate_by_scenario(rl_eval, validation_config)
    baseline_validation_by_scenario = validate_by_scenario(baseline_eval, validation_config)
    quality_gate = build_quality_gate(
        rl_summary,
        baseline_summary,
        rl_180s_summary,
        rl_validation_by_scenario,
        latency_summary,
    )
    decision_cards = []
    for _, row in warning_5min.loc[warning_5min["warning_5min"]].head(20).iterrows():
        sand_plug = float(row["max_predicted_sand_plug_probability"])
        abnormal = float(row["max_predicted_abnormal_probability"])
        main_risk = "砂堵风险" if sand_plug >= abnormal else "异常工况风险"
        decision_cards.append({
            "episode": int(row["episode"]),
            "step": int(row["step"]),
            "warning_horizon_seconds": int(row["warning_horizon_seconds"]),
            "risk_level": "high" if row["predicted_event_within_5min"] else "medium",
            "main_risk": main_risk,
            "recommendation": "降低砂比并限制排量增幅，保持监测后由工程师确认下一步动作。",
            "requires_confirmation": True,
            "confirmation_status": "waiting_confirmation",
            "evidence": {
                "max_abnormal_probability": abnormal,
                "max_sand_plug_probability": sand_plug,
                "max_bottomhole_pressure_mpa": float(row["max_predicted_bottomhole_pressure_mpa"]),
                "predicted_lead_seconds": (
                    None if pd.isna(row["predicted_lead_seconds"])
                    else float(row["predicted_lead_seconds"])
                ),
            },
        })
    (out / "human_machine_decisions.json").write_text(
        json.dumps(decision_cards, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "module": "Gymnasium lightweight hierarchical fracturing control agent" if args.hierarchical else "Gymnasium continuous-action fracturing control agent",
        "algorithm": args.algorithm.upper(),
        "source_fingerprint": source_fingerprint(),
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
        "completed_timesteps": int(model.num_timesteps),
        "resumed_from": args.resume_model,
        "eval_only": bool(args.eval_only),
        "evaluated_scenarios": scenario_names,
        "n_envs": int(args.n_envs),
        "policy_hidden_layers": [args.policy_hidden_dim, args.policy_hidden_dim],
        "scenario": scenario_spec,
        "scenario_training_weights": scenario_weights if args.scenario_source == "synthetic" else None,
        "response_surrogate_path": args.response_surrogate_path,
        "response_model": args.response_model,
        "train_samples": int(len(train_idx)),
        "test_samples": int(len(test_idx)),
        "environment": env_config.to_dict(),
        "reward_config": reward_config.to_dict(),
        "pump_schedule_constraint": schedule.to_dict(),
        "context_provenance": provenance,
        "rl_policy": rl_summary,
        "historical_baseline": baseline_summary,
        "rl_policy_by_scenario": summarize_by_scenario(rl_eval),
        "historical_baseline_by_scenario": summarize_by_scenario(baseline_eval),
        "validation_180s": {
            "rl_policy": rl_180s_summary,
            "historical_baseline": baseline_180s_summary,
            "rl_policy_by_scenario": rl_validation_by_scenario,
            "historical_baseline_by_scenario": baseline_validation_by_scenario,
        },
        "warning_5min": warning_5min_summary,
        "decision_latency": latency_summary,
        "human_machine_interaction": {
            "decision_cards": int(len(decision_cards)),
            "role": "engineer",
            "high_risk_requires_confirmation": True,
            "output": str(out / "human_machine_decisions.json"),
        },
        "quality_gate": quality_gate,
        "outputs": {
            "model": str(out / f"{args.algorithm}_fracturing_policy.zip"),
            "rl_evaluation": str(out / "rl_evaluation.csv"),
            "historical_evaluation": str(out / "historical_baseline_evaluation.csv"),
            "rl_180s_validation": str(out / "rl_180s_validation.csv"),
            "historical_baseline_180s_validation": str(out / "historical_baseline_180s_validation.csv"),
            "rl_180s_failures": str(out / "rl_180s_failures.csv"),
            "rl_180s_uncertain_windows": str(out / "rl_180s_uncertain_windows.csv"),
            "comparison_plot": str(out / "rl_vs_historical_reward.png"),
            "scenario_180s_safety_plot": str(out / "scenario_180s_safety.png"),
            "warning_5min_validation": str(out / "warning_5min_validation.csv"),
            "human_machine_decisions": str(out / "human_machine_decisions.json"),
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"run_dir": str(out), "algorithm": args.algorithm, "rl_policy": summary["rl_policy"], "historical_baseline": summary["historical_baseline"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
