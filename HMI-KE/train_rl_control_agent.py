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
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import ActionNoise
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
    estimate_sample_interval_seconds,
    load_or_discover_segment_frames,
    segment_split,
)
from pump_schedule_adapter import (
    SCHEDULE_NUMERIC_COLUMNS,
    attach_schedule_to_frame,
    identities_match,
    load_pump_schedule,
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


class DecayingNormalActionNoise(ActionNoise):
    """Dimension-specific TD3 noise that cools as the policy stabilizes."""

    def __init__(self, mean, sigma, final_sigma, decay_calls: int, seed: int):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.sigma = np.asarray(sigma, dtype=np.float32)
        self.final_sigma = np.asarray(final_sigma, dtype=np.float32)
        self.decay_calls = max(int(decay_calls), 1)
        self.calls = 0
        self.rng = np.random.default_rng(seed)

    def __call__(self) -> np.ndarray:
        progress = min(self.calls / self.decay_calls, 1.0)
        current_sigma = self.sigma + progress * (self.final_sigma - self.sigma)
        self.calls += 1
        return self.rng.normal(self.mean, current_sigma).astype(np.float32)

    def reset(self) -> None:
        self.calls = 0

    def __repr__(self) -> str:
        return f"DecayingNormalActionNoise(start={self.sigma}, final={self.final_sigma}, calls={self.calls})"


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
        ROOT / "decision_engine" / "pump_schedule_constraints.py",
        ROOT / "response_surrogate.py",
        ROOT / "data_pipeline.py",
        ROOT / "pump_schedule_adapter.py",
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
                "observed_current_sand_ratio_percent": info.get("pre_action_sand_ratio_percent"),
                "recommended_sand_ratio_percent": info.get("sand_ratio_percent"),
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


def audit_sand_action_sensitivity(
    env_factory,
    starts: np.ndarray,
    raw_sand_actions: tuple[float, ...] = (-1.0, 0.0, 1.0),
) -> pd.DataFrame:
    """Check whether the environment exposes a usable sand-action signal.

    This is deliberately separate from policy evaluation.  It holds the flow
    action fixed and probes the same state with decrease/hold/increase action
    coordinates.  If all three engineering outputs are identical, the issue
    is in action decoding or safety projection.  If the outputs differ but
    rewards do not, the issue is in response/reward sensitivity and should not
    be misdiagnosed as a TD3 architecture problem.
    """

    rows: list[dict] = []
    for probe_id, start in enumerate(np.asarray(starts, dtype=int)):
        for raw_sand in raw_sand_actions:
            env = env_factory()
            _, reset_info = env.reset(options={"start_index": int(start)})
            _, reward, terminated, truncated, info = env.step(
                np.asarray([0.0, raw_sand], dtype=np.float32)
            )
            rows.append(
                {
                    "probe_id": probe_id,
                    "start_index": int(start),
                    "raw_sand_action": float(raw_sand),
                    "scenario_name": str(info.get("scenario_name", "default")),
                    "high_level_option": info.get("high_level_option"),
                    "action_encoding": info.get("action_encoding"),
                    "current_sand_ratio_percent": float(info.get("current_sand_ratio_percent", np.nan)),
                    "recommended_sand_ratio_percent": float(info.get("recommended_sand_ratio_percent", np.nan)),
                    "sand_delta_from_current_percent": float(info.get("sand_delta_from_current_percent", np.nan)),
                    "reward": float(reward),
                    "simulated_pressure_mpa": float(info.get("simulated_pressure_mpa", np.nan)),
                    "simulated_half_length_m": float(info.get("simulated_half_length_m", np.nan)),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    **reset_info,
                }
            )
    return pd.DataFrame(rows)


def evaluate_conservative_baseline(
    env: FracturingControlEnv,
    episodes: int,
    scenario_name: str | None = None,
    episode_offset: int = 0,
) -> pd.DataFrame:
    """Evaluate a hold/reduce rule that never asks the policy to grow sand.

    This is deliberately independent of PPO and of the historical next-action
    labels.  It provides the safety reference needed to tell whether a learned
    policy actually improves the decision rather than merely replaying a safe
    historical window.
    """

    rows: list[dict] = []
    starts = env.evaluation_starts(episodes, scenario_name)
    for episode, start in enumerate(starts):
        env.reset(options={"start_index": int(start)})
        done = False
        step = 0
        while not done:
            row = env.meta.iloc[env._cursor]
            current_flow = float(pd.to_numeric(row.get("current_flow"), errors="coerce"))
            current_sand = float(pd.to_numeric(row.get("current_sand_ratio"), errors="coerce"))
            if not np.isfinite(current_flow):
                current_flow = float(env._current_flow)
            if not np.isfinite(current_sand):
                current_sand = float(env._current_sand)
            context = env._base_context(env._cursor)
            risk = max(
                float(np.nan_to_num(context.get("abnormal_probability", 0.0), nan=0.0)),
                float(np.nan_to_num(context.get("sand_plug_probability", 0.0), nan=0.0)),
            )
            pressure = float(np.nan_to_num(context.get("bottomhole_pressure_mpa", 0.0), nan=0.0))
            if risk >= env.config.abnormal_probability_max * 0.8 or pressure >= env.reward_config.bottomhole_pressure_max_mpa * 0.92:
                target_flow = max(0.0, current_flow - 0.5 * env.schedule.max_flow_step_m3_min)
                target_sand = max(0.0, current_sand - 0.5 * env.schedule.max_sand_decrease_percent)
            else:
                target_flow = current_flow
                target_sand = current_sand
            action = env.encode_engineering_action(target_flow, target_sand)
            _, reward, terminated, truncated, info = env.step(action)
            rows.append({"episode": episode + episode_offset, "step": step, "reward": reward, **info})
            done = terminated or truncated
            step += 1
    return pd.DataFrame(rows)


def summarize_action_safety(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"available": False, "reason": "empty_evaluation"}

    def finite_series(name: str, default: float = 0.0) -> pd.Series:
        if name not in frame:
            return pd.Series(default, index=frame.index, dtype=float)
        return pd.to_numeric(frame[name], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)

    sand = finite_series("sand_ratio_percent")
    sand_delta = finite_series("sand_delta_from_reference_percent")
    flow_delta = finite_series("flow_m3_min") - finite_series("pre_action_flow_m3_min")
    hard_limit_configured = bool(
        frame.get("sand_ratio_hard_limit_configured", pd.Series(False, index=frame.index))
        .fillna(False)
        .astype(bool)
        .any()
    )
    hard_limit = finite_series("sand_ratio_hard_limit_percent", np.nan).dropna()
    hard_limit_value = float(hard_limit.iloc[0]) if hard_limit_configured and not hard_limit.empty else None
    hard_limit_violations = (
        int((sand > hard_limit_value + 1e-9).sum())
        if hard_limit_value is not None else None
    )
    return {
        "available": True,
        "max_observed_sand_ratio_percent": float(sand.max()),
        "p95_sand_ratio_percent": float(sand.quantile(0.95)),
        "high_sand_warning_percent": 10.0,
        "high_sand_ratio_fraction": float((sand >= 10.0).mean()),
        "high_sand_ratio_count": int((sand >= 10.0).sum()),
        "hard_sand_limit_configured": hard_limit_configured,
        "hard_sand_limit_percent": hard_limit_value,
        "hard_sand_limit_violation_count": hard_limit_violations,
        "sand_increase_over_step_count": int((sand_delta > 0.5 + 1e-9).sum()),
        "max_sand_delta_from_reference_percent": float(sand_delta.max()),
        "p95_sand_delta_from_reference_percent": float(sand_delta.quantile(0.95)),
        "sand_non_hold_fraction": float(
            (finite_series("absolute_sand_change_percent") > 1.0e-3).mean()
        ),
        "sand_increase_fraction": float(
            (finite_series("sand_delta_from_current_percent") > 1.0e-3).mean()
        ),
        "sand_decrease_fraction": float(
            (finite_series("sand_delta_from_current_percent") < -1.0e-3).mean()
        ),
        "p95_absolute_sand_change_percent": float(
            finite_series("absolute_sand_change_percent").quantile(0.95)
        ),
        "mean_abs_raw_sand_action": float(
            finite_series("raw_sand_action").abs().mean()
        ),
        "raw_sand_boundary_fraction": float(
            (finite_series("raw_sand_action").abs() >= 0.95).mean()
        ),
        "raw_sand_center_fraction": float(
            (finite_series("raw_sand_action").abs() <= 0.10).mean()
        ),
        "max_absolute_flow_change_m3_min": float(flow_delta.abs().max()),
        "p95_absolute_flow_change_m3_min": float(flow_delta.abs().quantile(0.95)),
        "action_clipped_rate": float(finite_series("action_clipped").mean()),
        "confirmation_rate": float(finite_series("sand_ratio_requires_confirmation").mean()),
        "surrogate_fallback_rate": float(finite_series("surrogate_fallback").mean()),
        "unsafe_rate": float(finite_series("unsafe").mean()),
        # Optional surrogate fields are intentionally NaN when no surrogate is
        # connected.  They are not a failed policy output; populated surrogate
        # fields are checked by the DT environment's safety gate.
        "nan_or_inf_output": bool(
            not np.isfinite(
                frame[
                    [
                        column for column in frame.columns
                        if column not in {
                            "episode", "step", "segment_id", "scenario_name",
                            "response_model", "condition_probability_source",
                            "sand_control_mode", "high_level_option",
                            "action_encoding",
                            "unsafe_reasons", "uncertainty_reasons",
                            "surrogate_fallback_reason", "sand_ratio_hard_limit_percent",
                            # These are optional digital-twin/EnKF context
                            # fields.  They can be absent for historical
                            # replay and are not policy action outputs.
                            "posterior_fracture_toughness_pa_sqrt_m",
                            "cluster_geometry_spread",
                            "cluster_balance_source",
                            "cluster_balance_degree",
                            "cluster_balance_improvement",
                            "cluster_balance_available",
                            "posterior_width_m",
                            "posterior_fracture_volume_m3",
                            "sand_transport_factor",
                            "fracture_width_m",
                            "fracture_volume_m3",
                            "fracture_width_improvement",
                            "fracture_volume_improvement",
                        }
                        and not str(column).startswith("surrogate_")
                    ]
                ].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            ).all()
        ),
    }


def summarize(frame: pd.DataFrame) -> dict:
    episode_reward = frame.groupby("episode")["reward"].sum()
    fields = [
        "integrated_reward",
        "effectiveness_reward",
        "pressure_safety_penalty",
        "abnormal_risk_penalty",
        "construction_cost_penalty",
        "action_boundary_penalty",
        "cluster_balance_reward",
        "fracture_width_reward",
    ]
    return {
        "episode_reward_mean": float(episode_reward.mean()),
        "episode_reward_std": float(episode_reward.std(ddof=0)),
        "step_reward_mean": float(frame["reward"].mean()),
        "unsafe_rate": float(frame["unsafe"].mean()),
        "uncertain_rate": float(frame["uncertain"].mean()) if "uncertain" in frame else 0.0,
        "action_clipped_rate": float(frame["action_clipped"].mean()),
        "safety_audit": summarize_action_safety(frame),
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
    component_cols = [
        "effectiveness_reward",
        "cluster_balance_reward",
        "fracture_width_reward",
        "pressure_safety_penalty",
        "abnormal_risk_penalty",
        "construction_cost_penalty",
    ]
    component_cols = [column for column in component_cols if column in rl.columns and column in baseline.columns]
    x = np.arange(len(component_cols))
    width = 0.36
    axes[1].bar(x - width / 2, [baseline[c].mean() for c in component_cols], width, label="历史动作")
    axes[1].bar(x + width / 2, [rl[c].mean() for c in component_cols], width, label="强化学习")
    labels = {
        "effectiveness_reward": "改造效果",
        "cluster_balance_reward": "分簇均衡奖励",
        "fracture_width_reward": "缝宽效果",
        "pressure_safety_penalty": "压力惩罚",
        "abnormal_risk_penalty": "异常惩罚",
        "construction_cost_penalty": "施工成本",
    }
    axes[1].set_xticks(x, [labels[column] for column in component_cols], rotation=15)
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

    def safety_rate(summary: dict) -> float:
        # Very short smoke runs (or a stage shorter than the 180-second
        # window) legitimately have no estimable rate.  Keep the bar empty
        # instead of aborting report generation with a KeyError.
        value = summary.get("safe_within_180s_rate")
        return float(value) if value is not None and np.isfinite(value) else np.nan

    ax.bar(x - width / 2, [safety_rate(baseline_summary.get(name, {})) for name in names], width, label="历史动作")
    ax.bar(x + width / 2, [safety_rate(rl_summary.get(name, {})) for name in names], width, label="强化学习")
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
    conservative_summary: dict | None,
    rl_180s_summary: dict,
    scenario_validation: dict[str, dict],
    latency_summary: dict | None = None,
    warning_summary: dict | None = None,
) -> dict:
    preventive_pass = rl_180s_summary.get("pass_preventive_180s_safety")
    if preventive_pass is None:
        preventive_pass = rl_180s_summary.get("pass_180s_safety", False)
    conservative_summary = conservative_summary or baseline_summary
    rl_audit = rl_summary.get("safety_audit", {})
    historical_audit = baseline_summary.get("safety_audit", {})
    conservative_audit = conservative_summary.get("safety_audit", {})
    warning_summary = warning_summary or {}
    warning_estimable = int(warning_summary.get("event_windows", 0)) > 0
    checks = {
        "reward_not_worse_than_baseline": rl_summary["episode_reward_mean"] >= baseline_summary["episode_reward_mean"],
        "unsafe_rate_not_worse_than_historical": rl_summary["unsafe_rate"] <= baseline_summary["unsafe_rate"],
        "unsafe_rate_not_worse_than_conservative": rl_summary["unsafe_rate"] <= conservative_summary["unsafe_rate"],
        "high_sand_exposure_not_worse_than_historical": rl_audit.get("high_sand_ratio_fraction", np.inf)
        <= historical_audit.get("high_sand_ratio_fraction", np.inf) + 1e-9,
        "high_sand_exposure_not_worse_than_conservative": rl_audit.get("high_sand_ratio_fraction", np.inf)
        <= conservative_audit.get("high_sand_ratio_fraction", np.inf) + 1e-9,
        "no_agent_generated_sand_limit_violation": (
            not rl_audit.get("hard_sand_limit_configured", False)
            or rl_audit.get("hard_sand_limit_violation_count", 1) == 0
        ),
        "no_agent_generated_sand_step_violation": rl_audit.get("sand_increase_over_step_count", 1) == 0,
        "finite_policy_outputs": not rl_audit.get("nan_or_inf_output", True),
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
        "warning_5min_event_windows_available": warning_estimable,
    }
    notes = []
    if not warning_estimable:
        notes.append("5分钟预警不可估计：评估集没有真实异常事件窗口，不能宣称预警通过。")
    if not conservative_audit:
        notes.append("未提供保守规则基线。")
    return {
        "checks": checks,
        "passed": bool(all(checks.values())),
        "status": "acceptance_candidate" if all(checks.values()) else "development_only",
        "notes": notes,
    }


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    configure_plot_fonts()
    parser = argparse.ArgumentParser(description="Train PPO/SAC/TD3 on the action-conditioned fracturing control environment.")
    parser.add_argument("--algorithm", choices=["ppo", "sac", "td3"], default="ppo")
    parser.add_argument("--data-path", default=str(PROJECT_ROOT / "Data" / "raw_frac"))
    parser.add_argument(
        "--reference-header-path",
        default=None,
        help="可选真实表头文件；缺省自动从 FDBH 原始文件发现 53 列标准表头。",
    )
    parser.add_argument(
        "--pump-schedule-path",
        default=None,
        help="可选施工泵序表。仅允许绑定身份匹配的井段，不匹配时直接停止。",
    )
    parser.add_argument(
        "--include-schedule-context",
        action="store_true",
        help="将泵序排量、砂比和阶段进度加入观测特征；需要重新训练模型。",
    )
    parser.add_argument("--dt-context-csv", default=None)
    parser.add_argument("--abnormal-probability-csv", default=None)
    parser.add_argument(
        "--cluster-balance-csv",
        default=None,
        help="Optional FracMonitor balance_degree or per-cluster share export used by the reward context.",
    )
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
    parser.add_argument(
        "--action-encoding",
        choices=["auto", "legacy", "centered_delta"],
        default="auto",
        help=(
            "Normalized action mapping. auto uses centered_delta for new TD3 "
            "training and legacy for existing PPO/SAC compatibility."
        ),
    )
    parser.add_argument(
        "--td3-learning-starts",
        type=int,
        default=3000,
        help="TD3 random-action warm-up steps before gradient updates.",
    )
    parser.add_argument(
        "--td3-flow-noise-sigma",
        type=float,
        default=0.18,
        help="TD3 normalized exploration noise for the flow action.",
    )
    parser.add_argument(
        "--td3-sand-noise-sigma",
        type=float,
        default=0.35,
        help="TD3 normalized exploration noise for the sand-ratio action.",
    )
    parser.add_argument(
        "--td3-final-flow-noise-sigma",
        type=float,
        default=0.08,
        help="TD3 flow-noise floor after the exploration warm-up.",
    )
    parser.add_argument(
        "--td3-final-sand-noise-sigma",
        type=float,
        default=0.12,
        help="TD3 sand-noise floor after the exploration warm-up.",
    )
    parser.add_argument(
        "--td3-noise-decay-steps",
        type=int,
        default=0,
        help="Number of TD3 action-noise calls over which exploration cools; 0 uses total timesteps.",
    )
    parser.add_argument(
        "--td3-initial-action-sigma",
        type=float,
        default=0.35,
        help="Centered Gaussian sigma for TD3's random warm-up actions; 0 restores uniform sampling.",
    )
    parser.add_argument(
        "--td3-schedule-reward-weight",
        type=float,
        default=0.75,
        help="Weak historical-action alignment weight for TD3; safety and response rewards remain primary.",
    )
    parser.add_argument(
        "--td3-action-boundary-weight",
        type=float,
        default=0.35,
        help="Risk-aware soft penalty for saturated centered actions during TD3 training.",
    )
    parser.add_argument(
        "--td3-policy-delay",
        type=int,
        default=2,
        help="TD3 delayed actor update interval.",
    )
    parser.add_argument(
        "--td3-target-policy-noise",
        type=float,
        default=0.20,
        help="TD3 target-policy smoothing noise in normalized action units.",
    )
    parser.add_argument(
        "--td3-target-noise-clip",
        type=float,
        default=0.50,
        help="TD3 target-policy smoothing noise clip in normalized action units.",
    )
    parser.add_argument("--resume-model", default=None, help="Existing SB3 .zip model to continue training.")
    parser.add_argument("--eval-only", action="store_true", help="Load --resume-model and only run strict evaluation.")
    parser.add_argument("--max-samples", type=int, default=30000)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--max-rows-per-file", type=int, default=0)
    parser.add_argument(
        "--frame-cache-path",
        default=None,
        help="Optional local cache for parsed segment frames; use only for repeat experiments, never as release data.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--run-dir", default=str(ROOT / "runs" / "rl_control_agent"))
    args = parser.parse_args()
    if args.eval_only and not args.resume_model:
        parser.error("--eval-only requires --resume-model")
    if args.include_schedule_context and not args.pump_schedule_path:
        parser.error("--include-schedule-context requires --pump-schedule-path")

    frames = load_or_discover_segment_frames(
        args.data_path,
        args.reference_header_path,
        args.segment_column,
        args.time_column,
        ["SGBY", "PL", "SB", args.label_column],
        ["WITHfiltered", "便签数据", "综合", "aggregate", "combined"],
        args.max_files,
        args.max_rows_per_file,
        args.frame_cache_path,
    )
    schedule_metadata = {"enabled": False}
    if args.pump_schedule_path:
        schedule = load_pump_schedule(args.pump_schedule_path)
        matched = {
            key: frame for key, frame in frames.items()
            if identities_match(frame, schedule)
        }
        if not matched:
            raise ValueError(
                "施工泵序表没有匹配到当前数据中的井段；"
                f"泵序身份={schedule.well_name or '-'}/段{schedule.stage_id or '-'}。"
                "请提供对应井段泵序表，不能把一个阶段的泵序套到全平台。"
            )
        frames = {}
        for key, frame in matched.items():
            frames[key], metadata = attach_schedule_to_frame(frame, schedule, args.time_column)
            schedule_metadata = {"enabled": True, "source": str(args.pump_schedule_path), **metadata}
        print(json.dumps({"pump_schedule": schedule_metadata}, ensure_ascii=False), flush=True)
    interval = estimate_sample_interval_seconds(frames, args.time_column, args.sample_interval_seconds)
    state_columns = ["SGBY", "PL", "SB"]
    if args.include_schedule_context:
        state_columns.extend(SCHEDULE_NUMERIC_COLUMNS)
    bundle = build_dataset(
        frames,
        state_columns,
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
        len(bundle.x),
        args.dt_context_csv,
        args.abnormal_probability_csv,
        args.reward_alignment_mode,
        args.cluster_balance_csv,
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
    provenance["pump_schedule"] = schedule_metadata
    provenance["state_columns"] = state_columns
    provenance["schedule_context_enabled"] = bool(args.include_schedule_context)
    provenance["available_components"].append("multi_condition_scenario")
    train_idx, _, test_idx = segment_split(bundle.meta, 0.75, 0.1, args.seed)
    if not len(test_idx):
        test_idx = train_idx[-min(1000, len(train_idx)):]
    reward_config = IntegratedRewardConfig()
    resolved_schedule_reward_weight = (
        args.td3_schedule_reward_weight if args.algorithm == "td3" else 0.25
    )
    resolved_action_boundary_weight = (
        args.td3_action_boundary_weight if args.algorithm == "td3" else 0.35
    )
    resolved_action_encoding = (
        "centered_delta"
        if args.action_encoding == "auto" and args.algorithm == "td3"
        else "legacy"
        if args.action_encoding == "auto"
        else args.action_encoding
    )
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
                action_encoding=resolved_action_encoding,
                initial_action_sampling_sigma=(args.td3_initial_action_sigma if args.algorithm == "td3" else 0.0),
                schedule_reward_weight=resolved_schedule_reward_weight,
                action_boundary_weight=resolved_action_boundary_weight,
            )
            env_class = HierarchicalDigitalTwinFracturingControlEnv
        else:
            env_config = DigitalTwinEnvConfig(
                episode_steps=args.episode_steps,
                terminate_on_unsafe=args.terminate_on_unsafe,
                action_seconds=args.action_seconds,
                action_encoding=resolved_action_encoding,
                initial_action_sampling_sigma=(args.td3_initial_action_sigma if args.algorithm == "td3" else 0.0),
                schedule_reward_weight=resolved_schedule_reward_weight,
                action_boundary_weight=resolved_action_boundary_weight,
            )
            env_class = DigitalTwinFracturingControlEnv
    elif args.hierarchical:
        env_config = HierarchicalFracturingEnvConfig(
            episode_steps=args.episode_steps,
            terminate_on_unsafe=args.terminate_on_unsafe,
            high_level_interval_steps=args.high_level_interval_steps,
            action_encoding=resolved_action_encoding,
            initial_action_sampling_sigma=(args.td3_initial_action_sigma if args.algorithm == "td3" else 0.0),
            schedule_reward_weight=resolved_schedule_reward_weight,
            action_boundary_weight=resolved_action_boundary_weight,
        )
        env_class = HierarchicalFracturingControlEnv
    else:
        env_config = FracturingEnvConfig(
            episode_steps=args.episode_steps,
            terminate_on_unsafe=args.terminate_on_unsafe,
            action_encoding=resolved_action_encoding,
            initial_action_sampling_sigma=(args.td3_initial_action_sigma if args.algorithm == "td3" else 0.0),
            schedule_reward_weight=resolved_schedule_reward_weight,
            action_boundary_weight=resolved_action_boundary_weight,
        )
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
        model_class = {"ppo": PPO, "sac": SAC, "td3": TD3}[args.algorithm]
        model = model_class.load(args.resume_model, env=vec_env, device="cpu")
    elif args.algorithm == "ppo":
        model = PPO("MlpPolicy", vec_env, learning_rate=3e-4, n_steps=512, batch_size=256, gamma=0.99, gae_lambda=0.95, ent_coef=0.01, policy_kwargs=policy_kwargs, verbose=1, seed=args.seed, device="cpu")
    elif args.algorithm == "sac":
        model = SAC("MlpPolicy", vec_env, learning_rate=3e-4, buffer_size=200000, batch_size=256, gamma=0.99, learning_starts=2000, policy_kwargs=policy_kwargs, verbose=1, seed=args.seed, device="cpu")
    else:
        action_noise = DecayingNormalActionNoise(
            mean=np.zeros(2, dtype=np.float32),
            sigma=np.asarray(
                [args.td3_flow_noise_sigma, args.td3_sand_noise_sigma],
                dtype=np.float32,
            ),
            final_sigma=np.asarray(
                [args.td3_final_flow_noise_sigma, args.td3_final_sand_noise_sigma],
                dtype=np.float32,
            ),
            decay_calls=args.td3_noise_decay_steps or args.total_timesteps,
            seed=args.seed + 100,
        )
        model = TD3(
            "MlpPolicy",
            vec_env,
            learning_rate=3e-4,
            buffer_size=200000,
            batch_size=256,
            gamma=0.99,
            learning_starts=args.td3_learning_starts,
            action_noise=action_noise,
            policy_delay=args.td3_policy_delay,
            target_policy_noise=args.td3_target_policy_noise,
            target_noise_clip=args.td3_target_noise_clip,
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=args.seed,
            device="cpu",
        )
    checkpoint = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // max(args.n_envs, 1), 1),
        save_path=str(out / "checkpoints"),
        name_prefix=f"{args.algorithm}_policy",
    )
    if not args.eval_only:
        model.learn(total_timesteps=args.total_timesteps, callback=checkpoint, progress_bar=False, reset_num_timesteps=not bool(args.resume_model))

    test_env = make_env(test_idx, False)
    sensitivity_starts = test_env.evaluation_starts(
        min(max(args.eval_episodes_per_scenario, 1), 12)
    )
    sensitivity = audit_sand_action_sensitivity(
        lambda: make_env(test_idx, False, seed_offset=1000),
        sensitivity_starts,
    )
    sensitivity.to_csv(out / "sand_action_sensitivity.csv", index=False, encoding="utf-8-sig")
    sensitivity_summary = {
        "available": not sensitivity.empty,
        "probe_count": int(len(sensitivity)),
        "distinct_engineering_outputs": bool(
            sensitivity["recommended_sand_ratio_percent"].nunique(dropna=True) > 1
        ) if not sensitivity.empty else False,
        "nonzero_delta_fraction": float(
            (sensitivity["sand_delta_from_current_percent"].abs() > 1.0e-3).mean()
        ) if not sensitivity.empty else 0.0,
        "mean_reward_increase_vs_hold": (
            float(
                sensitivity.loc[sensitivity["raw_sand_action"] == 1.0, "reward"].mean()
                - sensitivity.loc[sensitivity["raw_sand_action"] == 0.0, "reward"].mean()
            )
            if not sensitivity.empty
            and (sensitivity["raw_sand_action"] == 1.0).any()
            and (sensitivity["raw_sand_action"] == 0.0).any()
            else None
        ),
        "interpretation": (
            "action_decoder_and_response_are_sand_sensitive"
            if not sensitivity.empty
            and sensitivity["recommended_sand_ratio_percent"].nunique(dropna=True) > 1
            else "sand_action_path_is_not_sensitive"
        ),
    }
    scenario_names = sorted(test_env.meta.get("scenario_name", pd.Series(["default"])).astype(str).unique())
    if args.scenario == "all":
        rl_parts = []
        baseline_parts = []
        conservative_parts = []
        offset = 0
        for scenario_name in scenario_names:
            rl_parts.append(evaluate(model, test_env, args.eval_episodes_per_scenario, scenario_name=scenario_name, episode_offset=offset))
            baseline_parts.append(evaluate_historical_baseline(test_env, bundle.y[test_idx], args.eval_episodes_per_scenario, scenario_name=scenario_name, episode_offset=offset))
            conservative_parts.append(evaluate_conservative_baseline(test_env, args.eval_episodes_per_scenario, scenario_name=scenario_name, episode_offset=offset))
            offset += args.eval_episodes_per_scenario
        rl_eval = pd.concat(rl_parts, ignore_index=True)
        baseline_eval = pd.concat(baseline_parts, ignore_index=True)
        conservative_eval = pd.concat(conservative_parts, ignore_index=True)
    else:
        rl_eval = evaluate(model, test_env, args.eval_episodes)
        baseline_eval = evaluate_historical_baseline(test_env, bundle.y[test_idx], args.eval_episodes)
        conservative_eval = evaluate_conservative_baseline(test_env, args.eval_episodes)
    model.save(out / f"{args.algorithm}_fracturing_policy")
    rl_eval.to_csv(out / "rl_evaluation.csv", index=False, encoding="utf-8-sig")
    baseline_eval.to_csv(out / "historical_baseline_evaluation.csv", index=False, encoding="utf-8-sig")
    conservative_eval.to_csv(out / "conservative_rule_evaluation.csv", index=False, encoding="utf-8-sig")
    validation_config = Validation180sConfig(action_seconds=args.action_seconds)
    warning_config = Warning5MinConfig(action_seconds=args.action_seconds)
    warning_5min, warning_5min_summary = annotate_5min_warnings(rl_eval, warning_config)
    warning_5min.to_csv(out / "warning_5min_validation.csv", index=False, encoding="utf-8-sig")
    latency_summary = summarize_decision_latency(rl_eval)
    rl_180s, rl_180s_summary = validate_180s(rl_eval, validation_config)
    baseline_180s, baseline_180s_summary = validate_180s(baseline_eval, validation_config)
    conservative_180s, conservative_180s_summary = validate_180s(conservative_eval, validation_config)
    rl_180s.to_csv(out / "rl_180s_validation.csv", index=False, encoding="utf-8-sig")
    baseline_180s.to_csv(out / "historical_baseline_180s_validation.csv", index=False, encoding="utf-8-sig")
    conservative_180s.to_csv(out / "conservative_rule_180s_validation.csv", index=False, encoding="utf-8-sig")
    if "unsafe_within_180s" in rl_180s.columns:
        rl_180s.loc[rl_180s["unsafe_within_180s"]].to_csv(
            out / "rl_180s_failures.csv", index=False, encoding="utf-8-sig"
        )
    else:
        # A deliberately short smoke dataset may have no complete 180-second
        # window. Keep the run machine-readable instead of failing while
        # writing an empty failure report.
        pd.DataFrame().to_csv(out / "rl_180s_failures.csv", index=False, encoding="utf-8-sig")
    if "uncertain_within_180s" in rl_180s.columns:
        rl_180s.loc[rl_180s["uncertain_within_180s"]].to_csv(
            out / "rl_180s_uncertain_windows.csv", index=False, encoding="utf-8-sig"
        )
    else:
        pd.DataFrame().to_csv(out / "rl_180s_uncertain_windows.csv", index=False, encoding="utf-8-sig")
    plot_comparison(rl_eval, baseline_eval, out / "rl_vs_historical_reward.png")
    plot_safety_by_scenario(rl_eval, baseline_eval, validation_config, out / "scenario_180s_safety.png")
    rl_summary = summarize(rl_eval)
    baseline_summary = summarize(baseline_eval)
    conservative_summary = summarize(conservative_eval)
    rl_validation_by_scenario = validate_by_scenario(rl_eval, validation_config)
    baseline_validation_by_scenario = validate_by_scenario(baseline_eval, validation_config)
    conservative_validation_by_scenario = validate_by_scenario(conservative_eval, validation_config)
    quality_gate = build_quality_gate(
        rl_summary,
        baseline_summary,
        conservative_summary,
        rl_180s_summary,
        rl_validation_by_scenario,
        latency_summary,
        warning_5min_summary,
    )
    quality_gate.setdefault("checks", {})["sand_action_path_sensitive"] = bool(
        sensitivity_summary.get("distinct_engineering_outputs", False)
        and sensitivity_summary.get("nonzero_delta_fraction", 0.0) > 0.0
    )
    if not quality_gate["checks"]["sand_action_path_sensitive"]:
        quality_gate["passed"] = False
        quality_gate.setdefault("notes", []).append(
            "砂比动作路径不可区分：不得将该策略作为有效训练结果。"
        )
    decision_cards = []
    warning_rows = (
        warning_5min.loc[warning_5min["warning_5min"]].head(20)
        if "warning_5min" in warning_5min.columns
        else pd.DataFrame()
    )
    for _, row in warning_rows.iterrows():
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
        "state": (
            "current 300s pressure/rate/sand ratio plus available DT/risk context"
            + (" and aligned pump-schedule flow/sand/progress" if args.include_schedule_context else "")
        ),
        "state_columns": state_columns,
        "pump_schedule": schedule_metadata,
        "action": "next 60s mean rate and sand ratio",
        "reward": "fracture effectiveness - pressure risk - abnormal risk - high-sand/action-change safety penalty - construction cost",
        "total_timesteps": args.total_timesteps,
        "completed_timesteps": int(model.num_timesteps),
        "resumed_from": args.resume_model,
        "eval_only": bool(args.eval_only),
        "evaluated_scenarios": scenario_names,
        "n_envs": int(args.n_envs),
        "policy_hidden_layers": [args.policy_hidden_dim, args.policy_hidden_dim],
        "action_encoding": resolved_action_encoding,
        "td3_training": {
            "learning_starts": args.td3_learning_starts if args.algorithm == "td3" else None,
            "flow_noise_sigma": args.td3_flow_noise_sigma if args.algorithm == "td3" else None,
            "sand_noise_sigma": args.td3_sand_noise_sigma if args.algorithm == "td3" else None,
            "final_flow_noise_sigma": args.td3_final_flow_noise_sigma if args.algorithm == "td3" else None,
            "final_sand_noise_sigma": args.td3_final_sand_noise_sigma if args.algorithm == "td3" else None,
            "noise_decay_steps": (
                args.td3_noise_decay_steps or args.total_timesteps
                if args.algorithm == "td3" else None
            ),
            "initial_action_sampling_sigma": (
                args.td3_initial_action_sigma if args.algorithm == "td3" else None
            ),
            "schedule_reward_weight": (
                resolved_schedule_reward_weight if args.algorithm == "td3" else None
            ),
            "action_boundary_weight": (
                resolved_action_boundary_weight if args.algorithm == "td3" else None
            ),
            "policy_delay": args.td3_policy_delay if args.algorithm == "td3" else None,
            "target_policy_noise": args.td3_target_policy_noise if args.algorithm == "td3" else None,
            "target_noise_clip": args.td3_target_noise_clip if args.algorithm == "td3" else None,
            "rationale": (
                "centered sand residual, centered warm-up and decaying dimension-specific exploration to avoid boundary policies"
                if args.algorithm == "td3" else None
            ),
        },
        "sand_action_sensitivity": sensitivity_summary,
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
        "conservative_rule_baseline": conservative_summary,
        "rl_policy_by_scenario": summarize_by_scenario(rl_eval),
        "historical_baseline_by_scenario": summarize_by_scenario(baseline_eval),
        "conservative_rule_by_scenario": summarize_by_scenario(conservative_eval),
        "validation_180s": {
            "rl_policy": rl_180s_summary,
            "historical_baseline": baseline_180s_summary,
            "conservative_rule_baseline": conservative_180s_summary,
            "rl_policy_by_scenario": rl_validation_by_scenario,
            "historical_baseline_by_scenario": baseline_validation_by_scenario,
            "conservative_rule_by_scenario": conservative_validation_by_scenario,
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
            "conservative_rule_evaluation": str(out / "conservative_rule_evaluation.csv"),
            "rl_180s_validation": str(out / "rl_180s_validation.csv"),
            "historical_baseline_180s_validation": str(out / "historical_baseline_180s_validation.csv"),
            "conservative_rule_180s_validation": str(out / "conservative_rule_180s_validation.csv"),
            "rl_180s_failures": str(out / "rl_180s_failures.csv"),
            "rl_180s_uncertain_windows": str(out / "rl_180s_uncertain_windows.csv"),
            "comparison_plot": str(out / "rl_vs_historical_reward.png"),
            "scenario_180s_safety_plot": str(out / "scenario_180s_safety.png"),
            "warning_5min_validation": str(out / "warning_5min_validation.csv"),
            "sand_action_sensitivity": str(out / "sand_action_sensitivity.csv"),
            "human_machine_decisions": str(out / "human_machine_decisions.json"),
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"run_dir": str(out), "algorithm": args.algorithm, "rl_policy": summary["rl_policy"], "historical_baseline": summary["historical_baseline"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
