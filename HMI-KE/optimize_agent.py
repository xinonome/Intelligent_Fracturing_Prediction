from __future__ import annotations

"""Run the safety-first Contract-3 agent benchmark.

The command deliberately keeps the 240-record HMI package out of training. It
is an independent output audit because the package contains decisions, not the
300-second feature windows required by the RL environment. The raw construction
segments remain the training source and are split by segment_id inside the
training entry point.
"""

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DEFAULT_HMI_HOLDOUT = PROJECT_ROOT / "deliverables" / "third_part_hmi_delivery" / "data" / "hmi_demo_sample.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def holdout_audit(path: Path) -> dict:
    payload = load_json(path)
    records = payload.get("records", [])
    if not records:
        return {"available": False, "reason": "no_records", "path": str(path)}

    current_sand = [float(row.get("current_sand_ratio_percent", 0.0)) for row in records]
    recommended_sand = [float(row.get("recommended_sand_ratio_percent", 0.0)) for row in records]
    deltas = [
        float(row.get("sand_delta_from_reference_percent", rec - cur))
        for row, cur, rec in zip(records, current_sand, recommended_sand)
    ]
    event_labels = [
        row.get("risk_level") in {"high", "medium"} or bool(row.get("unsafe", False))
        for row in records
    ]
    violations = [
        index for index, (sand, delta) in enumerate(zip(recommended_sand, deltas))
        if delta > 0.5 + 1e-9
    ]
    return {
        "available": True,
        "path": str(path),
        "record_count": len(records),
        "schema_version": payload.get("schema_version"),
        "source_record_count": payload.get("data_scope", {}).get("source_record_count"),
        "training_input": False,
        "independent_feature_windows_available": False,
        "evaluation_scope": "HMI decision-output safety audit; not a new policy rollout evaluation",
        "event_like_records": int(sum(event_labels)),
        "max_current_sand_ratio_percent": max(current_sand),
        "max_recommended_sand_ratio_percent": max(recommended_sand),
        "p95_recommended_sand_ratio_percent": _percentile(recommended_sand, 0.95),
        "max_sand_delta_from_reference_percent": max(deltas),
        "high_sand_recommended_fraction": sum(value >= 10.0 for value in recommended_sand) / len(records),
        "hard_limit_configured": False,
        "hard_limit_violations": None,
        "step_limit_violations": len([value for value in deltas if value > 0.5 + 1e-9]),
        "violation_indices": violations[:20],
        "safety_audit_pass": not violations,
    }


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


def run_seed(seed: int, algorithm: str, args, root: Path) -> dict:
    seed_dir = root / algorithm / f"seed_{seed}"
    log_path = root / algorithm / f"seed_{seed}.log"
    seed_dir.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(ROOT / "train_rl_control_agent.py"),
        "--algorithm", algorithm,
        "--scenario", "all",
        "--scenario-source", args.scenario_source,
        "--response-model", "digital_twin",
        "--hierarchical",
        "--total-timesteps", str(args.total_timesteps),
        "--max-samples", str(args.max_samples),
        "--max-files", str(args.max_files),
        "--max-rows-per-file", str(args.max_rows_per_file),
        "--frame-cache-path", str(args.frame_cache_path or (root / "shared_frame_cache.pkl")),
        "--episode-steps", str(args.episode_steps),
        "--eval-episodes-per-scenario", str(args.eval_episodes_per_scenario),
        "--n-envs", str(args.n_envs),
        "--seed", str(seed),
        "--run-dir", str(seed_dir),
    ]
    if args.pump_schedule_path:
        command.extend(["--pump-schedule-path", str(resolve_project_path(args.pump_schedule_path))])
    if args.include_schedule_context:
        command.append("--include-schedule-context")
    if args.action_encoding != "auto":
        command.extend(["--action-encoding", args.action_encoding])
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT)
    summaries = sorted(seed_dir.rglob("summary.json"), key=lambda p: p.stat().st_mtime)
    if completed.returncode or not summaries:
        return {
            "seed": seed,
            "algorithm": algorithm,
            "status": "failed",
            "returncode": completed.returncode,
            "log": str(log_path),
        }
    summary_path = summaries[-1]
    summary = load_json(summary_path)
    return {
        "seed": seed,
        "algorithm": algorithm,
        "status": "ok",
        "summary_path": str(summary_path),
        "model": summary.get("outputs", {}).get("model"),
        "summary": summary,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_model_card(root: Path, aggregate: dict) -> None:
    status = aggregate["default_decision"]["status"]
    text = f"""# 合同第三部分智能体安全优先优化记录

## 当前状态

- 默认在线架构：PKN+EnKF。
- 智能体角色：离线训练、在线建议式输出，不直接控制现场设备。
- 安全规则优先于候选策略、残差代理和效果奖励。
- 本次评估状态：`{status}`。

## 评价对象

- 历史施工动作；
- 保守规则策略（保持当前值，风险时降排量/降砂比）；
- 分层候选策略：{", ".join(aggregate.get("candidate_algorithms", []))}；
- 240 条 HMI 评估输出仅作为独立安全审计，不作为训练输入，也不冒充原始光纤数据。

## 砂比政策

推荐砂比以当前实测砂比为锚点；单步增加不超过 0.5 个百分点是本轮暂定动作步长，不是物理安全定律。历史数据中的 14% 仅记录为观测值，不作为绝对上限。工程硬上限只有在现场配置明确提供后才启用。高风险、不确定性或代理模型 OOD 时，进入保持/降砂/人工确认路径。

## 准入结论

只有所有随机种子、硬安全约束、180 秒验证、可估计的 5 分钟事件预警、推理耗时和非劣性条件均通过，才允许替换默认模型。否则默认仍为 PKN+EnKF + 保守规则，候选模型仅保留为实验版本。

详细机器可读结果见 `optimization_summary.json`、`model_comparison.csv`、`scenario_metrics.csv`、`action_safety_audit.csv` 和 `holdout_metrics.json`。
"""
    (root / "model_card.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Safety-first multi-baseline Contract-3 agent optimization benchmark")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028])
    parser.add_argument("--algorithm", choices=["ppo", "sac", "td3"], default=None, help="Backward-compatible single-candidate shortcut.")
    parser.add_argument("--algorithms", nargs="+", choices=["ppo", "sac", "td3"], default=["ppo", "sac", "td3"], help="Candidate algorithms to compare under identical splits and seeds.")
    parser.add_argument("--total-timesteps", type=int, default=100000)
    parser.add_argument("--episode-steps", type=int, default=60)
    parser.add_argument(
        "--scenario-source",
        choices=["historical", "synthetic", "fsl_real"],
        default="synthetic",
        help=(
            "Training source for candidate policies. synthetic is the default "
            "for TD3 retraining because it exposes normal and stress counterfactuals; "
            "historical remains available for a strict replay comparison."
        ),
    )
    parser.add_argument(
        "--action-encoding",
        choices=["auto", "legacy", "centered_delta"],
        default="auto",
        help="Forward the normalized action parameterization to the training entry point.",
    )
    parser.add_argument("--eval-episodes-per-scenario", type=int, default=10)
    parser.add_argument("--n-envs", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=0, help="Forwarded data cap; keep 0 for the authorized full training set.")
    parser.add_argument("--max-files", type=int, default=0, help="Forwarded file cap; keep 0 for the authorized full training set.")
    parser.add_argument("--max-rows-per-file", type=int, default=0, help="Forwarded per-file row cap; keep 0 for the authorized full training set.")
    parser.add_argument(
        "--frame-cache-path",
        default=None,
        help="Optional validated local parsed-frame cache shared across repeated benchmark batches.",
    )
    parser.add_argument(
        "--pump-schedule-path",
        default=None,
        help="可选的、与单一井段身份匹配的施工泵序表；不匹配时拒绝训练。",
    )
    parser.add_argument(
        "--include-schedule-context",
        action="store_true",
        help="将泵序排量/砂比/进度作为观测特征，启用后必须重新训练。",
    )
    parser.add_argument("--hmi-holdout-path", default=str(DEFAULT_HMI_HOLDOUT))
    parser.add_argument("--run-dir", default=str(PROJECT_ROOT / "outputs" / "hmi" / "optimization"))
    parser.add_argument("--existing-summaries", nargs="*", default=None, help="Skip training and aggregate existing summary.json files.")
    args = parser.parse_args()
    algorithms = [args.algorithm] if args.algorithm else list(dict.fromkeys(args.algorithms))

    root = resolve_project_path(args.run_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    root.mkdir(parents=True, exist_ok=True)
    runs = []
    if args.existing_summaries:
        for path in args.existing_summaries:
            summary_path = resolve_project_path(path)
            summary = load_json(summary_path)
            runs.append({"seed": None, "algorithm": str(summary.get("algorithm", "unknown")).lower(), "status": "ok", "summary_path": str(summary_path), "model": None, "summary": summary})
    else:
        for algorithm in algorithms:
            for seed in args.seeds:
                runs.append(run_seed(seed, algorithm, args, root))

    successful = [run for run in runs if run["status"] == "ok"]
    scenario_rows = []
    audit_rows = []
    for run in successful:
        summary = run["summary"]
        seed = run["seed"]
        for policy_key, display_name in (
            ("rl_policy_by_scenario", f"{run.get('algorithm', 'candidate').upper()}智能体"),
            ("historical_baseline_by_scenario", "历史施工策略"),
            ("conservative_rule_by_scenario", "保守规则策略"),
        ):
            for scenario, metrics in summary.get(policy_key, {}).items():
                scenario_rows.append({"algorithm": run.get("algorithm"), "seed": seed, "policy": display_name, "scenario": scenario, **metrics})
        for policy_key, display_name in (
            ("rl_policy", f"{run.get('algorithm', 'candidate').upper()}智能体"),
            ("historical_baseline", "历史施工策略"),
            ("conservative_rule_baseline", "保守规则策略"),
        ):
            metrics = summary.get(policy_key, {})
            audit_rows.append({"algorithm": run.get("algorithm"), "seed": seed, "policy": display_name, **metrics.get("safety_audit", {})})

    write_csv(root / "scenario_metrics.csv", scenario_rows)
    write_csv(root / "action_safety_audit.csv", audit_rows)
    holdout = holdout_audit(resolve_project_path(args.hmi_holdout_path))
    (root / "holdout_metrics.json").write_text(json.dumps(holdout, ensure_ascii=False, indent=2), encoding="utf-8")

    model_rows = []
    for algorithm in algorithms:
        candidate_runs = [run for run in successful if run.get("algorithm") == algorithm]
        candidate_summaries = [run["summary"] for run in candidate_runs]
        rewards = [summary.get("rl_policy", {}).get("episode_reward_mean") for summary in candidate_summaries]
        unsafe_rates = [summary.get("rl_policy", {}).get("unsafe_rate") for summary in candidate_summaries]
        quality = [bool(summary.get("quality_gate", {}).get("passed", False)) for summary in candidate_summaries]
        validation = [summary.get("validation_180s", {}).get("rl_policy", {}) for summary in candidate_summaries]
        latency = [summary.get("decision_latency", {}) for summary in candidate_summaries]
        audits = [summary.get("rl_policy", {}).get("safety_audit", {}) for summary in candidate_summaries]
        historical_policies = [summary.get("historical_baseline", {}) for summary in candidate_summaries]
        conservative_policies = [summary.get("conservative_rule_baseline", {}) for summary in candidate_summaries]
        historical_audits = [item.get("safety_audit", {}) for item in historical_policies]
        conservative_audits = [item.get("safety_audit", {}) for item in conservative_policies]
        mean_reward = float(np.mean(rewards)) if rewards else None
        mean_historical_reward = float(np.mean([
            item.get("episode_reward_mean", np.nan) for item in historical_policies
        ])) if historical_policies else None
        mean_conservative_reward = float(np.mean([
            item.get("episode_reward_mean", np.nan) for item in conservative_policies
        ])) if conservative_policies else None
        mean_high_sand = float(np.mean([
            item.get("high_sand_ratio_fraction", np.nan) for item in audits
        ])) if audits else None
        mean_historical_high_sand = float(np.mean([
            item.get("high_sand_ratio_fraction", np.nan) for item in historical_audits
        ])) if historical_audits else None
        mean_conservative_high_sand = float(np.mean([
            item.get("high_sand_ratio_fraction", np.nan) for item in conservative_audits
        ])) if conservative_audits else None
        model_rows.append({
            "algorithm": algorithm,
            "successful_seeds": len(candidate_runs),
            "requested_seeds": len(args.seeds),
            "quality_gate_pass_rate": (sum(quality) / len(quality)) if quality else 0.0,
            "mean_episode_reward": mean_reward,
            "mean_historical_episode_reward": mean_historical_reward,
            "mean_conservative_episode_reward": mean_conservative_reward,
            "reward_gain_vs_historical": (
                mean_reward - mean_historical_reward
                if mean_reward is not None and mean_historical_reward is not None else None
            ),
            "reward_gain_vs_conservative": (
                mean_reward - mean_conservative_reward
                if mean_reward is not None and mean_conservative_reward is not None else None
            ),
            "mean_unsafe_rate": float(np.mean(unsafe_rates)) if unsafe_rates else None,
            "mean_preventive_safe_within_180s_rate": float(np.mean([
                item.get("preventive_safe_within_180s_rate", np.nan) for item in validation
            ])) if validation else None,
            "mean_p95_decision_seconds": float(np.mean([
                item.get("p95_seconds", np.nan) for item in latency
            ])) if latency else None,
            "mean_p95_sand_ratio_percent": float(np.mean([
                item.get("p95_sand_ratio_percent", np.nan) for item in audits
            ])) if audits else None,
            "mean_high_sand_ratio_fraction": mean_high_sand,
            "mean_historical_high_sand_ratio_fraction": mean_historical_high_sand,
            "mean_conservative_high_sand_ratio_fraction": mean_conservative_high_sand,
            "high_sand_fraction_delta_vs_historical": (
                mean_high_sand - mean_historical_high_sand
                if mean_high_sand is not None and mean_historical_high_sand is not None else None
            ),
            "high_sand_fraction_delta_vs_conservative": (
                mean_high_sand - mean_conservative_high_sand
                if mean_high_sand is not None and mean_conservative_high_sand is not None else None
            ),
            "status": "candidate_evaluation_complete" if len(candidate_runs) == len(args.seeds) else "incomplete",
        })
    write_csv(root / "model_comparison.csv", model_rows)

    gates = []
    for run in successful:
        gate = run["summary"].get("quality_gate", {})
        gates.append(bool(gate.get("passed", False)))
    all_seed_gates_pass = bool(successful and len(successful) == len(runs) and all(gates))
    aggregate = {
        "run_root": str(root),
        "plan": "contract3_agent_safety_first_20260817",
        "candidate_algorithms": algorithms,
        "model_comparison_policy": "same grouped split, same scenarios, same safety projection, same seeds; no automatic promotion",
        "training_data": "Data/raw_frac full authorized construction segments; split by segment_id inside each training run",
        "hmi_holdout": holdout,
        "runs": [
            {key: value for key, value in run.items() if key != "summary"}
            for run in runs
        ],
        "successful_runs": len(successful),
        "all_seed_quality_gates_pass": all_seed_gates_pass,
        "default_decision": {
            "status": "experimental_only_fallback_to_pkn_enkf_and_conservative_rules",
            "default_online_model": "PKN+EnKF",
            "direct_device_control": False,
            "promotion_candidate": None,
            "reason": "This run compares candidate policies only; no model is promoted automatically.",
        },
        "outputs": {
            "model_comparison": str(root / "model_comparison.csv"),
            "scenario_metrics": str(root / "scenario_metrics.csv"),
            "action_safety_audit": str(root / "action_safety_audit.csv"),
            "holdout_metrics": str(root / "holdout_metrics.json"),
            "model_card": str(root / "model_card.md"),
        },
    }
    (root / "optimization_summary.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    make_model_card(root, aggregate)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
