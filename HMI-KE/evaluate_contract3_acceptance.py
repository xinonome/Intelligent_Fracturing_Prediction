from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from stable_baselines3 import PPO


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decision_engine.integrated_reward import IntegratedRewardConfig
from decision_engine.pump_schedule_constraints import get_schedule_constraint
from rl.digital_twin_env import HierarchicalDigitalTwinEnvConfig, HierarchicalDigitalTwinFracturingControlEnv
from simulator.contract_acceptance import (
    Warning5MinConfig,
    annotate_5min_warnings,
    evaluate_direct_5min_warning,
)
from simulator.validation_180s import Validation180sConfig, validate_180s


def latest_file(pattern: str) -> Path | None:
    candidates = [path for path in PROJECT_ROOT.glob(pattern) if path.is_file()]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def benchmark_online_step(model_path: str | Path, repeats: int = 100) -> dict:
    model = PPO.load(model_path, device="cpu")
    obs_shape = tuple(int(v) for v in model.observation_space.shape)
    obs = np.zeros(obs_shape, dtype=np.float32)
    policy_times = []
    for _ in range(repeats):
        started = perf_counter()
        model.predict(obs, deterministic=True)
        policy_times.append(perf_counter() - started)

    feature_width = max(obs_shape[0] - 5, 1)
    size = max(repeats + 4, 16)
    features = np.zeros((size, feature_width), dtype=np.float32)
    meta = pd.DataFrame(
        {
            "segment_id": ["acceptance_benchmark"] * size,
            "current_pressure": np.full(size, 75.0),
            "current_flow": np.full(size, 12.0),
            "current_sand_ratio": np.full(size, 4.0),
        }
    )
    context = pd.DataFrame(
        {
            "posterior_total_half_length_m": np.linspace(150.0, 180.0, size),
            "posterior_error": np.full(size, 0.10),
            "bottomhole_pressure_mpa": np.full(size, 85.0),
            "net_pressure_mpa": np.full(size, 20.0),
            "abnormal_probability": np.full(size, 0.08),
            "sand_plug_probability": np.full(size, 0.03),
        }
    )
    env = HierarchicalDigitalTwinFracturingControlEnv(
        features,
        meta,
        context,
        get_schedule_constraint("continuous"),
        IntegratedRewardConfig(),
        HierarchicalDigitalTwinEnvConfig(
            episode_steps=size - 2,
            ensemble_size=48,
            action_seconds=60.0,
        ),
        random_start=False,
    )
    env.reset(seed=2026)
    response_times = []
    for _ in range(repeats):
        started = perf_counter()
        _, _, terminated, truncated, _ = env.step(np.zeros(2, dtype=np.float32))
        response_times.append(perf_counter() - started)
        if terminated or truncated:
            env.reset(seed=2026)

    policy = np.asarray(policy_times)
    response = np.asarray(response_times)
    combined = policy + response
    return {
        "scientific_status": "representative_local_online_path_benchmark",
        "repeats": repeats,
        "policy_p95_seconds": float(np.percentile(policy, 95)),
        "pkn_enkf_response_p95_seconds": float(np.percentile(response, 95)),
        "combined_p50_seconds": float(np.percentile(combined, 50)),
        "combined_p95_seconds": float(np.percentile(combined, 95)),
        "combined_max_seconds": float(np.max(combined)),
        "limit_seconds": 15.0,
        "pass_15s": bool(np.percentile(combined, 95) <= 15.0),
        "note": "Excludes one-time Excel loading and report rendering; field hardware must be re-benchmarked.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate contract-part-3 acceptance evidence from an existing rollout.")
    parser.add_argument("--evaluation-csv", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--warning-predictions-csv", default=None)
    parser.add_argument("--run-dir", default=str(PROJECT_ROOT / "outputs" / "hmi" / "contract3_acceptance"))
    parser.add_argument("--benchmark-repeats", type=int, default=100)
    args = parser.parse_args()

    evaluation_path = Path(args.evaluation_csv).resolve() if args.evaluation_csv else (
        latest_file("outputs/hmi/safety_shield_eval/**/rl_evaluation.csv")
        or latest_file("outputs/hmi/**/rl_evaluation.csv")
    )
    if evaluation_path is None or not evaluation_path.exists():
        parser.error("No evaluation CSV found; provide --evaluation-csv")
    model_path = Path(args.model).resolve() if args.model else evaluation_path.with_name("ppo_fracturing_policy.zip")
    if not model_path.exists():
        parser.error(f"No policy model found: {model_path}; provide --model")
    warning_predictions_path = (
        Path(args.warning_predictions_csv).resolve()
        if args.warning_predictions_csv
        else latest_file("outputs/hmi/warning_surrogate_300s/**/predictions.csv")
    )

    evaluation = pd.read_csv(evaluation_path)
    validation_rows, validation_summary = validate_180s(evaluation, Validation180sConfig())
    warning_rows, warning_summary = annotate_5min_warnings(evaluation, Warning5MinConfig())
    direct_warning_summary = (
        evaluate_direct_5min_warning(pd.read_csv(warning_predictions_path))
        if warning_predictions_path else {"available": False, "reason": "not_provided"}
    )
    latency = benchmark_online_step(model_path, args.benchmark_repeats)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(args.run_dir) / timestamp
    output.mkdir(parents=True, exist_ok=True)
    validation_rows.to_csv(output / "validation_180s.csv", index=False, encoding="utf-8-sig")
    warning_rows.to_csv(output / "warning_5min.csv", index=False, encoding="utf-8-sig")

    decision_cards = []
    for _, row in warning_rows.loc[warning_rows["warning_5min"]].head(30).iterrows():
        event = bool(row["predicted_event_within_5min"])
        decision_cards.append(
            {
                "episode": int(row["episode"]),
                "step": int(row["step"]),
                "risk_level": "high" if event else "medium",
                "warning_horizon_seconds": 300,
                "recommendation": "限制排量增幅并降低砂比，保持监测；由工程师确认后执行。",
                "requires_confirmation": True,
                "confirmation_status": "waiting_confirmation",
                "evidence": {
                    "abnormal_probability": float(row["max_predicted_abnormal_probability"]),
                    "sand_plug_probability": float(row["max_predicted_sand_plug_probability"]),
                    "bottomhole_pressure_mpa": float(row["max_predicted_bottomhole_pressure_mpa"]),
                },
            }
        )
    (output / "human_machine_decisions.json").write_text(
        json.dumps(decision_cards, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    preventive_pass = bool(validation_summary.get("pass_preventive_180s_safety", False))
    report = {
        "contract_item": "knowledge-embedded agent real-time control module",
        "validation_scope": "offline historical replay plus calibrated digital-twin simulation",
        "sources": {
            "evaluation_csv": str(evaluation_path),
            "policy_model": str(model_path),
            "warning_predictions_csv": str(warning_predictions_path) if warning_predictions_path else None,
        },
        "warning_5min": warning_summary,
        "direct_warning_model_300s": direct_warning_summary,
        "effect_computation_15s": latency,
        "safety_180s": validation_summary,
        "human_machine_interaction": {
            "available": True,
            "decision_cards": len(decision_cards),
            "high_risk_requires_confirmation": True,
            "output": str(output / "human_machine_decisions.json"),
        },
        "gates": {
            "offline_5min_warning_available": bool(warning_summary.get("available", False)),
            "strict_5min_lead_demonstrated": bool(
                direct_warning_summary.get("pass_5min_warning_recall", False)
            ),
            "local_effect_computation_within_15s": bool(latency["pass_15s"]),
            "preventive_180s_windows_all_safe": preventive_pass,
            "field_acceptance_complete": False,
        },
        "status": (
            "offline_acceptance_candidate"
            if preventive_pass and latency["pass_15s"] and direct_warning_summary.get("pass_5min_warning_recall", False)
            else "development_only"
        ),
        "boundary": "Field acceptance still requires live synchronized data, five-minute prospective labels, and on-site 180-second follow-up.",
    }
    (output / "contract3_acceptance_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "gates": report["gates"], "status": report["status"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
