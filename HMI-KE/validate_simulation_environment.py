from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from decision_engine.integrated_reward import IntegratedRewardConfig
from decision_engine.pump_schedule_constraints import get_schedule_constraint
from rl.digital_twin_env import DigitalTwinEnvConfig, DigitalTwinFracturingControlEnv
from simulator.scenario_generator import apply_scenario, available_scenarios


ROOT = Path(__file__).resolve().parent


def base_data(size: int) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    features = np.zeros((size, 9), dtype=np.float32)
    meta = pd.DataFrame(
        {
            "segment_id": ["validation_segment"] * size,
            "current_pressure": np.linspace(72.0, 76.0, size),
            "current_flow": np.full(size, 12.0),
            "current_sand_ratio": np.full(size, 5.0),
        }
    )
    context = pd.DataFrame(
        {
            "posterior_total_half_length_m": np.linspace(120.0, 165.0, size),
            "posterior_error": np.full(size, 0.10),
            "bottomhole_pressure_mpa": np.linspace(84.0, 88.0, size),
            "net_pressure_mpa": np.linspace(14.0, 18.0, size),
            "abnormal_probability": np.full(size, 0.05),
            "sand_plug_probability": np.full(size, 0.03),
        }
    )
    return features, meta, context


def rollout(scenario_name: str, action: np.ndarray, steps: int, seed: int) -> pd.DataFrame:
    features, meta, context = base_data(max(steps + 3, 12))
    features, meta, context, _ = apply_scenario(features, meta, context, scenario_name)
    env = DigitalTwinFracturingControlEnv(
        features,
        meta,
        context,
        get_schedule_constraint("continuous"),
        IntegratedRewardConfig(),
        DigitalTwinEnvConfig(episode_steps=steps, ensemble_size=24, terminate_on_unsafe=False),
        seed=seed,
        random_start=False,
    )
    env.reset(seed=seed, options={"start_index": 0})
    rows = []
    for step in range(steps):
        _, reward, terminated, truncated, info = env.step(action)
        rows.append({"step": step, "reward": reward, **info})
        if terminated or truncated:
            break
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate directional behavior of all HMI digital-twin scenarios.")
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--run-dir", default=str(ROOT.parent / "outputs" / "hmi" / "environment_validation"))
    args = parser.parse_args()

    out = Path(args.run_dir).resolve() / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for scenario_name in available_scenarios():
        conservative = rollout(scenario_name, np.array([-1.0, -1.0], dtype=np.float32), args.steps, args.seed)
        aggressive = rollout(scenario_name, np.array([1.0, 1.0], dtype=np.float32), args.steps, args.seed)
        finite = bool(
            np.isfinite(conservative.select_dtypes(include=[np.number]).to_numpy()).all()
            and np.isfinite(aggressive.select_dtypes(include=[np.number]).to_numpy()).all()
        )
        length_nondecreasing = bool(
            (conservative["simulated_half_length_m"].diff().fillna(0.0) >= -1e-6).all()
            and (aggressive["simulated_half_length_m"].diff().fillna(0.0) >= -1e-6).all()
        )
        conservative_risk = float(
            conservative[["abnormal_probability", "sand_plug_probability"]].max(axis=1).mean()
        )
        aggressive_risk = float(
            aggressive[["abnormal_probability", "sand_plug_probability"]].max(axis=1).mean()
        )
        risk_direction_ok = conservative_risk <= aggressive_risk + 1e-6
        rows.append(
            {
                "scenario_name": scenario_name,
                "finite_outputs": finite,
                "length_nondecreasing": length_nondecreasing,
                "conservative_mean_risk": conservative_risk,
                "aggressive_mean_risk": aggressive_risk,
                "risk_direction_ok": risk_direction_ok,
                "conservative_unsafe_rate": float(conservative["unsafe"].mean()),
                "aggressive_unsafe_rate": float(aggressive["unsafe"].mean()),
                "conservative_uncertain_rate": float(conservative["uncertain"].mean()),
                "aggressive_uncertain_rate": float(aggressive["uncertain"].mean()),
                "passed": bool(finite and length_nondecreasing and risk_direction_ok),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(out / "scenario_response_validation.csv", index=False, encoding="utf-8-sig")
    summary = {
        "run_dir": str(out),
        "scenario_count": int(len(result)),
        "passed_count": int(result["passed"].sum()),
        "all_passed": bool(result["passed"].all()),
        "results": rows,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.strict and not summary["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
