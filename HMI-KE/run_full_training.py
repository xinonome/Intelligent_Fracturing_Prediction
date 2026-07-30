from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible multi-seed HMI training and strict 180s validation.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028])
    parser.add_argument("--total-timesteps", type=int, default=100000)
    parser.add_argument("--algorithm", choices=["ppo", "sac"], default="ppo")
    parser.add_argument("--max-samples", type=int, default=30000)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--episode-steps", type=int, default=60)
    parser.add_argument("--eval-episodes-per-scenario", type=int, default=10)
    parser.add_argument("--n-envs", type=int, default=2)
    parser.add_argument("--run-dir", default=str(PROJECT_ROOT / "outputs" / "hmi" / "full_training"))
    args = parser.parse_args()

    root = Path(args.run_dir).resolve() / datetime.now().strftime("%Y%m%d_%H%M%S")
    root.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in args.seeds:
        seed_dir = root / f"seed_{seed}"
        command = [
            sys.executable,
            str(ROOT / "train_rl_control_agent.py"),
            "--algorithm", args.algorithm,
            "--scenario", "all",
            "--response-model", "digital_twin",
            "--hierarchical",
            "--total-timesteps", str(args.total_timesteps),
            "--max-samples", str(args.max_samples),
            "--max-files", str(args.max_files),
            "--episode-steps", str(args.episode_steps),
            "--eval-episodes-per-scenario", str(args.eval_episodes_per_scenario),
            "--n-envs", str(args.n_envs),
            "--seed", str(seed),
            "--run-dir", str(seed_dir),
        ]
        log_path = root / f"seed_{seed}.log"
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT)
        summaries = sorted(seed_dir.rglob("summary.json"), key=lambda path: path.stat().st_mtime)
        if completed.returncode or not summaries:
            runs.append({"seed": seed, "status": "failed", "returncode": completed.returncode, "log": str(log_path)})
            continue
        summary = json.loads(summaries[-1].read_text(encoding="utf-8"))
        validation = summary["validation_180s"]["rl_policy"]
        runs.append(
            {
                "seed": seed,
                "status": "ok",
                "summary": str(summaries[-1]),
                "model": summary["outputs"]["model"],
                "reward_mean": summary["rl_policy"]["episode_reward_mean"],
                "unsafe_rate": summary["rl_policy"]["unsafe_rate"],
                "safe_180s_rate": validation["safe_within_180s_rate"],
                "pass_180s_safety": validation["pass_180s_safety"],
            }
        )

    successful = [run for run in runs if run["status"] == "ok"]
    aggregate = {
        "run_root": str(root),
        "required_gate": "all seeds must have safe_180s_rate=1.0",
        "runs": runs,
        "successful_seeds": len(successful),
        "safe_180s_rate_mean": float(np.mean([run["safe_180s_rate"] for run in successful])) if successful else None,
        "safe_180s_rate_worst": float(np.min([run["safe_180s_rate"] for run in successful])) if successful else None,
        "all_seeds_pass": bool(successful and len(successful) == len(args.seeds) and all(run["pass_180s_safety"] for run in successful)),
    }
    (root / "multi_seed_summary.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
