from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DEFAULT_STAGES = [
    ("normal_growth", 10000),
    ("baseline", 10000),
    ("pressure_limit", 10000),
    ("cluster_imbalance", 10000),
    ("sand_plug_risk", 20000),
    ("diversion_stage", 10000),
    ("all", 30000),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Curriculum training from stable operation to six-scenario control.")
    parser.add_argument("--algorithm", choices=["ppo", "sac"], default="ppo")
    parser.add_argument("--stage-timesteps", nargs="+", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--episode-steps", type=int, default=60)
    parser.add_argument("--eval-episodes-per-scenario", type=int, default=6)
    parser.add_argument("--max-samples", type=int, default=30000)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=2)
    parser.add_argument("--run-dir", default=str(PROJECT_ROOT / "outputs" / "hmi" / "curriculum"))
    args = parser.parse_args()

    if args.stage_timesteps is not None and len(args.stage_timesteps) != len(DEFAULT_STAGES):
        parser.error(f"--stage-timesteps requires {len(DEFAULT_STAGES)} values")
    stage_steps = args.stage_timesteps or [steps for _, steps in DEFAULT_STAGES]
    stages = [(DEFAULT_STAGES[index][0], stage_steps[index]) for index in range(len(DEFAULT_STAGES))]
    root = Path(args.run_dir).resolve() / datetime.now().strftime("%Y%m%d_%H%M%S")
    root.mkdir(parents=True, exist_ok=True)
    previous_model: Path | None = None
    stage_results = []

    for index, (scenario, timesteps) in enumerate(stages, start=1):
        stage_root = root / f"{index:02d}_{scenario}"
        command = [
            sys.executable,
            str(ROOT / "train_rl_control_agent.py"),
            "--algorithm", args.algorithm,
            "--scenario", scenario,
            "--response-model", "digital_twin",
            "--hierarchical",
            "--total-timesteps", str(timesteps),
            "--episode-steps", str(args.episode_steps),
            "--eval-episodes-per-scenario", str(args.eval_episodes_per_scenario),
            "--max-samples", str(args.max_samples),
            "--max-files", str(args.max_files),
            "--n-envs", str(args.n_envs),
            "--seed", str(args.seed),
            "--run-dir", str(stage_root),
        ]
        if previous_model is not None:
            command.extend(["--resume-model", str(previous_model)])
        log_path = root / f"{index:02d}_{scenario}.log"
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT)
        summaries = sorted(stage_root.rglob("summary.json"), key=lambda path: path.stat().st_mtime)
        if completed.returncode or not summaries:
            stage_results.append(
                {"stage": index, "scenario": scenario, "status": "failed", "returncode": completed.returncode, "log": str(log_path)}
            )
            break
        summary_path = summaries[-1]
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        previous_model = Path(summary["outputs"]["model"])
        stage_results.append(
            {
                "stage": index,
                "scenario": scenario,
                "status": "ok",
                "timesteps": timesteps,
                "completed_timesteps": summary["completed_timesteps"],
                "model": str(previous_model),
                "summary": str(summary_path),
                "reward_mean": summary["rl_policy"]["episode_reward_mean"],
                "unsafe_rate": summary["rl_policy"]["unsafe_rate"],
                "safe_180s_rate": summary["validation_180s"]["rl_policy"]["safe_within_180s_rate"],
                "quality_gate_passed": summary["quality_gate"]["passed"],
            }
        )

    final = {
        "run_root": str(root),
        "curriculum": [name for name, _ in stages],
        "stages": stage_results,
        "completed": len(stage_results) == len(stages) and all(item["status"] == "ok" for item in stage_results),
        "final_model": str(previous_model) if previous_model else None,
        "final_quality_gate_passed": bool(stage_results and stage_results[-1].get("quality_gate_passed", False)),
    }
    (root / "curriculum_summary.json").write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(final, ensure_ascii=False, indent=2))
    if not final["completed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
