from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd

from simulator.scenario_generator import DEFAULT_CONFIG_PATH, available_scenarios


ROOT = Path(__file__).resolve().parent


def configure_plot_fonts() -> None:
    for path in (Path(r"C:\Windows\Fonts\msyh.ttc"), Path(r"C:\Windows\Fonts\simhei.ttf")):
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(path)).get_name()
            plt.rcParams["axes.unicode_minus"] = False
            return


def newest_summary(root: Path) -> Path:
    candidates = sorted(root.glob("*/summary.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No summary.json generated under {root}")
    return candidates[-1]


def main() -> None:
    configure_plot_fonts()
    parser = argparse.ArgumentParser(description="Train and compare the control agent across all configured scenarios.")
    parser.add_argument("--algorithm", choices=["ppo", "sac"], default="ppo")
    parser.add_argument("--scenarios", nargs="*", default=None)
    parser.add_argument("--scenario-config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--total-timesteps", type=int, default=5000)
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--eval-episodes", type=int, default=4)
    parser.add_argument("--episode-steps", type=int, default=30)
    parser.add_argument("--hierarchical", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--run-dir", default=str(ROOT / "runs" / "scenario_suite"))
    args = parser.parse_args()

    scenarios = args.scenarios or available_scenarios(args.scenario_config)
    suite_root = Path(args.run_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for scenario in scenarios:
        scenario_root = suite_root / "runs" / scenario
        cmd = [
            sys.executable, str(ROOT / "train_rl_control_agent.py"),
            "--algorithm", args.algorithm,
            "--scenario", scenario,
            "--scenario-config", args.scenario_config,
            "--total-timesteps", str(args.total_timesteps),
            "--max-samples", str(args.max_samples),
            "--eval-episodes", str(args.eval_episodes),
            "--episode-steps", str(args.episode_steps),
            "--run-dir", str(scenario_root),
        ]
        if args.hierarchical:
            cmd.append("--hierarchical")
        completed = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, encoding="utf-8", errors="replace")
        (suite_root / f"{scenario}.log").write_text(completed.stdout + "\n[stderr]\n" + completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            rows.append({"scenario": scenario, "status": "failed", "returncode": completed.returncode})
            if not args.continue_on_error:
                raise RuntimeError(f"Scenario {scenario} failed; see {suite_root / f'{scenario}.log'}")
            continue
        summary_path = newest_summary(scenario_root)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rl = summary["rl_policy"]
        baseline = summary["historical_baseline"]
        validation = summary["validation_180s"]
        rows.append({
            "scenario": scenario,
            "display_name": summary["scenario"]["display_name"],
            "status": "ok",
            "rl_episode_reward": rl["episode_reward_mean"],
            "baseline_episode_reward": baseline["episode_reward_mean"],
            "reward_gain": rl["episode_reward_mean"] - baseline["episode_reward_mean"],
            "rl_unsafe_rate": rl["unsafe_rate"],
            "baseline_unsafe_rate": baseline["unsafe_rate"],
            "rl_safe_180s_rate": validation["rl_policy"]["safe_within_180s_rate"],
            "baseline_safe_180s_rate": validation["historical_baseline"]["safe_within_180s_rate"],
            "summary_path": str(summary_path),
        })

    frame = pd.DataFrame(rows)
    frame.to_csv(suite_root / "scenario_suite_summary.csv", index=False, encoding="utf-8-sig")
    ok = frame[frame["status"] == "ok"].copy()
    if not ok.empty:
        labels = ok["display_name"].tolist()
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        x = range(len(ok))
        axes[0].bar([v - 0.2 for v in x], ok["baseline_episode_reward"], 0.4, label="Historical")
        axes[0].bar([v + 0.2 for v in x], ok["rl_episode_reward"], 0.4, label="RL")
        axes[0].set_title("Episode reward by scenario")
        axes[0].set_xticks(list(x), labels, rotation=20)
        axes[0].legend()
        axes[1].bar([v - 0.2 for v in x], ok["baseline_safe_180s_rate"], 0.4, label="Historical")
        axes[1].bar([v + 0.2 for v in x], ok["rl_safe_180s_rate"], 0.4, label="RL")
        axes[1].set_ylim(0, 1.05)
        axes[1].set_title("180-second safety rate")
        axes[1].set_xticks(list(x), labels, rotation=20)
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(suite_root / "scenario_suite_comparison.png", dpi=180)
        plt.close(fig)

    suite_summary = {
        "run_root": str(suite_root),
        "algorithm": args.algorithm.upper(),
        "hierarchical": args.hierarchical,
        "scenario_count": len(scenarios),
        "successful_scenarios": int((frame["status"] == "ok").sum()),
        "scientific_status": "Offline multi-condition surrogate benchmark; not field closed-loop validation.",
        "outputs": {
            "table": str(suite_root / "scenario_suite_summary.csv"),
            "figure": str(suite_root / "scenario_suite_comparison.png"),
        },
    }
    (suite_root / "summary.json").write_text(json.dumps(suite_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(suite_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
