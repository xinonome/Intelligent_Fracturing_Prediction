"""Unified command line entry for the three contract workstreams."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import webbrowser
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Data"
OUTPUTS = ROOT / "outputs"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def run(script: Path, arguments: list[str], cwd: Path | None = None) -> int:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    python_path = [str(ROOT / "DT-Crack"), str(ROOT / "FSL-Expert"), str(ROOT / "HMI-KE")]
    env["PYTHONPATH"] = os.pathsep.join(python_path + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    command = [sys.executable, str(script), *arguments]
    print(json.dumps({"command": command, "cwd": str(cwd or ROOT)}, ensure_ascii=False, indent=2))
    return subprocess.run(command, cwd=cwd or ROOT, env=env).returncode


def fsl_command(action: str, extra: list[str]) -> int:
    module = ROOT / "FSL-Expert"
    if action == "knowledge-graph":
        index = module / "knowledge_graph" / "full_book_qwen_output" / "index_full_book_qwen.html"
        if not index.exists():
            raise FileNotFoundError(f"Knowledge-graph page not found: {index}")
        webbrowser.open(index.resolve().as_uri())
        print(index)
        return 0
    scripts = {
        "train": "train_hierarchical_grouped_abnormal_lgbm.py",
        "gnn": "train_frac_gnn.py",
        "transfer": "train_frac_gnn_transfer.py",
        "direct": "train_direct_grouped_working_type_lgbm.py",
        "transition": "train_working_type_transition_lgbm.py",
    }
    defaults = ["--data-path", str(DATA / "raw_frac")]
    return run(module / scripts[action], [*defaults, "--run-dir", str(OUTPUTS / "fsl" / action), *extra])


def dt_command(action: str, extra: list[str]) -> int:
    module = ROOT / "DT-Crack"
    common = [
        "--frac-monitor-text", str(DATA / "3Dfrac" / "光纤本井监测08.txt"),
        "--construction-pressure-xls", str(DATA / "3Dfrac" / "JY84-Z1-stage08-f1.xls"),
    ]
    if action == "validate":
        return run(module / "inversion" / "validate_direct_observations.py", [*common, *extra])
    if action == "benchmark":
        benchmark_args = [
            "--frac-monitor-text", str(DATA / "3Dfrac" / "光纤本井监测08.txt"),
            "--well-trajectory-csv", str(DATA / "3Dfrac" / "JY84-Z1HF-1011.csv"),
            "--construction-pressure-xls", str(DATA / "3Dfrac" / "JY84-Z1-stage08-f1.xls"),
        ]
        return run(module / "inversion" / "benchmark_forward_models.py", [*benchmark_args, *extra])
    visualization_args = [
        "--backend", "plotly-html",
        "--frac-monitor-text", str(DATA / "3Dfrac" / "光纤本井监测08.txt"),
        "--well-trajectory-csv", str(DATA / "3Dfrac" / "JY84-Z1HF-1011.csv"),
        "--construction-pressure-xls", str(DATA / "3Dfrac" / "JY84-Z1-stage08-f1.xls"),
        "--html", str(OUTPUTS / "dt" / "digital_twin_3d.html"),
    ]
    return run(module / "visualization" / "digital_twin_3d.py", [*visualization_args, *extra])


def hmi_command(action: str, extra: list[str]) -> int:
    module = ROOT / "HMI-KE"
    if action == "train":
        defaults = [
            "--data-path", str(DATA / "raw_frac"),
            "--response-model", "digital_twin",
            "--run-dir", str(OUTPUTS / "hmi" / "training"),
        ]
        return run(module / "train_rl_control_agent.py", [*defaults, *extra], module)
    return run(
        module / "run_scenario_suite.py",
        ["--hierarchical", "--run-dir", str(OUTPUTS / "hmi" / "scenarios"), *extra],
        module,
    )


def test_all() -> int:
    commands = [
        [sys.executable, "-m", "compileall", "-q", "FSL-Expert", "DT-Crack", "HMI-KE", "App"],
        [sys.executable, "-m", "unittest", "discover", "DT-Crack/tests"],
        [sys.executable, "-m", "pytest", "-q", "HMI-KE/tests"],
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "DT-Crack"), str(ROOT / "FSL-Expert"), str(ROOT / "HMI-KE")])
    for command in commands:
        print(" ".join(command))
        completed = subprocess.run(command, cwd=ROOT, env=env)
        if completed.returncode:
            return completed.returncode
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Intelligent fracturing unified runner")
    parser.add_argument("module", choices=["fsl", "dt", "hmi", "app", "test"])
    parser.add_argument("action", nargs="?")
    args, extra = parser.parse_known_args()
    if args.module == "fsl":
        action = args.action or "train"
        if action not in {"train", "gnn", "transfer", "direct", "transition", "knowledge-graph"}:
            parser.error(f"unsupported FSL action: {action}")
        code = fsl_command(action, extra)
    elif args.module == "dt":
        action = args.action or "validate"
        if action not in {"validate", "benchmark", "visualize"}:
            parser.error(f"unsupported DT action: {action}")
        code = dt_command(action, extra)
    elif args.module == "hmi":
        action = args.action or "train"
        if action not in {"train", "scenarios"}:
            parser.error(f"unsupported HMI action: {action}")
        code = hmi_command(action, extra)
    elif args.module == "app":
        code = run(ROOT / "App" / "run_app.py", ([args.action] if args.action else []) + extra)
    else:
        code = test_all()
    raise SystemExit(code)


if __name__ == "__main__":
    main()
