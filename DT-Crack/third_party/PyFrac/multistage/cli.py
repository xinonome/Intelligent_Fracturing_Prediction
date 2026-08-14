from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config, load_inputs
from .exceptions import MultistageError
from .multistage_runner import latest_run, run_project
from .sensitivity import run_sensitivity
from .report import generate_report
from .convergence import run_convergence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m multistage.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "run"):
        p = sub.add_parser(command); p.add_argument("--config", required=True); p.add_argument("--resume", action="store_true")
    p = sub.add_parser("sensitivity"); p.add_argument("--config", required=True); p.add_argument("--matrix", required=True)
    p = sub.add_parser("report"); p.add_argument("--run-dir", required=False); p.add_argument("--config", required=False)
    p = sub.add_parser("convergence"); p.add_argument("--config", required=True); p.add_argument("--output-dir", required=False)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "validate":
            config = load_config(args.config); logs, trajectory = load_inputs(config)
            print(json.dumps({"status": "PASSED", "stages": [s.__dict__ for s in config.stages], "logs": str(config.logs_path), "trajectory": str(config.trajectory_path), "warnings": list(config.warnings)}, ensure_ascii=False, indent=2))
            return 0
        if args.command == "run":
            run_dir, metadata, _ = run_project(args.config, resume=args.resume)
            report = generate_report(run_dir)
            print(json.dumps({"status": metadata["status"], "run_dir": str(run_dir), "report": str(report)}, ensure_ascii=False, indent=2))
            return 0 if metadata["status"] == "PASSED" else 2
        if args.command == "sensitivity":
            summary = run_sensitivity(args.config, args.matrix)
            print(json.dumps({"status": "PASSED", "summary": str(summary)}, ensure_ascii=False, indent=2))
            return 0
        if args.command == "convergence":
            summary = run_convergence(args.config, args.output_dir)
            print(json.dumps({"status": "PASSED", "summary": str(summary)}, ensure_ascii=False, indent=2))
            return 0
        run_dir = Path(args.run_dir) if args.run_dir else latest_run(load_config(args.config)) if args.config else None
        if run_dir is None:
            raise MultistageError("report requires --run-dir or --config with an existing output")
        print(json.dumps({"status": "PASSED", "report": str(generate_report(run_dir))}, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
