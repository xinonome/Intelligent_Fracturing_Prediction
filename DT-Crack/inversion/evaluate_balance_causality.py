"""CLI for matched/control evaluation of balance improvement."""

from __future__ import annotations

import argparse
import json

from inversion.balance_causal_evaluation import run_causal_evaluation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Piggy-Bank balance improvement with a declared comparison design.")
    parser.add_argument("--stage-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--treatment-label", default="piggy_bank")
    parser.add_argument("--control-label", default="baseline")
    parser.add_argument("--design-type", choices=["observational", "randomized", "matched_control", "difference_in_differences"], default="observational")
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_causal_evaluation(
        args.stage_csv,
        args.output_dir,
        treatment_label=args.treatment_label,
        control_label=args.control_label,
        design_type=args.design_type,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
