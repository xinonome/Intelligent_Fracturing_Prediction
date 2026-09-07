"""Validate a multi-stage Piggy-Bank ledger from a stage plan CSV.

This command is intentionally explicit about provenance:

* ``field_observed`` requires actual executed stage volumes and can validate a
  real historical sequence;
* ``simulated`` uses the recommendations as executed volumes and validates the
  ledger mechanics only;
* ``shadow`` never mutates the reserve and is suitable for a live read-only
  rehearsal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from inversion.stage_liquid_ledger import StageLedgerConfig, replay_stage_sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a multi-stage Piggy-Bank liquid ledger.")
    parser.add_argument("--stage-csv", required=True, help="Stage sequence CSV.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--execution-mode", choices=["field_observed", "simulated", "shadow"], default="field_observed")
    parser.add_argument("--max-adjustment-fraction", type=float, default=0.10)
    parser.add_argument("--max-reserve-m3", type=float, default=1.0e9)
    parser.add_argument("--allow-unapproved", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    stages = pd.read_csv(args.stage_csv)
    if args.execution_mode == "field_observed" and "actual_liquid_m3" not in stages.columns:
        raise ValueError("field_observed mode requires actual_liquid_m3")
    config = StageLedgerConfig(
        max_stage_adjustment_fraction=max(float(args.max_adjustment_fraction), 0.0),
        max_reserve_m3=max(float(args.max_reserve_m3), 0.0),
    )
    decisions, summary = replay_stage_sequence(
        stages,
        config=config,
        execution_mode=args.execution_mode,
        approve_all=not args.allow_unapproved,
    )
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    decisions.to_csv(output / "piggy_bank_stage_ledger.csv", index=False, encoding="utf-8-sig")
    summary = {
        **summary,
        "scientific_status": (
            "historical_field_sequence_validation"
            if args.execution_mode == "field_observed"
            else "ledger_mechanics_validation"
        ),
        "causal_effect_proven": False,
        "field_control_written": False,
        "outputs": {"stage_ledger": str(output / "piggy_bank_stage_ledger.csv")},
        "limitations": [
            "A ledger conservation pass does not prove that the policy improved fracture balance.",
            "Field control is never written by this command.",
            "Causal improvement requires a valid baseline/control comparison and post-decision outcomes.",
        ],
    }
    (output / "piggy_bank_sequence_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
