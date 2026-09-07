"""Run the stage-level Piggy-Bank controller in open/shadow/manual/closed mode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from inversion.closed_loop_controller import ClosedLoopConfig, ClosedLoopController
from inversion.pump_control import DryRunPumpAdapter, HttpPumpAdapter, PumpSafetyGate, PumpSafetyLimits, PumpState


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a safety-gated stage-level Piggy-Bank controller.")
    parser.add_argument("--stage-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["open_loop", "shadow", "manual", "closed_loop"], default="shadow")
    parser.add_argument("--pump-endpoint", help="Vendor endpoint; required for closed_loop mode.")
    parser.add_argument("--allow-write", action="store_true", help="Explicitly arm the HTTP adapter.")
    parser.add_argument("--approve", action="store_true", help="Simulate operator approval for each stage.")
    parser.add_argument("--commit-actual", action="store_true", help="Commit actual_liquid_m3 in manual/closed mode.")
    parser.add_argument("--stage-duration-s", type=float, default=60.0)
    parser.add_argument("--max-adjustment-fraction", type=float, default=0.10)
    parser.add_argument("--max-reserve-m3", type=float, default=1.0e9)
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    stages = pd.read_csv(args.stage_csv)
    limits = PumpSafetyLimits(require_manual_approval=args.mode != "closed_loop")
    gate = PumpSafetyGate(limits)
    if args.pump_endpoint:
        adapter = HttpPumpAdapter(args.pump_endpoint, gate=gate, write_enabled=bool(args.allow_write))
    else:
        if args.mode == "closed_loop":
            raise ValueError("closed_loop mode requires --pump-endpoint and --allow-write")
        adapter = DryRunPumpAdapter(Path(args.output_dir) / "pump_commands.jsonl", gate=gate)
    controller = ClosedLoopController(
        adapter,
        config=ClosedLoopConfig(
            mode=args.mode,
            stage_duration_s=max(float(args.stage_duration_s), 1.0e-9),
            max_stage_adjustment_fraction=max(float(args.max_adjustment_fraction), 0.0),
            max_reserve_m3=max(float(args.max_reserve_m3), 0.0),
        ),
        gate=gate,
    )
    for _, row in stages.sort_values("stage_order" if "stage_order" in stages else stages.index.name or stages.columns[0]).iterrows():
        state = PumpState(
            stage_id=str(row["stage_id"]),
            flow_rate_m3_s=float(row.get("flow_rate_m3_s", 0.0)),
            pressure_mpa=float(row.get("pressure_mpa", 0.0)),
            sand_ratio_pct=float(row.get("sand_ratio_pct", 0.0)),
            data_quality_valid=bool(row.get("data_quality_valid", True)),
        )
        actual = row.get("actual_liquid_m3") if args.commit_actual else None
        controller.process_stage(
            stage_id=str(row["stage_id"]),
            stage_order=int(row.get("stage_order", len(controller.audit))),
            response_class=str(row["response_class"]),
            planned_volume_m3=float(row["planned_liquid_m3"]),
            requested_delta_fraction=float(row.get("requested_delta_fraction", 0.0)),
            current_state=state,
            data_quality_valid=bool(row.get("data_quality_valid", True)),
            uncertainty_high=bool(row.get("uncertainty_high", False)),
            safety_ok=bool(row.get("safety_ok", True)),
            approved=bool(args.approve or args.mode == "closed_loop"),
            actual_volume_m3=actual,
        )
    paths = controller.write_audit(args.output_dir)
    summary = json.loads(Path(paths["summary_json"]).read_text(encoding="utf-8"))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
