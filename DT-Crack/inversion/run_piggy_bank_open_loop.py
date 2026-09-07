"""Run a leakage-free stage-liquid advisory open-loop replay.

The command reads interpreted FracMonitor/DAS cluster observations, uses only
data available at each decision point, calculates a next-window total-liquid
recommendation for the current stage, and records the following historical
observation for context.  It does not alter historical controls and it never
writes a recommendation to a field-control interface.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


DT_ROOT = Path(__file__).resolve().parents[1]
if str(DT_ROOT) not in sys.path:
    sys.path.insert(0, str(DT_ROOT))

from data_fusion import load_frac_monitor_text
from data_fusion.observation_quality import validate_cluster_controls
from data_fusion.well_trajectory_adapter import load_well_trajectory, make_cluster_trajectory_positions
from inversion.piggy_bank_allocator import PiggyBankAllocator, PiggyBankConfig
from inversion.stage_liquid_ledger import StageLedgerConfig, replay_stage_sequence
from inversion.segment_response_metrics import (
    ResponseMetricConfig,
    balance_indices,
    compute_segment_response_metrics,
    response_metric_snapshot,
)


def _load_geometry(args: argparse.Namespace, n_clusters: int) -> tuple[pd.DataFrame | None, str]:
    if args.cluster_geometry_csv:
        geometry = pd.read_csv(args.cluster_geometry_csv)
        if "cluster_id" not in geometry.columns:
            raise ValueError("cluster geometry CSV must contain cluster_id")
        geometry["geometry_source"] = "configured_cluster_geometry"
        return geometry, "configured_cluster_geometry"
    if args.trajectory_csv:
        trajectory = load_well_trajectory(args.trajectory_csv)
        geometry = make_cluster_trajectory_positions(
            trajectory,
            n_clusters,
            stage_md_start_m=args.stage_md_start_m,
            stage_md_end_m=args.stage_md_end_m,
        )
        geometry["geometry_source"] = "trajectory_interpolated"
        return geometry, "trajectory_interpolated"
    return None, "geometry_unavailable"


def _load_base_history(path: str | None) -> pd.DataFrame | None:
    if not path:
        return None
    frame = pd.read_csv(path)
    if "cluster_id" not in frame.columns:
        raise ValueError("cluster history must contain cluster_id")
    if "time_s" not in frame.columns and "source_step" not in frame.columns:
        raise ValueError("cluster history must contain time_s or source_step")
    if "time_s" not in frame.columns:
        frame["time_s"] = pd.to_numeric(frame["source_step"], errors="coerce")
    frame["time_s"] = pd.to_numeric(frame["time_s"], errors="coerce")
    frame["cluster_id"] = pd.to_numeric(frame["cluster_id"], errors="coerce")
    return frame.dropna(subset=["time_s", "cluster_id"]).sort_values(["time_s", "cluster_id"])


def _base_weights(
    current: pd.DataFrame,
    as_of_step: int,
    cluster_history: pd.DataFrame | None,
    cluster_ids: list[int],
) -> np.ndarray:
    if cluster_history is not None:
        available = cluster_history[cluster_history["time_s"] <= float(as_of_step)]
        if not available.empty:
            latest = available.sort_values("time_s").groupby("cluster_id", as_index=False).tail(1).set_index("cluster_id")
            for column in ("posterior_model_liquid_allocation", "recommended_weight", "base_weight"):
                if column in latest.columns:
                    values = pd.to_numeric(latest[column], errors="coerce").reindex(cluster_ids).to_numpy(dtype=float)
                    if np.isfinite(values).all() and float(np.nansum(values)) > 1.0e-12:
                        return np.clip(values, 0.0, None) / float(np.nansum(values))
    cumulative = current.groupby("cluster_id")["cumulative_liquid_volume_m3"].last().reindex(cluster_ids).to_numpy(dtype=float)
    cumulative = np.clip(np.nan_to_num(cumulative, nan=0.0), 0.0, None)
    return cumulative / float(cumulative.sum()) if float(cumulative.sum()) > 1.0e-12 else np.full(len(cluster_ids), 1.0 / len(cluster_ids))


def _future_observed_shares(controls: pd.DataFrame, step: int, cluster_ids: list[int]) -> np.ndarray | None:
    future = controls[controls["step"] > int(step)]
    if future.empty:
        return None
    next_step = int(future["step"].min())
    row = future[future["step"] == next_step]
    values = row.groupby("cluster_id")["cumulative_liquid_volume_m3"].last().reindex(cluster_ids).to_numpy(dtype=float)
    values = np.clip(np.nan_to_num(values, nan=0.0), 0.0, None)
    total = float(values.sum())
    return values / total if total > 1.0e-12 else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Open-loop stage-total liquid advisory with Piggy-Bank.")
    parser.add_argument("--frac-monitor-text")
    parser.add_argument("--stage-plan-csv", help="Optional multi-stage plan with actual_liquid_m3 for ledger replay.")
    parser.add_argument("--ledger-execution-mode", choices=["field_observed", "simulated", "shadow"], default="field_observed")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cluster-history")
    parser.add_argument("--cluster-geometry-csv")
    parser.add_argument("--trajectory-csv")
    parser.add_argument("--stage-md-start-m", type=float)
    parser.add_argument("--stage-md-end-m", type=float)
    parser.add_argument("--default-step-seconds", type=float, default=1.0)
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--max-adjustment-fraction", type=float, default=0.10)
    parser.add_argument("--max-reserve-m3", type=float, default=1.0e9)
    parser.add_argument("--score-threshold", type=float, default=0.20)
    parser.add_argument("--target-shares", nargs="*", type=float)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--allow-invalid-observations", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.stage_plan_csv:
        stages = pd.read_csv(args.stage_plan_csv)
        if args.ledger_execution_mode == "field_observed" and "actual_liquid_m3" not in stages.columns:
            raise ValueError("field_observed ledger replay requires actual_liquid_m3")
        decisions, ledger_summary = replay_stage_sequence(
            stages,
            config=StageLedgerConfig(
                max_stage_adjustment_fraction=max(float(args.max_adjustment_fraction), 0.0),
                max_reserve_m3=max(float(args.max_reserve_m3), 0.0),
            ),
            execution_mode=args.ledger_execution_mode,
        )
        output = Path(args.output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        decisions.to_csv(output / "piggy_bank_stage_ledger.csv", index=False, encoding="utf-8-sig")
        summary = {
            **ledger_summary,
            "scientific_status": "multi_stage_piggy_bank_ledger_validation",
            "field_control_written": False,
            "causal_effect_proven": False,
            "stage_plan_input": str(Path(args.stage_plan_csv).resolve()),
            "outputs": {"stage_ledger": str(output / "piggy_bank_stage_ledger.csv")},
            "limitations": [
                "Ledger conservation does not prove balance improvement.",
                "This entry point never writes to a pump-control interface.",
                "Use the closed-loop controller only after a vendor interface and safety approval are available.",
            ],
        }
        (output / "open_loop_validation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        return summary
    if not args.frac_monitor_text:
        raise ValueError("either --frac-monitor-text or --stage-plan-csv is required")
    monitor = load_frac_monitor_text(args.frac_monitor_text, default_step_seconds=args.default_step_seconds)
    controls, quality = validate_cluster_controls(
        monitor.controls,
        expected_clusters=6,
        pressure_times=None,
    )
    valid_controls = controls[controls["qc_valid"]].drop(columns=["qc_valid", "qc_reason"], errors="ignore")
    if not args.allow_invalid_observations and valid_controls.empty:
        raise ValueError("no valid cluster observations after QC")
    # The open-loop allocator only consumes valid rows.  The QC report remains
    # in the output and invalid time steps are never forward filled.
    controls = valid_controls if not args.allow_invalid_observations else controls.drop(columns=["qc_valid", "qc_reason"], errors="ignore")
    controls["step"] = pd.to_numeric(controls["step"], errors="coerce").astype(int)
    controls["cluster_id"] = pd.to_numeric(controls["cluster_id"], errors="coerce").astype(int)
    # The adapter retains a zero-origin bookkeeping row.  The field replay
    # contract is elapsed time 1..4435 s, so step zero is never a decision
    # point and must not enter response or allocation metrics.
    controls = controls[controls["step"] >= 1].copy()
    cluster_ids = sorted(controls["cluster_id"].unique().tolist())
    if len(cluster_ids) != 6:
        raise ValueError(f"Piggy-Bank requires six clusters; received {cluster_ids}")
    geometry, geometry_source = _load_geometry(args, len(cluster_ids))
    cluster_history = _load_base_history(args.cluster_history)
    target = args.target_shares if args.target_shares else None
    metric_config = ResponseMetricConfig(target_shares=None if target is None else tuple(target))
    all_events, _, all_metric_meta = compute_segment_response_metrics(controls, geometry, config=metric_config)
    allocator = PiggyBankAllocator(
        PiggyBankConfig(
            max_single_cluster_adjustment_fraction=max(float(args.max_adjustment_fraction), 0.0),
            max_total_reserve_m3=max(float(args.max_reserve_m3), 0.0),
            score_threshold=max(float(args.score_threshold), 0.0),
            require_geometry_valid=True,
        )
    )
    steps = sorted(controls["step"].unique().tolist())
    if args.max_steps:
        steps = steps[: max(int(args.max_steps), 1)]
    audit_frames: list[pd.DataFrame] = []
    metric_frames: list[pd.DataFrame] = []
    balance_rows: list[dict[str, object]] = []
    decision_rows: list[dict[str, object]] = []
    for index, step in enumerate(steps):
        current = controls[controls["step"] == int(step)].sort_values("cluster_id")
        if current.empty:
            continue
        metrics, balance, metric_meta = response_metric_snapshot(
            all_events,
            current,
            config=metric_config,
            as_of_step=int(step),
        )
        if not metrics.empty:
            metrics = metrics.copy()
            metrics.insert(0, "as_of_step", int(step))
            metric_frames.append(metrics)
        latest_balance = dict(balance) if balance else {}
        balance_rows.append({"as_of_step": int(step), **latest_balance})
        base = _base_weights(current, int(step), cluster_history, cluster_ids)
        total_rate_m3_s = max(float(pd.to_numeric(current["flow_rate_m3_min"], errors="coerce").sum()) / 60.0, 0.0)
        result = allocator.allocate(
            base,
            total_rate_m3_s,
            args.window_seconds,
            metrics,
            target_shares=target,
            geometry_valid=bool(metric_meta.get("cluster_geometry_available", False)),
        )
        audit = allocator.audit_frame(result, cluster_ids, as_of_step=int(step))
        audit_frames.append(audit)
        future = _future_observed_shares(controls, int(step), cluster_ids)
        next_balance = None
        if future is not None:
            next_balance = balance_indices(future, target)
        decision_rows.append({
            "as_of_step": int(step),
            "total_rate_m3_s": total_rate_m3_s,
            "window_seconds": float(args.window_seconds),
            "status": result["status"],
            "geometry_source": geometry_source,
            "recommendation_direction": result["recommendation_direction"],
            "stage_adjustment_fraction": float(result["stage_adjustment_fraction"]),
            "stage_liquid_delta_m3": float(result["stage_liquid_delta_m3"]),
            "recommended_stage_volume_m3": float(result["recommended_stage_volume_m3"]),
            "recommended_stage_rate_m3_s": float(result["recommended_stage_rate_m3_s"]),
            "virtual_stage_reserve_m3": float(result["virtual_stage_reserve_m3"]),
            "bank_balance_m3": 0.0,
            "cluster_allocation_actionable": bool(result["cluster_allocation_actionable"]),
            "transfer_total_m3": 0.0,
            "total_weight_error": float(result["total_weight_error"]),
            "next_observation_available": future is not None,
            "next_balance_degree": None if next_balance is None else float(next_balance["balance_degree"]),
            "current_balance_degree": latest_balance.get("balance_degree"),
        })
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    metric_frame = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    audit_frame = pd.concat(audit_frames, ignore_index=True) if audit_frames else pd.DataFrame()
    balance_frame = pd.DataFrame(balance_rows)
    decision_frame = pd.DataFrame(decision_rows)
    metric_frame.to_csv(output / "segment_response_metrics.csv", index=False, encoding="utf-8-sig")
    balance_frame.to_csv(output / "balance_history.csv", index=False, encoding="utf-8-sig")
    audit_frame.to_csv(output / "piggy_bank_history.csv", index=False, encoding="utf-8-sig")
    decision_frame.to_csv(output / "open_loop_decisions.csv", index=False, encoding="utf-8-sig")
    transfer_mask = pd.Series(False, index=audit_frame.index, dtype=bool) if not audit_frame.empty else pd.Series(dtype=bool)
    summary = {
        "scientific_status": "open_loop_engineering_validation",
        "observation_source": "interpreted_DAS_or_FracMonitor_cluster_observation",
        "input": str(Path(args.frac_monitor_text).resolve()),
        "cluster_geometry_source": geometry_source,
        "cluster_count": len(cluster_ids),
        "decision_count": int(len(decision_frame)),
        "replay_start_step": int(min(steps)) if steps else None,
        "replay_end_step": int(max(steps)) if steps else None,
        "valid_qc_steps": int(quality.valid_steps),
        "valid_replay_steps": int(controls["step"].nunique()),
        "invalid_qc_steps": int(quality.invalid_steps),
        "geometry_available_for_all_clusters": bool(all_metric_meta.get("cluster_geometry_available", False)),
        "stage_liquid_decrease_recommendation_count": int(decision_frame["recommendation_direction"].eq("decrease").sum()) if not decision_frame.empty else 0,
        "stage_liquid_increase_recommendation_count": int(decision_frame["recommendation_direction"].eq("increase").sum()) if not decision_frame.empty else 0,
        "stage_liquid_hold_count": int(decision_frame["recommendation_direction"].eq("hold").sum()) if not decision_frame.empty else 0,
        "total_transfer_rows": int(transfer_mask.sum()),
        "total_weight_error_max": float(pd.to_numeric(decision_frame["total_weight_error"], errors="coerce").max()) if not decision_frame.empty else 0.0,
        "stage_recommendation_causality_valid": False,
        "physical_parameters_modified": False,
        "field_control_written": False,
        "roll_back_available": True,
        "limitations": [
            "Open-loop replay does not alter historical controls or prove closed-loop causality.",
            "Piggy-Bank only recommends a current-stage total liquid change; it cannot command a specific cluster.",
            "Cluster response metrics explain the stage recommendation but are not cluster-level field-control commands.",
            "No stage recommendation is issued when geometry or response quality is unknown.",
            "Current FracMonitor input is interpreted cluster data, not raw DAS amplitude/strain.",
        ],
        "outputs": {
            "segment_response_metrics": str(output / "segment_response_metrics.csv"),
            "balance_history": str(output / "balance_history.csv"),
            "piggy_bank_history": str(output / "piggy_bank_history.csv"),
            "open_loop_decisions": str(output / "open_loop_decisions.csv"),
        },
        "config": {
            "window_seconds": float(args.window_seconds),
            "max_adjustment_fraction": float(args.max_adjustment_fraction),
            "max_reserve_m3": float(args.max_reserve_m3),
            "score_threshold": float(args.score_threshold),
        },
        "observation_quality": quality.to_dict(),
    }
    (output / "open_loop_validation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
