"""Run a same-seed comparison of the KG-EnKF prior modes.

The script deliberately launches the existing direct-observation validator for
each mode.  This keeps the comparison on one physical forward operator,
observation operator, data split, and random seed; only the knowledge-graph
bridge changes.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


MODES = ("off", "uncertainty_only", "soft_prior", "soft_correlated")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare KG-EnKF prior modes on the same validation split.")
    parser.add_argument("--frac-monitor-text", required=True)
    parser.add_argument("--construction-pressure-xls", required=True)
    parser.add_argument("--modes", default=",".join(MODES), help="Comma-separated modes to run.")
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--ensemble-size", type=int, default=80)
    parser.add_argument("--physics-profile", choices=["legacy", "enhanced"], default="enhanced")
    parser.add_argument("--validation-mode", choices=["frozen", "online"], default="frozen")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--knowledge-guided-strength", type=float, default=0.35)
    parser.add_argument("--output-root", default="outputs/dt/kg_enkf_mode_comparison")
    return parser


def newest_run(directory: Path) -> Path:
    candidates = [item for item in directory.iterdir() if item.is_dir()]
    if not candidates:
        raise RuntimeError(f"No validator run was created under {directory}")
    return max(candidates, key=lambda item: item.stat().st_mtime)


def main() -> None:
    args = build_parser().parse_args()
    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    unknown = sorted(set(modes) - set(MODES))
    if not modes or unknown:
        raise SystemExit(f"Invalid --modes: {unknown or 'empty'}; expected {', '.join(MODES)}")

    script = Path(__file__).resolve().parent / "validate_direct_observations.py"
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    run_records: list[dict[str, object]] = []

    for mode in modes:
        mode_root = output_root / mode
        command = [
            sys.executable,
            str(script),
            "--frac-monitor-text",
            str(Path(args.frac_monitor_text).resolve()),
            "--construction-pressure-xls",
            str(Path(args.construction_pressure_xls).resolve()),
            "--max-steps",
            str(args.max_steps),
            "--ensemble-size",
            str(args.ensemble_size),
            "--physics-profile",
            args.physics_profile,
            "--validation-mode",
            args.validation_mode,
            "--seed",
            str(args.seed),
            "--knowledge-guided-mode",
            mode,
            "--knowledge-guided-strength",
            str(args.knowledge_guided_strength),
            "--run-dir",
            str(mode_root),
        ]
        print(f"\n=== KG-EnKF mode: {mode} ===")
        print(" ".join(command))
        subprocess.run(command, check=True)
        run_root = newest_run(mode_root)
        summary_path = run_root / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics = summary.get("metrics", {})
        # The validator keeps the full KG metadata inside metrics so that the
        # summary is self-contained with the other validation metrics.
        kg_meta = metrics.get("knowledge_guided_prior", {})
        row = {
            "mode": mode,
            "run_root": str(run_root),
            "validation_liquid_tvd": metrics.get("validation_liquid_tvd_mean"),
            "validation_sand_tvd": metrics.get("validation_sand_tvd_mean"),
            "validation_bhp_error": metrics.get("validation_bhp_relative_error_mean"),
            "validation_all_observations_within_15_rate": metrics.get(
                "validation_all_observations_within_15_percent_rate"
            ),
            "compute_p50_ms": metrics.get("all_steps_compute_p50_ms"),
            "compute_p95_ms": metrics.get("all_steps_compute_p95_ms"),
            "length_shrink_gt_5_percent_count": metrics.get("length_shrink_gt_5_percent_count"),
            "risk_score": kg_meta.get("signals", {}).get("risk_score"),
            "rule_matches": ";".join(kg_meta.get("signals", {}).get("rule_matches", [])),
            "pressure_observation_multiplier": kg_meta.get("observation_noise_multiplier", {}).get("pressure"),
        }
        rows.append(row)
        run_records.append({"mode": mode, "command": command, "summary": str(summary_path)})

    fields = list(rows[0]) if rows else []
    csv_path = output_root / "kg_enkf_mode_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "experiment": "same-seed KG-EnKF prior mode comparison",
        "modes": modes,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "ensemble_size": args.ensemble_size,
        "physics_profile": args.physics_profile,
        "validation_mode": args.validation_mode,
        "rows": rows,
        "runs": run_records,
        "interpretation": (
            "Only the KG prior bridge changes between modes. The observation update, PKN forward model, "
            "validation split, and random seed are held fixed."
        ),
    }
    json_path = output_root / "kg_enkf_mode_comparison.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_root": str(output_root), "csv": str(csv_path), "json": str(json_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
