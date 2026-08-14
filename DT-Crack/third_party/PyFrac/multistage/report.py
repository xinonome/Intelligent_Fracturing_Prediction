from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def generate_report(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    config = {}
    lock = run_dir / "config.lock.yaml"
    if lock.is_file():
        import yaml
        config = yaml.safe_load(lock.read_text(encoding="utf-8")) or {}
    lines = [f"# Multi-stage PyFrac run report", "", f"- Project: `{manifest.get('project')}`", f"- Run: `{manifest.get('run_id')}`", f"- Status: **{manifest.get('status')}**", f"- PyFrac commit: `{manifest.get('pyfrac_commit')}`", f"- Project commit: `{manifest.get('project_commit')}`", "", "## Input ranges", ""]
    for field, path_key in (("logs", "logs_csv"), ("trajectory", "trajectory_csv")):
        path = config.get("inputs", {}).get(path_key)
        line = f"- {field}: `{path}`"
        try:
            source = Path(run_dir).parents[2] / path
            frame = pd.read_csv(source)
            if "MD_m" in frame:
                line += f"; MD range {frame.MD_m.min():g}–{frame.MD_m.max():g} m"
            if field == "logs" and "sigma_hmin_Pa" in frame:
                line += f"; stress range {frame.sigma_hmin_Pa.min():g}–{frame.sigma_hmin_Pa.max():g} Pa"
            if field == "trajectory" and "TVD_m" in frame:
                line += f"; TVD range {frame.TVD_m.min():g}–{frame.TVD_m.max():g} m"
        except Exception:
            line += "; range unavailable"
        lines.append(line)
    lines.extend(["", "## Stage definitions", ""])
    for stage in config.get("stages", []):
        lines.append(f"- {stage.get('id')}: MD {stage.get('md_start_m')}–{stage.get('md_end_m')} m")
    lines.extend(["", "## Solver settings", "", f"- Mesh: `{config.get('mesh')}`", f"- Fluid: `{config.get('fluid')}`", f"- Injection: `{config.get('injection')}`", f"- Simulation: `{config.get('simulation')}`", "", "## Stages", ""])
    for stage_id, entry in manifest.get("stages", {}).items():
        lines.append(f"### {stage_id}")
        lines.append(f"- Status: {entry.get('status')}")
        if entry.get("error"): lines.append(f"- Error: `{entry['error']}`")
        metadata = entry.get("metadata", {})
        lines.append(f"- Handover time: {metadata.get('handover_time_s')}")
        lines.append(f"- Representative material: `{metadata.get('stage_material')}`")
        lines.append(f"- Mass balance: `{metadata.get('mass_balance')}`")
        lines.append("")
    combined_metrics = run_dir / "combined" / "all_stages_metrics.csv"
    if combined_metrics.is_file():
        frame = pd.read_csv(combined_metrics)
        lines.extend(["## Final metrics", "", frame.groupby("stage_id").tail(1).to_markdown(index=False), ""])
    sensitivity = run_dir / "sensitivity" / "summary.csv"
    if sensitivity.is_file():
        lines.extend(["## Sensitivity comparison", "", pd.read_csv(sensitivity).to_markdown(index=False), ""])
    lines.extend(["## Scientific boundary", "", "V1 runs each stage independently in a planar PyFrac model and maps results to a common well-scale coordinate system. It does not model stress shadow, fracture interaction, proppant transport, or RL optimization.", "", "## Warnings", ""])
    lines.extend([f"- {warning}" for warning in manifest.get("warnings", [])] or ["- None recorded."])
    report_dir = run_dir / "report"; report_dir.mkdir(exist_ok=True)
    output = report_dir / "run_report.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output
