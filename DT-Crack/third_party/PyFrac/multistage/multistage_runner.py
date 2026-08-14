from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import load_config, load_inputs
from .export import export_combined
from .io import sha256_json
from .manifest import archive_run_metadata, git_commit, make_run_metadata
from .plotting import plot_combined, plot_stage
from .stage_runner import ProjectContext, run_stage


def _new_run_dir(config, suffix: str | None = None) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short = sha256_json(config.raw)[:8]
    return config.output_root / f"{stamp}_{short}{('_' + suffix) if suffix else ''}"


def latest_run(config) -> Path | None:
    if not config.output_root.is_dir():
        return None
    dirs = [p for p in config.output_root.iterdir() if p.is_dir() and (p / "manifest.json").is_file()]
    return sorted(dirs, key=lambda p: p.stat().st_mtime, reverse=True)[0] if dirs else None


def run_project(config_or_path, overrides: dict[str, Any] | None = None, output_dir: Path | None = None, resume: bool | None = None):
    config = load_config(config_or_path) if not hasattr(config_or_path, "stages") else config_or_path
    logs, trajectory = load_inputs(config)
    # A new CLI run is reproducible and gets a new run_id. Cache reuse is an
    # explicit user action via --resume; the config field documents that the
    # project supports resume but does not silently hide a new run.
    if bool(resume) and output_dir is None:
        candidate = latest_run(config)
        if candidate is not None:
            import json
            from .io import sha256_file
            existing = json.loads((candidate / "manifest.json").read_text(encoding="utf-8"))
            expected_inputs = {config.logs_path.name: sha256_file(config.logs_path), config.trajectory_path.name: sha256_file(config.trajectory_path)}
            expected_pyfrac = git_commit(config.project_root)
            expected_project = git_commit(config.project_root.parents[2])
            if (existing.get("status") == "PASSED" and existing.get("config_sha256") == sha256_json(config.raw)
                    and existing.get("inputs") == expected_inputs
                    and existing.get("pyfrac_commit") == expected_pyfrac
                    and existing.get("project_commit") == expected_project):
                return candidate, existing, {}
    run_dir = output_dir or _new_run_dir(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = make_run_metadata(config, config.logs_path, config.trajectory_path, run_dir.name)
    archive_run_metadata(run_dir, config, metadata)
    context = ProjectContext(config, logs, trajectory, run_dir)
    results = {}
    failed = None
    for stage in config.stages:
        try:
            result = run_stage(stage, context, overrides=overrides)
            results[stage.stage_id] = result
            metadata["stages"][stage.stage_id] = {"status": "PASSED", "metadata": result.metadata}
            from .io import write_json
            write_json(run_dir / stage.stage_id / "stage_manifest.json", {"stage_id": stage.stage_id, "status": "PASSED", "config_hash": metadata["config_sha256"], "input_hashes": metadata["inputs"], "pyfrac_commit": metadata["pyfrac_commit"], "project_commit": metadata["project_commit"], "metadata": result.metadata})
            plot_stage(result, run_dir / stage.stage_id)
        except Exception as exc:
            metadata["stages"][stage.stage_id] = {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"}
            (run_dir / stage.stage_id).mkdir(parents=True, exist_ok=True)
            from .io import write_json
            write_json(run_dir / stage.stage_id / "stage_manifest.json", {"stage_id": stage.stage_id, "status": "FAILED", "config_hash": metadata["config_sha256"], "input_hashes": metadata["inputs"], "pyfrac_commit": metadata["pyfrac_commit"], "project_commit": metadata["project_commit"], "error": f"{type(exc).__name__}: {exc}"})
            failed = exc
            break
    if results:
        export_combined(results, run_dir)
        plot_combined(results, trajectory, run_dir)
    metadata["status"] = "FAILED" if failed else "PASSED"
    metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
    archive_run_metadata(run_dir, config, metadata)
    if failed:
        raise failed
    return run_dir, metadata, results
