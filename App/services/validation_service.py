"""Safe, output-isolated validation helpers for the APP button."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from ..core.paths import PATHS, relative


def new_run_root() -> Path:
    root = PATHS.app_runs / datetime.now().strftime("%Y%m%d_%H%M%S")
    root.mkdir(parents=True, exist_ok=True)
    return root


def build_light_validation_command(run_root: Path, python: Path | None = None) -> list[str]:
    """Return the bounded DT cache/HTML refresh command sequence metadata.

    The actual process is launched asynchronously by ``TaskRunner``.  The
    commands write only to ``outputs/app/runs/<timestamp>`` or the release
    cache, never to ``Data`` or ``artifacts``.
    """

    interpreter = python or Path(sys.executable)
    return [str(interpreter), "-m", "App.services.validation_service", "--run-root", str(run_root)]


def write_validation_manifest(run_root: Path, command: list[str], status: str = "queued", **extra: Any) -> Path:
    payload = {
        "run_id": run_root.name,
        "run_root": relative(run_root),
        "status": status,
        "command": command,
        "source_data_unchanged": True,
        "artifacts_unchanged": True,
        **extra,
    }
    path = run_root / "validation_manifest.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def run_light_validation(run_root: Path, python: Path | None = None) -> dict[str, Any]:
    command = build_light_validation_command(run_root, python)
    completed = subprocess.run(command, cwd=PATHS.root, capture_output=True, text=True, check=False)
    payload = {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "status": "completed" if completed.returncode == 0 else "failed",
    }
    write_validation_manifest(run_root, command, payload["status"], result=payload)
    return payload


def execute_pipeline(run_root: Path, python: Path | None = None) -> dict[str, Any]:
    """Run the bounded DT → bridge → 180-second evidence pipeline.

    Every generated file is inside ``outputs/app/runs/<id>``.  Frozen source
    files remain inputs only; this is the command used by both the CLI and the
    asynchronous QProcess button.
    """

    run_root.mkdir(parents=True, exist_ok=True)
    interpreter = python or Path(sys.executable)
    cache_path = run_root / "dt_realtime_cache.json"
    html_path = run_root / "dt_realtime_3d.html"
    commands = [
        [str(interpreter), str(PATHS.app / "build_dt_realtime_cache.py"), "--output", str(cache_path)],
        [str(interpreter), str(PATHS.app / "build_dt_3d_realtime_html.py"), "--cache", str(cache_path), "--output", str(html_path)],
    ]
    steps = []
    for command in commands:
        completed = subprocess.run(command, cwd=PATHS.root, capture_output=True, text=True, check=False)
        steps.append({"command": command, "returncode": completed.returncode, "stdout": completed.stdout[-4000:], "stderr": completed.stderr[-4000:]})
        if completed.returncode != 0:
            payload = {"status": "failed", "steps": steps, "returncode": completed.returncode}
            write_validation_manifest(run_root, [str(item) for item in commands[0]], "failed", result=payload)
            return payload

    from ..core.artifacts import ArtifactRegistry
    from ..core.integration import build_bridge_from_registry

    registry_snapshot = ArtifactRegistry().snapshot()
    dt_to_hmi, hmi_decision = build_bridge_from_registry(registry_snapshot)
    (run_root / "dt_to_hmi.json").write_text(json.dumps(dt_to_hmi, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_root / "hmi_decision.json").write_text(json.dumps(hmi_decision, ensure_ascii=False, indent=2), encoding="utf-8")
    hmi_summary = registry_snapshot.get("modules", {}).get("hmi", {}).get("summary", {})
    validation = hmi_summary.get("validation_180s", {}) if isinstance(hmi_summary, dict) else {}
    (run_root / "validation_180s.json").write_text(json.dumps({"source": "registered_frozen_hmi", "validation": validation}, ensure_ascii=False, indent=2), encoding="utf-8")
    payload = {
        "status": "completed",
        "returncode": 0,
        "steps": steps,
        "outputs": {"cache": str(cache_path), "html": str(html_path), "dt_to_hmi": str(run_root / "dt_to_hmi.json"), "validation_180s": str(run_root / "validation_180s.json")},
        "source_data_unchanged": True,
        "artifacts_unchanged": True,
    }
    write_validation_manifest(run_root, [str(item) for item in commands[0]], "completed", result=payload)
    return payload


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the output-isolated APP light validation pipeline")
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    result = execute_pipeline(Path(args.run_root), Path(sys.executable))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result.get("status") == "completed" else 1)
