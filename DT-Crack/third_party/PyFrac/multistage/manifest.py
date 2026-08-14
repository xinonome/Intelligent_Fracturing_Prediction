from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import sha256_file, sha256_json, write_json, write_yaml


def git_commit(path: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def make_run_metadata(config, logs_path: Path, trajectory_path: Path, run_id: str) -> dict[str, Any]:
    return {
        "project": config.raw.get("project", {}).get("name", "well_x"),
        "run_id": run_id,
        "status": "RUNNING",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pyfrac_commit": git_commit(config.project_root),
        "project_commit": git_commit(config.project_root.parents[2]),
        "config_sha256": sha256_json(config.raw),
        "inputs": {logs_path.name: sha256_file(logs_path), trajectory_path.name: sha256_file(trajectory_path)},
        "stages": {},
        "warnings": list(config.warnings),
    }


def archive_run_metadata(run_dir: Path, config, metadata: dict[str, Any]) -> None:
    write_json(run_dir / "manifest.json", metadata)
    write_yaml(run_dir / "config.lock.yaml", config.raw)
    write_json(run_dir / "input_hashes.json", metadata["inputs"])
    write_json(run_dir / "git_info.json", {"pyfrac_commit": metadata.get("pyfrac_commit"), "project_commit": metadata.get("project_commit")})
    assumptions = Path(config.project_root) / "ASSUMPTIONS.md"
    if assumptions.is_file():
        (run_dir / "assumptions_snapshot.md").write_text(assumptions.read_text(encoding="utf-8"), encoding="utf-8")
