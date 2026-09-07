"""Run one native PyFrac continuation window in an isolated worker.

The legacy PyFrac solver contains numerical paths that can spend an
unbounded amount of time inside a single continuation window.  This worker
is deliberately small: the parent process supplies a serialized last-valid
fracture state, the worker advances it once, and a durable accepted
checkpoint is written before the result is returned.  The parent can then
terminate a slow worker without terminating the whole real-time experiment.

This is an execution-isolation boundary, not a second physical model.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys
import time

import numpy as np


def _json_default(value: object) -> object:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    args = parser.parse_args()
    spec_path = Path(args.spec).resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    result_path = Path(spec["result_path"]).resolve()
    started = time.perf_counter()

    try:
        project_root = Path(spec["project_root"]).resolve()
        dt_root = project_root / "DT-Crack"
        if str(dt_root) not in sys.path:
            sys.path.insert(0, str(dt_root))
        # Load PyFrac's legacy top-level modules before dill resolves the
        # classes contained in a retained Fracture object.
        from forward_models.pyfrac_adapter import PyFracAdapter, PyFracNativeSession  # noqa: E402
        from forward_models.pyfrac_config import PyFracConfig  # noqa: E402

        import dill  # noqa: E402

        config_values = dict(spec["pyfrac_config"])
        if "mesh_extension_directions" in config_values:
            config_values["mesh_extension_directions"] = tuple(
                bool(value) for value in config_values["mesh_extension_directions"]
            )
        if "assimilation_max_parameter_step" in config_values:
            config_values["assimilation_max_parameter_step"] = tuple(
                float(value) for value in config_values["assimilation_max_parameter_step"]
            )
        adapter = PyFracAdapter(PyFracConfig(**config_values), project_root=project_root)
        adapter._load_modules()
        with Path(spec["initial_fracture_path"]).open("rb") as handle:
            initial_fracture = dill.load(handle)
        schedule = np.load(Path(spec["schedule_path"]), allow_pickle=False)

        session = PyFracNativeSession(
            adapter,
            schedule,
            float(spec["native_initial_time_s"]),
            **dict(spec["state_parameters"]),
            max_time_steps=int(spec["max_time_steps"]),
            domain_time_s=float(spec["domain_time_s"]),
            checkpoint_dir=Path(spec["checkpoint_dir"]),
            checkpoint_interval_s=0.0,
            initial_fracture=initial_fracture,
            initial_parameters=dict(spec.get("initial_parameters") or {}),
            initial_successful_steps=int(spec.get("initial_successful_steps", 0)),
            initial_failed_steps=int(spec.get("initial_failed_steps", 0)),
            initial_step_limit_s=float(spec["continuation_step_limit_s"]),
            initial_consecutive_failures=int(spec.get("consecutive_failures", 0)),
            initial_success_streak=int(spec.get("continuation_success_streak", 0)),
            initial_front_metadata_repair_count=int(spec.get("front_metadata_repair_count", 0)),
        )
        result = session.advance_to(
            float(spec["target_time_s"]),
            schedule,
            **dict(spec["state_parameters"]),
            allow_partial=True,
        )
        checkpoint_path = None
        if result.success:
            # Partial progress is an accepted state too.  Force the durable
            # write so the parent can safely rebuild this member after the
            # worker exits, even when the normal 60 s cadence was not met.
            session._persist_accepted_state(force=True)
            if session.disk_checkpoint_paths:
                checkpoint_path = session.disk_checkpoint_paths[-1]
        payload = result.to_dict()
        payload.update(
            {
                "worker_status": "completed" if result.success else "failed",
                "worker_pid": int(__import__("os").getpid()),
                "worker_runtime_s": float(time.perf_counter() - started),
                "checkpoint_path": checkpoint_path,
            }
        )
        _write(result_path, payload)
        return 0 if result.success else 2
    except BaseException as exc:  # worker must always return a machine-readable failure
        _write(
            result_path,
            {
                "worker_status": "exception",
                "worker_pid": int(__import__("os").getpid()),
                "worker_runtime_s": float(time.perf_counter() - started),
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
