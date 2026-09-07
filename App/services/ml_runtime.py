"""Resolve a Python interpreter that can run the local ML jobs.

The desktop UI and the numerical/ML jobs may be installed in different
conda environments.  Keeping the resolution here prevents a Qt-only
interpreter from silently launching a job that immediately fails on numpy or
stable-baselines imports.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


# Checking the module spec avoids importing TensorFlow/Stable-Baselines just
# to select an interpreter.  The actual child process still imports the full
# stack and reports a precise error through the merged QProcess channel.
_REQUIRED_IMPORTS = "import numpy, pandas, importlib.util; assert importlib.util.find_spec('stable_baselines3')"


def _candidates() -> list[Path]:
    values: list[Path] = []
    configured = os.environ.get("FRACTURING_ML_PYTHON", "").strip()
    if configured:
        values.append(Path(configured))
    values.append(Path(sys.executable))
    # Common conda layout: <conda>/envs/<qt-env>/python.exe.  The base
    # interpreter is a useful fallback when the Qt environment is UI-only.
    current = Path(sys.executable).resolve()
    if current.parent.parent.name.lower() == "envs":
        values.append(current.parent.parent.parent / "python.exe")
    result: list[Path] = []
    seen: set[str] = set()
    for value in values:
        key = str(value).lower()
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def resolve_ml_python() -> tuple[str | None, str]:
    """Return ``(executable, diagnostic)`` for a usable ML interpreter."""

    diagnostics: list[str] = []
    for candidate in _candidates():
        if not candidate.exists():
            diagnostics.append(f"{candidate}: 文件不存在")
            continue
        try:
            completed = subprocess.run(
                [str(candidate), "-c", _REQUIRED_IMPORTS],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(Path(__file__).resolve().parents[2]),
                env={**os.environ, "TF_CPP_MIN_LOG_LEVEL": "2"},
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            diagnostics.append(f"{candidate}: {type(exc).__name__}")
            continue
        if completed.returncode == 0:
            return str(candidate), f"已选择机器学习运行环境：{candidate}"
        detail = (completed.stderr or completed.stdout or "依赖检查失败").strip().splitlines()
        diagnostics.append(f"{candidate}: {detail[-1] if detail else '依赖检查失败'}")
    return None, "；".join(diagnostics) or "未找到可用机器学习运行环境"


__all__ = ["resolve_ml_python"]
