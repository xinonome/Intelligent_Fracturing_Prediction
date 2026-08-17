"""Canonical project paths used by the acceptance application.

Keeping path resolution in one place is important for the frozen-demo mode:
the UI may be launched from any working directory, while registered paths are
always relative to the project root.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve(value: str | Path | None, root: Path = PROJECT_ROOT) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value).replace("/", os.sep).replace("\\", os.sep))
    return path if path.is_absolute() else root / path


def relative(value: str | Path | None, root: Path = PROJECT_ROOT) -> str | None:
    path = resolve(value, root)
    if path is None:
        return None
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


@dataclass(frozen=True)
class ProjectPaths:
    root: Path = PROJECT_ROOT

    @property
    def app(self) -> Path:
        return self.root / "App"

    @property
    def config(self) -> Path:
        return self.app / "config"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"

    @property
    def app_outputs(self) -> Path:
        return self.outputs / "app"

    @property
    def app_runs(self) -> Path:
        return self.app_outputs / "runs"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def data(self) -> Path:
        return self.root / "Data"

    @property
    def registry(self) -> Path:
        return self.config / "demo_registry.json"

    @property
    def ui_config(self) -> Path:
        return self.config / "ui_config.json"

    @property
    def runtime_config(self) -> Path:
        return self.config / "runtime_config.json"

    @property
    def dt_cache(self) -> Path:
        return self.app_outputs / "dt_realtime_cache.json"

    @property
    def dt_html(self) -> Path:
        return self.app_outputs / "dt_realtime_3d.html"


PATHS = ProjectPaths()
