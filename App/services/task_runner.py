"""Thin QProcess wrapper used for non-blocking light validation."""

from __future__ import annotations

from pathlib import Path


class TaskRunner:
    def __init__(self, parent=None) -> None:
        from PySide6.QtCore import QProcess

        self.process = QProcess(parent)

    def start(self, program: str, arguments: list[str], cwd: Path) -> None:
        self.process.setProgram(program)
        self.process.setArguments(arguments)
        self.process.setWorkingDirectory(str(cwd))
        self.process.start()

    def cancel(self) -> None:
        from PySide6.QtCore import QProcess

        if self.process.state() != QProcess.NotRunning:
            self.process.kill()
