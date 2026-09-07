"""Thin QProcess wrapper used for non-blocking light validation."""

from __future__ import annotations

from pathlib import Path


class TaskRunner:
    def __init__(self, parent=None) -> None:
        from PySide6.QtCore import QProcess

        self.process = QProcess(parent)

    def start(
        self,
        program: str,
        arguments: list[str],
        cwd: Path,
        environment: dict[str, str] | None = None,
    ) -> None:
        from PySide6.QtCore import QProcess

        if environment:
            from PySide6.QtCore import QProcessEnvironment

            process_environment = QProcessEnvironment.systemEnvironment()
            for key, value in environment.items():
                if key == "PYTHONPATH" and process_environment.value(key):
                    value = value + ";" + process_environment.value(key)
                process_environment.insert(key, value)
            self.process.setProcessEnvironment(process_environment)
        # Surface import and data errors in the same stream consumed by the
        # page.  Previously a failed child process wrote only to stderr and
        # looked like a button that did nothing.
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.setProgram(program)
        self.process.setArguments(arguments)
        self.process.setWorkingDirectory(str(cwd))
        self.process.start()

    def cancel(self) -> None:
        from PySide6.QtCore import QProcess

        if self.process.state() != QProcess.NotRunning:
            self.process.kill()
