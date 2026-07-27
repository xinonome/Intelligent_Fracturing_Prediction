"""Compact PySide dashboard for the three contract workstreams."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
OUTPUTS = ROOT / "outputs" / "app"


def load_json(path: Path) -> dict:
    if not path.exists():
        return {"status": "not available", "path": str(path)}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "invalid JSON", "path": str(path), "error": str(exc)}


def project_status() -> dict:
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "fsl": {
            "transfer": load_json(ARTIFACTS / "fsl" / "transfer_learning" / "transfer_metrics.json"),
            "two_stage": load_json(ARTIFACTS / "fsl" / "two_stage_model" / "summary.json"),
        },
        "dt": load_json(ARTIFACTS / "dt" / "direct_observation_enkf" / "summary.json"),
        "hmi": load_json(ARTIFACTS / "hmi" / "ppo_policy" / "summary.json"),
    }


def write_summary() -> Path:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS / "app_summary.json"
    path.write_text(json.dumps(project_status(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def launch_command(module: str, action: str) -> None:
    subprocess.Popen([sys.executable, str(ROOT / "run_project.py"), module, action], cwd=ROOT)


def run_gui() -> int:
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import (
            QApplication,
            QHBoxLayout,
            QLabel,
            QMainWindow,
            QPushButton,
            QTabWidget,
            QTextEdit,
            QVBoxLayout,
            QWidget,
        )
    except Exception as exc:
        print(
            "PySide6/Qt 无法加载，GUI 未启动。请安装 requirements-ui.txt，"
            f"或使用 --no-gui。\n原因：{exc}",
            file=sys.stderr,
        )
        return 2

    status = project_status()
    app = QApplication(sys.argv)
    window = QMainWindow()
    window.setWindowTitle("智能压裂预测合同主线演示")
    window.resize(1200, 760)
    tabs = QTabWidget()

    pages = [
        ("第一部分：小样本与知识图谱", "fsl", "knowledge-graph", status["fsl"]),
        ("第二部分：PKN-EnKF数字孪生", "dt", "visualize", status["dt"]),
        ("第三部分：强化学习工况仿真", "hmi", "scenarios", status["hmi"]),
    ]
    for title, module, action, payload in pages:
        page = QWidget()
        layout = QVBoxLayout(page)
        heading = QLabel(title)
        heading.setStyleSheet("font-size: 22px; font-weight: 700; padding: 12px;")
        layout.addWidget(heading)
        text = QTextEdit()
        text.setReadOnly(True)
        text.setPlainText(json.dumps(payload, ensure_ascii=False, indent=2))
        layout.addWidget(text, 1)
        row = QHBoxLayout()
        button = QPushButton("运行当前主线")
        button.clicked.connect(lambda _checked=False, m=module, a=action: launch_command(m, a))
        row.addWidget(button)
        row.addStretch(1)
        layout.addLayout(row)
        tabs.addTab(page, title.split("：", 1)[0])

    footer = QLabel("当前页面只展示合同三部分正式主线；运行结果统一写入 outputs。")
    footer.setAlignment(Qt.AlignCenter)
    footer.setStyleSheet("padding: 8px; color: #506070;")
    central = QWidget()
    central_layout = QVBoxLayout(central)
    central_layout.addWidget(tabs, 1)
    central_layout.addWidget(footer)
    window.setCentralWidget(central)
    window.show()
    return app.exec()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compact intelligent fracturing dashboard")
    parser.add_argument("--no-gui", action="store_true")
    args = parser.parse_args()
    summary = write_summary()
    print(json.dumps({"app_summary": str(summary)}, ensure_ascii=False, indent=2))
    if not args.no_gui:
        raise SystemExit(run_gui())


if __name__ == "__main__":
    main()
