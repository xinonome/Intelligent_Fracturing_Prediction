"""Acceptance dashboard for the three intelligent-fracturing workstreams."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import webbrowser
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
OUTPUTS = ROOT / "outputs" / "app"
BASE_PYTHON = Path(os.environ.get("FRACTURING_ALGORITHM_PYTHON", r"C:\Users\xinonome\anaconda3\python.exe"))
QT_PYTHON = Path(r"C:\Users\xinonome\anaconda3\envs\frac_app\python.exe")


def load_json(path: Path) -> dict:
    if not path.exists():
        return {"status": "not_available", "path": str(path)}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "invalid_json", "path": str(path), "error": str(exc)}


def nested(data: dict, *keys, default=None):
    value = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def project_status() -> dict:
    transfer = load_json(ARTIFACTS / "fsl" / "transfer_learning" / "transfer_metrics.json")
    two_stage = load_json(ARTIFACTS / "fsl" / "two_stage_model" / "summary.json")
    dt = load_json(ARTIFACTS / "dt" / "direct_observation_enkf" / "summary.json")
    hmi = load_json(ARTIFACTS / "hmi" / "ppo_policy" / "summary.json")
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "environment": {
            "ui_python": sys.executable,
            "algorithm_python": str(BASE_PYTHON),
            "project_root": str(ROOT),
        },
        "fsl": {"transfer": transfer, "two_stage": two_stage},
        "dt": dt,
        "hmi": hmi,
    }


def write_summary() -> Path:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS / "app_summary.json"
    path.write_text(json.dumps(project_status(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def qt_import_error() -> str | None:
    try:
        from PySide6.QtCore import qVersion

        qVersion()
        return None
    except Exception as exc:
        return str(exc)


def relaunch_with_qt_env(arguments: list[str]) -> int | None:
    if os.environ.get("FRACTURING_APP_RELAUNCHED") == "1" or not QT_PYTHON.exists():
        return None
    probe = subprocess.run(
        [str(QT_PYTHON), "-c", "from PySide6.QtCore import qVersion; print(qVersion())"],
        capture_output=True,
        text=True,
    )
    if probe.returncode:
        return None
    env = os.environ.copy()
    env["FRACTURING_APP_RELAUNCHED"] = "1"
    env["HOME"] = env.get("USERPROFILE", str(Path.home()))
    return subprocess.call([str(QT_PYTHON), str(Path(__file__).resolve()), *arguments], cwd=ROOT, env=env)


def run_gui() -> int:
    from PySide6.QtCore import QProcess, Qt, Signal
    from PySide6.QtGui import QColor, QFont, QPixmap
    from PySide6.QtWidgets import (
        QApplication,
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QSplitter,
        QStackedWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )

    status = project_status()

    class MetricCard(QFrame):
        def __init__(self, title: str, value: str, caption: str, accent: str):
            super().__init__()
            self.setObjectName("metricCard")
            layout = QVBoxLayout(self)
            layout.setContentsMargins(18, 14, 18, 14)
            title_label = QLabel(title)
            title_label.setObjectName("cardTitle")
            value_label = QLabel(value)
            value_label.setStyleSheet(f"font-size: 28px; font-weight: 800; color: {accent};")
            caption_label = QLabel(caption)
            caption_label.setWordWrap(True)
            caption_label.setObjectName("muted")
            layout.addWidget(title_label)
            layout.addWidget(value_label)
            layout.addWidget(caption_label)

    class ImagePanel(QFrame):
        def __init__(self, title: str, path: Path):
            super().__init__()
            self.path = path
            layout = QVBoxLayout(self)
            heading = QLabel(title)
            heading.setObjectName("sectionTitle")
            self.image = QLabel()
            self.image.setAlignment(Qt.AlignCenter)
            self.image.setMinimumHeight(280)
            self.image.setStyleSheet("background:#f7f9fb; border:1px solid #dbe2ea;")
            layout.addWidget(heading)
            layout.addWidget(self.image, 1)
            self.refresh()

        def refresh(self):
            if not self.path.exists():
                self.image.setText(f"暂无图表\n{self.path}")
                return
            pixmap = QPixmap(str(self.path))
            self.image.setPixmap(pixmap.scaled(900, 520, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    class Dashboard(QMainWindow):
        log_line = Signal(str)

        def __init__(self):
            super().__init__()
            self.process: QProcess | None = None
            self.setWindowTitle("智能压裂预测联合验收系统")
            self.resize(1500, 920)
            self.setMinimumSize(1100, 720)
            self.pages = QStackedWidget()
            self.navigation = QListWidget()
            self.navigation.setFixedWidth(230)
            self.navigation.setSpacing(4)
            self.log = QTextEdit()
            self.log.setReadOnly(True)
            self.log.setMinimumHeight(150)
            self.log_line.connect(self.log.append)
            self._build_ui()
            self._apply_style()

        def _build_ui(self):
            root = QWidget()
            root_layout = QVBoxLayout(root)
            root_layout.setContentsMargins(0, 0, 0, 0)

            header = QFrame()
            header.setObjectName("header")
            h = QHBoxLayout(header)
            title = QLabel("智能压裂预测联合验收系统")
            title.setStyleSheet("font-size:24px;font-weight:800;color:white;")
            subtitle = QLabel("知识图谱与小样本学习  |  裂缝数字孪生  |  知识嵌入智能体")
            subtitle.setStyleSheet("color:#dce9f5;font-size:14px;")
            h.addWidget(title)
            h.addSpacing(24)
            h.addWidget(subtitle)
            h.addStretch(1)
            badge = QLabel("验收演示模式")
            badge.setStyleSheet("background:#f04b3f;color:white;padding:7px 12px;border-radius:5px;font-weight:700;")
            h.addWidget(badge)
            root_layout.addWidget(header)

            body = QSplitter()
            body.addWidget(self.navigation)
            body.addWidget(self.pages)
            body.setStretchFactor(1, 1)
            root_layout.addWidget(body, 1)
            self.setCentralWidget(root)

            page_specs = [
                ("项目总览", self._overview_page()),
                ("第一部分  小样本学习", self._fsl_page()),
                ("第二部分  数字孪生", self._dt_page()),
                ("第三部分  智能决策", self._hmi_page()),
                ("联合演示", self._integration_page()),
                ("运行日志", self._log_page()),
                ("验收与边界", self._acceptance_page()),
            ]
            for text, page in page_specs:
                self.navigation.addItem(QListWidgetItem(text))
                self.pages.addWidget(page)
            self.navigation.currentRowChanged.connect(self.pages.setCurrentIndex)
            self.navigation.setCurrentRow(0)

        def _scroll_page(self, title: str, subtitle: str) -> tuple[QScrollArea, QVBoxLayout]:
            container = QWidget()
            layout = QVBoxLayout(container)
            layout.setContentsMargins(28, 24, 28, 28)
            heading = QLabel(title)
            heading.setObjectName("pageTitle")
            desc = QLabel(subtitle)
            desc.setObjectName("muted")
            desc.setWordWrap(True)
            layout.addWidget(heading)
            layout.addWidget(desc)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.NoFrame)
            scroll.setWidget(container)
            return scroll, layout

        def _overview_page(self):
            page, layout = self._scroll_page("项目总览", "按照合同三条主线组织成果、指标、演示入口和技术边界。")
            cards = QGridLayout()
            fsl_f1 = nested(status, "fsl", "two_stage", "window_results", "4", "test_two_stage_grouped_macro_f1", default=0)
            dt_pass = nested(status, "dt", "metrics", "validation_all_observations_within_15_percent_rate", default=0)
            dt_p95 = nested(status, "dt", "metrics", "all_steps_compute_p95_ms", default=0)
            hmi_safe = nested(status, "hmi", "validation_180s", "rl_policy", "safe_within_180s_rate", default=None)
            cards.addWidget(MetricCard("两阶段工况识别", f"{fsl_f1:.3f}", "测试集五类Macro-F1；异常长尾仍是限制", "#d9363e"), 0, 0)
            cards.addWidget(MetricCard("数字孪生观测验证", f"{dt_pass*100:.0f}%", "留出阶段三类观测误差在15%以内", "#168aad"), 0, 1)
            cards.addWidget(MetricCard("闭环单步P95", f"{dt_p95:.1f} ms", "计算时间不含3D界面渲染", "#168aad"), 0, 2)
            hmi_text = "待正式训练" if hmi_safe is None else f"{hmi_safe*100:.1f}%"
            cards.addWidget(MetricCard("180秒安全验证", hmi_text, "当前代表性策略为阶段产物，砂堵场景仍需提升", "#d9921e"), 0, 3)
            layout.addLayout(cards)

            pipeline = QFrame()
            pipeline.setObjectName("panel")
            p = QHBoxLayout(pipeline)
            for index, (name, detail) in enumerate([
                ("1  感知", "施工秒点\n光纤分簇\n施工压力"),
                ("2  识别", "GNN/两阶段\n工况概率\n知识规则"),
                ("3  孪生", "PKN正演\nEnKF参数更新\n3D状态"),
                ("4  决策", "未来60秒动作\n风险与奖励\n人工确认"),
            ]):
                card = QLabel(f"<b>{name}</b><br><span style='color:#607080'>{detail}</span>")
                card.setAlignment(Qt.AlignCenter)
                card.setStyleSheet("background:white;border:1px solid #d8e0e8;padding:22px;border-radius:8px;")
                p.addWidget(card, 1)
                if index < 3:
                    arrow = QLabel("→")
                    arrow.setStyleSheet("font-size:26px;color:#9aa8b6;")
                    p.addWidget(arrow)
            layout.addWidget(pipeline)

            notes = QLabel("主演示顺序：知识图谱与工况识别结果 → PKN-EnKF参数反演与3D裂缝 → 智能体动作、奖励和180秒验证。训练任务不在现场重新运行。")
            notes.setWordWrap(True)
            notes.setObjectName("notice")
            layout.addWidget(notes)
            layout.addStretch(1)
            return page

        def _button_row(self, buttons: list[tuple[str, callable]]):
            row = QHBoxLayout()
            for text, callback in buttons:
                button = QPushButton(text)
                button.clicked.connect(callback)
                row.addWidget(button)
            row.addStretch(1)
            return row

        def _fsl_page(self):
            page, layout = self._scroll_page("第一部分：融合专家知识的小样本学习", "展示全书知识图谱、两阶段工况识别和迁移学习；不在验收现场重新训练模型。")
            metrics = nested(status, "fsl", "two_stage", "window_results", "4", default={}) or {}
            transfer = nested(status, "fsl", "transfer", "metrics", default={}) or {}
            cards = QGridLayout()
            cards.addWidget(MetricCard("正常/异常 Macro-F1", f"{metrics.get('test_binary_macro_f1', 0):.3f}", "两阶段第一层", "#d9363e"), 0, 0)
            cards.addWidget(MetricCard("五类 Accuracy", f"{metrics.get('test_two_stage_grouped_accuracy', 0)*100:.1f}%", "两阶段最终输出", "#d9363e"), 0, 1)
            before = nested(transfer, "transfer_query_before_finetune", "accuracy", default=0)
            after = nested(transfer, "transfer_query_after_finetune", "accuracy", default=0)
            cards.addWidget(MetricCard("迁移查询集 Accuracy", f"{before*100:.1f}% → {after*100:.1f}%", "支持集微调前后", "#d9363e"), 0, 2)
            layout.addLayout(cards)
            layout.addWidget(ImagePanel("两阶段五类测试集混淆矩阵", ARTIFACTS / "fsl" / "two_stage_model" / "window_4" / "test_grouped_two_stage_confusion.png"))
            layout.addLayout(self._button_row([
                ("打开全书知识图谱", self.open_knowledge_graph),
                ("打开FSL结果目录", lambda: self.open_path(ARTIFACTS / "fsl")),
            ]))
            return page

        def _dt_page(self):
            page, layout = self._scroll_page("第二部分：裂缝实时扩展数字孪生", "真实分簇液砂、施工压力和井轨迹进入coupled PKN；EnKF更新物理参数后重新正演。")
            m = nested(status, "dt", "metrics", default={}) or {}
            cards = QGridLayout()
            cards.addWidget(MetricCard("液量TVD", f"{m.get('validation_liquid_tvd_mean',0)*100:.2f}%", "留出阶段观测空间误差", "#168aad"), 0, 0)
            cards.addWidget(MetricCard("砂量TVD", f"{m.get('validation_sand_tvd_mean',0)*100:.2f}%", "留出阶段观测空间误差", "#168aad"), 0, 1)
            cards.addWidget(MetricCard("井底压力误差", f"{m.get('validation_bhp_relative_error_mean',0)*100:.2f}%", "留出阶段平均相对误差", "#168aad"), 0, 2)
            cards.addWidget(MetricCard("单步P95", f"{m.get('all_steps_compute_p95_ms',0):.1f} ms", "15秒指标通过", "#168aad"), 0, 3)
            layout.addLayout(cards)
            layout.addWidget(ImagePanel("PKN-EnKF直接观测空间验证", ARTIFACTS / "dt" / "direct_observation_enkf" / "direct_observation_validation.png"))
            layout.addLayout(self._button_row([
                ("运行轻量验证", lambda: self.start_job("dt", "validate")),
                ("生成并打开3D数字孪生", lambda: self.start_job("dt", "visualize", open_after=ROOT / "outputs" / "dt" / "digital_twin_3d.html")),
                ("打开DT结果目录", lambda: self.open_path(ARTIFACTS / "dt")),
            ]))
            return page

        def _hmi_page(self):
            page, layout = self._scroll_page("第三部分：知识嵌入智能决策", "状态为历史300秒压力/排量/砂比及孪生风险，动作为未来60秒目标排量和砂比。")
            h = status["hmi"]
            rl = h.get("rl_policy", {}) if isinstance(h, dict) else {}
            validation = nested(h, "validation_180s", "rl_policy", default={}) or {}
            cards = QGridLayout()
            cards.addWidget(MetricCard("算法", str(h.get('algorithm','PPO')), "Gymnasium连续动作环境", "#d9921e"), 0, 0)
            cards.addWidget(MetricCard("训练步数", str(h.get('total_timesteps','-')), "当前归档代表性策略", "#d9921e"), 0, 1)
            cards.addWidget(MetricCard("平均综合奖励", f"{rl.get('mean_integrated_reward',0):.2f}", "效果-压力-异常-成本", "#d9921e"), 0, 2)
            cards.addWidget(MetricCard("180秒安全率", f"{validation.get('safe_within_180s_rate',0)*100:.1f}%", "归档模型口径，非最终验收结论", "#d9921e"), 0, 3)
            layout.addLayout(cards)
            layout.addWidget(ImagePanel("强化学习策略与历史动作奖励对比", ARTIFACTS / "hmi" / "ppo_policy" / "rl_vs_historical_reward.png"))
            warning = QLabel("当前策略文件用于说明接口与验证流程。正式验收应替换为完成课程训练、风险加权回放及多随机种子门禁后的策略快照。")
            warning.setObjectName("warning")
            warning.setWordWrap(True)
            layout.addWidget(warning)
            layout.addLayout(self._button_row([
                ("运行场景轻量验证", lambda: self.start_job("hmi", "scenarios")),
                ("打开HMI结果目录", lambda: self.open_path(ARTIFACTS / "hmi")),
            ]))
            return page

        def _integration_page(self):
            page, layout = self._scroll_page("联合演示", "按验收节奏依次展示已训练成果；一键演示不会启动长时间训练。")
            steps = QFrame()
            steps.setObjectName("panel")
            step_layout = QVBoxLayout(steps)
            for title, detail in [
                ("步骤1  知识与识别", "打开全书知识图谱；展示两阶段混淆矩阵和迁移前后指标。"),
                ("步骤2  正演与反演", "运行第8段轻量验证；说明EnKF更新E′、CL、μ、σmin和分簇参数。"),
                ("步骤3  三维孪生", "生成WebGL页面；拖动井轨迹和多簇裂缝，切换时间步。"),
                ("步骤4  决策与安全", "展示未来60秒排量/砂比建议、奖励分解、人工确认和180秒验证。"),
            ]:
                label = QLabel(f"<b>{title}</b><br><span style='color:#5b6977'>{detail}</span>")
                label.setStyleSheet("background:white;border-bottom:1px solid #e4e9ee;padding:14px;")
                label.setWordWrap(True)
                step_layout.addWidget(label)
            layout.addWidget(steps)
            layout.addLayout(self._button_row([
                ("一键准备验收演示", self.prepare_demo),
                ("生成验收摘要", self.export_acceptance_summary),
                ("查看运行日志", lambda: self.navigation.setCurrentRow(5)),
            ]))
            layout.addStretch(1)
            return page

        def _log_page(self):
            page, layout = self._scroll_page("运行日志", "算法子进程的命令、标准输出和错误信息在此显示。")
            layout.addWidget(self.log, 1)
            layout.addLayout(self._button_row([
                ("停止当前任务", self.stop_job),
                ("清空日志", self.log.clear),
                ("打开outputs目录", lambda: self.open_path(ROOT / "outputs")),
            ]))
            return page

        def _acceptance_page(self):
            page, layout = self._scroll_page("验收指标与技术边界", "把已验证、阶段性结果和待补现场数据分开说明。")
            text = QTextEdit()
            text.setReadOnly(True)
            text.setHtml("""
            <h3>已形成的可验收能力</h3>
            <ul><li>三部分统一数据目录、CLI/API和展示入口。</li>
            <li>知识图谱、两阶段识别、迁移学习的代表性结果可追溯。</li>
            <li>PKN-EnKF留出阶段观测空间误差低于15%，单步计算低于15秒。</li>
            <li>强化学习环境、奖励、安全投影和180秒验证链路已形成。</li></ul>
            <h3>不得混淆的口径</h3>
            <ul><li>光纤液砂份额约束分簇响应，不等同于真实裂缝缝长标签。</li>
            <li>EnKF更新PKN物理参数，不直接把观测缝长写入结果。</li>
            <li>智能体当前为建议模式，任何高风险动作均需人工确认。</li>
            <li>砂堵和极少异常仍是数据与模型短板。</li></ul>
            <h3>正式验收前需冻结</h3>
            <ul><li>最终模型权重、配置、数据manifest和源码指纹。</li>
            <li>固定演示井段、固定随机种子和离线演示产物。</li>
            <li>准备断网、Qt异常和现场数据接口不可用的预案。</li></ul>
            """)
            layout.addWidget(text, 1)
            return page

        def _apply_style(self):
            self.setStyleSheet("""
                QMainWindow, QWidget { background:#f3f6f8; color:#17212b; font-family:'Microsoft YaHei'; font-size:14px; }
                #header { background:#13324b; padding:12px 20px; }
                QListWidget { background:#172b3a; color:#dbe6ee; border:0; padding:14px 8px; font-size:15px; }
                QListWidget::item { padding:14px 12px; border-radius:6px; }
                QListWidget::item:selected { background:#d9363e; color:white; font-weight:700; }
                #pageTitle { font-size:28px; font-weight:800; color:#17344b; }
                #sectionTitle { font-size:18px; font-weight:700; color:#17344b; padding:5px 0; }
                #muted { color:#667684; }
                #metricCard, #panel { background:white; border:1px solid #dce3e9; border-radius:9px; }
                #cardTitle { font-weight:700; color:#304657; }
                #notice { background:#e8f2f6; border-left:4px solid #168aad; padding:14px; }
                #warning { background:#fff3dd; border-left:4px solid #d9921e; padding:14px; }
                QPushButton { background:#d9363e; color:white; border:0; padding:10px 18px; border-radius:5px; font-weight:700; }
                QPushButton:hover { background:#b92830; }
                QTextEdit { background:white; border:1px solid #dce3e9; border-radius:6px; padding:8px; }
                QScrollArea { background:#f3f6f8; }
            """)

        def start_job(self, module: str, action: str, open_after: Path | None = None):
            if self.process and self.process.state() != QProcess.NotRunning:
                QMessageBox.warning(self, "任务正在运行", "请等待当前任务结束或先停止任务。")
                return
            python = BASE_PYTHON if BASE_PYTHON.exists() else Path(sys.executable)
            args = [str(ROOT / "run_project.py"), module, action]
            self.process = QProcess(self)
            self.process.setProgram(str(python))
            self.process.setArguments(args)
            self.process.setWorkingDirectory(str(ROOT))
            env = self.process.processEnvironment()
            env.insert("PYTHONUTF8", "1")
            env.insert("PYTHONIOENCODING", "utf-8")
            self.process.setProcessEnvironment(env)
            self.process.readyReadStandardOutput.connect(lambda: self._read_process(False))
            self.process.readyReadStandardError.connect(lambda: self._read_process(True))
            self.process.finished.connect(lambda code, _status: self._job_finished(code, open_after))
            self.log_line.emit(f"<b>启动：</b>{python} {' '.join(args)}")
            self.navigation.setCurrentRow(5)
            self.process.start()

        def _read_process(self, error: bool):
            if not self.process:
                return
            raw = self.process.readAllStandardError() if error else self.process.readAllStandardOutput()
            text = bytes(raw).decode("utf-8", errors="replace").rstrip()
            if text:
                color = "#b92830" if error else "#253745"
                self.log_line.emit(f"<span style='color:{color};white-space:pre'>{text}</span>")

        def _job_finished(self, code: int, open_after: Path | None):
            self.log_line.emit(f"<b>任务结束，退出码：{code}</b>")
            if code == 0 and open_after and open_after.exists():
                webbrowser.open(open_after.resolve().as_uri())
            write_summary()

        def stop_job(self):
            if self.process and self.process.state() != QProcess.NotRunning:
                self.process.kill()
                self.log_line.emit("任务已由用户停止。")

        def open_path(self, path: Path):
            path.mkdir(parents=True, exist_ok=True) if path.suffix == "" else None
            os.startfile(str(path))

        def open_knowledge_graph(self):
            path = ROOT / "FSL-Expert" / "knowledge_graph" / "full_book_qwen_output" / "index_full_book_qwen.html"
            if path.exists():
                webbrowser.open(path.resolve().as_uri())
            else:
                QMessageBox.warning(self, "文件不存在", str(path))

        def prepare_demo(self):
            write_summary()
            self.open_knowledge_graph()
            dt_html = ROOT / "outputs" / "dt" / "digital_twin_3d.html"
            if dt_html.exists():
                webbrowser.open(dt_html.resolve().as_uri())
            else:
                self.start_job("dt", "visualize", open_after=dt_html)
            self.log_line.emit("验收摘要已刷新；知识图谱已打开；正在准备数字孪生页面。")

        def export_acceptance_summary(self):
            path = write_summary()
            QMessageBox.information(self, "已生成", f"验收摘要：\n{path}")

    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 10))
    window = Dashboard()
    window.show()
    return app.exec()


def main() -> None:
    parser = argparse.ArgumentParser(description="Intelligent fracturing acceptance dashboard")
    parser.add_argument("--no-gui", action="store_true", help="Only write the integrated summary")
    parser.add_argument("--preflight", action="store_true", help="Check Qt and required artifacts")
    parser.add_argument("--no-auto-env", action="store_true", help="Do not relaunch in frac_app")
    args = parser.parse_args()
    summary = write_summary()
    qt_error = qt_import_error()
    checks = {
        "summary": str(summary),
        "qt_ok": qt_error is None,
        "qt_error": qt_error,
        "qt_python": str(QT_PYTHON),
        "algorithm_python": str(BASE_PYTHON),
        "artifacts": {
            "fsl": (ARTIFACTS / "fsl").exists(),
            "dt": (ARTIFACTS / "dt").exists(),
            "hmi": (ARTIFACTS / "hmi").exists(),
        },
    }
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    if args.no_gui or args.preflight:
        return
    if qt_error:
        if not args.no_auto_env:
            code = relaunch_with_qt_env(["--no-auto-env"])
            if code is not None:
                raise SystemExit(code)
        print(
            "PySide6 QtCore无法加载。请使用 App\\launch_app.ps1 启动。\n"
            f"原因：{qt_error}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    raise SystemExit(run_gui())


if __name__ == "__main__":
    main()
