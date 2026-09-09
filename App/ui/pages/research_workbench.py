"""Operational model-development workbenches for the integrated desktop APP."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shlex
import sys

from ...core.paths import PATHS
from ...services.ml_runtime import resolve_ml_python
from ...services.task_runner import TaskRunner
from ..widgets.status_card import Panel
from ..widgets.pyfrac_workbench import create_pyfrac_workbench


@dataclass(frozen=True)
class TaskSpec:
    group: str
    name: str
    module: str
    action: str
    output: Path
    needs_ml: bool = True
    note: str = ""


_TASKS = {
    "fsl": (
        TaskSpec("数据与规则", "分层异常模型训练", "fsl", "train", PATHS.outputs / "fsl" / "train"),
        TaskSpec("模型训练", "图神经网络训练", "fsl", "gnn", PATHS.outputs / "fsl" / "gnn"),
        TaskSpec("模型训练", "工况直接分类训练", "fsl", "direct", PATHS.outputs / "fsl" / "direct"),
        TaskSpec("模型训练", "工况转移模型训练", "fsl", "transition", PATHS.outputs / "fsl" / "transition"),
        TaskSpec("风险训练", "砂堵风险概率训练", "fsl", "risk", PATHS.outputs / "fsl" / "risk", note="需要原始施工表与对应增强标注表"),
        TaskSpec("风险训练", "未来绿黄红风险等级训练", "fsl", "risk-levels", PATHS.outputs / "fsl" / "risk_levels", note="附加参数需指定 --enhanced-file 与 --reference-risk-run"),
        TaskSpec("风险训练", "风险边界校准", "fsl", "risk-boundaries", PATHS.outputs / "fsl" / "risk_boundaries", note="附加参数需指定 --prediction-run"),
        TaskSpec("风险训练", "同井参数与风险联合评估", "fsl", "risk-evaluate", PATHS.outputs / "fsl" / "risk_evaluation", note="附加参数需指定 --enhanced-file 与 --risk-run"),
        TaskSpec("风险训练", "Z6HF→Z7HF风险等级迁移", "fsl", "risk-transfer", PATHS.outputs / "fsl" / "risk_transfer", note="使用已登记Z6HF权重和目标井支持集"),
        TaskSpec("迁移训练", "跨井迁移训练", "fsl", "transfer", PATHS.outputs / "fsl" / "transfer"),
    ),
    "dt": (
        TaskSpec("观测融合", "KG-EnKF观测同化验证", "dt", "validate", PATHS.outputs / "dt"),
        TaskSpec("正演对比", "PKN / PyFrac正演对比", "dt", "benchmark", PATHS.outputs / "dt"),
        TaskSpec("裂缝推演", "PyFrac内生推演与三维可视化", "dt", "visualize", PATHS.outputs / "dt" / "digital_twin_3d.html"),
        TaskSpec("液量调控", "Piggy-Bank阶段液量计算", "dt", "piggy-bank", PATHS.outputs / "dt" / "piggy_bank_open_loop"),
    ),
    "hmi": (
        TaskSpec("响应代理", "训练响应代理模型", "hmi", "train-surrogate", PATHS.outputs / "hmi" / "response_surrogate"),
        TaskSpec("策略训练", "训练分层智能体", "hmi", "train", PATHS.outputs / "hmi" / "training"),
        TaskSpec("策略训练", "运行完整训练流程", "hmi", "full-train", PATHS.outputs / "hmi"),
        TaskSpec("策略优化", "智能体参数优化", "hmi", "optimize", PATHS.outputs / "hmi"),
        TaskSpec("策略优化", "课程学习训练", "hmi", "curriculum", PATHS.outputs / "hmi"),
        TaskSpec("安全验证", "仿真环境验证", "hmi", "validate-env", PATHS.outputs / "hmi"),
        TaskSpec("安全验证", "多场景离线验证", "hmi", "scenarios", PATHS.outputs / "hmi" / "scenarios"),
        TaskSpec("安全验证", "综合离线检查", "hmi", "acceptance", PATHS.outputs / "hmi"),
    ),
}


_TITLES = {
    "fsl": "工况与风险",
    "dt": "裂缝与参数",
    "hmi": "智能调控",
}


_DT_FIXED_INPUTS = {
    "validate": (
        ("光纤监测", PATHS.root / "Data" / "3Dfrac" / "光纤本井监测08.txt"),
        ("施工压力", PATHS.root / "Data" / "3Dfrac" / "JY84-Z1-stage08-f1.xls"),
    ),
    "benchmark": (
        ("光纤监测", PATHS.root / "Data" / "3Dfrac" / "光纤本井监测08.txt"),
        ("井轨迹", PATHS.root / "Data" / "3Dfrac" / "JY84-Z1HF-1011.csv"),
        ("施工压力", PATHS.root / "Data" / "3Dfrac" / "JY84-Z1-stage08-f1.xls"),
    ),
    "visualize": (
        ("光纤监测", PATHS.root / "Data" / "3Dfrac" / "光纤本井监测08.txt"),
        ("井轨迹", PATHS.root / "Data" / "3Dfrac" / "JY84-Z1HF-1011.csv"),
        ("施工压力", PATHS.root / "Data" / "3Dfrac" / "JY84-Z1-stage08-f1.xls"),
    ),
    "piggy-bank": (
        ("光纤监测", PATHS.root / "Data" / "3Dfrac" / "光纤本井监测08.txt"),
        ("井轨迹", PATHS.root / "Data" / "3Dfrac" / "JY84-Z1HF-1011.csv"),
        (
            "分簇历史",
            PATHS.root
            / "outputs"
            / "dt"
            / "second_part_kg_enkf_20260824"
            / "20260824_004357"
            / "cluster_share_history.csv",
        ),
    ),
}


def _relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PATHS.root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _actual_input_text(area: str, task: TaskSpec | None) -> str:
    if task is None:
        return "尚未选择任务"
    if area == "dt":
        lines = ["当前任务输入：JY84-Z1 · Stage 08（脚本固定输入）"]
        for label, path in _DT_FIXED_INPUTS.get(task.action, ()):
            status = "已找到" if path.is_file() else "缺失"
            lines.append(f"{label}：{_relative_path(path)}　[{status}]")
        return "\n".join(lines)
    if area in {"fsl", "hmi"}:
        base = (
            "实际输入范围：Data/raw_frac/（按脚本参数读取独立井段数据）\n"
            f"当前任务输出：{_relative_path(task.output)}"
        )
        return base + (f"\n运行条件：{task.note}" if task.note else "")
    return "实际输入：由任务脚本参数确定"


def build_research_workbench(area: str, registry, controller):
    """Build a real QProcess-backed training/validation launcher."""

    from PySide6.QtCore import QProcess, QUrl, Qt
    from PySide6.QtGui import QDesktopServices, QTextCursor
    from PySide6.QtWidgets import (
        QComboBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QPlainTextEdit,
        QPushButton,
        QScrollArea,
        QSpinBox,
        QTabWidget,
        QVBoxLayout,
        QWidget,
    )

    tasks = tuple(_TASKS.get(area, ()))
    page = QScrollArea()
    page.setObjectName(f"{area}ResearchScroll")
    page.setWidgetResizable(True)
    content = QWidget()
    page.setWidget(content)
    layout = QVBoxLayout(content)
    layout.setContentsMargins(20, 16, 20, 18)
    layout.setSpacing(10)

    title = QLabel(_TITLES.get(area, "模型研发"))
    title.setObjectName("pageTitle")
    layout.addWidget(title)

    context = QLabel("")
    context.setObjectName("notice")
    context.setWordWrap(True)
    layout.addWidget(context)

    tabs = QTabWidget()
    tabs.setObjectName(f"{area}ResearchTabs")
    layout.addWidget(tabs)
    selected = {"task": tasks[0] if tasks else None}
    select_buttons = []
    selected_button = {"widget": None}

    for group in dict.fromkeys(task.group for task in tasks):
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        tab_layout.setContentsMargins(10, 10, 10, 10)
        tab_layout.setSpacing(8)
        for task in (item for item in tasks if item.group == group):
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            name = QLabel(task.name)
            name.setObjectName("value")
            command = QLabel(f"{task.module}  /  {task.action}")
            command.setObjectName("muted")
            choose = QPushButton("选择任务")
            choose.setProperty("taskAction", task.action)
            row_layout.addWidget(name, 2)
            row_layout.addWidget(command, 1)
            row_layout.addWidget(choose)
            tab_layout.addWidget(row)
            select_buttons.append((choose, task))
        tab_layout.addStretch(1)
        tabs.addTab(tab, group)

    runner_panel, runner_layout = Panel.create("任务运行")
    selected_label = QLabel("尚未选择任务")
    selected_label.setObjectName("sectionTitle")
    runner_layout.addWidget(selected_label)
    input_files = QLabel("")
    input_files.setObjectName("researchInputFiles")
    input_files.setWordWrap(True)
    input_files.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
    runner_layout.addWidget(input_files)
    parameter_row = QHBoxLayout()
    parameter_row.setSpacing(8)
    algorithm_label = QLabel("算法")
    algorithm = QComboBox()
    algorithm.setObjectName("researchAlgorithm")
    primary_label = QLabel("训练步数")
    primary_value = QSpinBox()
    primary_value.setObjectName("researchPrimaryParameter")
    primary_value.setRange(1, 10_000_000)
    primary_value.setSingleStep(1000)
    secondary_label = QLabel("微调轮次")
    secondary_value = QSpinBox()
    secondary_value.setObjectName("researchSecondaryParameter")
    secondary_value.setRange(1, 100_000)
    seed_label = QLabel("随机种子")
    seed_value = QSpinBox()
    seed_value.setObjectName("researchSeed")
    seed_value.setRange(0, 2_147_483_647)
    for widget in (
        algorithm_label,
        algorithm,
        primary_label,
        primary_value,
        secondary_label,
        secondary_value,
        seed_label,
        seed_value,
    ):
        parameter_row.addWidget(widget)
    parameter_row.addStretch(1)
    runner_layout.addLayout(parameter_row)
    extra_row = QHBoxLayout()
    extra_row.setSpacing(8)
    extra_row.addWidget(QLabel("附加参数"))
    extra_arguments = QLineEdit()
    extra_arguments.setObjectName("researchExtraArguments")
    extra_arguments.setPlaceholderText("可选，例如：--enhanced-file 文件名.xlsx --risk-run outputs/fsl/risk/运行目录")
    extra_arguments.setToolTip("参数直接传给所选 Python 脚本；路径含空格时请使用双引号")
    extra_row.addWidget(extra_arguments, 1)
    runner_layout.addLayout(extra_row)
    parameter_spec: dict[str, object] = {}
    actions = QHBoxLayout()
    start = QPushButton("开始运行")
    start.setObjectName("researchRunButton")
    stop = QPushButton("停止任务")
    stop.setEnabled(False)
    open_output = QPushButton("打开输出目录")
    for button in (start, stop, open_output):
        actions.addWidget(button)
    actions.addStretch(1)
    runner_layout.addLayout(actions)
    task_status = QLabel("未启动")
    task_status.setObjectName("muted")
    task_status.setProperty("role", "researchTaskStatus")
    task_status.setWordWrap(True)
    runner_layout.addWidget(task_status)
    log = QPlainTextEdit()
    log.setObjectName("researchTaskLog")
    log.setReadOnly(True)
    log.setMaximumBlockCount(4000)
    log.setMinimumHeight(260)
    runner_layout.addWidget(log)
    layout.addWidget(runner_panel, 1)

    # The old visualize entry only launched the Plotly HTML generator and
    # exposed its console log. Make the native PyFrac workbench the primary
    # surface for this task so accepted internal points can be inspected and
    # replayed without inventing intermediate states.
    native_workbench = create_pyfrac_workbench(content, registry.dataset())
    native_workbench.setVisible(False)
    layout.addWidget(native_workbench, 2)

    runner = TaskRunner(page)

    def refresh_context(dataset_id: str | None = None):
        dataset_id = str(dataset_id or getattr(registry, "dataset_id", "") or "")
        dataset = registry.dataset(dataset_id) if dataset_id else {}
        identity = dataset.get("display_name") or dataset.get("stage_id") or dataset_id or "未选择井段"
        if area == "dt":
            context.setText(
                "当前任务输入：JY84-Z1 · Stage 08　·　正演/同化脚本使用固定登记文件"
            )
        else:
            scope = "训练脚本按已登记数据目录运行。"
            context.setText(f"当前核对井段：{identity}　·　{scope}")
        if area == "dt":
            native_workbench.set_dataset(dataset)

    def set_task_surface(task: TaskSpec | None):
        native_visible = bool(area == "dt" and task is not None and task.action == "visualize")
        runner_panel.setVisible(not native_visible)
        native_workbench.setVisible(native_visible)
        if native_visible:
            native_workbench.set_dataset(registry.dataset())

    def configure_parameters(task: TaskSpec | None):
        nonlocal parameter_spec
        parameter_spec = {}
        for widget in (
            algorithm_label,
            algorithm,
            primary_label,
            primary_value,
            secondary_label,
            secondary_value,
            seed_label,
            seed_value,
        ):
            widget.setVisible(False)
        if task is None:
            return

        if task.module == "fsl":
            seed_label.setVisible(True)
            seed_value.setVisible(True)
            seed_value.setValue(42)
            parameter_spec["seed"] = "--seed"
            if task.action in {"train", "direct", "transition"}:
                primary_label.setText("树数量")
                primary_value.setRange(10, 5000)
                primary_value.setSingleStep(50)
                primary_value.setValue(300)
                parameter_spec["primary"] = "--n-estimators"
            elif task.action == "gnn":
                primary_label.setText("训练轮次")
                primary_value.setRange(1, 10_000)
                primary_value.setSingleStep(10)
                primary_value.setValue(50)
                parameter_spec["primary"] = "--epochs"
            elif task.action == "risk":
                primary_label.setText("训练轮次")
                primary_value.setRange(1, 1000)
                primary_value.setSingleStep(2)
                primary_value.setValue(12)
                parameter_spec["primary"] = "--epochs"
            elif task.action == "risk-transfer":
                primary_label.setText("目标井微调轮次")
                primary_value.setRange(1, 1000)
                primary_value.setSingleStep(1)
                primary_value.setValue(5)
                parameter_spec["primary"] = "--finetune-epochs"
            elif task.action == "transfer":
                primary_label.setText("基础训练轮次")
                primary_value.setRange(1, 10_000)
                primary_value.setSingleStep(5)
                primary_value.setValue(20)
                secondary_label.setText("目标井微调轮次")
                secondary_value.setValue(10)
                secondary_label.setVisible(True)
                secondary_value.setVisible(True)
                parameter_spec["primary"] = "--pretrain-epochs"
                parameter_spec["secondary"] = "--finetune-epochs"
            primary_label.setVisible("primary" in parameter_spec)
            primary_value.setVisible("primary" in parameter_spec)
            return

        if task.module != "hmi":
            return
        if task.action in {"train", "full-train", "optimize", "curriculum", "scenarios"}:
            choices = ["PPO", "SAC"] if task.action in {"full-train", "scenarios"} else ["PPO", "SAC", "TD3"]
            algorithm.clear()
            algorithm.addItems(choices)
            algorithm_label.setVisible(True)
            algorithm.setVisible(True)
            parameter_spec["algorithm"] = "--algorithm"
        if task.action in {"train", "full-train", "optimize", "scenarios"}:
            primary_label.setText("训练步数")
            primary_value.setRange(100, 10_000_000)
            primary_value.setSingleStep(5000)
            primary_value.setValue(100_000 if task.action != "scenarios" else 5000)
            primary_label.setVisible(True)
            primary_value.setVisible(True)
            parameter_spec["primary"] = "--total-timesteps"
        elif task.action == "curriculum":
            primary_label.setText("每阶段步数")
            primary_value.setRange(100, 10_000_000)
            primary_value.setSingleStep(5000)
            primary_value.setValue(20_000)
            primary_label.setVisible(True)
            primary_value.setVisible(True)
            parameter_spec["primary"] = "--stage-timesteps"
        elif task.action == "train-surrogate":
            primary_label.setText("最大样本数")
            primary_value.setRange(100, 5_000_000)
            primary_value.setSingleStep(5000)
            primary_value.setValue(50_000)
            primary_label.setVisible(True)
            primary_value.setVisible(True)
            parameter_spec["primary"] = "--max-samples"
        elif task.action == "validate-env":
            primary_label.setText("验证步数")
            primary_value.setRange(1, 100_000)
            primary_value.setSingleStep(5)
            primary_value.setValue(6)
            primary_label.setVisible(True)
            primary_value.setVisible(True)
            parameter_spec["primary"] = "--steps"
        if task.action in {"train", "full-train", "optimize", "curriculum", "train-surrogate", "validate-env"}:
            seed_label.setVisible(True)
            seed_value.setVisible(True)
            seed_value.setValue(2026)
            parameter_spec["seed"] = "--seeds" if task.action in {"full-train", "optimize"} else "--seed"

    def choose_task(task: TaskSpec, button=None, *, reveal: bool = True):
        previous = selected_button["widget"]
        if previous is not None and previous is not button:
            previous.setText("选择任务")
        if button is not None:
            button.setText("已选择")
            selected_button["widget"] = button
        selected["task"] = task
        selected_label.setText(task.name)
        input_files.setText(_actual_input_text(area, task))
        task_status.setText(f"准备运行：{task.module} {task.action}")
        configure_parameters(task)
        set_task_surface(task)
        if reveal:
            # The task runner is below the task tabs.  Selecting a task should
            # produce visible feedback and take the operator to the next step
            # instead of silently changing controls below the fold.
            target_panel = runner_panel if runner_panel.isVisible() else native_workbench
            page.ensureWidgetVisible(target_panel)

    def choose_group(index: int):
        group = tabs.tabText(index)
        task = next((item for item in tasks if item.group == group), None)
        if task is not None:
            button = next((widget for widget, item in select_buttons if item == task), None)
            choose_task(task, button)

    for button, task in select_buttons:
        button.clicked.connect(lambda _checked=False, item=task, widget=button: choose_task(item, widget))
    tabs.currentChanged.connect(choose_group)

    def selected_parameters() -> list[str]:
        values: list[str] = []
        if "algorithm" in parameter_spec:
            values.extend([str(parameter_spec["algorithm"]), algorithm.currentText().lower()])
        if "primary" in parameter_spec:
            values.extend([str(parameter_spec["primary"]), str(primary_value.value())])
        if "secondary" in parameter_spec:
            values.extend([str(parameter_spec["secondary"]), str(secondary_value.value())])
        if "seed" in parameter_spec:
            values.extend([str(parameter_spec["seed"]), str(seed_value.value())])
        if extra_arguments.text().strip():
            tokens = shlex.split(extra_arguments.text(), posix=False)
            values.extend(
                token[1:-1]
                if len(token) >= 2 and token[0] == token[-1] and token[0] in {'"', "'"}
                else token
                for token in tokens
            )
        return values

    def start_task():
        task = selected.get("task")
        if task is None or runner.process.state() != QProcess.NotRunning:
            return
        executable = str(sys.executable)
        runtime_note = f"使用当前运行环境：{executable}"
        if task.needs_ml:
            resolved, runtime_note = resolve_ml_python()
            if not resolved:
                task_status.setText(f"任务未启动：{runtime_note}")
                return
            executable = resolved
        try:
            parsed_args = selected_parameters()
        except ValueError as exc:
            task_status.setText(f"附加参数格式错误：{exc}")
            return
        arguments = [str(PATHS.root / "run_project.py"), task.module, task.action, *parsed_args]
        log.clear()
        log.appendPlainText(f"$ {executable} {' '.join(arguments)}")
        task_status.setText(f"正在运行：{task.name}　·　{runtime_note}")
        start.setEnabled(False)
        stop.setEnabled(True)
        runner.start(
            executable,
            arguments,
            PATHS.root,
            environment={"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "TF_CPP_MIN_LOG_LEVEL": "2"},
        )

    def read_output():
        output = bytes(runner.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if output:
            log.moveCursor(QTextCursor.End)
            log.insertPlainText(output)
            log.moveCursor(QTextCursor.End)

    def finished(exit_code, _status):
        read_output()
        task = selected.get("task")
        start.setEnabled(True)
        stop.setEnabled(False)
        if exit_code == 0:
            task_status.setText(f"运行完成：{task.name if task else ''}。请在输出目录核对本次结果。")
        else:
            task_status.setText(f"运行失败或已停止（返回码 {exit_code}）；日志保留在本页。")

    def open_result():
        task = selected.get("task")
        path = task.output if task else PATHS.app_outputs
        target = path if path.is_dir() else path.parent
        target.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

    def task_error(_error):
        start.setEnabled(True)
        stop.setEnabled(False)
        task_status.setText(f"任务进程未能启动：{runner.process.errorString()}")

    start.clicked.connect(start_task)
    stop.clicked.connect(runner.cancel)
    open_output.clicked.connect(open_result)
    runner.process.readyReadStandardOutput.connect(read_output)
    runner.process.finished.connect(finished)
    runner.process.errorOccurred.connect(task_error)

    if tasks:
        first_button = select_buttons[0][0] if select_buttons else None
        choose_task(tasks[0], first_button, reveal=False)
    refresh_context()

    def set_global_dataset(dataset_id):
        refresh_context(dataset_id)
        set_task_surface(selected.get("task"))

    page.set_global_dataset = set_global_dataset
    page.dataset_selection_locked = lambda: area == "dt"
    page.dataset_selection_message = lambda: (
        "当前模型研发任务使用 JY84-Z1 · Stage 08 的固定登记输入；如需支持其他井段，需先完成对应数据文件映射。"
        if area == "dt"
        else ""
    )
    page.pause_playback = lambda: None
    page.prepare_dataset_change = lambda: None
    page._research_runner = runner
    page._native_workbench = native_workbench
    return page


__all__ = ["build_research_workbench", "TaskSpec"]
