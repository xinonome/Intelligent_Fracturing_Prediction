"""Acceptance dashboard for the three intelligent-fracturing workstreams."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

try:
    from core.artifacts import ArtifactRegistry, create_app_run, relative_path, resolve_project_path
except ImportError:
    from App.core.artifacts import ArtifactRegistry, create_app_run, relative_path, resolve_project_path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
OUTPUTS = ROOT / "outputs" / "app"
BASE_PYTHON = Path(os.environ.get("FRACTURING_ALGORITHM_PYTHON", r"C:\Users\xinonome\anaconda3\python.exe"))
QT_PYTHON = Path(os.environ.get("FRACTURING_QT_PYTHON", r"C:\Users\xinonome\anaconda3\envs\frac_app\python.exe"))


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


def _float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _truthy(value: object) -> bool:
    """Parse CSV boolean fields without treating non-empty strings as True."""

    return str(value).strip().lower() in {"1", "true", "yes", "y", "是"}


def _build_replay_decision(hmi: dict[str, str], fallback: dict) -> dict:
    """Create a time-local HMI card from one RL evaluation window.

    ``human_machine_decisions.json`` contains aggregate decision cards for
    acceptance reporting.  It is not a frame-by-frame decision stream.  The
    replay therefore derives the visible card from the corresponding row in
    ``rl_evaluation.csv`` and keeps the aggregate card only as a fallback.
    """

    option = str(hmi.get("high_level_option", "")).strip().lower()
    abnormal = _float(hmi.get("abnormal_probability"), 0.0)
    sand_plug = _float(hmi.get("sand_plug_probability"), 0.0)
    posterior_error = _float(hmi.get("posterior_error"), 0.0)
    unsafe = _truthy(hmi.get("unsafe")) or _truthy(hmi.get("severe_pressure_violation"))
    uncertain = _truthy(hmi.get("uncertain")) or _truthy(hmi.get("pkn_update_skipped"))

    # The high-level policy mode is the primary decision.  The probabilities
    # and flags add the operational risk context shown beside that action.
    if option == "safe" or unsafe or abnormal >= 0.45 or sand_plug >= 0.25:
        risk = "high"
        main_risk = "压力或异常风险接近安全边界"
        recommendation = "降低排量和砂比，暂停激进调整并请求工程师确认。"
        requires_confirmation = True
    elif option == "divert":
        risk = "medium"
        main_risk = "分簇进液/携砂不均衡"
        recommendation = "限制砂比增幅，优先进行分簇均衡并观察关键簇响应。"
        requires_confirmation = True
    elif option == "hold" or uncertain or posterior_error > 0.30:
        risk = "medium"
        main_risk = "模型不确定性偏高"
        recommendation = "保持当前动作，等待下一观测更新后再决定是否调整。"
        requires_confirmation = True
    else:
        risk = "low"
        main_risk = "当前未触发异常风险规则"
        recommendation = "允许小幅增加排量/砂比，继续监测压力和分簇响应。"
        requires_confirmation = False

    result = dict(fallback or {})
    result.update(
        {
            "risk_level": risk,
            "main_risk": main_risk,
            "recommendation": recommendation,
            "requires_confirmation": requires_confirmation,
            "uncertainty": "medium" if uncertain or posterior_error > 0.30 else "low",
            "evidence": {
                "max_abnormal_probability": abnormal,
                "max_sand_plug_probability": sand_plug,
                "posterior_error": posterior_error,
                "unsafe": unsafe,
                "uncertain": uncertain,
            },
            "source": "rl_evaluation.csv（按DT时间进度归一化映射）",
            "episode": hmi.get("episode", "--"),
            "step": hmi.get("step", "--"),
            "high_level_option": option or "--",
        }
    )
    return result


def _read_csv_rows(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _enrich_hmi_control_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Attach the current control state to each future-action record.

    New evaluations contain ``pre_action_*`` fields from the environment.
    Older frozen evaluations do not, so the previous action in the same
    episode is used as a clearly marked compatibility fallback.
    """

    last_by_episode: dict[str, tuple[str, str]] = {}
    enriched: list[dict[str, str]] = []
    for row in rows:
        item = dict(row)
        episode = str(item.get("episode", "0"))
        action_flow = item.get("flow_m3_min", "")
        action_sand = item.get("sand_ratio_percent", "")
        current_flow = item.get("pre_action_flow_m3_min", "")
        current_sand = item.get("pre_action_sand_ratio_percent", "")
        if not str(current_flow).strip():
            current_flow = last_by_episode.get(episode, (action_flow, action_sand))[0]
            item["current_control_source"] = "previous_action_fallback"
        else:
            item["current_control_source"] = "environment_pre_action_state"
        if not str(current_sand).strip():
            current_sand = last_by_episode.get(episode, (action_flow, action_sand))[1]
        item["current_flow_m3_min"] = current_flow
        item["current_sand_ratio_percent"] = current_sand
        last_by_episode[episode] = (action_flow, action_sand)
        enriched.append(item)
    return enriched


def _registry_table_path(module: dict, filename: str) -> Path | None:
    for item in nested(module, "files", "tables", default=[]) or []:
        if item.get("path", "").endswith(filename):
            return resolve_project_path(item.get("path"))
    return None


def _registry_output_path(module: dict, output_name: str) -> Path | None:
    """Resolve an output listed in a module summary, including non-table outputs."""

    path = nested(module, "summary", "outputs", output_name)
    if path:
        resolved = resolve_project_path(path)
        if resolved:
            return resolved
    return None


def _dt_cache_matches_registry(cache_path: Path) -> bool:
    """Avoid replaying a cache generated from an older DT run."""

    if not cache_path.exists():
        return False
    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        registered = ArtifactRegistry().module("dt")
        registered_history = _registry_table_path(registered, "direct_observation_history.csv")
        cached_history = resolve_project_path(nested(cache, "sources", "dt_history"))
        return bool(registered_history and cached_history and registered_history.resolve() == cached_history.resolve())
    except (OSError, json.JSONDecodeError, TypeError):
        return False


def load_playback_frames() -> list[dict]:
    """Build a replay stream containing every available HMI decision window.

    DT stores a smaller set of state snapshots, while HMI stores one row per
    60-second decision window.  The replay therefore uses the larger HMI
    stream as its clock and maps the DT snapshots to it by normalized progress.
    This keeps all decision cards visible without claiming that the two files
    have native synchronized timestamps.
    """

    registry = ArtifactRegistry().snapshot().get("modules", {})
    dt_module = registry.get("dt", {})
    hmi_module = registry.get("hmi", {})
    dt_history = _read_csv_rows(_registry_table_path(dt_module, "direct_observation_history.csv"))
    cluster_history = _read_csv_rows(_registry_table_path(dt_module, "cluster_share_history.csv"))
    hmi_eval_path = _registry_table_path(hmi_module, "rl_evaluation.csv")
    if hmi_eval_path is None:
        # rl_evaluation.csv is a large diagnostic table and is intentionally
        # not part of the compact registry tables list.  It is still the
        # correct frame-level source for the dynamic HMI replay card.
        hmi_eval_path = _registry_output_path(hmi_module, "rl_evaluation")
    hmi_rows = _enrich_hmi_control_rows(_read_csv_rows(hmi_eval_path))
    decision_path = _registry_table_path(hmi_module, "human_machine_decisions.json")
    decisions = load_json(decision_path) if decision_path else []
    if isinstance(decisions, dict):
        decisions = [decisions]
    by_time: dict[float, list[dict[str, str]]] = {}
    for row in cluster_history:
        by_time.setdefault(_float(row.get("time_s")), []).append(row)

    replay_length = max(len(dt_history), len(hmi_rows), len(decisions))
    frames: list[dict] = []
    for index in range(replay_length):
        dt_index = (
            round(index / max(replay_length - 1, 1) * max(len(dt_history) - 1, 0))
            if dt_history
            else 0
        )
        row = dt_history[dt_index] if dt_history else {}
        time_s = _float(row.get("time_s"))
        clusters = by_time.get(time_s, [])
        hmi_index = index if hmi_rows else 0
        if hmi_rows:
            hmi_index = min(hmi_index, len(hmi_rows) - 1)
        hmi = hmi_rows[hmi_index] if hmi_rows else {}
        decision_index = (
            round(index / max(replay_length - 1, 1) * max(len(decisions) - 1, 0))
            if decisions
            else 0
        )
        decision = decisions[decision_index] if decisions else {}
        # The acceptance decision JSON is sparse and aggregate.  Use the
        # aligned RL row for the dynamic card so risk and action can change
        # during playback instead of repeating one static summary card.
        decision = _build_replay_decision(hmi, decision)
        prior_total_length = _float(row.get("prior_total_half_length_m"))
        liquid_weights = [max(_float(item.get("observed_liquid_share")), 1.0e-12) ** 0.6 for item in clusters]
        weight_sum = sum(liquid_weights) or 1.0
        for item, weight in zip(clusters, liquid_weights):
            # New DT runs persist the exact prior per cluster. Keep the
            # liquid-share reconstruction only for older result folders.
            exact_prior = _float(item.get("prior_half_length_m"))
            item["prior_half_length_m"] = (
                exact_prior
                if exact_prior > 0.0
                else prior_total_length * weight / weight_sum
            )
        frames.append(
            {
                "index": index,
                "replay_index": index + 1,
                "replay_total": replay_length,
                "dt_index": dt_index,
                "time_s": time_s,
                "phase": row.get("phase", "unknown"),
                "prior_bhp": _float(row.get("prior_bottomhole_pressure_mpa")),
                "posterior_bhp": _float(row.get("posterior_bottomhole_pressure_mpa")),
                "observed_bhp": _float(row.get("observed_bottomhole_pressure_mpa")),
                "prior_liquid_error": _float(row.get("prior_liquid_tvd")),
                "posterior_liquid_error": _float(row.get("posterior_liquid_tvd")),
                "prior_sand_error": _float(row.get("prior_sand_tvd")),
                "posterior_sand_error": _float(row.get("posterior_sand_tvd")),
                "kalman_gain": _float(row.get("mean_abs_kalman_gain")),
                "posterior_eprime": _float(row.get("posterior_eprime_gpa")),
                "posterior_leakoff": _float(row.get("posterior_leakoff_m_sqrt_s")),
                "posterior_viscosity": _float(row.get("posterior_viscosity_pa_s")),
                "posterior_min_stress": _float(row.get("posterior_min_stress_mpa")),
                "within_15": str(row.get("posterior_all_observations_within_15_percent", "false")).lower() == "true",
                "clusters": [
                    {
                        "id": int(_float(item.get("cluster_id"))),
                        "prior_length": _float(item.get("prior_half_length_m")),
                        "length": _float(item.get("posterior_half_length_m")),
                        "liquid": _float(item.get("posterior_liquid_share")),
                        "sand": _float(item.get("posterior_sand_share")),
                    }
                    for item in sorted(clusters, key=lambda value: _float(value.get("cluster_id")))
                ],
                "current_flow": _float(hmi.get("current_flow_m3_min")),
                "current_sand": _float(hmi.get("current_sand_ratio_percent")),
                "action_flow": _float(hmi.get("flow_m3_min")),
                "action_sand": _float(hmi.get("sand_ratio_percent")),
                "hmi_pressure": _float(hmi.get("bottomhole_pressure_mpa")),
                "hmi_abnormal": _float(hmi.get("abnormal_probability")),
                "hmi_sand_plug": _float(hmi.get("sand_plug_probability")),
                "hmi_reward": _float(hmi.get("integrated_reward")),
                "hmi_option": hmi.get("high_level_option", "--"),
                "hmi_episode": hmi.get("episode", "--"),
                "hmi_step": hmi.get("step", "--"),
                "decision": decision,
            }
        )
    return frames


def project_status() -> dict:
    registry = ArtifactRegistry().snapshot()
    fsl = registry["modules"].get("fsl", {})
    dt = registry["modules"].get("dt", {})
    hmi = registry["modules"].get("hmi", {})
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "environment": {
            "ui_python": sys.executable,
            "algorithm_python": str(BASE_PYTHON),
            "project_root": str(ROOT),
        },
        "registry": registry,
        "fsl": {"transfer": fsl.get("supporting", {}).get("transfer", {}), "two_stage": fsl.get("summary", {}), "artifact": fsl},
        "dt": {**dt.get("summary", {}), "artifact": dt},
        "hmi": {**hmi.get("summary", {}), "artifact": hmi},
    }


def write_summary() -> Path:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    run = create_app_run({"project_status": project_status()})
    path = OUTPUTS / "app_summary.json"
    path.write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
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


def run_gui(smoke: bool = False) -> int:
    from PySide6.QtCore import QProcess, QRect, QTimer, Qt, Signal
    from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
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
        QComboBox,
        QScrollArea,
        QSizePolicy,
        QSlider,
        QSplitter,
        QStackedWidget,
        QSpinBox,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )

    try:
        from App.dt_realtime import CACHE_PATH, HTML_PATH, build_dt_realtime_panel
    except ImportError:
        from dt_realtime import CACHE_PATH, HTML_PATH, build_dt_realtime_panel

    # Keep the main dashboard on the OS font registry. The dynamic DT panel
    # has its own fallback for stripped-down/offscreen Qt environments.
    ui_font_family = "Microsoft YaHei"

    # The Qt environment is deliberately UI-only.  Build the synchronized
    # cache with the algorithm environment only when a release copy is absent.
    if not _dt_cache_matches_registry(CACHE_PATH) and BASE_PYTHON.exists():
        subprocess.run(
            [str(BASE_PYTHON), str(ROOT / "App" / "build_dt_realtime_cache.py"), "--output", str(CACHE_PATH)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    if BASE_PYTHON.exists() and (not HTML_PATH.exists() or (CACHE_PATH.exists() and HTML_PATH.stat().st_mtime < CACHE_PATH.stat().st_mtime)):
        subprocess.run(
            [str(BASE_PYTHON), str(ROOT / "App" / "build_dt_3d_realtime_html.py"), "--cache", str(CACHE_PATH), "--output", str(HTML_PATH)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
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

    class ReplayChart(QFrame):
        def __init__(self, frames: list[dict]):
            super().__init__()
            self.frames = frames
            self.index = 0
            self.setMinimumHeight(560)
            self.setObjectName("replayChart")

        def set_index(self, index: int):
            self.index = max(0, min(index, max(len(self.frames) - 1, 0)))
            self.update()

        @staticmethod
        def _points(values: list[float], rect, lower: float, upper: float) -> list[tuple[int, int]]:
            if not values:
                return []
            span = max(upper - lower, 1e-9)
            count = max(len(values) - 1, 1)
            return [
                (
                    int(rect.left() + rect.width() * i / count),
                    int(rect.bottom() - (value - lower) / span * rect.height()),
                )
                for i, value in enumerate(values)
            ]

        def _plot(self, painter: QPainter, rect, series, title: str, percent: bool = False):
            painter.setPen(QPen(QColor("#d8e2ec"), 1))
            painter.setBrush(QColor("#fbfdff"))
            painter.drawRect(rect)
            values = [value for _, values_line, _ in series for value in values_line]
            if not values:
                return
            lower = min(values)
            upper = max(values)
            if math.isclose(lower, upper):
                lower -= 1.0
                upper += 1.0
            painter.setPen(QPen(QColor("#35516a"), 1))
            painter.drawText(int(rect.left() + 10), int(rect.top() + 20), title)
            for tick in range(1, 4):
                y = int(rect.top() + rect.height() * tick / 4)
                painter.setPen(QPen(QColor("#e7eef5"), 1))
                painter.drawLine(int(rect.left()), y, int(rect.right()), y)
            for name, values_line, color in series:
                points = self._points(values_line, rect.adjusted(10, 28, -10, -12), lower, upper)
                if len(points) < 2:
                    continue
                painter.setPen(QPen(QColor(color), 2))
                for first, second in zip(points, points[1:]):
                    painter.drawLine(first[0], first[1], second[0], second[1])
                painter.drawText(points[-1][0] - 80, points[-1][1] - 4, name)
            marker_x = int(rect.left() + rect.width() * self.index / max(len(self.frames) - 1, 1))
            painter.setPen(QPen(QColor("#dc2626"), 2, Qt.DashLine))
            painter.drawLine(marker_x, int(rect.top()), marker_x, int(rect.bottom()))
            painter.setPen(QPen(QColor("#64748b"), 1))
            painter.drawText(int(rect.left() + 10), int(rect.bottom() - 4), f"当前 {self.index + 1}/{len(self.frames)}")

        def paintEvent(self, _event):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.fillRect(self.rect(), QColor("#ffffff"))
            if not self.frames:
                painter.setPen(QColor("#b42318"))
                painter.drawText(20, 40, "暂无可播放的 DT 历史结果")
                return
            prior_bhp = [frame["prior_bhp"] for frame in self.frames]
            observed_bhp = [frame["observed_bhp"] for frame in self.frames]
            posterior_bhp = [frame["posterior_bhp"] for frame in self.frames]
            pressure_error = [
                abs(frame["posterior_bhp"] - frame["observed_bhp"])
                / max(abs(frame["observed_bhp"]), 1.0)
                * 100
                for frame in self.frames
            ]
            margin = 18
            width = self.width() - 2 * margin
            first = (margin, 8, width, int(self.height() * 0.24))
            second = (margin, int(self.height() * 0.28), width, int(self.height() * 0.22))
            third = (margin, int(self.height() * 0.52), width, int(self.height() * 0.22))
            fourth = (margin, int(self.height() * 0.76), width, int(self.height() * 0.22))
            self._plot(
                painter,
                QRect(*first),
                [("PKN先验", prior_bhp, "#f97316"), ("观测", observed_bhp, "#334155"), ("EnKF后验", posterior_bhp, "#0f9f9a")],
                "井底压力：正演先验 → 观测 → 参数更新后再正演（MPa）",
            )
            self._plot(
                painter,
                QRect(*second),
                [("当前排量", [frame.get("current_flow", 0.0) for frame in self.frames], "#334155"),
                 ("未来60秒建议", [frame.get("action_flow", 0.0) for frame in self.frames], "#f97316")],
                "第三部分动作响应：排量（m³/min）",
            )
            self._plot(
                painter,
                QRect(*third),
                [("当前砂比", [frame.get("current_sand", 0.0) for frame in self.frames], "#334155"),
                 ("未来60秒建议", [frame.get("action_sand", 0.0) for frame in self.frames], "#0f9f9a")],
                "第三部分动作响应：砂比（%）",
            )
            self._plot(
                painter,
                QRect(*fourth),
                [("井底压力误差", pressure_error, "#dc2626")],
                "后验误差：井底压力相对误差（%）",
            )

    class HMIActionChart(QFrame):
        """Compare the current control state with the future policy action."""

        def __init__(self, rows: list[dict[str, str]]):
            super().__init__()
            self.rows = rows
            self.setMinimumHeight(360)
            self.setObjectName("hmiActionChart")

        @staticmethod
        def _points(values: list[float], rect, lower: float, upper: float) -> list[tuple[int, int]]:
            span = max(upper - lower, 1e-9)
            count = max(len(values) - 1, 1)
            return [
                (
                    int(rect.left() + rect.width() * i / count),
                    int(rect.bottom() - (value - lower) / span * rect.height()),
                )
                for i, value in enumerate(values)
            ]

        def _plot(self, painter: QPainter, rect, series, title: str):
            painter.setPen(QPen(QColor("#d8e2ec"), 1))
            painter.setBrush(QColor("#fbfdff"))
            painter.drawRect(rect)
            values = [value for _, line, _ in series for value in line]
            if not values:
                return
            lower = min(values)
            upper = max(values)
            pad = max((upper - lower) * 0.08, 0.2)
            lower -= pad
            upper += pad
            plot_rect = rect.adjusted(10, 28, -10, -14)
            painter.setPen(QColor("#35516a"))
            painter.drawText(rect.left() + 10, rect.top() + 20, title)
            for tick in range(1, 4):
                y = int(plot_rect.top() + plot_rect.height() * tick / 4)
                painter.setPen(QPen(QColor("#e7eef5"), 1))
                painter.drawLine(plot_rect.left(), y, plot_rect.right(), y)
            for name, line, color in series:
                points = self._points(line, plot_rect, lower, upper)
                if len(points) < 2:
                    continue
                painter.setPen(QPen(QColor(color), 2))
                for first, second in zip(points, points[1:]):
                    painter.drawLine(first[0], first[1], second[0], second[1])
                painter.setPen(QColor(color))
                painter.drawText(points[-1][0] - 130, points[-1][1] - 4, name)

        def paintEvent(self, _event):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.fillRect(self.rect(), QColor("#ffffff"))
            if not self.rows:
                painter.setPen(QColor("#b42318"))
                painter.drawText(20, 40, "暂无第三部分动作记录")
                return
            current_flow = [_float(row.get("current_flow_m3_min")) for row in self.rows]
            action_flow = [_float(row.get("flow_m3_min")) for row in self.rows]
            current_sand = [_float(row.get("current_sand_ratio_percent")) for row in self.rows]
            action_sand = [_float(row.get("sand_ratio_percent")) for row in self.rows]
            margin = 18
            width = self.width() - 2 * margin
            upper = (margin, 8, width, int(self.height() * 0.43))
            lower = (margin, int(self.height() * 0.52), width, int(self.height() * 0.43))
            self._plot(
                painter,
                QRect(*upper),
                [("当前/历史输入", current_flow, "#334155"), ("未来60秒策略动作", action_flow, "#f97316")],
                "当前排量 vs 未来60秒建议排量（m³/min）",
            )
            self._plot(
                painter,
                QRect(*lower),
                [("当前/历史输入", current_sand, "#334155"), ("未来60秒策略动作", action_sand, "#0f9f9a")],
                "当前砂比 vs 未来60秒建议砂比（%）",
            )

    class ClusterBars(QFrame):
        def __init__(self, global_max_length: float = 1.0):
            super().__init__()
            self.frame: dict = {}
            self.global_max_length = max(float(global_max_length), 1.0)
            self.setMinimumHeight(260)
            self.setObjectName("clusterBars")

        def set_frame(self, frame: dict):
            self.frame = frame
            self.update()

        def paintEvent(self, _event):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.fillRect(self.rect(), QColor("#ffffff"))
            painter.setPen(QColor("#18354d"))
            painter.drawText(
                16,
                24,
                f"当前时间步分簇缝长对比（m）｜橙=PKN先验，青=EnKF后验｜全程统一上限 {self.global_max_length:.1f}",
            )
            clusters = self.frame.get("clusters", [])
            if not clusters:
                painter.setPen(QColor("#b42318"))
                painter.drawText(16, 55, "暂无分簇结果")
                return
            chart = (48, 62, max(self.width() - 70, 100), max(self.height() - 95, 100))
            painter.setBrush(QColor("#f97316"))
            painter.setPen(QPen(QColor("#f97316"), 1))
            painter.drawRect(16, 42, 12, 8)
            painter.setPen(QColor("#475569"))
            painter.setFont(QFont("Microsoft YaHei", 8))
            painter.drawText(34, 50, "PKN先验")
            painter.setBrush(QColor("#0f9f9a"))
            painter.setPen(QPen(QColor("#0f9f9a"), 1))
            painter.drawRect(92, 42, 12, 8)
            painter.setPen(QColor("#475569"))
            painter.drawText(110, 50, "EnKF后验")
            # Keep the y-scale fixed to the maximum across the whole replay;
            # otherwise a changing frame makes a shrinking bar look stable.
            max_value = self.global_max_length
            slot = chart[2] / len(clusters)
            for i, item in enumerate(clusters):
                prior = max(float(item.get("prior_length", 0.0)), 0.0)
                posterior = max(float(item.get("length", 0.0)), 0.0)
                bar_width = max(int(slot * 0.25), 9)
                gap = max(int(slot * 0.05), 3)
                center = chart[0] + slot * (i + 0.5)
                prior_h = prior / max_value * chart[3]
                posterior_h = posterior / max_value * chart[3]
                prior_y = chart[1] + chart[3] - prior_h
                posterior_y = chart[1] + chart[3] - posterior_h
                painter.setBrush(QColor("#f97316"))
                painter.setPen(QPen(QColor("#ea580c"), 1))
                painter.drawRoundedRect(
                    int(center - bar_width - gap / 2),
                    int(prior_y),
                    bar_width,
                    max(int(prior_h), 1),
                    4,
                    4,
                )
                painter.setBrush(QColor("#0f9f9a"))
                painter.setPen(QPen(QColor("#087f7a"), 1))
                painter.drawRoundedRect(
                    int(center + gap / 2),
                    int(posterior_y),
                    bar_width,
                    max(int(posterior_h), 1),
                    5,
                    5,
                )
                label_y = max(chart[1] + 16, int(posterior_y) - 6)
                # Keep the single posterior label compact so values remain
                # fully visible even when six clusters share a narrow panel.
                painter.setFont(QFont("Microsoft YaHei", 6))
                painter.setPen(QColor("#087f7a"))
                painter.drawText(
                    QRect(int(center - slot * 0.45), label_y - 15, int(slot * 0.9), 15),
                    Qt.AlignCenter,
                    f"{posterior:.1f}",
                )
                painter.setPen(QColor("#18354d"))
                painter.setFont(QFont("Microsoft YaHei", 8))
                painter.drawText(
                    QRect(int(center - slot * 0.45), chart[1] + chart[3] + 8, int(slot * 0.9), 18),
                    Qt.AlignCenter,
                    f"C{item['id']}",
                )

    class ReplayDashboard(QFrame):
        def __init__(self, frames: list[dict]):
            super().__init__()
            self.frames = frames
            self.index = 0
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.next_frame)
            self.setObjectName("replayPanel")
            root_layout = QVBoxLayout(self)
            root_layout.setContentsMargins(14, 14, 14, 14)
            heading = QHBoxLayout()
            title = QLabel("动态联调回放：正演 → 观测 → EnKF参数更新 → 智能体决策")
            title.setObjectName("replayTitle")
            self.time_label = QLabel("t=--")
            self.time_label.setObjectName("replayTime")
            heading.addWidget(title)
            heading.addStretch(1)
            heading.addWidget(self.time_label)
            root_layout.addLayout(heading)

            controls = QHBoxLayout()
            self.play_button = QPushButton("▶ 播放")
            self.play_button.clicked.connect(self.toggle_play)
            self.speed = QComboBox()
            self.speed.addItem("0.5×", 2000)
            self.speed.addItem("1×", 1000)
            self.speed.addItem("2×", 500)
            self.speed.addItem("4×", 250)
            self.speed.setCurrentIndex(1)
            self.speed.currentIndexChanged.connect(self.update_speed)
            self.slider = QSlider(Qt.Horizontal)
            self.slider.setRange(0, max(len(frames) - 1, 0))
            self.slider.valueChanged.connect(self.set_frame)
            self.jump = QSpinBox()
            self.jump.setRange(0, max(len(frames) - 1, 0))
            self.jump.valueChanged.connect(self.set_frame)
            controls.addWidget(self.play_button)
            controls.addWidget(QLabel("倍速"))
            controls.addWidget(self.speed)
            controls.addWidget(QLabel("时间轴"))
            controls.addWidget(self.slider, 1)
            controls.addWidget(QLabel("跳转帧"))
            controls.addWidget(self.jump)
            root_layout.addLayout(controls)

            self.kpi = QLabel()
            self.kpi.setObjectName("replayKpi")
            self.kpi.setWordWrap(True)
            root_layout.addWidget(self.kpi)
            body = QHBoxLayout()
            self.chart = ReplayChart(frames)
            body.addWidget(self.chart, 3)
            global_max_length = max(
                [
                    max(float(item.get("prior_length", 0.0)), float(item.get("length", 0.0)))
                    for frame in frames
                    for item in frame.get("clusters", [])
                ]
                or [1.0]
            )
            self.bars = ClusterBars(global_max_length)
            body.addWidget(self.bars, 1)
            root_layout.addLayout(body)
            bottom = QHBoxLayout()
            self.decision = QLabel()
            self.decision.setObjectName("decisionCard")
            self.decision.setWordWrap(True)
            bottom.addWidget(self.decision, 1)
            self.update_view(0)
            root_layout.addLayout(bottom)

        def toggle_play(self):
            if self.timer.isActive():
                self.timer.stop()
                self.play_button.setText("▶ 播放")
            else:
                self.timer.start(int(self.speed.currentData()))
                self.play_button.setText("⏸ 暂停")

        def update_speed(self):
            if self.timer.isActive():
                self.timer.start(int(self.speed.currentData()))

        def next_frame(self):
            if not self.frames:
                return
            self.set_frame((self.index + 1) % len(self.frames))

        def set_frame(self, index: int):
            if not self.frames:
                return
            self.index = max(0, min(int(index), len(self.frames) - 1))
            self.update_view(self.index)

        def update_view(self, index: int):
            if not self.frames:
                self.time_label.setText("暂无历史回放数据")
                return
            frame = self.frames[index]
            self.slider.blockSignals(True)
            self.jump.blockSignals(True)
            self.slider.setValue(index)
            self.jump.setValue(index)
            self.slider.blockSignals(False)
            self.jump.blockSignals(False)
            self.chart.set_index(index)
            self.bars.set_frame(frame)
            self.time_label.setText(f"t={frame['time_s']:.0f}s  |  阶段={frame['phase']}  |  {index + 1}/{len(self.frames)}")
            target = "达标" if frame["within_15"] else "未达标"
            target_color = "#087f5b" if frame["within_15"] else "#b42318"
            self.kpi.setText(
                f"<b>EnKF后验：</b>井底压力 {frame['posterior_bhp']:.2f} MPa　"
                f"Kalman Gain {frame['kalman_gain']:.4f}　"
                f"E′ {frame['posterior_eprime']:.2f} GPa　"
                f"<span style='color:{target_color}'><b>15%观测目标：{target}</b></span>"
            )
            decision = frame.get("decision", {}) or {}
            evidence = decision.get("evidence", {}) or {}
            risk = decision.get("risk_level", "unknown")
            option_labels = {
                "grow": "grow 增长",
                "hold": "hold 保持",
                "divert": "divert 分流/均衡",
                "safe": "safe 安全",
            }
            option = option_labels.get(frame.get("hmi_option", ""), frame.get("hmi_option", "--"))
            risk_colors = {"low": "#087f5b", "medium": "#b45309", "high": "#b42318"}
            risk_color = risk_colors.get(risk, "#475569")
            posterior_error = evidence.get("posterior_error", frame.get("hmi_posterior_error", 0.0))
            self.decision.setText(
                f"<b>HMI-KE 动态决策卡片</b>　动作：<b>{option}</b>　"
                f"风险：<span style='color:{risk_color}'><b>{risk}</b></span>　"
                f"不确定性：{decision.get('uncertainty', '--')}　"
                f"人工确认：{'是' if decision.get('requires_confirmation', True) else '否'}<br>"
                f"主要风险：{decision.get('main_risk', '--')}　"
                f"建议：{decision.get('recommendation', '暂无决策卡片')}<br>"
                f"证据：异常概率 {evidence.get('max_abnormal_probability', frame['hmi_abnormal']):.3f}；"
                f"砂堵概率 {evidence.get('max_sand_plug_probability', frame['hmi_sand_plug']):.3f}；"
                f"后验误差 {posterior_error:.3f}　"
                f"决策卡 {frame.get('replay_index', '--')}/{frame.get('replay_total', '--')}；"
                f"窗口 episode={decision.get('episode', frame.get('hmi_episode', '--'))}, step={decision.get('step', frame.get('hmi_step', '--'))}<br>"
                f"来源：{decision.get('source', '聚合决策卡片')}<br>"
                f"未来60秒动作：排量 {frame['action_flow']:.2f} m³/min，砂比 {frame['action_sand']:.2f}%"
            )

    class Dashboard(QMainWindow):
        log_line = Signal(str)

        def __init__(self):
            super().__init__()
            self.process: QProcess | None = None
            self.job_log_path: Path | None = None
            app_snapshot = load_json(OUTPUTS / "app_summary.json")
            self.app_run_root = resolve_project_path(app_snapshot.get("run_root"))
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
            h.addWidget(title)
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
            fsl_status = nested(status, "fsl", "artifact", "status", default="not_available")
            dt_status = nested(status, "dt", "artifact", "status", default="not_available")
            hmi_status = nested(status, "hmi", "artifact", "status", default="not_available")
            cards.addWidget(MetricCard("两阶段工况识别", f"{fsl_f1:.3f}", f"测试集五类Macro-F1 | 产物：{fsl_status}", "#d9363e"), 0, 0)
            cards.addWidget(MetricCard("数字孪生观测验证", f"{dt_pass*100:.0f}%", f"留出阶段观测误差 | 产物：{dt_status}", "#168aad"), 0, 1)
            cards.addWidget(MetricCard("闭环单步P95", f"{dt_p95:.1f} ms", "不含3D渲染；指标口径为模型计算", "#168aad"), 0, 2)
            hmi_text = "待正式训练" if hmi_safe is None else f"{hmi_safe*100:.1f}%"
            cards.addWidget(MetricCard("180秒安全验证", hmi_text, f"当前结果：{hmi_status}；砂堵场景仍需提升", "#d9921e"), 0, 3)
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
            fsl_meta = nested(status, "fsl", "artifact", default={}) or {}
            fsl_figure = resolve_project_path((fsl_meta.get("figures") or [None])[0]) or (ARTIFACTS / "fsl" / "two_stage_model" / "window_4" / "test_grouped_two_stage_confusion.png")
            layout.addWidget(ImagePanel("两阶段五类测试集混淆矩阵", fsl_figure))
            fsl_note = QLabel(f"产物状态：{fsl_meta.get('status', 'not_available')}。指标来源：{fsl_meta.get('summary_path', '--')}。口径：{fsl_meta.get('metrics_scope', '--')}")
            fsl_note.setObjectName("notice")
            fsl_note.setWordWrap(True)
            layout.addWidget(fsl_note)
            layout.addLayout(self._button_row([
                ("打开全书知识图谱", self.open_knowledge_graph),
                ("打开FSL结果目录", lambda: self.open_path(resolve_project_path("artifacts/fsl") or ARTIFACTS / "fsl")),
            ]))
            return page

        def _dt_page(self):
            page, layout = self._scroll_page(
                "第二部分：数字孪生闭环",
                "动态展示真实数据演化、PKN输入输出、EnKF参数更新、计算性能和误差指标。",
            )
            layout.addWidget(build_dt_realtime_panel())
            layout.addLayout(self._button_row([
                ("运行轻量验证", lambda: self.start_job("dt", "validate")),
                ("打开实时3D模型", lambda: webbrowser.open(HTML_PATH.resolve().as_uri()) if HTML_PATH.exists() else None),
                ("打开DT结果目录", lambda: self.open_path(resolve_project_path("artifacts/dt") or ARTIFACTS / "dt")),
            ]))
            return page

        def _hmi_page(self):
            page, layout = self._scroll_page(
                "第三部分：知识嵌入智能决策",
                "当前动作建议只对显式异常风险进行分级；正常施工阶段作为安全基线，压力安全和模型不确定性单独展示。",
            )
            h = status["hmi"]
            rl = h.get("rl_policy", {}) if isinstance(h, dict) else {}
            validation = nested(h, "validation_180s", "rl_policy", default={}) or {}
            cards = QGridLayout()
            cards.addWidget(MetricCard("算法", str(h.get('algorithm','PPO')), "Gymnasium连续动作环境", "#d9921e"), 0, 0)
            cards.addWidget(MetricCard("训练步数", str(h.get('total_timesteps','-')), "当前归档代表性策略", "#d9921e"), 0, 1)
            cards.addWidget(MetricCard("平均综合奖励", f"{rl.get('mean_integrated_reward',0):.2f}", "效果-压力-异常-成本", "#d9921e"), 0, 2)
            cards.addWidget(MetricCard("180秒安全率", f"{_float(validation.get('safe_within_180s_rate'), 0.0)*100:.1f}%", "归档模型口径，非最终验收结论", "#d9921e"), 0, 3)
            warning5 = h.get("warning_5min", {}) or {}
            latency = h.get("decision_latency", {}) or {}
            warning_recall = warning5.get("event_warning_recall")
            recall_text = "无完整事件" if warning_recall is None else f"{_float(warning_recall)*100:.1f}%"
            cards.addWidget(MetricCard("5分钟预警召回", recall_text, "当前数据没有完整5分钟事件样本" if warning_recall is None else "离线数字孪生回放口径", "#d9921e"), 1, 0)
            cards.addWidget(MetricCard("决策P95", f"{_float(latency.get('p95_seconds'), 0.0):.2f} s", "调整效果计算延迟，不含训练", "#d9921e"), 1, 1)
            cards.addWidget(MetricCard("质量门禁", "通过" if nested(h, "quality_gate", "passed", default=False) else "未通过", "当前必须如实显示阶段结果", "#d9363e"), 1, 2)
            cards.addWidget(MetricCard("动作定义", "未来60秒", "输入为当前300秒历史状态", "#d9921e"), 1, 3)
            layout.addLayout(cards)

            hmi_artifact = nested(status, "hmi", "artifact", default={}) or {}
            hmi_eval_path = _registry_output_path(hmi_artifact, "rl_evaluation")
            hmi_action_rows = _enrich_hmi_control_rows(_read_csv_rows(hmi_eval_path))
            action_note = QLabel(
                "控制量口径：深色线是当前/历史施工输入，橙色或青绿色线是智能体针对未来60秒的建议平均排量和砂比；"
                "这里不是累计液量/累计砂量，累计量只在第二部分数字孪生页面中展示。"
            )
            action_note.setObjectName("notice")
            action_note.setWordWrap(True)
            layout.addWidget(action_note)
            layout.addWidget(HMIActionChart(hmi_action_rows))

            criteria_panel = QFrame()
            criteria_panel.setObjectName("riskCriteriaPanel")
            criteria_layout = QVBoxLayout(criteria_panel)
            criteria_title = QLabel("人机协同风险判定口径")
            criteria_title.setObjectName("sectionTitle")
            criteria_layout.addWidget(criteria_title)
            criteria_grid = QGridLayout()
            criteria_items = [
                (
                    "异常工况目标",
                    "显式异常标签",
                    "砂堵、缝口/缝内暂堵、延伸受阻、滤失过大、缝高延伸、新缝开启、其他/异常",
                ),
                (
                    "正常基线",
                    "不计入异常正类",
                    "正常、正常工况、主缝延伸作为正常施工基线；空白或“??”不作为异常正类，但应补充标注",
                ),
                (
                    "压力安全",
                    "BHP 45–110 MPa",
                    "净压力安全范围为 0–35 MPa；超出范围进入压力处罚和人工复核，不改变工况标签",
                ),
                (
                    "5分钟预警",
                    "异常≥0.30；砂堵≥0.22",
                    "压力达到上限的90%也触发预警：BHP≥99 MPa 或净压力≥31.5 MPa",
                ),
                (
                    "事件与180秒安全",
                    "异常>0.45；砂堵>0.35",
                    "同时检查 BHP>110 MPa、净压力>35 MPa；任一超限则该窗口不安全",
                ),
                (
                    "模型不确定性",
                    "残差≤3 / 3–8 / >8",
                    "分别记为低/中/高不确定性；它表示模型证据不足，不等同于异常工况",
                ),
            ]
            for index, (title, value, caption) in enumerate(criteria_items):
                card = QFrame()
                card.setObjectName("riskCriteriaCard")
                card_layout = QVBoxLayout(card)
                card_layout.setContentsMargins(14, 12, 14, 12)
                label = QLabel(title)
                label.setObjectName("riskCriteriaTitle")
                value_label = QLabel(value)
                value_label.setObjectName("riskCriteriaValue")
                value_label.setWordWrap(True)
                caption_label = QLabel(caption)
                caption_label.setObjectName("riskCriteriaCaption")
                caption_label.setWordWrap(True)
                card_layout.addWidget(label)
                card_layout.addWidget(value_label)
                card_layout.addWidget(caption_label)
                criteria_grid.addWidget(card, index // 2, index % 2)
            criteria_layout.addLayout(criteria_grid)
            criteria_note = QLabel(
                "风险分层顺序：显式异常概率/压力边界 → 砂堵专项概率 → 分簇不均衡 → EnKF后验误差与模型越界。"
                "高风险或控制、停泵、参数修改类动作必须人工确认；当前系统只输出建议，不直接下发设备控制。"
            )
            criteria_note.setObjectName("notice")
            criteria_note.setWordWrap(True)
            criteria_layout.addWidget(criteria_note)
            layout.addWidget(criteria_panel)

            hmi_meta = nested(status, "hmi", "artifact", default={}) or {}
            hmi_figure = resolve_project_path((hmi_meta.get("figures") or [None])[0]) or (ARTIFACTS / "hmi" / "ppo_policy" / "rl_vs_historical_reward.png")
            layout.addWidget(ImagePanel("强化学习策略与历史动作奖励对比", hmi_figure))
            warning = QLabel(
                f"当前产物状态：{hmi_meta.get('status', 'not_available')}。{hmi_meta.get('status_reason', '')}\n"
                f"指标来源：{hmi_meta.get('summary_path', '--')}。{'; '.join(hmi_meta.get('limitations', []))}\n"
                "风险标签口径已切换为显式异常标签；当前结果为历史回放/离线响应代理，不能替代现场在线闭环验收。"
            )
            warning.setObjectName("warning")
            warning.setWordWrap(True)
            layout.addWidget(warning)
            reward = rl
            hmi_detail = QLabel(
                "<b>状态</b>：当前300秒压力/排量/砂比及DT风险上下文；<b>动作</b>：未来60秒平均排量与砂比。<br>"
                "<b>异常风险口径</b>：异常概率只由显式异常工况标签训练得到，正常/主缝延伸不作为异常；"
                "压力越界、分簇不均衡和EnKF残差作为独立动态证据。<br>"
                f"<b>奖励分解</b>：改造效果 {reward.get('mean_effectiveness_reward', 0):.3f}；"
                f"压力安全项 {reward.get('mean_pressure_safety_penalty', 0):.3f}；"
                f"异常风险项 {reward.get('mean_abnormal_risk_penalty', 0):.3f}；"
                f"施工成本项 {reward.get('mean_construction_cost_penalty', 0):.3f}。<br>"
                f"<b>180秒口径</b>：候选窗口 {validation.get('candidate_windows', '--')}，"
                f"完整窗口 {validation.get('eligible_complete_windows', '--')}，"
                f"砂堵概率最大值 {validation.get('max_sand_plug_probability_180s', 0):.3f}。"
            )
            hmi_detail.setObjectName("notice")
            hmi_detail.setWordWrap(True)
            layout.addWidget(hmi_detail)
            layout.addLayout(self._button_row([
                ("运行场景轻量验证", lambda: self.start_job("hmi", "scenarios")),
                ("打开HMI结果目录", lambda: self.open_path(resolve_project_path("outputs/hmi/contract3_acceptance") or ROOT / "outputs" / "hmi")),
            ]))
            return page

        def _integration_page(self):
            page, layout = self._scroll_page("联合动态演示", "冻结结果按时间步回放，不重新训练模型；每一帧同时更新数字孪生状态、观测误差和智能体建议。")
            self.replay_dashboard = ReplayDashboard(load_playback_frames())
            layout.addWidget(self.replay_dashboard)
            flow_note = QLabel(
                "动态展示口径：第8段DT结果提供状态快照，联合回放完整读取 HMI 的全部 720 个决策窗口；两者按归一化进度对齐。红色时间线表示当前帧，左侧曲线展示井底压力和观测误差，右侧展示分簇后验缝长，底部展示当前决策卡、动作和人工确认。右侧缝长图使用全程最大值固定纵轴，3D模型使用全程最大裂缝尺度固定坐标轴；六簇位于垂深3000 m后的北向77–1650 m展示区间。"
            )
            flow_note.setObjectName("notice")
            flow_note.setWordWrap(True)
            layout.addWidget(flow_note)
            layout.addLayout(self._button_row([
                ("一键准备验收演示", self.prepare_demo),
                ("运行轻量联调验证", lambda: self.start_job("dt", "validate")),
                ("生成HMI场景验证", lambda: self.start_job("hmi", "scenarios")),
                ("生成验收摘要", self.export_acceptance_summary),
            ]))
            layout.addStretch(1)
            return page

        def _apply_style(self):
            self.setStyleSheet("""
                QMainWindow, QWidget { background:#f4f7fb; color:#17212b; font-family:'__UI_FONT__'; font-size:14px; }
                QLabel { color:#17344b; background:transparent; }
                #header { background:#0b2942; padding:12px 20px; border-bottom:3px solid #14b8a6; }
                #header QLabel { color:#f8fbff; background:transparent; }
                QListWidget { background:#102b40; color:#e7f0f7; border:0; padding:14px 8px; font-size:15px; }
                QListWidget::item { padding:14px 12px; border-radius:6px; }
                QListWidget::item:selected { background:#e23b45; color:#ffffff; font-weight:700; border:1px solid #ff9f9f; }
                #pageTitle { font-size:28px; font-weight:800; color:#0b3558; background:transparent; }
                #sectionTitle { font-size:18px; font-weight:700; color:#0b3558; padding:5px 0; background:transparent; }
                #muted { color:#405b70; background:transparent; }
                #metricCard, #panel { background:#ffffff; border:1px solid #cbd9e6; border-radius:9px; }
                #cardTitle { font-weight:700; color:#17344b; background:#edf3f8; padding:2px 4px; }
                #notice { background:#e7f6f5; color:#164e63; border:1px solid #8dd8d0; border-left:5px solid #0f9f9a; padding:14px; }
                #warning { background:#fff4df; color:#713f12; border:1px solid #f2c77c; border-left:5px solid #d9921e; padding:14px; }
                QPushButton { background:#0f766e; color:#ffffff; border:0; padding:10px 18px; border-radius:5px; font-weight:700; }
                QPushButton:hover { background:#115e59; }
                QPushButton:disabled { background:#9aa9b5; color:#eef3f6; }
                QComboBox, QSpinBox { background:#ffffff; color:#17344b; border:1px solid #9bb0c2; padding:7px 10px; border-radius:5px; }
                QSlider::groove:horizontal { height:7px; background:#d7e3ed; border-radius:3px; }
                QSlider::sub-page:horizontal { background:#0f9f9a; border-radius:3px; }
                QSlider::handle:horizontal { width:18px; margin:-6px 0; background:#0f766e; border-radius:9px; }
                #replayPanel { background:#ffffff; border:1px solid #b9cedd; border-radius:12px; }
                #replayTitle { font-size:19px; font-weight:800; color:#0b3558; }
                #replayTime { background:#e7f6f5; color:#075e5b; border:1px solid #84d3cb; padding:7px 12px; border-radius:6px; font-weight:700; }
                #replayKpi { background:#eef6fb; color:#17344b; border:1px solid #bdd1e2; padding:11px; border-radius:6px; }
                #decisionCard { background:#fff4df; color:#713f12; border:1px solid #efc16b; border-left:5px solid #d9921e; padding:13px; border-radius:6px; }
                 #replayChart, #clusterBars { background:#ffffff; border:1px solid #d2e0eb; border-radius:8px; }
                 #dtRealtime { background:#eef3f7; border:1px solid #c5d4df; border-radius:10px; }
                 #dtPanel { background:#ffffff; border:1px solid #c5d4df; border-radius:9px; }
                 #dtSectionTitle { font-size:16px; font-weight:800; color:#0b3558; padding:2px 0 5px; }
                 #dtControlPanel { background:#102b40; border:1px solid #244c69; border-radius:8px; }
                 #dtControlPanel QLabel { color:#f5f9fc; font-weight:700; }
                 #dtStatus { background:#e5f4f3; color:#14545c; border-left:4px solid #0f9f9a; padding:8px 10px; }
                 #dtTotalTime { color:#f5f9fc; background:transparent; font-weight:700; padding:4px 8px; }
                 #dtKey { color:#52687a; background:#f1f5f8; padding:4px 6px; }
                 #dtValue { color:#102f4a; background:#ffffff; padding:4px 6px; font-weight:700; }
                 #dtMetric { background:#ffffff; border:1px solid #d2dee8; border-radius:7px; padding:4px; }
                 #dtMetricTitle { color:#52687a; font-weight:700; }
                 #dtMetricValue { color:#0b6e75; font-size:21px; font-weight:800; }
                 #dtMetricNote { color:#66788a; font-size:11px; }
                 QTextEdit { background:white; border:1px solid #dce3e9; border-radius:6px; padding:8px; }
                QScrollArea { background:#f4f7fb; border:0; }
            """.replace("__UI_FONT__", ui_font_family))

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
            env.insert("HOME", os.environ.get("HOME", os.environ.get("USERPROFILE", str(Path.home()))))
            env.insert("PYTHONUTF8", "1")
            env.insert("PYTHONIOENCODING", "utf-8")
            self.process.setProcessEnvironment(env)
            log_root = self.app_run_root / "logs" if self.app_run_root else OUTPUTS / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S") / "logs"
            log_root.mkdir(parents=True, exist_ok=True)
            self.job_log_path = log_root / f"{module}_{action}.log"
            self.job_log_path.write_text("> " + " ".join([str(python), *args]) + "\n", encoding="utf-8")
            if self.app_run_root:
                app_run_path = self.app_run_root / "app_run.json"
                app_run = load_json(app_run_path)
                commands = app_run.get("commands", []) if isinstance(app_run.get("commands", []), list) else []
                commands.append({"module": module, "action": action, "python": str(python), "args": args, "log": relative_path(self.job_log_path)})
                app_run["commands"] = commands
                app_run_path.write_text(json.dumps(app_run, ensure_ascii=False, indent=2), encoding="utf-8")
                (self.app_run_root / "commands.json").write_text(json.dumps(commands, ensure_ascii=False, indent=2), encoding="utf-8")
            self.process.readyReadStandardOutput.connect(lambda: self._read_process(False))
            self.process.readyReadStandardError.connect(lambda: self._read_process(True))
            self.process.finished.connect(lambda code, _status: self._job_finished(code, open_after))
            self.log_line.emit(f"<b>启动：</b>{python} {' '.join(args)}")
            self.process.start()

        def _read_process(self, error: bool):
            if not self.process:
                return
            raw = self.process.readAllStandardError() if error else self.process.readAllStandardOutput()
            text = bytes(raw).decode("utf-8", errors="replace").rstrip()
            if text:
                color = "#b92830" if error else "#253745"
                self.log_line.emit(f"<span style='color:{color};white-space:pre'>{text}</span>")
                if self.job_log_path:
                    with self.job_log_path.open("a", encoding="utf-8") as handle:
                        handle.write(("[stderr] " if error else "[stdout] ") + text + "\n")

        def _job_finished(self, code: int, open_after: Path | None):
            self.log_line.emit(f"<b>任务结束，退出码：{code}</b>")
            if self.job_log_path:
                with self.job_log_path.open("a", encoding="utf-8") as handle:
                    handle.write(f"[exit] {code}\n")
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
            meta = ArtifactRegistry().module("fsl")
            path = resolve_project_path(meta.get("html")) or (ROOT / "FSL-Expert" / "knowledge_graph" / "full_book_qwen_output" / "index_full_book_qwen.html")
            if path.exists():
                webbrowser.open(path.resolve().as_uri())
            else:
                QMessageBox.warning(self, "文件不存在", str(path))

        def prepare_demo(self):
            write_summary()
            self.open_knowledge_graph()
            dt_meta = ArtifactRegistry().module("dt")
            dt_html = resolve_project_path(dt_meta.get("html")) or (ROOT / "outputs" / "dt" / "digital_twin_3d.html")
            offline_ready = False
            if dt_html.exists():
                try:
                    offline_ready = "<script src=" not in dt_html.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    offline_ready = False
            if offline_ready:
                webbrowser.open(dt_html.resolve().as_uri())
            else:
                self.start_job("dt", "visualize", open_after=dt_html)
            self.log_line.emit("验收摘要已刷新；知识图谱已打开；正在准备自包含数字孪生页面。")

        def export_acceptance_summary(self):
            path = write_summary()
            QMessageBox.information(self, "已生成", f"验收摘要：\n{path}")

    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 10))
    window = Dashboard()
    window.show()
    if smoke:
        QTimer.singleShot(1200, app.quit)
    return app.exec()


def main() -> None:
    parser = argparse.ArgumentParser(description="Intelligent fracturing acceptance dashboard")
    parser.add_argument("--no-gui", action="store_true", help="Only write the integrated summary")
    parser.add_argument("--preflight", action="store_true", help="Check Qt and required artifacts")
    parser.add_argument("--no-auto-env", action="store_true", help="Do not relaunch in frac_app")
    parser.add_argument("--smoke-gui", action="store_true", help="Construct the PySide UI and exit after a short smoke check")
    args = parser.parse_args()
    summary = write_summary()
    summary_payload = load_json(summary)
    qt_error = qt_import_error()
    checks = {
        "summary": str(summary),
        "qt_ok": qt_error is None,
        "qt_error": qt_error,
        "qt_python": str(QT_PYTHON),
        "algorithm_python": str(BASE_PYTHON),
        "preflight": summary_payload.get("preflight", {}),
        "registry": nested(summary_payload, "registry", "modules", default={}),
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
    raise SystemExit(run_gui(smoke=args.smoke_gui))


if __name__ == "__main__":
    main()
