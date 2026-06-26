from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
METRICS = ROOT / "runs" / "frac_lgbm_split_search_normal_f1" / "20260428_194324" / "seed_1_metrics.json"
OUT_MAIN = ROOT / "report_figures" / "13_seed1_multiclass_per_class_metrics.png"
OUT_FULL = ROOT / "report_figures" / "13_seed1_multiclass_per_class_metrics_full.png"


def setup_font() -> None:
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def plot(rows: list[tuple[str, dict]], out_path: Path, title: str, width: float) -> None:
    labels = [r[0] for r in rows]
    precision = [r[1]["precision"] for r in rows]
    recall = [r[1]["recall"] for r in rows]
    f1 = [r[1]["f1-score"] for r in rows]
    support = [int(r[1]["support"]) for r in rows]

    x = np.arange(len(labels))
    bar_width = 0.24
    fig, ax = plt.subplots(figsize=(width, 5.0), dpi=160)
    ax.bar(x - bar_width, precision, bar_width, label="Precision", color="#4C78A8")
    ax.bar(x, recall, bar_width, label="Recall", color="#F58518")
    ax.bar(x + bar_width, f1, bar_width, label="F1", color="#54A24B")
    ax.set_title(title)
    ax.set_ylabel("指标值")
    ax.set_ylim(0, 1.08)
    ax.set_xticks(x, labels)
    ax.legend(loc="upper center", frameon=True)
    for i, n in enumerate(support):
        ax.text(i, 1.035, f"n={n}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup_font()
    data = json.loads(METRICS.read_text(encoding="utf-8"))
    classes = data["classes"]
    report = data["metrics"]["test"]["report"]
    name_to_metric = {
        classes[int(label_idx)]: metric
        for label_idx, metric in report.items()
        if label_idx.isdigit() and isinstance(metric, dict)
    }
    main_labels = ["正常", "主缝延伸", "缝内暂堵", "缝口暂堵"]
    full_labels = ["正常", "主缝延伸", "砂堵", "缝内暂堵", "缝口暂堵", "滤失过大"]
    plot(
        [(label, name_to_metric[label]) for label in main_labels if label in name_to_metric],
        OUT_MAIN,
        "合并数据多分类：测试集各类别识别情况",
        8.5,
    )
    plot(
        [(label, name_to_metric[label]) for label in full_labels if label in name_to_metric],
        OUT_FULL,
        "合并数据多分类：测试集各类别识别情况（含低频类）",
        10.5,
    )
    print(OUT_MAIN)
    print(OUT_FULL)


if __name__ == "__main__":
    main()
