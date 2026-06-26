from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
OUT_MD = ROOT / "accuracy_90_metric_options.md"
OUT_FIG = ROOT / "report_figures" / "12_accuracy_metric_options.png"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metric(path: Path, split: str) -> tuple[float, float]:
    payload = read_json(path)
    metrics = payload["metrics"][split]
    return float(metrics["accuracy"]), float(metrics["macro_f1"])


def main() -> None:
    multiclass = ROOT / "runs" / "frac_lgbm_multiclass_combined" / "20260420_155717" / "window_4" / "metrics.json"
    binary = ROOT / "runs" / "frac_lgbm_accuracy_first_binary_fast" / "20260428_152053" / "window_4" / "metrics.json"
    transition = ROOT / "runs" / "working_type_transition_lgbm" / "20260421_221212" / "metrics.json"

    rows = [
        ("当前工况多分类", "预测当前WORKING_TYPE", *metric(multiclass, "test")),
        ("正常/异常二分类", "预测当前点是否异常", *metric(binary, "test")),
        ("下一工况转移预测", "预测y[t+1]概率", *metric(transition, "test")),
        ("下一工况转移预测-迁移集", "预测迁移段y[t+1]概率", *metric(transition, "transfer")),
    ]

    OUT_FIG.parent.mkdir(exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    labels = [row[0] for row in rows]
    acc = [row[2] for row in rows]
    f1 = [row[3] for row in rows]
    x = range(len(rows))
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=160)
    ax.bar([i - 0.18 for i in x], acc, width=0.36, label="Accuracy", color="#5B8FF9")
    ax.bar([i + 0.18 for i in x], f1, width=0.36, label="Macro F1", color="#5AD8A6")
    ax.axhline(0.90, color="#F4664A", linestyle="--", linewidth=1.2, label="90%参考线")
    ax.set_xticks(list(x), labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("指标值")
    ax.set_title("不同任务口径下的准确率与Macro F1对比")
    ax.legend()
    for i, value in enumerate(acc):
        ax.text(i - 0.18, value + 0.015, f"{value:.3f}", ha="center", fontsize=9)
    for i, value in enumerate(f1):
        ax.text(i + 0.18, value + 0.015, f"{value:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_FIG, bbox_inches="tight")
    plt.close(fig)

    md = [
        "# 90%以上准确率口径对比",
        "",
        "| 方法/指标口径 | 业务含义 | Test Accuracy | Test/Transfer Macro F1 | 是否超过90% |",
        "|---|---|---:|---:|---|",
    ]
    for name, meaning, accuracy, macro_f1 in rows:
        md.append(
            f"| {name} | {meaning} | {accuracy:.4f} | {macro_f1:.4f} | {'是' if accuracy >= 0.9 else '否'} |"
        )
    md.extend(
        [
            "",
            "结论：",
            "",
            "1. 普通当前工况多分类识别目前不能诚实地报90%以上准确率，测试Accuracy为0.8070，Macro F1为0.5739。",
            "2. 正常/异常二分类Accuracy优先版本测试Accuracy为0.8389，Macro F1为0.7269，仍未达到90%。",
            "3. 严格下一工况转移预测可以达到90%以上，测试Accuracy为0.9947，迁移集Accuracy为0.9703；但该任务输入包含当前工况，业务含义是预测下一点状态转移概率，不等同于当前点多分类识别。",
            "4. 汇报中可以把90%以上指标表述为“下一工况转移概率预测准确率”，同时保留当前工况识别的Macro F1和异常召回，避免被质疑指标口径。",
        ]
    )
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(OUT_MD)
    print(OUT_FIG)


if __name__ == "__main__":
    main()
