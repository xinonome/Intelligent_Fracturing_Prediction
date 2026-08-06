from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


def configure_font() -> None:
    for path in [Path(r"C:\Windows\Fonts\msyh.ttc"), Path(r"C:\Windows\Fonts\simhei.ttf"), Path(r"C:\Windows\Fonts\simsun.ttc")]:
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            name = font_manager.FontProperties(fname=str(path)).get_name()
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["axes.unicode_minus"] = False
            return


def load_summary(path: Path, label: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    return {
        "方案": label,
        "液量TVD/%": metrics["validation_liquid_tvd_mean"] * 100.0,
        "砂量TVD/%": metrics["validation_sand_tvd_mean"] * 100.0,
        "压力相对误差/%": metrics["validation_bhp_relative_error_mean"] * 100.0,
        "先验液量TVD/%": metrics["validation_prior_liquid_tvd_mean"] * 100.0,
        "先验砂量TVD/%": metrics["validation_prior_sand_tvd_mean"] * 100.0,
        "先验压力误差/%": metrics["validation_prior_bhp_relative_error_mean"] * 100.0,
        "单步P50/ms": metrics["all_steps_compute_p50_ms"],
        "单步P95/ms": metrics["all_steps_compute_p95_ms"],
        "15秒达标率/%": metrics["all_steps_under_15_seconds_rate"] * 100.0,
        "观测≤15%步次率/%": metrics["validation_all_observations_within_15_percent_rate"] * 100.0,
        "滤失比例/%": metrics["mean_posterior_leakoff_fraction"] * 100.0,
    }


def plot_comparison(frame: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    x = np.arange(len(frame))
    colors = ["#4C78A8", "#2A9D8F"]

    axes[0, 0].bar(x, frame["液量TVD/%"], color=colors)
    axes[0, 0].axhline(15, color="#D62728", linestyle="--", linewidth=1.2)
    axes[0, 0].set_title("留出液量分配误差")
    axes[0, 0].set_ylabel("TVD / %")

    axes[0, 1].bar(x, frame["砂量TVD/%"], color=colors)
    axes[0, 1].axhline(15, color="#D62728", linestyle="--", linewidth=1.2)
    axes[0, 1].set_title("留出砂量分配误差")
    axes[0, 1].set_ylabel("TVD / %")

    axes[1, 0].bar(x - 0.18, frame["压力相对误差/%"], width=0.36, color="#54A24B", label="后验")
    axes[1, 0].bar(x + 0.18, frame["先验压力误差/%"], width=0.36, color="#F58518", label="先验")
    axes[1, 0].axhline(15, color="#D62728", linestyle="--", linewidth=1.2)
    axes[1, 0].set_title("压力误差：校正前后")
    axes[1, 0].set_ylabel("相对误差 / %")
    axes[1, 0].legend()

    axes[1, 1].bar(x - 0.18, frame["单步P50/ms"], width=0.36, color="#72B7B2", label="P50")
    axes[1, 1].bar(x + 0.18, frame["单步P95/ms"], width=0.36, color="#E45756", label="P95")
    axes[1, 1].axhline(15000, color="#D62728", linestyle="--", linewidth=1.2)
    axes[1, 1].set_title("在线单步计算耗时")
    axes[1, 1].set_ylabel("耗时 / ms")
    axes[1, 1].legend()

    for ax in axes.flat:
        ax.set_xticks(x, frame["方案"])
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("PKN-EnKF：基线与物理预校准增强版对比", fontsize=15, weight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    configure_font()
    parser = argparse.ArgumentParser(description="Compare direct-observation PKN-EnKF runs.")
    parser.add_argument("--baseline-summary", required=True)
    parser.add_argument("--enhanced-summary", required=True)
    parser.add_argument("--out-dir", default="outputs/dt/model_compare_report")
    args = parser.parse_args()

    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        [
            load_summary(Path(args.baseline_summary), "原基线"),
            load_summary(Path(args.enhanced_summary), "物理预校准增强"),
        ]
    )
    frame.to_csv(out / "direct_observation_comparison.csv", index=False, encoding="utf-8-sig")
    plot_comparison(frame, out / "direct_observation_comparison.png")
    payload = {
        "baseline_summary": str(Path(args.baseline_summary).resolve()),
        "enhanced_summary": str(Path(args.enhanced_summary).resolve()),
        "comparison": frame.to_dict(orient="records"),
        "enhanced_design": [
            "前70%校准段执行物理参数批量MAP预校准",
            "预校准中心用于EnKF先验，后续仍由EnKF在线更新参数",
            "Carter滤失上限从50%放宽为85%，但仍受累计注入量质量守恒约束",
            "加入压力依赖滤失弱修正和簇间应力阴影对导流分配的反馈",
        ],
    }
    (out / "comparison_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"out_dir": str(out), "comparison": payload["comparison"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
