"""Plot historical and agent action trajectories from an evaluation run."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd


def configure_font() -> None:
    for font_path in (
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
    ):
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
            plt.rcParams["axes.unicode_minus"] = False
            return


def plot(rl_path: Path, baseline_path: Path, output: Path, episode: int | None = None) -> None:
    rl = pd.read_csv(rl_path)
    baseline = pd.read_csv(baseline_path)
    if episode is None:
        common = sorted(set(rl["episode"].unique()) & set(baseline["episode"].unique()))
        if not common:
            raise ValueError("No common episode exists in the two evaluation files")
        episode = int(common[0])
    rl = rl[rl["episode"] == episode].sort_values("step").reset_index(drop=True)
    baseline = baseline[baseline["episode"] == episode].sort_values("step").reset_index(drop=True)
    if rl.empty or baseline.empty:
        raise ValueError(f"Episode {episode} is missing from one evaluation file")

    output.parent.mkdir(parents=True, exist_ok=True)
    x_rl = rl["step"].to_numpy(dtype=float) * 60.0
    x_base = baseline["step"].to_numpy(dtype=float) * 60.0
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle("智能体未来60秒动作轨迹：历史基线与安全约束后策略", fontsize=14)
    axes[0].plot(x_base, baseline["flow_m3_min"], color="#202124", lw=1.5, label="历史动作基线")
    axes[0].plot(x_rl, rl["flow_m3_min"], color="#e67e22", lw=1.7, label="策略动作（约束后）")
    axes[0].set_ylabel("排量 (m³/min)")
    axes[0].set_title("未来60秒排量建议")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")
    axes[1].plot(x_base, baseline["sand_ratio_percent"], color="#202124", lw=1.5, label="历史动作基线")
    axes[1].plot(x_rl, rl["sand_ratio_percent"], color="#e67e22", lw=1.7, label="策略动作（约束后）")
    axes[1].set_xlabel("相对起始时刻 (s)")
    axes[1].set_ylabel("砂比 (%)")
    axes[1].set_title("未来60秒砂比建议")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")
    scenario = str(rl["scenario_name"].iloc[0]) if "scenario_name" in rl else "unknown"
    segment = str(rl["segment_id"].iloc[0]) if "segment_id" in rl else "unknown"
    fig.text(0.01, 0.01, f"episode={episode}  segment={segment}  scenario={scenario}  |  策略曲线为泵序/安全边界裁剪后的最终动作", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot agent flow/sand action trajectories")
    parser.add_argument("--rl", required=True, help="rl_evaluation.csv")
    parser.add_argument("--baseline", required=True, help="historical_baseline_evaluation.csv")
    parser.add_argument("--output", required=True, help="output PNG")
    parser.add_argument("--episode", type=int, default=None)
    args = parser.parse_args()
    configure_font()
    plot(Path(args.rl), Path(args.baseline), Path(args.output), args.episode)
    print(Path(args.output).resolve())


if __name__ == "__main__":
    main()
