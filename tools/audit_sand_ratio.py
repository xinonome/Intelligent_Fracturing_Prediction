"""Audit raw SB sand ratio against the HMI action sequence."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "Data" / "raw_frac" / "WITHfiltered2ASSELECTDISTINCTJTHJDFROMhagHAGMARKPOINTWHEREWORKIN_202511211131.xlsx"
HMI = ROOT / "outputs" / "hmi" / "bugfix_validation" / "ppo_20260812_184048" / "rl_evaluation.csv"
OUT = ROOT / "outputs" / "hmi" / "sand_ratio_audit"


def raw_segment(name: str) -> pd.DataFrame:
    columns = pd.read_excel(REF, nrows=0).columns.tolist()
    frame = pd.read_excel(ROOT / "Data" / "raw_frac" / name, header=None)
    frame.columns = columns
    frame["SB_numeric"] = pd.to_numeric(frame["SB"], errors="coerce")
    frame["PL_numeric"] = pd.to_numeric(frame["PL"], errors="coerce")
    return frame.dropna(subset=["SB_numeric"]).reset_index(drop=True)


def summary(values):
    series = pd.Series(values, dtype="float64").dropna()
    return {
        "count": int(series.size),
        "min_percent": float(series.min()),
        "max_percent": float(series.max()),
        "mean_percent": float(series.mean()),
        "median_percent": float(series.median()),
        "p10_percent": float(series.quantile(0.10)),
        "p90_percent": float(series.quantile(0.90)),
        "count_ge_14_percent": int((series >= 14.0).sum()),
        "count_eq_14_percent": int((series == 14.0).sum()),
    }


def main() -> None:
    hmi = pd.read_csv(HMI)
    hmi["decision_index"] = range(1, len(hmi) + 1)
    hmi_summary = {
        "current_sand_ratio": summary(hmi["pre_action_sand_ratio_percent"]),
        "recommended_sand_ratio": summary(hmi["sand_ratio_percent"]),
    }
    raw = {name: raw_segment(name) for name in ("FDBH15.xlsx", "FDBH18.xlsx")}
    raw_summary = {name: summary(frame["SB_numeric"]) for name, frame in raw.items()}

    OUT.mkdir(parents=True, exist_ok=True)
    audit = {
        "hmi_source": str(HMI.relative_to(ROOT)),
        "raw_sources": {name: str((ROOT / "Data" / "raw_frac" / name).relative_to(ROOT)) for name in raw},
        "hmi": hmi_summary,
        "raw": raw_summary,
        "interpretation": {
            "hmi_14_percent_is_action_limit": True,
            "raw_fdbh15_reaches_14_percent": raw_summary["FDBH15.xlsx"]["count_ge_14_percent"] > 0,
            "raw_fdbh18_reaches_14_percent": raw_summary["FDBH18.xlsx"]["count_ge_14_percent"] > 0,
            "note": "HMI推荐砂比是策略动作，不等同于原始施工SB；修复后推荐值以当前观测砂比为基准，仅允许保守微调，不自动跨入14%高砂比边界。",
        },
    }
    (OUT / "sand_ratio_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    hmi[["decision_index", "episode", "step", "segment_id", "pre_action_sand_ratio_percent", "sand_ratio_percent"]].to_csv(
        OUT / "hmi_sand_ratio_series.csv", index=False, encoding="utf-8-sig"
    )
    with (OUT / "raw_sand_ratio_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", *next(iter(raw_summary.values())).keys()])
        for name, values in raw_summary.items():
            writer.writerow([name, *values.values()])

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)
    ax = axes[0]
    ax.plot(hmi["decision_index"], hmi["pre_action_sand_ratio_percent"], label="HMI当前砂比", color="#20C7C2", linewidth=1.5)
    ax.plot(hmi["decision_index"], hmi["sand_ratio_percent"], label="HMI推荐/动作砂比", color="#4D9DE0", linewidth=1.5)
    for boundary in [60.5, 120.5, 180.5]:
        ax.axvline(boundary, color="#9EB2C1", linestyle="--", linewidth=0.8)
    ax.axhline(14, color="#E05252", linestyle=":", linewidth=1.5, label="14%动作上限")
    ax.set_title("HMI 240个决策点砂比变化")
    ax.set_xlabel("决策点")
    ax.set_ylabel("砂比 (%)")
    ax.set_xlim(1, len(hmi))
    ax.set_ylim(0, max(15, float(hmi[["pre_action_sand_ratio_percent", "sand_ratio_percent"]].max().max()) + 1))
    ax.grid(alpha=0.25)
    ax.legend(ncol=3)

    ax = axes[1]
    for name, frame, color in [("FDBH15 原始SB", raw["FDBH15.xlsx"], "#F2A93B"), ("FDBH18 原始SB", raw["FDBH18.xlsx"], "#7C3AED")]:
        ax.plot(range(1, len(frame) + 1), frame["SB_numeric"], label=name, linewidth=1.0, color=color, alpha=0.85)
    ax.axhline(14, color="#E05252", linestyle=":", linewidth=1.5, label="14%参考线")
    ax.set_title("原始数据 SB 砂比变化（当前HMI评估涉及的可追溯原始段）")
    ax.set_xlabel("原始采样点")
    ax.set_ylabel("SB (%)")
    ax.set_ylim(0, 16)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(OUT / "sand_ratio_audit.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"output": str(OUT), "hmi_rows": len(hmi), "raw": raw_summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
