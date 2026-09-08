"""Plot the parameter trajectory and hard-domain ranges of an EnKF run.

The source run stores the prior/posterior *ensemble means* at each assimilation
node.  This tool deliberately does not call those curves an ensemble envelope:
the full member-wise covariance history is not persisted by the current run.
Hard limits are taken from ``clip_augmented_state`` in the same validation code.

Outputs are three PNG small-multiple charts, a long CSV for inspection, a JSON
diagnostic, a short Markdown note, and (optionally) a self-contained SVG HTML
fragment for an inline view.
"""

from __future__ import annotations

import argparse
import json
import math
from html import escape
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "outputs/dt/second_part_kg_enkf_20260824/20260824_004357"

# The Windows execution environment has Chinese fonts, but matplotlib does
# not always select them when a script is launched from a fresh interpreter.
# Register one explicitly so the exported expert-facing charts do not contain
# tofu/square glyphs.
_CJK_FONT = Path(r"C:\Windows\Fonts\simhei.ttf")
if _CJK_FONT.is_file():
    font_manager.fontManager.addfont(str(_CJK_FONT))
    _CJK_FONT_NAME = font_manager.FontProperties(fname=str(_CJK_FONT)).get_name()
    plt.rcParams["font.family"] = [_CJK_FONT_NAME, "DejaVu Sans"]
else:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _finite(values: pd.Series) -> np.ndarray:
    return values.to_numpy(dtype=float)


def _bounds_count(values: np.ndarray, lower: float | None, upper: float | None) -> dict[str, int]:
    if lower is None or upper is None:
        return {"lower_hits": 0, "upper_hits": 0, "out_of_bounds": 0}
    tol = max(abs(upper - lower) * 1.0e-6, 1.0e-12)
    return {
        "lower_hits": int(np.sum(values <= lower + tol)),
        "upper_hits": int(np.sum(values >= upper - tol)),
        "out_of_bounds": int(np.sum((values < lower - tol) | (values > upper + tol))),
    }


def _format_value(value: float) -> str:
    if abs(value) >= 100 or (0 < abs(value) < 0.01):
        return f"{value:.3g}"
    return f"{value:.3f}"


def _load(run_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    history_path = run_dir / "direct_observation_history.csv"
    summary_path = run_dir / "summary.json"
    if not history_path.is_file():
        raise FileNotFoundError(f"missing history: {history_path}")
    if not summary_path.is_file():
        raise FileNotFoundError(f"missing summary: {summary_path}")
    history = pd.read_csv(history_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    required = {"time_s", "phase"}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"history missing columns: {sorted(missing)}")
    return history, summary


def _descriptors(history: pd.DataFrame, summary: dict[str, Any]) -> list[dict[str, Any]]:
    cfg = summary.get("config", {})
    base_eprime = float(cfg.get("base_eprime_pa", 3.2e10))
    base_leakoff = float(cfg.get("base_leakoff_m_sqrt_s", 1.0e-5))
    base_mu = float(cfg.get("base_viscosity_pa_s", 0.1))
    base_stress = float(cfg.get("base_min_stress_mpa", 60.0))
    base_kic = float(cfg.get("base_fracture_toughness_pa_sqrt_m", 5.0e5))

    def physical(name: str, scale: float = 1.0) -> tuple[Callable[[str], np.ndarray], str, float, float]:
        return (
            lambda prefix: _finite(history[f"{prefix}_{name}"]) * scale,
            name,
            0.0,
            0.0,
        )

    items: list[dict[str, Any]] = [
        {
            "group": "pressure",
            "name": "E'",
            "unit": "GPa",
            "columns": "eprime_gpa",
            "transform": lambda prefix: _finite(history[f"{prefix}_eprime_gpa"]),
            "lower": 0.45 * base_eprime / 1.0e9,
            "upper": 2.20 * base_eprime / 1.0e9,
            "bound_basis": "clip_augmented_state: E'/E'_0 ∈ [0.45, 2.20]",
        },
        {
            "group": "pressure",
            "name": "C_L",
            "unit": "×10⁻⁶ m·s⁻¹ᐟ²",
            "columns": "leakoff_m_sqrt_s",
            "transform": lambda prefix: _finite(history[f"{prefix}_leakoff_m_sqrt_s"]) * 1.0e6,
            "lower": 0.10 * base_leakoff * 1.0e6,
            "upper": 8.00 * base_leakoff * 1.0e6,
            "bound_basis": "clip_augmented_state: C_L/C_L0 ∈ [0.10, 8.00]",
        },
        {
            "group": "pressure",
            "name": "μ",
            "unit": "Pa·s",
            "columns": "viscosity_pa_s",
            "transform": lambda prefix: _finite(history[f"{prefix}_viscosity_pa_s"]),
            "lower": 0.20 * base_mu,
            "upper": 5.00 * base_mu,
            "bound_basis": "clip_augmented_state: μ/μ0 ∈ [0.20, 5.00]",
        },
        {
            "group": "pressure",
            "name": "σ_min",
            "unit": "MPa",
            "columns": "min_stress_mpa",
            "transform": lambda prefix: _finite(history[f"{prefix}_min_stress_mpa"]),
            "lower": 35.0,
            "upper": 90.0,
            "bound_basis": "clip_augmented_state: σ_min ∈ [35, 90] MPa",
        },
        {
            "group": "pressure",
            "name": "K_IC",
            "unit": "×10⁻⁵ Pa·m¹ᐟ²",
            "columns": "fracture_toughness_pa_sqrt_m",
            "transform": lambda prefix: _finite(history[f"{prefix}_fracture_toughness_pa_sqrt_m"]) * 1.0e-5,
            "lower": 0.25 * base_kic * 1.0e-5,
            "upper": 4.00 * base_kic * 1.0e-5,
            "bound_basis": "clip_augmented_state: K_IC/K_IC0 ∈ [0.25, 4.00]",
        },
    ]
    for cluster in range(1, 7):
        items.append(
            {
                "group": "allocation",
                "name": f"κ_C{cluster}",
                "unit": "归一化进液能力因子",
                "columns": f"intake_capacity_factor_c{cluster}",
                "transform": lambda prefix, c=cluster: _finite(history[f"{prefix}_intake_capacity_factor_c{c}"]),
                "lower": None,
                "upper": None,
                "bound_basis": "模型输出为归一化 κ；硬 guard 作用于 raw log κ_Ci ∈ [-6, 6]，不是 κ 的直接上下限",
            }
        )
    items.extend(
        [
            {
                "group": "interaction",
                "name": "γ_sh",
                "unit": "应力阴影尺度",
                "columns": "stress_shadow_scale",
                "transform": lambda prefix: _finite(history[f"{prefix}_stress_shadow_scale"]),
                "lower": math.exp(-4.0),
                "upper": math.exp(4.0),
                "bound_basis": "raw log guard ∈ [-4, 4]，映射到 γ_sh ∈ [e⁻⁴, e⁴]",
            },
            {
                "group": "interaction",
                "name": "γ_e",
                "unit": "边界释放尺度",
                "columns": "boundary_relief_scale",
                "transform": lambda prefix: _finite(history[f"{prefix}_boundary_relief_scale"]),
                "lower": math.exp(-4.0),
                "upper": math.exp(4.0),
                "bound_basis": "raw log guard ∈ [-4, 4]，映射到 γ_e ∈ [e⁻⁴, e⁴]",
            },
            {
                "group": "interaction",
                "name": "α_a",
                "unit": "分配指数",
                "columns": "allocation_exponent",
                "transform": lambda prefix: _finite(history[f"{prefix}_allocation_exponent"]),
                "lower": math.exp(-4.0),
                "upper": math.exp(4.0),
                "bound_basis": "raw log guard ∈ [-4, 4]，映射到 α_a ∈ [e⁻⁴, e⁴]",
            },
        ]
    )
    return items


def _prepare(history: pd.DataFrame, items: list[dict[str, Any]]) -> tuple[dict[str, Any], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    for item in items:
        prior = item["transform"]("prior")
        posterior = item["transform"]("posterior")
        if len(prior) != len(history) or len(posterior) != len(history):
            raise ValueError(f"length mismatch for {item['name']}")
        if not np.isfinite(prior).all() or not np.isfinite(posterior).all():
            raise ValueError(f"NaN/Inf in {item['name']}")
        low, high = item["lower"], item["upper"]
        combined = np.r_[prior, posterior]
        counts = _bounds_count(combined, low, high)
        diag = {
            "group": item["group"],
            "unit": item["unit"],
            "min_prior": float(prior.min()),
            "max_prior": float(prior.max()),
            "min_posterior": float(posterior.min()),
            "max_posterior": float(posterior.max()),
            "first_prior": float(prior[0]),
            "last_prior": float(prior[-1]),
            "first_posterior": float(posterior[0]),
            "last_posterior": float(posterior[-1]),
            "lower": None if low is None else float(low),
            "upper": None if high is None else float(high),
            "bound_basis": item["bound_basis"],
            **counts,
        }
        diagnostics[item["name"]] = diag
        for time_s, phase, p, q in zip(history["time_s"], history["phase"], prior, posterior):
            rows.append(
                {
                    "group": item["group"],
                    "parameter": item["name"],
                    "unit": item["unit"],
                    "time_s": float(time_s),
                    "phase": str(phase),
                    "prior": float(p),
                    "posterior": float(q),
                    "lower": None if low is None else float(low),
                    "upper": None if high is None else float(high),
                    "bound_basis": item["bound_basis"],
                }
            )
    return diagnostics, pd.DataFrame(rows)


def _style_axes(ax: Any, t: np.ndarray, split: float | None) -> None:
    ax.grid(True, alpha=0.23, linewidth=0.7)
    ax.set_xlim(float(t.min()), float(t.max()))
    if split is not None and t.min() < split < t.max():
        ax.axvline(split, color="#d97706", linestyle="--", linewidth=1.0, alpha=0.9)
    ax.tick_params(labelsize=8)


def _plot_group(
    history: pd.DataFrame,
    items: list[dict[str, Any]],
    output: Path,
    split: float | None,
    title: str,
    subtitle: str,
    filename: str,
    log_y: bool = False,
) -> None:
    n = len(items)
    cols = 2 if n <= 5 else 3
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13.4, 3.05 * rows), squeeze=False)
    axes_flat = list(axes.ravel())
    t = history["time_s"].to_numpy(dtype=float)
    colors = plt.get_cmap("tab10").colors
    for idx, (ax, item) in enumerate(zip(axes_flat, items)):
        color = colors[idx % len(colors)]
        prior = item["transform"]("prior")
        posterior = item["transform"]("posterior")
        low, high = item["lower"], item["upper"]
        if log_y:
            ax.set_yscale("log")
        if low is not None and high is not None:
            ax.axhspan(low, high, color="#94a3b8", alpha=0.09, zorder=0)
            ax.axhline(low, color="#64748b", linestyle=":", linewidth=0.9, label="下限")
            ax.axhline(high, color="#64748b", linestyle=":", linewidth=0.9, label="上限")
        ax.plot(t, prior, color=color, linewidth=1.25, label="先验（实线）")
        ax.plot(t, posterior, color=color, linewidth=1.25, linestyle="--", label="后验（虚线）")
        ax.axhline(1.0, color="#9ca3af", linewidth=0.8, alpha=0.8)
        _style_axes(ax, t, split)
        ax.set_title(item["name"], loc="left", fontsize=11, fontweight="bold")
        ax.set_ylabel(item["unit"], fontsize=8)
        ax.set_xlabel("时间 / s", fontsize=8)
        ax.text(
            0.99,
            0.03,
            f"范围 { _format_value(min(prior.min(), posterior.min())) }–{ _format_value(max(prior.max(), posterior.max())) }",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.5,
            color="#475569",
        )
    for ax in axes_flat[n:]:
        ax.axis("off")
    handles = [
        plt.Line2D([0], [0], color="#334155", linewidth=1.3, label="先验（实线）"),
        plt.Line2D([0], [0], color="#334155", linewidth=1.3, linestyle="--", label="后验（虚线）"),
        plt.Line2D([0], [0], color="#64748b", linewidth=0.9, linestyle=":", label="硬上下限（适用时）"),
    ]
    fig.legend(handles=handles, loc="upper right", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.99, 0.985))
    fig.suptitle(title, x=0.02, y=0.995, ha="left", fontsize=16, fontweight="bold")
    fig.text(0.02, 0.962, subtitle, ha="left", va="top", fontsize=9, color="#475569")
    fig.tight_layout(rect=(0, 0, 1, 0.935))
    fig.savefig(output / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _svg_path(t: np.ndarray, values: np.ndarray, x0: float, y0: float, width: float, height: float, ymin: float, ymax: float, log_y: bool) -> str:
    def tx(v: float) -> float:
        return x0 + (v - float(t.min())) / max(float(t.max() - t.min()), 1.0e-9) * width

    def ty(v: float) -> float:
        if log_y:
            a, b = math.log(max(ymin, 1.0e-12)), math.log(max(ymax, 1.0e-12))
            z = (math.log(max(float(v), 1.0e-12)) - a) / max(b - a, 1.0e-12)
        else:
            z = (float(v) - ymin) / max(ymax - ymin, 1.0e-12)
        return y0 + height - z * height

    points = [f"{tx(float(a)):.2f},{ty(float(b)):.2f}" for a, b in zip(t, values)]
    return "M " + " L ".join(points)


def _svg_panel(t: np.ndarray, item: dict[str, Any], x: float, y: float, w: float, h: float, color_index: int, log_y: bool) -> str:
    prior = item["transform"]("prior")
    posterior = item["transform"]("posterior")
    low, high = item["lower"], item["upper"]
    combined = np.r_[prior, posterior]
    if low is not None and high is not None:
        ymin, ymax = low, high
        pad = 0.08 if not log_y else 0.0
        if not log_y:
            ymin -= (ymax - ymin) * pad
            ymax += (ymax - ymin) * pad
    else:
        if log_y:
            ymin = max(float(combined.min()) * 0.65, 1.0e-3)
            ymax = float(combined.max()) * 1.55
        else:
            span = max(float(combined.max() - combined.min()), 0.1)
            ymin = float(combined.min()) - 0.12 * span
            ymax = float(combined.max()) + 0.12 * span
    px, py, pw, ph = x + 38, y + 25, w - 48, h - 50
    color = f"var(--viz-series-{(color_index % 6) + 1})"
    parts = [f'<g aria-label="{escape(item["name"])}">']
    parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="none" stroke="var(--border)" />')
    parts.append(f'<text x="{x+8:.1f}" y="{y+16:.1f}" class="panel-title">{escape(item["name"])}</text>')
    for frac in (0.0, 0.5, 1.0):
        yy = py + ph * (1 - frac)
        parts.append(f'<line x1="{px:.1f}" y1="{yy:.1f}" x2="{px+pw:.1f}" y2="{yy:.1f}" class="grid" />')
        val = (ymin + frac * (ymax - ymin)) if not log_y else math.exp(math.log(ymin) + frac * (math.log(ymax) - math.log(ymin)))
        parts.append(f'<text x="{px-5:.1f}" y="{yy+3:.1f}" text-anchor="end" class="tick">{escape(_format_value(val))}</text>')
    if low is not None and high is not None:
        def map_y(v: float) -> float:
            if log_y:
                z = (math.log(max(v, 1.0e-12)) - math.log(max(ymin, 1.0e-12))) / max(math.log(max(ymax, 1.0e-12)) - math.log(max(ymin, 1.0e-12)), 1.0e-12)
            else:
                z = (v-ymin)/max(ymax-ymin, 1.0e-12)
            return py+ph*(1-z)
        for val in (low, high):
            yy = map_y(float(val))
            parts.append(f'<line x1="{px:.1f}" y1="{yy:.1f}" x2="{px+pw:.1f}" y2="{yy:.1f}" class="bound" />')
    parts.append(f'<path d="{_svg_path(t, prior, px, py, pw, ph, ymin, ymax, log_y)}" fill="none" stroke="{color}" class="prior" />')
    parts.append(f'<path d="{_svg_path(t, posterior, px, py, pw, ph, ymin, ymax, log_y)}" fill="none" stroke="{color}" class="posterior" />')
    parts.append(f'<text x="{px:.1f}" y="{y+h-8:.1f}" class="tick">{float(t.min()):.0f}s</text>')
    parts.append(f'<text x="{px+pw:.1f}" y="{y+h-8:.1f}" text-anchor="end" class="tick">{float(t.max()):.0f}s</text>')
    parts.append("</g>")
    return "".join(parts)


def _write_html(path: Path, history: pd.DataFrame, groups: dict[str, list[dict[str, Any]]], split: float | None) -> None:
    t = history["time_s"].to_numpy(dtype=float)
    width = 900
    sections: list[str] = []
    specs = [
        ("pressure", "压力/裂缝物理参数", "实线=先验均值；虚线=后验均值；虚线灰线=适用的绝对域上下限。", False, 2),
        ("allocation", "六簇进液能力参数", "κ 为模型使用的归一化进液能力因子；raw log κ 的数值域为 [-6, 6]。", True, 3),
        ("interaction", "簇间影响与分配参数", "应力阴影、边界释放和分配指数；raw log 参数的数值域为 [-4, 4]。", True, 3),
    ]
    for key, title, subtitle, log_y, cols in specs:
        items = groups[key]
        rows = int(math.ceil(len(items) / cols))
        panel_w = (width - 20 * (cols - 1)) / cols
        panel_h = 190
        height = 60 + rows * (panel_h + 12)
        panels = []
        for i, item in enumerate(items):
            row, col = divmod(i, cols)
            panels.append(_svg_panel(t, item, col * (panel_w + 20), 45 + row * (panel_h + 12), panel_w, panel_h, i, log_y))
        svg = f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}参数更新轨迹"><title>{escape(title)}</title><desc>{escape(subtitle)}</desc>{"".join(panels)}</svg>'
        sections.append(f'<section class="enkf-section"><h2>{escape(title)}</h2><p>{escape(subtitle)}</p>{svg}</section>')
    css = """
    <style>
      #enkf-parameter-ranges-visual { color: var(--foreground); font-family: inherit; }
      #enkf-parameter-ranges-visual .enkf-section { margin: 0 0 1rem; }
      #enkf-parameter-ranges-visual h2 { margin: 0 0 .2rem; font-size: 1rem; font-weight: 500; }
      #enkf-parameter-ranges-visual p { margin: 0 0 .3rem; color: var(--muted-foreground); font-size: .82rem; }
      #enkf-parameter-ranges-visual svg { display: block; width: 100%; height: auto; overflow: visible; }
      #enkf-parameter-ranges-visual .panel-title { fill: var(--foreground); font-size: 12px; }
      #enkf-parameter-ranges-visual .tick { fill: var(--muted-foreground); font-size: 9px; }
      #enkf-parameter-ranges-visual .grid { stroke: var(--border); stroke-width: 1; opacity: .7; }
      #enkf-parameter-ranges-visual .bound { stroke: var(--muted-foreground); stroke-width: 1; stroke-dasharray: 3 3; opacity: .9; }
      #enkf-parameter-ranges-visual .prior { stroke-width: 1.6; }
      #enkf-parameter-ranges-visual .posterior { stroke-width: 1.6; stroke-dasharray: 5 3; }
    </style>
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '<div id="enkf-parameter-ranges-visual" role="figure" aria-label="EnKF 三类参数全流程更新范围">'
        + css
        + "".join(sections)
        + "</div>\n",
        encoding="utf-8",
    )


def _write_report(path: Path, run_dir: Path, history: pd.DataFrame, diagnostics: dict[str, Any], output: Path) -> None:
    valid = history["phase"].astype(str).eq("validation")
    split = float(history.loc[valid, "time_s"].min()) if valid.any() else None
    lines = [
        "# EnKF 三类参数全流程更新范围诊断",
        "",
        f"数据来源：`{run_dir.as_posix()}` 的 `direct_observation_history.csv`。",
        f"共 `{len(history)}` 个更新节点，时间范围 `{float(history.time_s.min()):.0f}–{float(history.time_s.max()):.0f} s`；校准 `{int((~valid).sum())}` 个节点，验证 `{int(valid.sum())}` 个节点。",
        "",
        "图中每条曲线是该更新节点的 EnKF ensemble 均值：实线为先验，虚线为后验。当前运行没有保存每个 ensemble 成员在每个节点的完整历史，因此图中不把均值曲线误称为 ensemble min–max 包络。",
        "",
        "## 专家建议",
        "",
        "- 先区分两类边界：图中的硬上下限是防止数值溢出的绝对状态域，不是单步参数变化限制；当前流程没有单步 delta 限幅。",
        "- 压力参数应重点检查接近边界的参数是否具有现场可辨识性；若长期贴近边界，应优先复核压力换算、观测噪声和参数敏感性，不宜直接放宽边界。",
        "- 六簇 κ 是归一化进液能力因子，不等于现场可独立控制的簇级排量；异常大的 κ 只能说明模型在当前观测下需要更强的相对进液能力，应结合分簇解释质量和井段几何复核。",
        "- γ_sh、γ_e、α_a 的 raw log guard 映射到很宽的物理域；若曲线长期接近 1 且不贴边，当前数据对这些参数的约束仍然偏弱，后续可做敏感性/可辨识性分析。",
        "",
        "## 参数统计",
        "",
        "| 类别 | 参数 | 后验范围 | 硬下限 | 硬上限 | 边界状态 |",
        "|---|---|---:|---:|---:|---|",
    ]
    for name, item in diagnostics.items():
        low = "—" if item["lower"] is None else _format_value(item["lower"])
        high = "—" if item["upper"] is None else _format_value(item["upper"])
        if item["lower"] is None:
            status = "raw log guard；无直接 κ 上下限"
        elif item["out_of_bounds"]:
            status = "越界：需排查"
        elif item["lower_hits"] or item["upper_hits"]:
            status = f"触边（下{item['lower_hits']} / 上{item['upper_hits']}）"
        else:
            status = "未触边"
        lines.append(
            f"| {item['group']} | `{name}` | {_format_value(item['min_posterior'])}–{_format_value(item['max_posterior'])} | {low} | {high} | {status} |"
        )
    lines.extend(
        [
            "",
            "## 文件说明",
            "",
            f"- 三张图：`{(output / 'enkf_parameter_ranges_pressure.png').name}`、`{(output / 'enkf_parameter_ranges_allocation.png').name}`、`{(output / 'enkf_parameter_ranges_interaction.png').name}`。",
            f"- 机器可读明细：`{(output / 'parameter_range_history.csv').name}`、`{(output / 'parameter_range_diagnostics.json').name}`。",
            "- 读取的是现有 KG-EnKF 有 DAS 全流程回放的先验/后验均值，不包含伪造数据。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--html-out", type=Path, default=None)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    output = (args.output_dir or (run_dir / "parameter_range_diagnostics")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    history, summary = _load(run_dir)
    items = _descriptors(history, summary)
    diagnostics, long_frame = _prepare(history, items)
    long_frame.to_csv(output / "parameter_range_history.csv", index=False, encoding="utf-8-sig")
    valid = history["phase"].astype(str).eq("validation")
    split = float(history.loc[valid, "time_s"].min()) if valid.any() else None
    group_map = {
        "pressure": [item for item in items if item["group"] == "pressure"],
        "allocation": [item for item in items if item["group"] == "allocation"],
        "interaction": [item for item in items if item["group"] == "interaction"],
    }
    _plot_group(history, group_map["pressure"], output, split, "EnKF 参数更新范围：压力/裂缝物理参数", "实线=先验均值，虚线=后验均值；灰色虚线=clip_augmented_state 的绝对域上下限。", "enkf_parameter_ranges_pressure.png")
    _plot_group(history, group_map["allocation"], output, split, "EnKF 参数更新范围：六簇进液能力", "κ 为模型使用的归一化进液能力因子；raw log κ 的数值域为 [-6, 6]，不把它误画成 κ 的直接上下限。", "enkf_parameter_ranges_allocation.png", log_y=True)
    _plot_group(history, group_map["interaction"], output, split, "EnKF 参数更新范围：簇间影响与分配", "γ_sh、γ_e、α_a 的 raw log 数值域为 [-4, 4]；物理值采用对数坐标显示。", "enkf_parameter_ranges_interaction.png", log_y=True)
    metadata = {
        "source_run": str(run_dir),
        "history_file": str(run_dir / "direct_observation_history.csv"),
        "update_nodes": int(len(history)),
        "time_start_s": float(history.time_s.min()),
        "time_end_s": float(history.time_s.max()),
        "calibration_nodes": int((~valid).sum()),
        "validation_nodes": int(valid.sum()),
        "validation_start_s": split,
        "curves_are_ensemble_means": True,
        "full_member_envelope_saved": False,
        "diagnostics": diagnostics,
        "outputs": {
            "pressure_png": str(output / "enkf_parameter_ranges_pressure.png"),
            "allocation_png": str(output / "enkf_parameter_ranges_allocation.png"),
            "interaction_png": str(output / "enkf_parameter_ranges_interaction.png"),
            "history_csv": str(output / "parameter_range_history.csv"),
            "report_md": str(output / "ENKF参数上下限更新范围诊断.md"),
        },
    }
    (output / "parameter_range_diagnostics.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(output / "ENKF参数上下限更新范围诊断.md", run_dir, history, diagnostics, output)
    if args.html_out:
        _write_html(args.html_out.resolve(), history, group_map, split)
    print(json.dumps(metadata["outputs"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
