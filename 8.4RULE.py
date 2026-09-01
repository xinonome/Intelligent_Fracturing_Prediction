# -*- coding: utf-8 -*-
"""
压裂施工小条统计 + 高级绘图（增强版）
- 地层破裂检测覆盖前置液、加砂台阶、隔离液
- 砂堵检测排除排量上升阶段
- 加砂台阶支撑剂类型从 ZCJLX 列提取
- 停泵后压力记录取最后一个有效点，并记录时间差（min）及说明
- 总体统计独立生成汇总 Excel 文件（每个井段一行）
- 曲线类型判断采用斜率法（分段线性回归），抗砂堵扰动
- 绘图显示台阶代表压力拟合直线/折线（与判断逻辑完全统一）
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    matplotlib = None

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Patch
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("警告：matplotlib未安装，将跳过绘图。")

# ==================== 输出列定义 ====================
OUTPUT_COLUMNS = [
    "来源文件", "段号", "小条序号", "小条类型", "砂比台阶",
    "开始数据序号", "结束数据序号", "数据点数",
    "开始时间", "结束时间", "最大压力", "最大排量",
    "总液量", "总砂量", "压力上升值", "压力下降值",
    "台阶代表压力", "加砂曲线判断", "顶替液量",
    "停泵压力", "停泵数据序号", "停泵时间",
    "停泵30min压力", "停泵30min数据序号", "停泵30min时间", "停泵30min压降",
    "原始SB均值", "原始SB最小值", "原始SB最大值",
    "砂堵总砂量", "砂堵最高砂比", "砂堵压力斜率",
    "暂堵最大压力差", "暂堵稳定压力差",
    "支撑剂类型",
    "停泵后时间(min)",
    "停泵说明",
]

WARNING_COLUMNS = [
    "来源文件", "段号",
    "开始数据序号", "结束数据序号",
    "开始时间", "结束时间",
    "预警类型", "压力变化(MPa)",
    "排量标准差", "砂比标准差",
    "波峰递增", "波动幅度增大", "描述",
]

FRACTURE_COLUMNS = [
    "来源文件", "段号", "破裂序号",
    "开始数据序号", "结束数据序号",
    "开始时间", "结束时间",
    "破裂压力(MPa)", "压力下降范围(MPa)", "压力下降斜率(MPa/min)",
]

OVERALL_STAT_COLUMNS = [
    "来源文件", "井段", "破裂压力(MPa)", "停泵压力(MPa)",
    "停泵压降(MPa)", "压降时间(min)", "总液量(m³)", "总砂量(m³)",
]

# ==================== 列名别名 ====================
COLUMN_ALIASES = {
    "sand_ratio": ("SB", "砂比", "沙比", "砂比(%)", "砂比%", "SAND_RATIO", "sand_ratio"),
    "pressure": ("SGBY", "施工泵压", "泵压", "压力", "PRESSURE"),
    "rate": ("PL", "排量", "流量", "RATE"),
    "cumulative_liquid": ("LJYL", "累计液量", "累计液体", "总液量", "CUM_LIQUID"),
    "cumulative_sand": ("LJSL", "累计砂量", "累计砂", "总砂量", "CUM_SAND"),
    "date": ("SGRQ", "施工日期", "日期", "DATE"),
    "time": ("SGSJ", "施工时间", "时间", "TIME"),
    "working_type": ("WORKING_TYPE", "工况类型", "工况", "ZDW_LB"),
    "liquid_type": ("ZCJLX", "支撑剂类型", "ZCJLX", "ProppantType"),
}

# ==================== 通用工具函数 ====================
def to_number(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip().str.replace("%", "", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")

def find_column(df: pd.DataFrame, field: str, required: bool = False) -> str | None:
    cols_lower = {col.strip().lower(): col for col in df.columns}
    for alias in COLUMN_ALIASES[field]:
        alias_clean = alias.strip().lower()
        if alias_clean in cols_lower:
            return cols_lower[alias_clean]
    if required:
        expected = " / ".join(COLUMN_ALIASES[field])
        raise KeyError(f"缺少字段 {field}，可接受列名：{expected}")
    return None

def resolve_columns(df: pd.DataFrame) -> dict[str, str | None]:
    return {
        "sand_ratio": find_column(df, "sand_ratio", required=True),
        "pressure": find_column(df, "pressure", required=True),
        "rate": find_column(df, "rate", required=True),
        "cumulative_liquid": find_column(df, "cumulative_liquid", required=True),
        "cumulative_sand": find_column(df, "cumulative_sand", required=True),
        "date": find_column(df, "date"),
        "time": find_column(df, "time"),
        "working_type": find_column(df, "working_type"),
        "liquid_type": find_column(df, "liquid_type"),
    }

def round_sand_ratio_to_step(value: Any) -> int | None:
    if pd.isna(value) or float(value) <= 0:
        return None
    return int(math.floor(float(value) + 0.5))

def round_value(value: Any, digits: int = 3) -> float | Any:
    if value is None or pd.isna(value):
        return pd.NA
    return round(float(value), digits)

def make_datetime_text(df: pd.DataFrame, row_index: int, columns: dict[str, str | None]) -> str | Any:
    date_col = columns["date"]
    time_col = columns["time"]
    if date_col is None or time_col is None:
        return pd.NA
    date_val = df.at[row_index, date_col]
    time_val = df.at[row_index, time_col]
    date_part = pd.to_datetime(date_val, errors="coerce")
    time_part = pd.to_datetime(time_val, errors="coerce")
    if pd.isna(date_part) or pd.isna(time_part):
        return pd.NA
    combined = date_part.normalize() + pd.Timedelta(hours=time_part.hour, minutes=time_part.minute, seconds=time_part.second)
    return combined.strftime("%Y-%m-%d %H:%M:%S")

def make_timestamp(df: pd.DataFrame, row_index: int, columns: dict[str, str | None]) -> pd.Timestamp | Any:
    date_col = columns["date"]
    time_col = columns["time"]
    if date_col is None or time_col is None:
        return pd.NA
    date_val = df.at[row_index, date_col]
    time_val = df.at[row_index, time_col]
    date_part = pd.to_datetime(date_val, errors="coerce")
    time_part = pd.to_datetime(time_val, errors="coerce")
    if pd.isna(date_part) or pd.isna(time_part):
        return pd.NA
    return date_part.normalize() + pd.Timedelta(hours=time_part.hour, minutes=time_part.minute, seconds=time_part.second)

def last_valid_number(df: pd.DataFrame, column: str, start: int, end: int) -> float | None:
    values = to_number(df.iloc[start: end + 1][column]).dropna()
    return None if values.empty else float(values.iloc[-1])

def cumulative_increment(df: pd.DataFrame, start: int, end: int, column: str) -> float | Any:
    end_value = last_valid_number(df, column, start, end)
    if end_value is None:
        return pd.NA
    previous_values = to_number(df.iloc[:start][column]).dropna()
    baseline = 0.0 if previous_values.empty else float(previous_values.iloc[-1])
    return round(end_value - baseline, 3)

def first_last_pressure(df: pd.DataFrame, start: int, end: int, pressure_col: str) -> tuple[float | None, float | None]:
    values = to_number(df.iloc[start: end + 1][pressure_col]).dropna()
    if values.empty:
        return None, None
    return float(values.iloc[0]), float(values.iloc[-1])

def empty_record() -> dict[str, Any]:
    return {column: pd.NA for column in OUTPUT_COLUMNS}

def make_base_record(df, columns, source_file, sheet_name, sequence, row_type, start, end):
    record = empty_record()
    record.update({
        "来源文件": source_file,
        "段号": str(sheet_name).strip(),
        "小条序号": sequence,
        "小条类型": row_type,
        "开始数据序号": start + 1,
        "结束数据序号": end + 1,
        "数据点数": end - start + 1,
        "开始时间": make_datetime_text(df, start, columns),
        "结束时间": make_datetime_text(df, end, columns),
    })
    return record

# ==================== 曲线类型判断（斜率法 + 抗扰动） ====================
def classify_curve(representative_pressures: list[Any], threshold: float) -> str | Any:
    """
    基于斜率法的压裂加砂曲线类型判断（增强抗扰动版）
    类型：上升、下降、平稳、V字型
    参数 threshold：用户设定的压力变化阈值（MPa），用于控制敏感度
    """
    values = [float(v) for v in representative_pressures if not pd.isna(v)]
    if len(values) < 2:
        return "平稳" if values else pd.NA

    OUTLIER_THRESH = 2.0
    # 斜率判定阈值缩小一半，改为 threshold / 6.0
    SLOPE_THRESH = threshold / 6.0
    V_DROP_THRESH = threshold
    V_RISE_THRESH = threshold
    V_FINAL_COND = 0.2

    p = np.array(values, dtype=float)
    p_clean = p.copy()
    n = len(p)
    for i in range(1, n - 1):
        if abs(p[i] - p[i - 1]) > OUTLIER_THRESH and abs(p[i + 1] - p[i - 1]) < 0.5:
            p_clean[i] = (p[i - 1] + p[i + 1]) / 2.0

    x = np.arange(len(p_clean))

    if len(p_clean) >= 2:
        k_all = np.polyfit(x, p_clean, 1)[0]
    else:
        k_all = 0.0

    if k_all > SLOPE_THRESH:
        return "上升"
    elif k_all < -SLOPE_THRESH:
        return "下降"

    min_idx = np.argmin(p_clean)
    if min_idx > 0 and min_idx < len(p_clean) - 1 and min_idx >= 1 and (len(p_clean) - min_idx) >= 2:
        if min_idx + 1 >= 2:
            k_left = np.polyfit(x[:min_idx + 1], p_clean[:min_idx + 1], 1)[0]
        else:
            k_left = 0.0
        if (len(p_clean) - min_idx) >= 2:
            k_right = np.polyfit(x[min_idx:], p_clean[min_idx:], 1)[0]
        else:
            k_right = 0.0

        is_v = (
            k_left < -SLOPE_THRESH and
            k_right > SLOPE_THRESH and
            (p_clean[0] - p_clean[min_idx]) >= V_DROP_THRESH and
            (p_clean[-1] - p_clean[min_idx]) >= V_RISE_THRESH and
            p_clean[-1] >= p_clean[0] - V_FINAL_COND
        )
        if is_v:
            return "V字型"

    return "平稳"

# ==================== 各类型记录填充 ====================
def fill_prepad_record(df, record, start, end, columns):
    block = df.iloc[start:end+1]
    pressures = to_number(block[columns["pressure"]]).dropna()
    rates = to_number(block[columns["rate"]]).dropna()
    if not pressures.empty:
        record["最大压力"] = round(float(pressures.max()), 3)
    if not rates.empty:
        record["最大排量"] = round(float(rates.max()), 3)

def fill_sand_step_record(df, record, start, end, step, columns):
    block = df.iloc[start:end+1]
    pressure_col = columns["pressure"]
    sand_ratio_col = columns["sand_ratio"]
    record["砂比台阶"] = step
    record["总液量"] = cumulative_increment(df, start, end, columns["cumulative_liquid"])
    record["总砂量"] = cumulative_increment(df, start, end, columns["cumulative_sand"])
    sp, ep = first_last_pressure(df, start, end, pressure_col)
    if sp is not None and ep is not None:
        record["压力下降值"] = round(max(sp - ep, 0.0), 3)
    pressures = to_number(block[pressure_col]).dropna()
    if not pressures.empty:
        record["台阶代表压力"] = round(float(pressures.mean()), 3)
    raw = to_number(block[sand_ratio_col]).dropna()
    if not raw.empty:
        record["原始SB均值"] = round(float(raw.mean()), 3)
        record["原始SB最小值"] = round(float(raw.min()), 3)
        record["原始SB最大值"] = round(float(raw.max()), 3)
    liquid_col = columns.get("liquid_type")
    if liquid_col is not None and liquid_col in df.columns:
        types = df.iloc[start:end+1][liquid_col].astype(str).str.strip()
        types = types[types.notna() & (types != "nan") & (types != "")]
        if not types.empty:
            mode_series = types.mode()
            record["支撑剂类型"] = mode_series.iloc[0] if not mode_series.empty else types.iloc[0]

def fill_isolation_record(df, record, start, end, columns):
    record["总液量"] = cumulative_increment(df, start, end, columns["cumulative_liquid"])
    sp, ep = first_last_pressure(df, start, end, columns["pressure"])
    if sp is not None and ep is not None:
        record["压力上升值"] = round(max(ep - sp, 0.0), 3)

def fill_displacement_stop_record(df, record, start, end, columns):
    pressure_col = columns["pressure"]
    rate_col = columns["rate"]
    record["顶替液量"] = cumulative_increment(df, start, end, columns["cumulative_liquid"])
    pressures = to_number(df.iloc[start:end+1][pressure_col]).dropna()
    if not pressures.empty:
        record["最大压力"] = round(float(pressures.max()), 3)
    rates = to_number(df.iloc[start:end+1][rate_col])
    stop_positions = np.where((rates.fillna(np.inf) <= 0).to_numpy())[0]
    if len(stop_positions) == 0:
        return
    stop_idx = start + int(stop_positions[0])
    stop_pressure = to_number(df[pressure_col]).iloc[stop_idx]
    record["停泵压力"] = round_value(stop_pressure)
    record["停泵数据序号"] = stop_idx + 1
    record["停泵时间"] = make_datetime_text(df, stop_idx, columns)
    stop_ts = make_timestamp(df, stop_idx, columns)
    if pd.isna(stop_ts):
        return

    last_valid_idx = None
    last_p = None
    last_ts = None
    for i in range(stop_idx + 1, len(df)):
        p_val = to_number(df[pressure_col]).iloc[i]
        if not pd.isna(p_val):
            last_p = p_val
            last_valid_idx = i
            last_ts = make_timestamp(df, i, columns)
    if last_valid_idx is not None and not pd.isna(last_p):
        if not pd.isna(last_ts):
            time_diff_min = (last_ts - stop_ts).total_seconds() / 60.0
        else:
            time_diff_min = pd.NA
        record["停泵30min压力"] = round_value(last_p)
        record["停泵30min数据序号"] = last_valid_idx + 1
        record["停泵30min时间"] = make_datetime_text(df, last_valid_idx, columns)
        if not pd.isna(stop_pressure) and not pd.isna(last_p):
            record["停泵30min压降"] = round(float(stop_pressure) - float(last_p), 3)
        record["停泵后时间(min)"] = round(time_diff_min, 1) if not pd.isna(time_diff_min) else pd.NA
        if not pd.isna(time_diff_min):
            if time_diff_min >= 30:
                record["停泵说明"] = "停泵30min"
            else:
                record["停泵说明"] = f"停泵后{round(time_diff_min, 1)}分钟"
        else:
            record["停泵说明"] = pd.NA
    else:
        record["停泵30min压力"] = pd.NA
        record["停泵30min数据序号"] = pd.NA
        record["停泵30min时间"] = pd.NA
        record["停泵30min压降"] = pd.NA
        record["停泵后时间(min)"] = pd.NA
        record["停泵说明"] = "无有效后续压力数据"

def fill_work_type_record(df, record, start, end, columns, work_type):
    pressure_col = columns["pressure"]
    sand_ratio_col = columns["sand_ratio"]
    sand_col = columns["cumulative_sand"]
    pressures = to_number(df.iloc[start:end+1][pressure_col]).dropna()
    sand_ratios = to_number(df.iloc[start:end+1][sand_ratio_col]).dropna()
    if work_type == "砂堵":
        record["砂堵总砂量"] = cumulative_increment(df, start, end, sand_col)
        if not sand_ratios.empty:
            record["砂堵最高砂比"] = round(float(sand_ratios.max()), 3)
        if len(pressures) >= 2:
            st = make_timestamp(df, start, columns)
            et = make_timestamp(df, end, columns)
            if not pd.isna(st) and not pd.isna(et):
                dt = (et - st).total_seconds()
                if dt > 0:
                    dp = float(pressures.iloc[-1]) - float(pressures.iloc[0])
                    record["砂堵压力斜率"] = round(dp / dt, 6)
    elif work_type == "暂堵":
        stable_before = pd.NA
        if start > 0:
            prev = to_number(df.iloc[:start][pressure_col]).dropna()
            if not prev.empty:
                stable_before = float(prev.iloc[-1])
        stable_after = pd.NA
        if end < len(df) - 1:
            nxt = to_number(df.iloc[end+1:][pressure_col]).dropna()
            if not nxt.empty:
                stable_after = float(nxt.iloc[0])
        max_p = pressures.max() if not pressures.empty else pd.NA
        if not pd.isna(max_p) and not pd.isna(stable_before):
            record["暂堵最大压力差"] = round(float(max_p) - stable_before, 3)
        if not pd.isna(stable_after) and not pd.isna(stable_before):
            record["暂堵稳定压力差"] = round(stable_after - stable_before, 3)

# ==================== 时间顺序修正 ====================
def ensure_chronological_order(df: pd.DataFrame, columns: dict[str, str | None]) -> tuple[pd.DataFrame, bool]:
    date_col = columns.get("date")
    time_col = columns.get("time")
    if date_col is None or time_col is None:
        return df, False
    timestamps = []
    for idx in range(len(df)):
        ts = make_timestamp(df, idx, columns)
        timestamps.append(ts)
    ts_series = pd.Series(timestamps)
    valid = ts_series.notna()
    if valid.sum() < 2:
        return df, False
    first_valid = ts_series[valid].iloc[0]
    last_valid = ts_series[valid].iloc[-1]
    if last_valid < first_valid:
        df_temp = df.copy()
        df_temp['_timestamp'] = ts_series
        df_sorted = df_temp.sort_values('_timestamp').drop(columns=['_timestamp']).reset_index(drop=True)
        liq_col = columns.get("cumulative_liquid")
        if liq_col is not None and liq_col in df_sorted.columns:
            liq_vals = to_number(df_sorted[liq_col])
            valid_mask = liq_vals.notna()
            valid_indices = liq_vals[valid_mask].index
            if len(valid_indices) > 1:
                new_liq = pd.Series(index=df_sorted.index, dtype=float)
                prev_val = liq_vals.loc[valid_indices[0]]
                new_liq.loc[valid_indices[0]] = prev_val
                for i in range(1, len(valid_indices)):
                    idx_cur = valid_indices[i]
                    cur_orig = liq_vals.loc[idx_cur]
                    delta = abs(cur_orig - prev_val)
                    new_val = prev_val + delta
                    new_liq.loc[idx_cur] = new_val
                    prev_val = new_val
                df_sorted[liq_col] = new_liq
        sand_col = columns.get("cumulative_sand")
        if sand_col is not None and sand_col in df_sorted.columns:
            sand_vals = to_number(df_sorted[sand_col])
            valid_mask = sand_vals.notna()
            valid_indices = sand_vals[valid_mask].index
            if len(valid_indices) > 1:
                new_sand = pd.Series(index=df_sorted.index, dtype=float)
                prev_val = sand_vals.loc[valid_indices[0]]
                new_sand.loc[valid_indices[0]] = prev_val
                for i in range(1, len(valid_indices)):
                    idx_cur = valid_indices[i]
                    cur_orig = sand_vals.loc[idx_cur]
                    delta = abs(cur_orig - prev_val)
                    new_val = prev_val + delta
                    new_sand.loc[idx_cur] = new_val
                    prev_val = new_val
                df_sorted[sand_col] = new_sand
        return df_sorted, True
    else:
        return df, False

# ==================== 砂堵预警检测 ====================
def detect_sand_warnings(
    df, columns, source_file, sheet_name,
    sand_start=None, sand_end=None,
    time_window_sec=60.0,
    q_std_threshold=0.2,
    s_std_threshold=1.0,
    pressure_rise_threshold=0.5,
    pressure_drop_threshold=5.0,
) -> list[dict[str, Any]]:
    warnings = []
    if df.empty or sand_start is None or sand_end is None or sand_start > sand_end:
        return warnings

    pressure_series = to_number(df[columns["pressure"]])
    rate_series = to_number(df[columns["rate"]])
    sand_series = to_number(df[columns["sand_ratio"]])

    timestamps = [make_timestamp(df, i, columns) for i in range(len(df))]
    ts_series = pd.Series(timestamps)
    valid = ts_series.notna()
    if valid.sum() < 2:
        return warnings
    diffs = ts_series[valid].diff().dropna()
    if len(diffs) == 0:
        return warnings
    dt_median = diffs.median().total_seconds()
    if dt_median <= 0:
        return warnings

    window_points = int(time_window_sec / dt_median) + 1
    n = len(df)
    if n < window_points:
        return warnings

    pressure = pressure_series.values
    rate = rate_series.values
    sand = sand_series.values
    prev_max_p = None
    prev_std_p = None

    for i in range(n - window_points + 1):
        j = i + window_points - 1
        if j < sand_start or i > sand_end:
            continue
        p_win = pressure[i:j+1]
        q_win = rate[i:j+1]
        s_win = sand[i:j+1]
        if np.isnan(p_win).sum() > len(p_win) * 0.3:
            continue
        q_start, q_end = q_win[0], q_win[-1]
        if not np.isnan(q_start) and not np.isnan(q_end):
            if q_end > q_start + 0.2 and np.nanmax(q_win) - np.nanmin(q_win) > 0.3:
                prev_max_p = None
                prev_std_p = None
                continue
        q_std = np.nanstd(q_win)
        s_std = np.nanstd(s_win)
        if q_std > q_std_threshold or s_std > s_std_threshold:
            prev_max_p = None
            prev_std_p = None
            continue

        p_start = p_win[0]
        p_end = p_win[-1]
        diff_p = p_end - p_start
        diff_arr = np.diff(p_win)
        is_monotonic_increasing = np.all(diff_arr >= 0)
        is_monotonic_decreasing = np.all(diff_arr <= 0)

        if is_monotonic_increasing and diff_p > pressure_rise_threshold:
            max_p = np.nanmax(p_win)
            std_p = np.nanstd(p_win)
            peak_inc = False
            amp_inc = False
            if prev_max_p is not None and max_p > prev_max_p:
                peak_inc = True
            if prev_std_p is not None and std_p > prev_std_p:
                amp_inc = True
            if peak_inc and amp_inc:
                start_ts = ts_series.iloc[i]
                end_ts = ts_series.iloc[j]
                if pd.notna(start_ts) and pd.notna(end_ts):
                    duration_sec = (end_ts - start_ts).total_seconds()
                    slope_per_min = diff_p / (duration_sec / 60.0) if duration_sec > 0 else 0.0
                    if duration_sec >= 60 or slope_per_min > 5.0:
                        start_time = make_datetime_text(df, i, columns)
                        end_time = make_datetime_text(df, j, columns)
                        warnings.append({
                            "来源文件": source_file,
                            "段号": str(sheet_name).strip(),
                            "开始数据序号": i + 1,
                            "结束数据序号": j + 1,
                            "开始时间": start_time,
                            "结束时间": end_time,
                            "预警类型": "砂堵迹象",
                            "压力变化(MPa)": round(diff_p, 3),
                            "排量标准差": round(q_std, 3),
                            "砂比标准差": round(s_std, 3),
                            "波峰递增": peak_inc,
                            "波动幅度增大": amp_inc,
                            "描述": f"压力持续上涨{diff_p:.2f}MPa，波峰递增且波动增大"
                        })
            prev_max_p = max_p
            prev_std_p = std_p
        elif is_monotonic_decreasing and diff_p < -pressure_drop_threshold:
            start_time = make_datetime_text(df, i, columns)
            end_time = make_datetime_text(df, j, columns)
            warnings.append({
                "来源文件": source_file,
                "段号": str(sheet_name).strip(),
                "开始数据序号": i + 1,
                "结束数据序号": j + 1,
                "开始时间": start_time,
                "结束时间": end_time,
                "预警类型": "砂堵风险",
                "压力变化(MPa)": round(diff_p, 3),
                "排量标准差": round(q_std, 3),
                "砂比标准差": round(s_std, 3),
                "波峰递增": False,
                "波动幅度增大": False,
                "描述": f"压力持续下降{abs(diff_p):.2f}MPa，排量砂比稳定"
            })
            prev_max_p = None
            prev_std_p = None
        else:
            prev_max_p = None
            prev_std_p = None

    if warnings:
        warnings.sort(key=lambda x: x["开始数据序号"])
        merged = []
        current = warnings[0]
        for w in warnings[1:]:
            if (w["开始数据序号"] <= current["结束数据序号"] + 1 and
                w["预警类型"] == current["预警类型"] and
                w["段号"] == current["段号"]):
                current["结束数据序号"] = w["结束数据序号"]
                current["结束时间"] = w["结束时间"]
                current["描述"] += f"; {w['描述']}"
            else:
                merged.append(current)
                current = w
        merged.append(current)
        return merged
    return warnings

# ==================== 地层破裂检测 ====================
def detect_fracture(
    df, columns, source_file, sheet_name,
    start_idx, end_idx,
    slope_threshold=5.0,
    window_sec=10.0,
    rate_rise_threshold=0.2,
) -> list[dict[str, Any]]:
    if df.empty or start_idx > end_idx:
        return []

    pressure_series = to_number(df[columns["pressure"]])
    rate_series = to_number(df[columns["rate"]])
    if pressure_series.isna().all() or rate_series.isna().all():
        return []

    timestamps = [make_timestamp(df, i, columns) for i in range(len(df))]
    ts_series = pd.Series(timestamps)
    valid = ts_series.notna()
    if valid.sum() < 2:
        return []
    diffs = ts_series[valid].diff().dropna()
    if len(diffs) == 0:
        return []
    dt_median = diffs.median().total_seconds()
    if dt_median <= 0:
        return []

    window_points = max(3, int(window_sec / dt_median) + 1)
    n = len(df)
    if n < window_points:
        return []

    events = []
    for i in range(start_idx, end_idx - window_points + 2):
        j = min(i + window_points - 1, end_idx)
        if j - i + 1 < 3:
            continue

        p_win = pressure_series.iloc[i:j+1].values
        q_win = rate_series.iloc[i:j+1].values
        t_win = ts_series.iloc[i:j+1]

        if np.isnan(p_win).sum() > len(p_win) * 0.3:
            continue

        q_start, q_end = q_win[0], q_win[-1]
        if np.isnan(q_start) or np.isnan(q_end):
            continue
        if not (q_end > q_start + rate_rise_threshold and np.nanmax(q_win) - np.nanmin(q_win) > 0.3):
            continue

        valid_p = ~np.isnan(p_win)
        if valid_p.sum() < 2:
            continue
        x = np.arange(len(p_win))[valid_p]
        y = p_win[valid_p]
        if len(x) < 2:
            continue
        slope, _ = np.polyfit(x, y, 1)
        slope_per_min = slope / dt_median * 60.0
        if slope_per_min < -slope_threshold:
            t_start = t_win.iloc[0]
            t_end = t_win.iloc[-1]
            if pd.isna(t_start) or pd.isna(t_end):
                continue
            time_diff_min = (t_end - t_start).total_seconds() / 60.0
            if time_diff_min <= 0:
                continue
            actual_slope = (p_win[-1] - p_win[0]) / time_diff_min if not np.isnan(p_win[0]) and not np.isnan(p_win[-1]) else slope_per_min
            if actual_slope >= -slope_threshold:
                continue

            peak_idx = i + np.argmax(p_win) if not np.isnan(p_win).all() else i
            valley_idx = i + np.argmin(p_win)
            peak_p = float(p_win[np.argmax(p_win)]) if not np.isnan(p_win).all() else np.nan
            valley_p = float(p_win[np.argmin(p_win)])
            if np.isnan(peak_p) or np.isnan(valley_p):
                continue
            drop = peak_p - valley_p
            if drop <= 0:
                continue

            events.append({
                "来源文件": source_file,
                "段号": str(sheet_name).strip(),
                "开始数据序号": i + 1,
                "结束数据序号": j + 1,
                "开始时间": make_datetime_text(df, i, columns),
                "结束时间": make_datetime_text(df, j, columns),
                "破裂压力(MPa)": round(peak_p, 3),
                "压力下降范围(MPa)": round(drop, 3),
                "压力下降斜率(MPa/min)": round(abs(actual_slope), 3),
                "peak_idx": i + np.argmax(p_win),
                "valley_idx": i + np.argmin(p_win),
            })

    if not events:
        return []
    events.sort(key=lambda x: x["开始数据序号"])
    merged = []
    current = events[0]
    for e in events[1:]:
        if e["开始数据序号"] <= current["结束数据序号"] + 2:
            current["结束数据序号"] = max(current["结束数据序号"], e["结束数据序号"])
            current["结束时间"] = e["结束时间"]
        else:
            merged.append(current)
            current = e
    merged.append(current)
    return merged

# ==================== 绘图函数（彻底统一判断与绘图） ====================
def plot_sheet(df, columns, records, warnings, fracture_events, sheet_name, output_dir, source_file, curve_threshold):
    if not MATPLOTLIB_AVAILABLE:
        return
    if df.empty or not records:
        return
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False

        timestamps = [make_timestamp(df, i, columns) for i in range(len(df))]
        ts_series = pd.Series(timestamps)
        valid = ts_series.notna()
        if valid.sum() < 2:
            return

        pressure = to_number(df[columns["pressure"]])
        rate = to_number(df[columns["rate"]])
        sand_ratio = to_number(df[columns["sand_ratio"]]).fillna(0)

        fig = plt.figure(figsize=(21, 11))
        gs = fig.add_gridspec(1, 2, width_ratios=[3.5, 1.2], wspace=0.15)
        ax_left = fig.add_subplot(gs[0])
        ax_right = fig.add_subplot(gs[1])
        ax_right.axis('off')

        ax_left.plot(ts_series, pressure, color='blue', linewidth=1.8, label='施工压力 (MPa)')
        ax_left.set_xlabel('时间')
        ax_left.set_ylabel('压力 (MPa)', color='blue')
        ax_left.tick_params(axis='y', labelcolor='blue')
        ax_left.plot(ts_series, rate, color='green', linewidth=1.2, label='排量 (m$^3$/min)')
        ax_left.tick_params(axis='y')
        ax_sand = ax_left.twinx()
        ax_sand.plot(ts_series, sand_ratio, color='red', linewidth=1.5, label='砂比 (%)')
        ax_sand.set_ylabel('砂比 (%)', color='red')
        ax_sand.tick_params(axis='y', labelcolor='red')
        if not sand_ratio.isna().all():
            ax_sand.set_ylim(0, max(100, sand_ratio.max() * 1.1))
        else:
            ax_sand.set_ylim(0, 100)

        if valid.sum() > 1:
            t_min = ts_series[valid].min()
            t_max = ts_series[valid].max()
            delta = (t_max - t_min) * 0.05
            ax_left.set_xlim(t_min - delta, t_max + delta)

        type_colors = {
            "前置液": "lightgreen",
            "加砂台阶": "orange",
            "隔离液": "yellow",
            "顶替液/停泵液": "lightblue",
            "砂堵": "red",
            "暂堵": "purple",
        }
        table_data = []
        headers = ["编号", "类型", "砂比台阶", "压力变化\n(MPa)", "总液量\n(m$^3$)", "总砂量\n(m$^3$)", "备注"]
        curve_trend = ""

        for idx, rec in enumerate(records, start=1):
            try:
                start_idx = rec["开始数据序号"] - 1
                end_idx = rec["结束数据序号"] - 1
                if start_idx < 0 or end_idx >= len(df):
                    continue
                start_time = ts_series.iloc[start_idx]
                end_time = ts_series.iloc[end_idx]
                if pd.isna(start_time) or pd.isna(end_time):
                    continue
                rec_type = rec["小条类型"]
                if pd.isna(rec_type):
                    continue
                color = type_colors.get(rec_type, "gray")
                alpha = 0.25 if rec_type in ["前置液", "隔离液", "顶替液/停泵液"] else 0.2
                ax_left.axvspan(start_time, end_time, alpha=alpha, color=color)
                mid_time = start_time + (end_time - start_time) / 2
                y_pos = ax_left.get_ylim()[1] * 0.92
                ax_left.annotate(str(idx), xy=(mid_time, y_pos),
                                 xytext=(0, 0), textcoords='offset points',
                                 ha='center', va='center',
                                 bbox=dict(boxstyle="circle,pad=0.15", facecolor='white', edgecolor='black', linewidth=1),
                                 fontsize=9, fontweight='bold')
                if rec_type == "前置液":
                    max_p = rec["最大压力"]
                    if not pd.isna(max_p):
                        seg_p = pressure.iloc[start_idx:end_idx+1]
                        if not seg_p.isna().all():
                            max_idx = seg_p.idxmax()
                            max_time = ts_series.iloc[max_idx]
                            ax_left.scatter(max_time, max_p, color='blue', s=50, zorder=5)
                            ax_left.text(max_time, max_p + 1, f'{max_p:.1f}', color='blue', fontsize=9, ha='center', va='bottom')
                    max_q = rec["最大排量"]
                    if not pd.isna(max_q):
                        seg_q = rate.iloc[start_idx:end_idx+1]
                        if not seg_q.isna().all():
                            max_idx = seg_q.idxmax()
                            max_time = ts_series.iloc[max_idx]
                            ax_left.scatter(max_time, max_q, color='green', s=50, zorder=5)
                            ax_left.text(max_time, max_q + 0.5, f'{max_q:.1f}', color='green', fontsize=9, ha='center', va='bottom')
                elif rec_type == "加砂台阶":
                    if pd.notna(rec.get("加砂曲线判断")):
                        curve_trend = rec["加砂曲线判断"]
                elif rec_type == "顶替液/停泵液":
                    max_p = rec["最大压力"]
                    if not pd.isna(max_p):
                        seg_p = pressure.iloc[start_idx:end_idx+1]
                        if not seg_p.isna().all():
                            max_idx = seg_p.idxmax()
                            max_time = ts_series.iloc[max_idx]
                            ax_left.scatter(max_time, max_p, color='red', s=50, zorder=5)
                            ax_left.text(max_time, max_p + 1, f'{max_p:.1f}', color='red', fontsize=9, ha='center', va='bottom')
                    stop_p = rec["停泵压力"]
                    stop_time_str = rec["停泵时间"]
                    if not pd.isna(stop_p) and not pd.isna(stop_time_str):
                        try:
                            stop_time = pd.to_datetime(stop_time_str)
                            if not pd.isna(stop_time):
                                ax_left.scatter(stop_time, stop_p, color='gray', s=50, zorder=5)
                                ax_left.text(stop_time, stop_p - 1, f'{stop_p:.1f}', color='gray', fontsize=9, ha='center', va='top')
                        except:
                            pass
                    displ_vol = rec["顶替液量"]
                    if not pd.isna(displ_vol):
                        ax_left.text(mid_time, (ax_left.get_ylim()[0] + ax_left.get_ylim()[1])/2,
                                     f'顶替液量 {displ_vol:.1f} m$^3$', color='black', fontsize=9,
                                     ha='center', va='center', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

                pressure_change = ""
                if rec_type == "加砂台阶":
                    if not pd.isna(rec["压力下降值"]):
                        pressure_change = f"-{round_value(rec['压力下降值'], 1)}"
                elif rec_type == "隔离液":
                    if not pd.isna(rec["压力上升值"]):
                        pressure_change = f"+{round_value(rec['压力上升值'], 1)}"
                row = {
                    "编号": str(idx),
                    "类型": rec_type,
                    "砂比台阶": rec["砂比台阶"] if not pd.isna(rec["砂比台阶"]) else "",
                    "压力变化": pressure_change,
                    "总液量": round_value(rec["总液量"], 1) if not pd.isna(rec["总液量"]) else "",
                    "总砂量": round_value(rec["总砂量"], 1) if not pd.isna(rec["总砂量"]) else "",
                    "备注": ""
                }
                if rec_type == "前置液":
                    row["备注"] = f"最大压力 {round_value(rec['最大压力'],1)} MPa"
                elif rec_type == "顶替液/停泵液":
                    row["备注"] = f"顶替液 {round_value(rec['顶替液量'],1)} m$^3$"
                elif rec_type == "砂堵":
                    row["备注"] = f"总砂 {round_value(rec['砂堵总砂量'],1)} m$^3$，最高砂比 {round_value(rec['砂堵最高砂比'],0)}%"
                elif rec_type == "暂堵":
                    row["备注"] = f"最大压差 {round_value(rec['暂堵最大压力差'],1)} MPa"
                else:
                    row["备注"] = ""
                if rec_type == "加砂台阶" and pd.notna(rec.get("支撑剂类型")):
                    row["备注"] = f"支撑剂：{rec['支撑剂类型']}"
                table_data.append(row)
            except Exception as e:
                print(f"    处理记录 {idx} 时出错：{e}，跳过")

        for w in warnings:
            try:
                w_type = w["预警类型"]
                start_idx = w["开始数据序号"] - 1
                end_idx = w["结束数据序号"] - 1
                if start_idx < 0 or end_idx >= len(df):
                    continue
                start_time = ts_series.iloc[start_idx]
                end_time = ts_series.iloc[end_idx]
                if pd.isna(start_time) or pd.isna(end_time):
                    continue
                hatch = '///' if w_type == "砂堵迹象" else 'xxx'
                alpha = 0.2 if w_type == "砂堵迹象" else 0.25
                ax_left.axvspan(start_time, end_time, facecolor='none', edgecolor='none', hatch=hatch, alpha=alpha)
                ax_left.axvspan(start_time, end_time, facecolor='gray', alpha=0.05, hatch=hatch)
            except Exception as e:
                print(f"    处理预警 {w.get('预警类型','未知')} 时出错：{e}")

        for fi, fe in enumerate(fracture_events, start=1):
            try:
                start_idx = fe["开始数据序号"] - 1
                end_idx = fe["结束数据序号"] - 1
                if start_idx < 0 or end_idx >= len(df):
                    continue
                start_time = ts_series.iloc[start_idx]
                end_time = ts_series.iloc[end_idx]
                if pd.isna(start_time) or pd.isna(end_time):
                    continue
                ax_left.axvspan(start_time, end_time, alpha=0.25, color='cyan', label='地层破裂' if fi==1 else "")
                mid_time = start_time + (end_time - start_time) / 2
                y_pos = ax_left.get_ylim()[1] * 0.85
                ax_left.annotate(f'F{fi}', xy=(mid_time, y_pos),
                                 xytext=(0, 0), textcoords='offset points',
                                 ha='center', va='center',
                                 bbox=dict(boxstyle="round,pad=0.1", facecolor='cyan', edgecolor='black', linewidth=0.5),
                                 fontsize=8, fontweight='bold', color='black')
                row = {
                    "编号": f"F{fi}",
                    "类型": "地层破裂",
                    "砂比台阶": "",
                    "压力变化": f"{fe['压力下降斜率(MPa/min)']} MPa/min",
                    "总液量": "",
                    "总砂量": "",
                    "备注": f"压力从 {fe['破裂压力(MPa)']:.1f} 降至 {fe['破裂压力(MPa)']-fe['压力下降范围(MPa)']:.1f} MPa"
                }
                table_data.append(row)
            except Exception as e:
                print(f"    处理破裂事件 {fi} 时出错：{e}")

        # ========== 核心修改：绘制加砂台阶代表压力（绘图与判断计算逻辑深度统一） ==========
        sand_records = [rec for rec in records if rec.get("小条类型") == "加砂台阶" and pd.notna(rec.get("台阶代表压力"))]
        if sand_records:
            # 按开始数据序号排序（保证顺序）
            sand_records_sorted = sorted(sand_records, key=lambda r: r["开始数据序号"])
            times = []
            pressures_vals = []
            for rec in sand_records_sorted:
                start_idx = rec["开始数据序号"] - 1
                end_idx = rec["结束数据序号"] - 1
                if start_idx < 0 or end_idx >= len(df):
                    continue
                start_time = ts_series.iloc[start_idx]
                end_time = ts_series.iloc[end_idx]
                if pd.isna(start_time) or pd.isna(end_time):
                    continue
                mid_time = start_time + (end_time - start_time) / 2
                times.append(mid_time)
                pressures_vals.append(rec["台阶代表压力"])

            if len(times) >= 2:
                # 获取顶部曲线判断结果
                curve_trend_label = curve_trend if curve_trend else "平稳"
                th = curve_threshold / 6.0
                
                n = len(pressures_vals)
                p = np.array(pressures_vals, dtype=float)
                # 复制判断逻辑中的抗扰动平滑
                p_clean = p.copy()
                for i in range(1, n - 1):
                    if abs(p[i] - p[i - 1]) > 2.0 and abs(p[i + 1] - p[i - 1]) < 0.5:
                        p_clean[i] = (p[i - 1] + p[i + 1]) / 2.0
                
                x = np.arange(n)
                # 计算整体直线（与 classify_curve 完全一致）
                coeffs_all = np.polyfit(x, p_clean, 1)
                
                # 根据逻辑判断结果执行相应的绘图策略
                if curve_trend_label in ["上升", "下降", "平稳"]:
                    # 单一趋势，画一条直线
                    y_start = coeffs_all[0] * 0 + coeffs_all[1]
                    y_end = coeffs_all[0] * (n - 1) + coeffs_all[1]
                    # 【修改】改为白色虚线，线宽加粗至2.0，透明度提升至0.8
                    ax_left.plot([times[0], times[-1]], [y_start, y_end], 
                                 color='white', linestyle='--', linewidth=2.0, alpha=0.8, label=f'拟合趋势({curve_trend_label})')
                elif curve_trend_label == "V字型":
                    # V字型趋势，寻找低谷点（完全复用 classify_curve 的逻辑）
                    min_idx = np.argmin(p_clean)
                    if min_idx > 0 and min_idx < n - 1:
                        # 绘制左段
                        x_l = x[:min_idx+1]
                        y_l = p_clean[:min_idx+1]
                        coeffs_l = np.polyfit(x_l, y_l, 1)
                        y_start_l = coeffs_l[0] * 0 + coeffs_l[1]
                        y_end_l = coeffs_l[0] * min_idx + coeffs_l[1]
                        ax_left.plot([times[0], times[min_idx]], [y_start_l, y_end_l], 
                                     color='white', linestyle='--', linewidth=2.0, alpha=0.8)
                        
                        # 绘制右段
                        x_r = x[min_idx:]
                        y_r = p_clean[min_idx:]
                        coeffs_r = np.polyfit(x_r, y_r, 1)
                        y_start_r = coeffs_r[0] * min_idx + coeffs_r[1]
                        y_end_r = coeffs_r[0] * (n - 1) + coeffs_r[1]
                        ax_left.plot([times[min_idx], times[-1]], [y_start_r, y_end_r], 
                                     color='white', linestyle='--', linewidth=2.0, alpha=0.8)
                        
                        # 标注转折点
                        ax_left.scatter([times[min_idx]], [y_end_l], color='red', s=30, zorder=6, label='转折点')
                        ax_left.plot([], [], color='white', linestyle='--', linewidth=2.0, alpha=0.8, label='拟合趋势(V字型)')
                    else:
                        # 异常兜底，画整体直线
                        y_start = coeffs_all[0] * 0 + coeffs_all[1]
                        y_end = coeffs_all[0] * (n - 1) + coeffs_all[1]
                        ax_left.plot([times[0], times[-1]], [y_start, y_end], 
                                     color='white', linestyle='--', linewidth=2.0, alpha=0.8, label='拟合趋势(V字型-回退)')
                
                # 绘制原始的台阶代表压力点（作为散点，用来对比拟合线）
                ax_left.scatter(times, pressures_vals, color='gray', s=40, zorder=5, alpha=0.7, label='台阶代表压力')
        
        # 更新左上角的文本
        if curve_trend:
            ax_left.text(0.02, 0.98, f'加砂曲线趋势：{curve_trend}', transform=ax_left.transAxes,
                         fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        if table_data:
            try:
                cell_text = []
                for row in table_data:
                    cell_text.append([
                        row["编号"],
                        row["类型"],
                        row["砂比台阶"],
                        row["压力变化"],
                        row["总液量"],
                        row["总砂量"],
                        row["备注"]
                    ])
                cell_text.insert(0, headers)
                table = ax_right.table(cellText=cell_text, loc='center', cellLoc='center',
                                       colWidths=[0.09, 0.17, 0.12, 0.17, 0.12, 0.12, 0.21])
                table.auto_set_font_size(False)
                table.set_fontsize(8)
                table.scale(1.2, 1.4)
                for (i, j), cell in table.get_celld().items():
                    if i == 0:
                        cell.set_facecolor('#40466e')
                        cell.set_text_props(color='white', fontweight='bold')
                    else:
                        typ = cell_text[i][1] if i < len(cell_text) else ""
                        if typ == "加砂台阶":
                            cell.set_facecolor('#ffcc99')
                        elif typ == "前置液":
                            cell.set_facecolor('#ccffcc')
                        elif typ == "隔离液":
                            cell.set_facecolor('#ffffcc')
                        elif typ == "顶替液/停泵液":
                            cell.set_facecolor('#ccffff')
                        elif typ == "砂堵":
                            cell.set_facecolor('#ffcccc')
                        elif typ == "暂堵":
                            cell.set_facecolor('#e6ccff')
                        elif typ == "地层破裂":
                            cell.set_facecolor('#b3e6ff')
                        else:
                            cell.set_facecolor('#f5f5f5')
                        cell.set_edgecolor('gray')
            except Exception as e:
                print(f"    绘制表格时出错：{e}")

        try:
            lines1, labels1 = ax_left.get_legend_handles_labels()
            lines2, labels2 = ax_sand.get_legend_handles_labels()
            combined = dict(zip(labels1 + labels2, lines1 + lines2))
            legend_elements = [
                Patch(facecolor='none', edgecolor='none', label='标注符号说明：'),
                Patch(facecolor='blue', edgecolor='blue', label='● 蓝色数字=最大压力(MPa)'),
                Patch(facecolor='green', edgecolor='green', label='● 绿色数字=最大排量(m$^3$/min)'),
                Patch(facecolor='brown', edgecolor='brown', label='砂量(棕) 液量(紫) 压降(青)'),
                Patch(facecolor='orange', edgecolor='orange', label='液量(橙) 压升(深蓝)'),
                Patch(facecolor='red', edgecolor='red', label='● 红色数字=最大压力'),
                Patch(facecolor='gray', edgecolor='gray', label='● 灰色数字=停泵压力'),
                Patch(facecolor='none', edgecolor='none', hatch='///', label='斜线阴影=砂堵迹象'),
                Patch(facecolor='none', edgecolor='none', hatch='xxx', label='交叉阴影=砂堵风险'),
                Patch(facecolor='cyan', edgecolor='cyan', label='青色阴影=地层破裂'),
            ]
            ax_left.legend(handles=list(combined.values()) + legend_elements,
                           labels=list(combined.keys()) + [e.get_label() for e in legend_elements],
                           loc='upper center', bbox_to_anchor=(0.5, 1.18), fontsize=8, ncol=3)
        except Exception as e:
            print(f"    绘制图例时出错：{e}")

        ax_left.set_title(f"{source_file} - {sheet_name} 施工曲线及分段统计", fontsize=14)
        ax_left.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        fig.autofmt_xdate()
        plt.subplots_adjust(top=0.82)

        output_dir.mkdir(parents=True, exist_ok=True)
        safe_sheet = str(sheet_name).replace('/', '_').replace('\\', '_')
        out_path = output_dir / f"{Path(source_file).stem}_{safe_sheet}_plot.png"
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        plt.close('all')
        print(f"  图表已保存：{out_path}")
    except Exception as e:
        print(f"  绘制 {sheet_name} 时发生严重错误：{e}，跳过该 Sheet 的绘图")
        try:
            plt.close('all')
        except:
            pass

# ==================== 表头检测和核心分析 ====================
def detect_header_row(workbook: pd.ExcelFile, sheet_name: str, nrows: int = 20) -> int:
    df_preview = pd.read_excel(workbook, sheet_name=sheet_name, header=None, nrows=nrows)
    keywords = ["SGBY", "PL", "SB", "LJYL", "LJSL", "SGRQ", "SGSJ", "砂比", "排量", "施工泵压"]
    best_row, best_count = 0, -1
    for idx in range(len(df_preview)):
        row = df_preview.iloc[idx].astype(str).str.strip().str.upper()
        matched = set()
        for kw in keywords:
            kw_upper = kw.upper()
            for cell in row:
                if kw_upper in cell:
                    matched.add(kw_upper)
                    break
        count = len(matched)
        if count > best_count:
            best_count, best_row = count, idx
            if count >= 3:
                break
    return best_row if best_count >= 2 else 0

def analyse_sheet(
    df: pd.DataFrame,
    source_file: str,
    sheet_name: str,
    curve_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    if df.empty:
        return [], [], df, [], {}

    columns = resolve_columns(df)
    df, is_reversed = ensure_chronological_order(df, columns)
    if is_reversed:
        print(f"  信息：{source_file} - {sheet_name} 数据为时间倒序，已翻转并修复累计列。")

    sand_ratio = to_number(df[columns["sand_ratio"]])
    positive_positions = np.where((sand_ratio.fillna(0) > 0).to_numpy())[0]
    if len(positive_positions) == 0:
        return [], [], df, [], {}

    first_sand = int(positive_positions[0])
    last_sand = int(positive_positions[-1])

    warnings = detect_sand_warnings(df, columns, source_file, sheet_name, sand_start=first_sand, sand_end=last_sand)

    records: list[dict[str, Any]] = []
    sand_record_indices: list[int] = []
    sequence = 1

    if first_sand > 0:
        record = make_base_record(df, columns, source_file, sheet_name, sequence, "前置液", 0, first_sand - 1)
        fill_prepad_record(df, record, 0, first_sand - 1, columns)
        records.append(record)
        sequence += 1

    pos = first_sand
    while pos <= last_sand:
        cur = sand_ratio.iloc[pos]
        if pd.notna(cur) and float(cur) > 0:
            step = round_sand_ratio_to_step(cur)
            start = pos
            pos += 1
            while pos <= last_sand:
                val = sand_ratio.iloc[pos]
                if pd.isna(val) or float(val) <= 0 or round_sand_ratio_to_step(val) != step:
                    break
                pos += 1
            end = pos - 1
            if end - start + 1 >= 5:
                rec = make_base_record(df, columns, source_file, sheet_name, sequence, "加砂台阶", start, end)
                fill_sand_step_record(df, rec, start, end, step, columns)
                records.append(rec)
                sand_record_indices.append(len(records) - 1)
                sequence += 1
        else:
            start = pos
            pos += 1
            while pos <= last_sand:
                val = sand_ratio.iloc[pos]
                if pd.notna(val) and float(val) > 0:
                    break
                pos += 1
            end = pos - 1
            rec = make_base_record(df, columns, source_file, sheet_name, sequence, "隔离液", start, end)
            fill_isolation_record(df, rec, start, end, columns)
            records.append(rec)
            sequence += 1

    if sand_record_indices:
        reps = [records[i]["台阶代表压力"] for i in sand_record_indices]
        records[sand_record_indices[-1]]["加砂曲线判断"] = classify_curve(reps, curve_threshold)

    if last_sand + 1 < len(df):
        rec = make_base_record(df, columns, source_file, sheet_name, sequence, "顶替液/停泵液", last_sand + 1, len(df) - 1)
        fill_displacement_stop_record(df, rec, last_sand + 1, len(df) - 1, columns)
        records.append(rec)
        sequence += 1

    wt_col = columns.get("working_type")
    new_work_records = []
    if wt_col is not None:
        wt_series = df[wt_col].astype(str).str.strip()
        is_sand_block = wt_series == "砂堵"
        is_temp = wt_series.isin(["缝内暂堵", "缝口暂堵"])
        for mask, type_name in [(is_sand_block, "砂堵"), (is_temp, "暂堵")]:
            in_block = False
            start_idx = 0
            for i, flag in enumerate(mask):
                if flag and not in_block:
                    in_block = True
                    start_idx = i
                elif not flag and in_block:
                    in_block = False
                    end_idx = i - 1
                    rec = make_base_record(df, columns, source_file, sheet_name, 0, type_name, start_idx, end_idx)
                    fill_work_type_record(df, rec, start_idx, end_idx, columns, type_name)
                    new_work_records.append(rec)
            if in_block:
                end_idx = len(df) - 1
                rec = make_base_record(df, columns, source_file, sheet_name, 0, type_name, start_idx, end_idx)
                fill_work_type_record(df, rec, start_idx, end_idx, columns, type_name)
                new_work_records.append(rec)

    all_records = records + new_work_records
    all_records.sort(key=lambda r: r["开始数据序号"])
    for idx, rec in enumerate(all_records, start=1):
        rec["小条序号"] = idx

    fracture_events: list[dict[str, Any]] = []
    for rec in records:
        if rec["小条类型"] in ["前置液", "加砂台阶", "隔离液"]:
            start_idx = rec["开始数据序号"] - 1
            end_idx = rec["结束数据序号"] - 1
            if start_idx < 0 or end_idx >= len(df):
                continue
            events = detect_fracture(df, columns, source_file, sheet_name, start_idx, end_idx)
            fracture_events.extend(events)
    if fracture_events:
        fracture_events.sort(key=lambda x: x["开始数据序号"])
        merged = []
        for ev in fracture_events:
            if not merged or ev["开始数据序号"] > merged[-1]["结束数据序号"] + 2:
                merged.append(ev)
            else:
                merged[-1]["结束数据序号"] = max(merged[-1]["结束数据序号"], ev["结束数据序号"])
                merged[-1]["结束时间"] = ev["结束时间"]
        fracture_events = merged

    sheet_stats = {
        "来源文件": source_file,
        "井段": sheet_name,
        "破裂压力(MPa)": None,
        "停泵压力(MPa)": None,
        "停泵压降(MPa)": None,
        "压降时间(min)": None,
        "总液量(m³)": None,
        "总砂量(m³)": None,
    }
    liq_col = columns.get("cumulative_liquid")
    sand_col = columns.get("cumulative_sand")
    if liq_col is not None and liq_col in df.columns:
        liq_vals = to_number(df[liq_col]).dropna()
        if not liq_vals.empty:
            sheet_stats["总液量(m³)"] = round(float(liq_vals.iloc[-1]), 3)
    if sand_col is not None and sand_col in df.columns:
        sand_vals = to_number(df[sand_col]).dropna()
        if not sand_vals.empty:
            sheet_stats["总砂量(m³)"] = round(float(sand_vals.iloc[-1]), 3)
    if fracture_events:
        sheet_stats["破裂压力(MPa)"] = fracture_events[0]["破裂压力(MPa)"]

    stop_rec = None
    for rec in all_records:
        if rec["小条类型"] == "顶替液/停泵液":
            stop_rec = rec
            break
    if stop_rec is not None:
        sheet_stats["停泵压力(MPa)"] = stop_rec.get("停泵压力")
        sheet_stats["停泵压降(MPa)"] = stop_rec.get("停泵30min压降")
        sheet_stats["压降时间(min)"] = stop_rec.get("停泵后时间(min)")

    return all_records, warnings, df, fracture_events, sheet_stats

# ==================== 工作簿处理 ====================
def analyse_workbook(input_path: Path, output_path: Path, curve_threshold: float, plot: bool) -> list[dict[str, Any]]:
    all_records = []
    all_warnings = []
    all_fractures = []
    overall_stats = []
    plot_dir = input_path.parent / "plots" if plot else None

    with pd.ExcelFile(input_path, engine="openpyxl") as workbook:
        for sheet_name in workbook.sheet_names:
            try:
                print(f"  处理 Sheet: {sheet_name}")
                header_row = detect_header_row(workbook, sheet_name)
                df = pd.read_excel(workbook, sheet_name=sheet_name, header=header_row)
                df.columns = [col.strip() if isinstance(col, str) else col for col in df.columns]
                print(f"    表头行索引 {header_row}，列名: {list(df.columns)}")

                records, warnings, df_sorted, fractures, stats = analyse_sheet(df, input_path.name, sheet_name, curve_threshold)
                all_records.extend(records)
                all_warnings.extend(warnings)
                all_fractures.extend(fractures)
                if stats:
                    overall_stats.append(stats)

                if plot and MATPLOTLIB_AVAILABLE and records:
                    try:
                        cols = resolve_columns(df_sorted)
                        # 将阈值传递给绘图函数，保证内部逻辑统一
                        plot_sheet(df_sorted, cols, records, warnings, fractures, sheet_name, plot_dir, input_path.name, curve_threshold)
                    except Exception as e:
                        print(f"    绘图时出错：{e}，跳过绘图")
            except Exception as e:
                print(f"    处理 Sheet {sheet_name} 时出错：{e}，跳过该 Sheet 继续。")
                continue

    result = pd.DataFrame(all_records, columns=OUTPUT_COLUMNS)
    warning_df = pd.DataFrame(all_warnings, columns=WARNING_COLUMNS)
    fracture_df = pd.DataFrame(all_fractures, columns=FRACTURE_COLUMNS)

    run_info = pd.DataFrame([
        {"项目": "输入文件", "值": str(input_path.resolve())},
        {"项目": "输出文件", "值": str(output_path.resolve())},
        {"项目": "小条切分", "值": "前置液；SB>0按四舍五入整数砂比连续切台阶（至少连续5点有效）；加砂中间SB<=0或空为隔离液；连续正砂比但不足5点的过渡段直接忽略；最后有效加砂后为顶替液/停泵液"},
        {"项目": "加砂台阶统计", "值": "总液量、总砂量、压力下降值=max(开始压力-结束压力,0)、台阶代表压力=台阶内平均施工泵压、支撑剂类型=区间内ZCJLX列众数"},
        {"项目": "隔离液统计", "值": "总液量、压力上升值=max(结束压力-开始压力,0)"},
        {"项目": "顶替液/停泵液统计", "值": "顶替液量、最大压力、停泵压力、停泵后最后一个有效压力及压降、记录时间"},
        {"项目": "加砂曲线判断", "值": "采用斜率法（分段线性回归）判断：上升/下降/平稳/V字型，阈值可调"},
        {"项目": "加砂曲线阈值", "值": f"{curve_threshold} MPa（用于V字型幅度和斜率换算）"},
        {"项目": "V字型判断口径", "值": "最低点在中间，左段斜率< -阈值/6，右段斜率>阈值/6，首末压力均比最低点高至少阈值，且末压力不低于首压力-0.2MPa"},
        {"项目": "上升/下降口径", "值": "总体斜率>阈值/6为上升，< -阈值/6为下降，否则为平稳（经V字型复核）"},
        {"项目": "砂比台阶有效长度", "值": "连续相同四舍五入整数台阶至少5个数据点才计为有效加砂台阶；不足5点的过渡段直接忽略"},
        {"项目": "砂堵统计", "值": "连续标注“砂堵”的区间；统计总砂量、最高砂比、压力斜率(MPa/s)"},
        {"项目": "暂堵统计", "值": "连续标注“缝内暂堵”或“缝口暂堵”的区间；统计最大压力差、稳定压力差"},
        {"项目": "时间顺序处理", "值": "自动检测时间正序/倒序，若为倒序则按时间升序重排，并修复累计液量/砂量列，确保增量为正"},
        {"项目": "读取方式", "值": "自动检测表头行（寻找包含SGBY/PL/SB等关键词的行），数据从表头下一行开始"},
        {"项目": "砂堵迹象检测", "值": "排量、砂比稳定窗口内，压力单调上涨＞0.5MPa，且波峰压力递增、波动标准差增大，同时持续时间≥60秒或压力上升斜率＞5MPa/min；窗口必须处于加砂台阶或隔离液阶段；排除排量上升阶段"},
        {"项目": "砂堵风险检测", "值": "排量、砂比稳定窗口内，压力单调下降＞5MPa，且窗口必须处于加砂台阶或隔离液阶段；排除排量上升阶段"},
        {"项目": "预警阈值", "值": f"排量标准差<{0.2} m$^3$/min，砂比标准差<{1}%，上涨阈值{0.5}MPa，下降阈值{5}MPa，窗口{60}秒"},
        {"项目": "地层破裂检测", "值": "在前置液、加砂台阶、隔离液阶段，排量上升且压力下降斜率 > 5 MPa/min 时标记为破裂；统计破裂压力、下降范围、下降斜率"},
        {"项目": "停泵后压力记录", "值": "记录停泵后最后一个有效压力数据点及其相对停泵的时间（分钟），压降=停泵压力-该点压力，并生成说明文字"},
    ])

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        result.to_excel(writer, sheet_name="小条统计", index=False)
        run_info.to_excel(writer, sheet_name="统计口径", index=False)
        if not warning_df.empty:
            warning_df.to_excel(writer, sheet_name="砂堵预警", index=False)
        if not fracture_df.empty:
            fracture_df.to_excel(writer, sheet_name="地层破裂", index=False)

    return overall_stats

# ==================== 命令行入口 ====================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="压裂施工小条统计 + 高级绘图（增强版）")
    parser.add_argument("input", type=Path, help="输入 xlsx 文件路径，或包含 .xlsx 文件的文件夹路径")
    parser.add_argument("-o", "--output", type=Path, help="输出 xlsx 文件路径（仅在输入为单个文件时有效）")
    parser.add_argument("--curve-threshold", type=float, default=1.0, help="加砂曲线判断阈值，默认 1.0 MPa")
    parser.add_argument("--plot", action="store_true", default=True, help="生成图表（默认开启）")
    parser.add_argument("--no-plot", dest="plot", action="store_false", help="禁用图表生成")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    input_path = args.input
    if not input_path.exists():
        raise FileNotFoundError(f"路径不存在：{input_path}")

    all_overall_stats = []

    if input_path.is_file():
        output_path = args.output or input_path.with_name(f"{input_path.stem}_小条统计_增强版.xlsx")
        stats = analyse_workbook(input_path, output_path, args.curve_threshold, args.plot)
        all_overall_stats.extend(stats)
    elif input_path.is_dir():
        xlsx_files = list(input_path.glob("*.xlsx"))
        if not xlsx_files:
            print(f"目录 {input_path} 中没有找到 .xlsx 文件")
            return
        print(f"找到 {len(xlsx_files)} 个 .xlsx 文件，开始批量处理...")
        for idx, file_path in enumerate(xlsx_files, 1):
            print(f"\n[{idx}/{len(xlsx_files)}] 正在处理: {file_path.name}")
            out_path = file_path.with_name(f"{file_path.stem}_小条统计_增强版.xlsx")
            try:
                stats = analyse_workbook(file_path, out_path, args.curve_threshold, args.plot)
                all_overall_stats.extend(stats)
            except Exception as e:
                print(f"  处理 {file_path.name} 时出错：{e}，跳过继续。")
        print("\n批量处理完成！")
    else:
        raise ValueError(f"输入路径既不是文件也不是目录：{input_path}")

    if all_overall_stats:
        overall_df = pd.DataFrame(all_overall_stats, columns=OVERALL_STAT_COLUMNS)
        summary_path = input_path.parent / "总体统计汇总.xlsx"
        overall_df.to_excel(summary_path, index=False)
        print(f"\n总体统计汇总已保存至：{summary_path.resolve()}")
    else:
        print("\n没有生成任何总体统计数据。")

if __name__ == "__main__":
    main()
