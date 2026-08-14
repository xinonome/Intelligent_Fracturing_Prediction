import pandas as pd
import matplotlib.pyplot as plt
import warnings
import numpy as np
import traceback
import os

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def has_continuous_decrease(arr, min_length=3, strict: bool = True, eps=0.1):
    """
    判断数组是否存在连续min_length个及以上数据递减
    :param arr: 输入的数值数组（np.array）
    :param min_length: 最小连续递减长度（默认3，指原数组的连续元素数）
    :param strict: 是否严格递减（True=后数<前数；False=后数≤前数）
    :param eps: 浮点误差阈值
    :return: 布尔值、递减片段列表（(起始索引, 结束索引, 连续长度)）
    """
    arr = np.asarray(arr, dtype=float)

    if len(arr) < min_length:
        return False, []
    if np.isnan(arr).any():
        return False, []

    diffs = arr[1:] - arr[:-1]

    if strict:
        decrease_mask = diffs < -eps
    else:
        decrease_mask = diffs <= eps

    continuous_segments = []
    current_start_diff = None

    for i, is_decrease in enumerate(decrease_mask):
        if is_decrease:
            if current_start_diff is None:
                current_start_diff = i
        else:
            if current_start_diff is not None:
                diff_seg_length = i - current_start_diff
                arr_seg_length = diff_seg_length + 1
                if arr_seg_length >= min_length:
                    arr_start = current_start_diff
                    arr_end = i
                    continuous_segments.append((arr_start, arr_end, arr_seg_length))
                current_start_diff = None

    if current_start_diff is not None:
        diff_seg_length = len(decrease_mask) - current_start_diff
        arr_seg_length = diff_seg_length + 1
        if arr_seg_length >= min_length:
            arr_start = current_start_diff
            arr_end = len(decrease_mask)
            continuous_segments.append((arr_start, arr_end, arr_seg_length))

    has_decrease = len(continuous_segments) > 0
    return has_decrease, continuous_segments


def has_continuous_increase(arr, min_length=3, strict: bool = True, eps=0.05):
    """
    判断数组是否存在连续min_length个及以上数据递增
    :param arr: 输入数组
    :param min_length: 最小连续递增长度
    :param strict: 是否严格递增
    :param eps: 浮点误差阈值
    :return: 布尔值、递增片段列表
    """
    arr = np.asarray(arr, dtype=float)

    if len(arr) < min_length:
        return False, []
    if np.isnan(arr).any():
        return False, []

    diffs = arr[1:] - arr[:-1]

    if strict:
        increase_mask = diffs > eps
    else:
        increase_mask = diffs >= -eps

    continuous_segments = []
    current_start_diff = None

    for i, is_increase in enumerate(increase_mask):
        if is_increase:
            if current_start_diff is None:
                current_start_diff = i
        else:
            if current_start_diff is not None:
                diff_seg_length = i - current_start_diff
                arr_seg_length = diff_seg_length + 1
                if arr_seg_length >= min_length:
                    arr_start = current_start_diff
                    arr_end = i
                    continuous_segments.append((arr_start, arr_end, arr_seg_length))
                current_start_diff = None

    if current_start_diff is not None:
        diff_seg_length = len(increase_mask) - current_start_diff
        arr_seg_length = diff_seg_length + 1
        if arr_seg_length >= min_length:
            arr_start = current_start_diff
            arr_end = len(increase_mask)
            continuous_segments.append((arr_start, arr_end, arr_seg_length))

    has_increase = len(continuous_segments) > 0
    return has_increase, continuous_segments


def find_continuous_increase_segments(series, threshold=0.05):
    """找到连续递增区段"""
    if len(series) < 2:
        return []

    segments = []
    start = None
    for i in range(1, len(series)):
        if series.iloc[i] - series.iloc[i - 1] >= threshold:
            if start is None:
                start = i - 1
        else:
            if start is not None and (i - 1 - start) >= 2:
                segments.append((start, i - 1))
            start = None

    if start is not None and (len(series) - 1 - start) >= 2:
        segments.append((start, len(series) - 1))
    return segments


def calc_slope_per_min(start_value, end_value, duration_seconds):
    """计算斜率，单位MPa/min"""
    if duration_seconds <= 0:
        return 0.0
    return (end_value - start_value) / (duration_seconds / 60.0)


def merge_segments_with_label(segments_with_labels):
    """合并重叠区段，按标签优先级输出最终标签"""
    if not segments_with_labels:
        return []

    priority = {
        'sand_plug': 5,
        'sand_plug_sand_flow': 4,
        'sand_plug_flow': 3,
        'sand_plug_sand': 2,
        'abnormal': 1
    }

    sorted_segs = sorted(segments_with_labels, key=lambda x: x[0])
    merged = []

    current_start = sorted_segs[0][0]
    current_end = sorted_segs[0][1]
    current_labels = [sorted_segs[0][2]]

    for seg in sorted_segs[1:]:
        s, e, label = seg
        if s <= current_end:
            current_end = max(current_end, e)
            current_labels.append(label)
        else:
            final_label = max(current_labels, key=lambda x: priority.get(x, 0))
            merged.append((current_start, current_end, final_label))
            current_start, current_end, current_labels = s, e, [label]

    final_label = max(current_labels, key=lambda x: priority.get(x, 0))
    merged.append((current_start, current_end, final_label))
    return merged


def read_and_plot_excel_data(file_path, segment_num):
    try:
        df = pd.read_excel(file_path)
        print(f"\n===== 处理第{segment_num}段数据 =====")

        # 1. 读取数据
        x_data = df["序号"]
        y1_data = df["施工压力(MPa)"]
        y2_data = df["排出流量(m³/min)"]
        y3_data = df["砂比(%)"]
        y4_data = df["粘度(mPa.s)"]

        # 2. 加砂区间
        sand_ratio_gt0 = df["砂比(%)"] > 0
        first_sand_idx = sand_ratio_gt0.idxmax() if sand_ratio_gt0.any() else None
        last_sand_zero_idx = None

        if first_sand_idx is not None:
            df_after_sand = df.loc[first_sand_idx:]
            sand_eq0 = df_after_sand["砂比(%)"] == 0
            last_sand_zero_idx = sand_eq0[::-1].idxmax() if sand_eq0.any() else df.index.max()

        start_seq = end_seq = None
        valid_df = pd.DataFrame()

        if first_sand_idx is not None and last_sand_zero_idx is not None:
            start_seq = df.loc[first_sand_idx, "序号"]
            end_seq = df.loc[last_sand_zero_idx, "序号"]
            valid_df = df.loc[first_sand_idx:last_sand_zero_idx].copy().reset_index(drop=True)

        # 3. 参数配置（按最终专家规则）
        block_threshold = 8.0                  # 基础触发：ΔP >= 8 MPa
        block_threshold_high = 10.0           # 强触发：ΔP >= 10 MPa
        pressure_slope_main = 1.0             # 基础触发：斜率 >= 1.0 MPa/min
        pressure_slope_high = 1.5             # 强触发：斜率 >= 1.5 MPa/min
        min_pressure_rise_duration = 30       # 基础触发：持续时间 >= 30 s

        flow_drop_threshold = 0.1             # 排量递减阈值
        sand_drop_threshold = 0.2             # 砂比递减阈值
        max_zero_count_for_sand_plug = 3      # zero_count <= 3 才可按砂堵判定

        flow_rise_eps = 0.05                  # 排量连续升高判定阈值
        flow_rise_total_threshold = 1.0       # 排量总升幅 >= 1.0 m³/min
        recent_sand_zero_exclude_seconds = 120  # 最近120秒砂比存在0，不算砂堵

        window_size = 300
        step_size = 50
        min_increase_step = 0.05
        avg_window = 10

        segments_with_labels = []
        filtered_decrease = 0
        filtered_no_growth = 0
        filtered_normal = 0
        filtered_by_flow_rise = 0
        filtered_by_recent_sand_zero = 0
        filtered_short_duration = 0
        processed_windows = set()

        if not valid_df.empty:
            n_rows = len(valid_df)
            for start_idx in range(0, n_rows - window_size + 1, step_size):
                end_idx = start_idx + window_size
                window_key = (start_idx, end_idx)
                if window_key in processed_windows:
                    continue
                processed_windows.add(window_key)

                refer_end_idx = min(end_idx + int(window_size / 5), n_rows)

                # 提取窗口数据
                window_pressure_raw = valid_df["施工压力(MPa)"].iloc[start_idx:end_idx]
                window_flow_raw = valid_df["排出流量(m³/min)"].iloc[start_idx:end_idx]
                window_sand_raw = valid_df["砂比(%)"].iloc[start_idx:end_idx]

                # 参考数据：窗口后续1/5窗口长度
                window_flow_refer_raw = valid_df["排出流量(m³/min)"].iloc[end_idx:refer_end_idx]
                window_sand_refer_raw = valid_df["砂比(%)"].iloc[end_idx:refer_end_idx]

                # 聚合方式保持原思路
                window_pressure = window_pressure_raw.groupby(window_pressure_raw.index // avg_window).mean()
                window_flow = window_flow_raw.groupby(window_flow_raw.index // 60).mean()
                window_sand = window_sand_raw.groupby(window_sand_raw.index // 60).mean()

                # 修正：这里应使用 refer_raw，而不是原窗口 raw
                window_refer_flow = window_flow_refer_raw.groupby(window_flow_refer_raw.index // 60).mean() \
                    if len(window_flow_refer_raw) > 0 else pd.Series(dtype=float)
                window_refer_sand = window_sand_refer_raw.groupby(window_sand_refer_raw.index // 60).mean() \
                    if len(window_sand_refer_raw) > 0 else pd.Series(dtype=float)

                # 找压力连续递增段
                pressure_segments = find_continuous_increase_segments(window_pressure, min_increase_step)

                original_start_idx = first_sand_idx + start_idx
                original_end_idx = first_sand_idx + end_idx - 1
                original_start_seq = df.loc[original_start_idx, "序号"] if original_start_idx < len(df) else df["序号"].iloc[-1]
                original_end_seq = df.loc[original_end_idx, "序号"] if original_end_idx < len(df) else df["序号"].iloc[-1]

                if not pressure_segments:
                    filtered_no_growth += 1
                    continue

                window_label = None

                for (p_start, p_end) in pressure_segments:
                    # 压力持续时间：window_pressure每个点对应avg_window秒
                    pressure_duration_sec = (p_end - p_start) * avg_window
                    if pressure_duration_sec < min_pressure_rise_duration:
                        filtered_short_duration += 1
                        continue

                    pressure_change = window_pressure.iloc[p_end] - window_pressure.iloc[p_start]
                    pressure_slope = calc_slope_per_min(
                        window_pressure.iloc[p_start],
                        window_pressure.iloc[p_end],
                        pressure_duration_sec
                    )

                    # 基础触发
                    pressure_main_trigger = (
                        pressure_change >= block_threshold and
                        pressure_slope >= pressure_slope_main
                    )

                    # 强触发
                    pressure_high_trigger = (
                        pressure_change >= block_threshold_high and
                        pressure_slope >= pressure_slope_high
                    )

                    if not pressure_main_trigger:
                        continue

                    # 构造排量、砂比判断序列
                    flow = np.concatenate([
                        window_flow.iloc[min(p_start, len(window_flow)):].values,
                        window_refer_flow.values
                    ]) if len(window_flow.iloc[min(p_start, len(window_flow)):]) > 0 or len(window_refer_flow) > 0 else np.array([])

                    sand = np.concatenate([
                        window_sand.iloc[min(p_start, len(window_sand)):].values,
                        window_refer_sand.values
                    ]) if len(window_sand.iloc[min(p_start, len(window_sand)):]) > 0 or len(window_refer_sand) > 0 else np.array([])

                    if len(flow) == 0 or len(sand) == 0:
                        continue

                    zero_count = np.sum(sand == 0)

                    # 前置排除1：排量连续升高，总升幅 >= 1.0，且砂比未下降
                    flow_has_increase, flow_increase_segments = has_continuous_increase(
                        flow, min_length=3, strict=True, eps=flow_rise_eps
                    )
                    sand_has_decrease, sand_segments = has_continuous_decrease(
                        sand, min_length=3, strict=False, eps=sand_drop_threshold
                    )

                    max_flow_rise = 0.0
                    if flow_has_increase:
                        for seg_start, seg_end, _ in flow_increase_segments:
                            rise_value = flow[seg_end] - flow[seg_start]
                            if rise_value > max_flow_rise:
                                max_flow_rise = rise_value

                    if flow_has_increase and max_flow_rise >= flow_rise_total_threshold and (not sand_has_decrease):
                        filtered_by_flow_rise += 1
                        print(f"窗口[{original_start_seq}-{original_end_seq}]：压力↑+排量连续升高(ΔQ={max_flow_rise:.2f})且砂比未下降 → 提排量，不标记")
                        window_label = None
                        break

                    # 前置排除2：压力异常附近最近120秒内，砂比存在0 → 不算砂堵
                    p_start_raw_idx = start_idx + p_start * avg_window
                    recent_start_raw_idx = max(0, p_start_raw_idx - recent_sand_zero_exclude_seconds)
                    recent_sand_raw = valid_df["砂比(%)"].iloc[recent_start_raw_idx:p_start_raw_idx + 1]
                    has_recent_zero_sand = (recent_sand_raw == 0).any() if len(recent_sand_raw) > 0 else False

                    if has_recent_zero_sand:
                        filtered_by_recent_sand_zero += 1
                        print(f"窗口[{original_start_seq}-{original_end_seq}]：最近120秒砂比存在0 → 井筒效应，不标记")
                        window_label = None
                        break

                    # 重新计算流量下降（在排除规则后用于分类）
                    flow_has_decrease, flow_segments = has_continuous_decrease(
                        flow, min_length=3, strict=True, eps=flow_drop_threshold
                    )

                    # 强触发：严重砂堵
                    if pressure_high_trigger and zero_count <= max_zero_count_for_sand_plug:
                        window_label = 'sand_plug'
                        print(f"窗口[{original_start_seq}-{original_end_seq}]：ΔP={pressure_change:.2f}MPa, 斜率={pressure_slope:.2f}MPa/min → 严重砂堵")
                        break

                    # 分类1-5
                    if zero_count <= max_zero_count_for_sand_plug:
                        if flow_has_decrease and sand_has_decrease:
                            window_label = 'sand_plug_sand_flow'
                            print(f"窗口[{original_start_seq}-{original_end_seq}]：压力↑+排量递减+砂比递减 → 砂堵(降排量降砂比)")
                            break
                        elif flow_has_decrease:
                            window_label = 'sand_plug_flow'
                            print(f"窗口[{original_start_seq}-{original_end_seq}]：压力↑+仅排量递减 → 砂堵(降排量)")
                            break
                        elif sand_has_decrease:
                            window_label = 'sand_plug_sand'
                            print(f"窗口[{original_start_seq}-{original_end_seq}]：压力↑+仅砂比递减 → 砂堵(降砂比)")
                            break
                        else:
                            window_label = 'abnormal'
                            print(f"窗口[{original_start_seq}-{original_end_seq}]：压力↑+排量和砂比均未递减 → 压力异常波动")
                            break
                    else:
                        window_label = 'abnormal'
                        print(f"窗口[{original_start_seq}-{original_end_seq}]：压力↑但zero_count={zero_count}>3 → 压力异常")
                        break

                if window_label is not None:
                    segments_with_labels.append((original_start_seq, original_end_seq, window_label))
                else:
                    filtered_normal += 1

        # 统计输出
        print(f"\n窗口过滤统计：")
        print(f"  压力递减窗口数：{filtered_decrease}")
        print(f"  压力无增长窗口数：{filtered_no_growth}")
        print(f"  压力上升持续不足30s窗口数：{filtered_short_duration}")
        print(f"  提排量排除窗口数：{filtered_by_flow_rise}")
        print(f"  最近120秒砂比含0排除窗口数：{filtered_by_recent_sand_zero}")
        print(f"  排量/砂比增加的正常窗口数：{filtered_normal}")

        # 合并区段
        merged_segments = merge_segments_with_label(segments_with_labels)
        print(f"\n合并后最终区段：")
        if merged_segments:
            for idx, (s, e, label) in enumerate(merged_segments):
                if label == 'sand_plug':
                    name = "砂堵"
                elif label == 'sand_plug_flow':
                    name = "砂堵(降排量)"
                elif label == 'sand_plug_sand':
                    name = "砂堵(降砂比)"
                elif label == 'sand_plug_sand_flow':
                    name = "砂堵(降排量降砂比)"
                elif label == 'abnormal':
                    name = "压力异常波动"
                else:
                    name = "压力异常"
                print(f"  区段{idx + 1}：序号{s}-{e} → {name}")
        else:
            print(f"  无异常区段")

        # =========================
        # 绘图：保持原代码风格和方式不变
        # =========================
        fig, ax1 = plt.subplots(figsize=(14, 8))
        ax1.plot(x_data, y1_data, 'tab:blue', label='施工压力(MPa)', linewidth=2)
        ax1.set_xlabel('序号（秒）')
        ax1.set_ylabel('施工压力(MPa)', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.set_ylim(0, 150)
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(x_data, y2_data, 'tab:red', label='排出流量(m³/min)', linewidth=2)
        ax2.set_ylabel('排出流量(m³/min)', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        ax2.set_ylim(0, 40)

        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        ax3.plot(x_data, y3_data, 'green', label='砂比(%)', linewidth=2)
        ax3.set_ylabel('砂比(%)', color='green')
        ax3.tick_params(axis='y', labelcolor='green')
        ax3.set_ylim(0, 60)

        ax4 = ax1.twinx()
        ax4.spines['right'].set_position(('outward', 120))
        ax4.plot(x_data, y4_data, 'orange', label='粘度(mPa.s)', linewidth=2)
        ax4.set_ylabel('粘度(mPa.s)', color='orange')
        ax4.tick_params(axis='y', labelcolor='orange')
        ax4.set_ylim(0, 160)

        if start_seq and end_seq:
            ax1.axvspan(start_seq, end_seq, color='lightgray', alpha=0.1, label='加砂有效区间')

        for idx, (seg_start, seg_end, label) in enumerate(merged_segments):
            if label == 'sand_plug':
                color, y_pos, name = 'darkred', 145, '砂堵风险较高'
            elif label == 'sand_plug_flow':
                color, y_pos, name = 'red', 140, '砂堵风险较高(降排量)'
            elif label == 'sand_plug_sand':
                color, y_pos, name = 'darkred', 130, '砂堵风险较高(降砂比)'
            elif label == 'sand_plug_sand_flow':
                color, y_pos, name = 'darkred', 120, '砂堵风险较高(降排量降砂比)'
            elif label == 'abnormal':
                color, y_pos, name = 'gold', 110, '压力异常波动'
            else:
                color, y_pos, name = 'gold', 120, '压力异常'

            ax1.axvspan(seg_start, seg_end, color=color, alpha=0.2)
            mid = (seg_start + seg_end) / 2
            ax1.text(
                mid, y_pos, f'{name}区段{idx+1}', fontsize=12, fontweight='bold',
                bbox=dict(facecolor=color, alpha=0.8, edgecolor='black')
            )

        ax1.legend(loc='upper left')
        fig.suptitle(f'四川盆地某页岩气井第{segment_num}段施工参数分析', fontsize=16, fontweight='bold')
        plt.tight_layout()
        save_path = f'四川盆地某页岩气井第{segment_num}段_分析.png'
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n图表已保存：{save_path}")
        plt.close(fig)

    except Exception as e:
        print(f"错误：第{segment_num}段 - {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    base_file_path = os.environ.get(
        "FSL_EXPERT_DATA_PATTERN",
        os.path.join("data", "segment_{segment}.xlsx"),
    )
    for segment in range(1, 12):
        read_and_plot_excel_data(base_file_path.format(segment=segment), segment)
    print("\n===== 所有11段处理完成！=====")
