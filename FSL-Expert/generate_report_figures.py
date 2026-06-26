from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

ROOT = Path(r'C:\Workspace\Graph')
OUT = ROOT / 'report_figures'
OUT.mkdir(exist_ok=True)

# Prefer a Chinese-capable font if installed.
font_candidates = [
    r'C:\Windows\Fonts\msyh.ttc',
    r'C:\Windows\Fonts\simhei.ttf',
    r'C:\Windows\Fonts\simsun.ttc',
]
for fp in font_candidates:
    if Path(fp).exists():
        font_manager.fontManager.addfont(fp)
        plt.rcParams['font.sans-serif'] = [font_manager.FontProperties(fname=fp).get_name()]
        break
plt.rcParams['axes.unicode_minus'] = False

old_counts = pd.Series({
    '正常': 161900,
    '砂堵': 1058,
    '主缝延伸': 633,
    '缝口暂堵': 312,
    '其他': 52,
})
new_counts = pd.Series({
    '正常': 25972,
    '缝内暂堵': 2508,
    '缝口暂堵': 2436,
    '主缝延伸': 2366,
    '缝高延伸': 320,
    '延伸受阻': 77,
    '滤失过大': 72,
    '其他': 49,
})
combined_counts = old_counts.add(new_counts, fill_value=0).astype(int).sort_values(ascending=False)

# 1. Old vs new vs combined normal/abnormal ratio.
ratio_df = pd.DataFrame([
    {'数据集': '旧数据', '正常': 161900, '异常': 2055},
    {'数据集': '新增数据', '正常': 25972, '异常': 7828},
    {'数据集': '合并后', '正常': 187872, '异常': 9883},
]).set_index('数据集')
ratio_pct = ratio_df.div(ratio_df.sum(axis=1), axis=0) * 100
fig, ax = plt.subplots(figsize=(9, 5))
bottom = np.zeros(len(ratio_pct))
colors = {'正常': '#4C78A8', '异常': '#F58518'}
for col in ['正常', '异常']:
    ax.bar(ratio_pct.index, ratio_pct[col], bottom=bottom, label=col, color=colors[col])
    for i, value in enumerate(ratio_pct[col]):
        if value >= 4:
            ax.text(i, bottom[i] + value / 2, f'{value:.1f}%', ha='center', va='center', color='white', fontsize=11, fontweight='bold')
    bottom += ratio_pct[col].values
ax.set_ylabel('占比 (%)')
ax.set_title('正常/异常占比变化：新增数据显著提高异常占比')
ax.legend(loc='upper right')
ax.set_ylim(0, 100)
fig.tight_layout()
fig.savefig(OUT / '01_normal_abnormal_ratio.png', dpi=180)
plt.close(fig)

# 2. Combined label distribution excluding normal.
abnormal_combined = combined_counts.drop(labels=['正常'], errors='ignore').sort_values()
fig, ax = plt.subplots(figsize=(9, 5.5))
bars = ax.barh(abnormal_combined.index, abnormal_combined.values, color='#E45756')
ax.set_title('合并后异常工况样本量分布')
ax.set_xlabel('样本点数')
for bar in bars:
    w = bar.get_width()
    ax.text(w + max(abnormal_combined.values) * 0.01, bar.get_y() + bar.get_height()/2, f'{int(w)}', va='center')
fig.tight_layout()
fig.savefig(OUT / '02_abnormal_label_distribution.png', dpi=180)
plt.close(fig)

# 3. Segment-label heatmap for new dataset.
seg_path = ROOT / 'new_dataset_segment_label_distribution.csv'
if seg_path.exists():
    seg = pd.read_csv(seg_path, index_col=0)
    # Rename first garbled/normal column if needed.
    cols = list(seg.columns)
    if cols and cols[0] not in {'正常', 'total'}:
        cols[0] = '正常'
        seg.columns = cols
    heat_cols = [c for c in seg.columns if c != 'total']
    heat = seg[heat_cols].copy()
    fig, ax = plt.subplots(figsize=(11, 8))
    im = ax.imshow(heat.values, aspect='auto', cmap='YlOrRd')
    ax.set_xticks(np.arange(len(heat.columns)))
    ax.set_xticklabels(heat.columns, rotation=35, ha='right')
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels(heat.index)
    ax.set_title('新增数据：各段工况标签分布热力图')
    ax.set_xlabel('工况标签')
    ax.set_ylabel('段号')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('样本点数')
    fig.tight_layout()
    fig.savefig(OUT / '03_new_segment_label_heatmap.png', dpi=180)
    plt.close(fig)

# 4. Model comparison.
model_df = pd.DataFrame([
    {'方案': 'GNN二分类', 'Test Macro F1': 0.5170},
    {'方案': 'LightGBM二分类', 'Test Macro F1': 0.5237},
    {'方案': '加砂后+动态特征二分类', 'Test Macro F1': 0.5533},
    {'方案': '合并数据多分类', 'Test Macro F1': 0.5739},
    {'方案': '剔除纯正常段多分类', 'Test Macro F1': 0.4389},
])
fig, ax = plt.subplots(figsize=(10, 5.5))
colors_list = ['#72B7B2', '#72B7B2', '#54A24B', '#2F855A', '#E45756']
bars = ax.bar(model_df['方案'], model_df['Test Macro F1'], color=colors_list)
ax.set_ylim(0, 0.7)
ax.set_ylabel('Test Macro F1')
ax.set_title('模型路线对比：新增数据多分类当前效果最好')
ax.tick_params(axis='x', rotation=20)
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.015, f'{h:.3f}', ha='center', fontweight='bold')
fig.tight_layout()
fig.savefig(OUT / '04_model_macro_f1_comparison.png', dpi=180)
plt.close(fig)

# 5. Per-class metrics for multiclass combined test.
metrics_path = ROOT / 'runs' / 'frac_lgbm_multiclass_combined' / '20260420_155717' / 'window_4' / 'metrics.json'
if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
    readable_classes = ['正常', '主缝延伸', '其他', '延伸受阻', '滤失过大', '砂堵', '缝内暂堵', '缝口暂堵', '缝高延伸']
    rows = []
    report = metrics['metrics']['test']['report']
    for idx, name in enumerate(readable_classes):
        item = report.get(str(idx))
        if not item or item.get('support', 0) <= 0:
            continue
        rows.append({'类别': name, 'Precision': item['precision'], 'Recall': item['recall'], 'F1': item['f1-score'], 'Support': item['support']})
    per = pd.DataFrame(rows)
    x = np.arange(len(per))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - width, per['Precision'], width, label='Precision', color='#4C78A8')
    ax.bar(x, per['Recall'], width, label='Recall', color='#F58518')
    ax.bar(x + width, per['F1'], width, label='F1', color='#54A24B')
    ax.set_xticks(x)
    ax.set_xticklabels(per['类别'])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('指标值')
    ax.set_title('合并数据多分类：测试集各类别识别情况')
    ax.legend()
    for i, support in enumerate(per['Support']):
        ax.text(i, 1.02, f'n={int(support)}', ha='center', fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / '05_multiclass_per_class_metrics.png', dpi=180)
    plt.close(fig)

# 6. Transition probabilities from normal to abnormal classes.
trans_path = ROOT / 'runs' / 'working_type_transitions_combined' / 'working_type_transition_probabilities.csv'
if trans_path.exists():
    trans = pd.read_csv(trans_path, index_col=0)
    if '正常' in trans.index:
        normal_row = trans.loc['正常'].drop(labels=['正常'], errors='ignore').sort_values(ascending=True) * 100
        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.barh(normal_row.index, normal_row.values, color='#B279A2')
        ax.set_xlabel('转移概率 (%)')
        ax.set_title('从正常工况转移到各异常工况的历史概率')
        for bar in bars:
            w = bar.get_width()
            ax.text(w + max(normal_row.values) * 0.02, bar.get_y() + bar.get_height()/2, f'{w:.4f}%', va='center')
        fig.tight_layout()
        fig.savefig(OUT / '06_normal_to_abnormal_transition_probs.png', dpi=180)
        plt.close(fig)

# 7. Pure normal removal comparison.
fig, ax = plt.subplots(figsize=(7.5, 5))
labels = ['保留纯正常段', '剔除纯正常段']
values = [0.5739, 0.4389]
bars = ax.bar(labels, values, color=['#2F855A', '#E45756'])
ax.set_ylim(0, 0.7)
ax.set_ylabel('Test Macro F1')
ax.set_title('纯正常段不能直接删除：删除后测试效果下降')
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.015, f'{h:.3f}', ha='center', fontweight='bold')
fig.tight_layout()
fig.savefig(OUT / '07_pure_normal_segment_ablation.png', dpi=180)
plt.close(fig)

print(f'Generated figures under: {OUT}')
for p in sorted(OUT.glob('*.png')):
    print(p.name)
