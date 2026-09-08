from __future__ import annotations

"""Rebuild the patent/software-copyright submission documents in the reference style.

The reference set contains two visual languages:
* patent documents: Chinese technical manuscript with an ``附件10.3`` cover;
* software-copyright documents: clean manual cover/TOC and dark code blocks.

This script deliberately keeps the current project's factual content, while making the
DOCX structure consistent and keeping the duplicated ``word_submission`` package in sync.
"""

from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor


ROOT = Path(r"C:\Workspace\Intelligent_Fracturing_Prediction")
DELIVERABLES = ROOT / "专利与软著" / "deliverables"
PATENT_DIR = DELIVERABLES / "专利"
COPYRIGHT_DIR = DELIVERABLES / "软著"
PATENT_PACKAGE = DELIVERABLES / "patent_part2_part3" / "word_submission"
COPYRIGHT_PACKAGE = DELIVERABLES / "software_copyright_part2_part3" / "word_submission"
COPYRIGHT_SOURCE_ARCHIVE = DELIVERABLES / "software_copyright_part2_part3" / "source_code_submission"
ASSET_DIR = ROOT / ".docx_qa_20260830" / "submission_assets"
ASSET_DIR.mkdir(parents=True, exist_ok=True)

REF_LOGO = ROOT / ".docx_qa_20260830" / "assets" / "ref_image1.png"
PATENT_FIGURES = ROOT / ".docx_qa_20260830" / "patent_figures"

NAVY = "173553"
BLUE = "2F75B5"
LIGHT_BLUE = "E8F0F7"
TEXT = "203B59"
MUTED = "777777"
CODE_BG = "111315"
CODE_TEXT = "E9EEF3"


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=90, start=110, bottom=90, end=110) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for m, v in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{m}"))
        if node is None:
            node = OxmlElement(f"w:{m}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(v))
        node.set(qn("w:type"), "dxa")


def set_row_cant_split(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    node = OxmlElement("w:cantSplit")
    tr_pr.append(node)


def set_row_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    node = OxmlElement("w:tblHeader")
    node.set(qn("w:val"), "true")
    tr_pr.append(node)


def set_run(run, *, font="宋体", size=12, bold=False, italic=False, color=TEXT, east_asia=None) -> None:
    run.font.name = font
    run._element.rPr.rFonts.set(qn("w:eastAsia"), east_asia or font)
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


def set_para(p, *, align=WD_ALIGN_PARAGRAPH.JUSTIFY, line=1.5, before=0, after=0,
             first_indent=0.75, left=0, right=0, keep=False) -> None:
    fmt = p.paragraph_format
    fmt.alignment = align
    fmt.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    fmt.line_spacing = line
    fmt.space_before = Pt(before)
    fmt.space_after = Pt(after)
    fmt.first_line_indent = Cm(first_indent) if first_indent else Cm(0)
    fmt.left_indent = Cm(left) if left else Cm(0)
    fmt.right_indent = Cm(right) if right else Cm(0)
    fmt.keep_with_next = keep


def add_text(doc, text="", *, size=12, bold=False, color=TEXT, align=WD_ALIGN_PARAGRAPH.JUSTIFY,
             first=0.75, before=0, after=0, keep=False, font="宋体", italic=False):
    p = doc.add_paragraph()
    set_para(p, align=align, first_indent=first, before=before, after=after, keep=keep)
    if text:
        r = p.add_run(text)
        set_run(r, font=font, size=size, bold=bold, italic=italic, color=color)
    return p


def add_heading(doc, text, level=1):
    sizes = {1: 16, 2: 14, 3: 12.5}
    p = doc.add_paragraph()
    set_para(p, align=WD_ALIGN_PARAGRAPH.LEFT, first_indent=0, before=8 if level == 1 else 4,
             after=3, keep=True)
    r = p.add_run(text)
    set_run(r, font="黑体", size=sizes.get(level, 12), bold=True, color=NAVY)
    return p


def add_bullet(doc, text, *, size=12, color=TEXT, level=0):
    p = doc.add_paragraph()
    set_para(p, align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_indent=-0.55, left=1.0 + level * 0.5,
             line=1.35, after=0)
    r = p.add_run("•  " + text)
    set_run(r, size=size, color=color)
    return p


def add_numbered(doc, number, text, *, size=12):
    p = doc.add_paragraph()
    set_para(p, align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_indent=-0.55, left=1.0, line=1.45)
    r = p.add_run(f"{number}.  {text}")
    set_run(r, size=size, color=TEXT)
    return p


def add_table(doc, headers: Iterable[str], rows: Iterable[Iterable[str]], widths: Iterable[float],
              header_fill=LIGHT_BLUE, font_size=10.5):
    headers = list(headers)
    rows = [list(row) for row in rows]
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    for i, width in enumerate(widths):
        for cell in table.columns[i].cells:
            cell.width = Cm(width)
    for i, value in enumerate(headers):
        cell = table.rows[0].cells[i]
        set_cell_shading(cell, header_fill)
        set_cell_margins(cell)
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        p = cell.paragraphs[0]
        set_para(p, align=WD_ALIGN_PARAGRAPH.CENTER, line=1.1, first_indent=0)
        r = p.add_run(value)
        set_run(r, font="宋体", size=font_size, bold=True, color=NAVY)
    set_row_cant_split(table.rows[0])
    set_row_header(table.rows[0])
    for row_values in rows:
        row = table.add_row()
        set_row_cant_split(row)
        for i, value in enumerate(row_values):
            cell = row.cells[i]
            set_cell_margins(cell)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            p = cell.paragraphs[0]
            set_para(p, align=WD_ALIGN_PARAGRAPH.LEFT, line=1.1, first_indent=0)
            r = p.add_run(str(value))
            set_run(r, font="宋体", size=font_size, color=TEXT)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)
    return table


def add_callout(doc, label, text):
    table = doc.add_table(rows=1, cols=1)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = table.cell(0, 0)
    set_cell_shading(cell, "F3F6F9")
    set_cell_margins(cell, top=130, start=160, bottom=130, end=160)
    p = cell.paragraphs[0]
    set_para(p, align=WD_ALIGN_PARAGRAPH.LEFT, line=1.35, first_indent=0)
    r = p.add_run(label + "：")
    set_run(r, font="宋体", size=11.5, bold=True, color=NAVY)
    r = p.add_run(text)
    set_run(r, font="宋体", size=11.5, color=TEXT)
    set_row_cant_split(table.rows[0])
    set_row_header(table.rows[0])
    doc.add_paragraph().paragraph_format.space_after = Pt(0)
    return table


def add_picture(doc, path: Path, width=15.0, caption=None, alt_text=""):
    if not path.exists():
        return
    p = doc.add_paragraph()
    set_para(p, align=WD_ALIGN_PARAGRAPH.CENTER, line=1.0, first_indent=0, keep=True)
    run = p.add_run()
    run.add_picture(str(path), width=Cm(width))
    inline = run._r.xpath(".//wp:inline")
    if inline:
        doc_pr = inline[0].find(qn("wp:docPr"))
        if doc_pr is not None:
            doc_pr.set("descr", alt_text or caption or path.stem)
    if caption:
        p = add_text(doc, caption, size=10.5, color=MUTED, align=WD_ALIGN_PARAGRAPH.CENTER,
                     first=0, before=1, after=3, keep=True)
    return p


def set_a4(section, *, left=2.54, right=2.54, top=2.35, bottom=2.35):
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.left_margin = Cm(left)
    section.right_margin = Cm(right)
    section.top_margin = Cm(top)
    section.bottom_margin = Cm(bottom)
    section.header_distance = Cm(1.15)
    section.footer_distance = Cm(1.15)


def add_page_number(paragraph):
    run = paragraph.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    fld_sep = OxmlElement("w:fldChar")
    fld_sep.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = "1"
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_sep)
    run._r.append(text)
    run._r.append(fld_end)
    set_run(run, font="宋体", size=9, color=MUTED)


def save_and_sync(doc: Document, outputs: list[Path]) -> Path:
    """Save once, then copy the canonical bytes to every delivery location."""
    primary = outputs[0]
    primary.parent.mkdir(parents=True, exist_ok=True)
    doc.save(primary)
    payload = primary.read_bytes()
    for out in outputs[1:]:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(payload)
    return primary


def add_manual_header_footer(doc):
    for section in doc.sections:
        header = section.header
        p = header.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        if REF_LOGO.exists():
            run = p.add_run()
            run.add_picture(str(REF_LOGO), width=Cm(1.25))
            inline = run._r.xpath(".//wp:inline")
            if inline:
                doc_pr = inline[0].find(qn("wp:docPr"))
                if doc_pr is not None:
                    doc_pr.set("descr", "中国石化标识")
        r = p.add_run("  智能压裂项目 · 用户手册")
        set_run(r, font="宋体", size=9, color=MUTED)
        footer = section.footer
        p = footer.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        r = p.add_run("Internal submission draft · Page ")
        set_run(r, font="宋体", size=9, color=MUTED)
        add_page_number(p)


def add_patent_header_footer(doc, title):
    for section in doc.sections:
        header = section.header
        p = header.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        r = p.add_run("智能压裂项目 · " + title)
        set_run(r, font="宋体", size=9, color=MUTED)
        footer = section.footer
        p = footer.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        r = p.add_run("—")
        set_run(r, font="宋体", size=9, color=MUTED)
        add_page_number(p)
        r = p.add_run("—")
        set_run(r, font="宋体", size=9, color=MUTED)


def set_core(doc, title, subject):
    doc.core_properties.title = title
    doc.core_properties.subject = subject
    doc.core_properties.author = ""
    doc.core_properties.comments = "依据项目示例材料统一整理的交付稿"


def add_cover_meta(doc, rows, callout_label, callout_text):
    add_table(doc, ["项目", "智能压裂预测与联合动态联调平台"], [], [3.6, 13.0]) if False else None
    add_table(doc, ["项目", "智能压裂预测与联合动态联调平台"], [], [3.6, 13.0])
    # Replace the one-row helper table with a complete metadata table by adding rows.
    table = doc.tables[-1]
    for key, value in rows:
        row = table.add_row()
        set_row_cant_split(row)
        for i, text in enumerate((key, value)):
            cell = row.cells[i]
            set_cell_margins(cell)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            if i == 0:
                set_cell_shading(cell, LIGHT_BLUE)
            p = cell.paragraphs[0]
            set_para(p, align=WD_ALIGN_PARAGRAPH.LEFT, line=1.1, first_indent=0)
            r = p.add_run(text)
            set_run(r, size=10.5, bold=(i == 0), color=NAVY if i == 0 else TEXT)
    add_callout(doc, callout_label, callout_text)


def build_patent04() -> Path:
    doc = Document()
    set_a4(doc.sections[0], left=2.35, right=2.35, top=2.05, bottom=2.0)
    add_patent_header_footer(doc, "专利实施例与实验依据")
    set_core(doc, "实施例与实验依据", "专利补充材料")

    add_text(doc, "专利实施例", size=13, bold=True, color=BLUE, align=WD_ALIGN_PARAGRAPH.LEFT,
             first=0, before=2, after=1)
    add_text(doc, "实施例与实验依据", size=25, bold=True, color=NAVY, align=WD_ALIGN_PARAGRAPH.LEFT,
             first=0, before=0, after=2, font="黑体")
    add_text(doc, "双场景数字孪生、KG-EnKF、Piggy-Bank 与建议式决策", size=13, color=MUTED,
             align=WD_ALIGN_PARAGRAPH.LEFT, first=0, after=8)
    add_cover_meta(
        doc,
        [
            ("材料版本", "V1.0 · 2026-08-30"),
            ("材料性质", "专利技术交底书的实施例、实验数据和代码依据补充"),
            ("数据边界", "仅含脱敏代码、合成验证和公开技术描述"),
        ],
        "递交定位",
        "本材料用于说明方法如何实现、如何验证以及当前证据边界；申请主体、发明人和法律性表述由申请人及专利代理师最终审定。",
    )

    add_heading(doc, "1. 适用对象与证据边界", 1)
    add_text(doc, "本材料围绕主交底书所述的双场景压裂数字孪生与阶段总液量均衡调控方法，给出可复现的实施流程、数学定义、代码映射和实验结果。材料将现场已经具备的施工压力、排量、砂比、累计液量、阶段信息、井轨迹以及可选的 DAS/FracMonitor 解释结果区分开来，不把模型推导结果写成现场直接测量值。")
    add_callout(doc, "证据等级", "函数与流程测试、合成账本守恒和单段离线对照可作为实现依据；现场泵控闭环、直接单簇液量控制和旧版 PyFrac 长时动态验收仍属于后续验证范围。")

    add_heading(doc, "2. 实施例一：无 DAS 压力在线校正", 1)
    add_text(doc, "无 DAS 场景只使用施工曲线和压力换算结果，不加载分簇观测。系统从井口压力、静液柱、管柱摩阻、射孔压降、压力偏置和累计注入量构造井底压力，再将压力序列送入压力-only 的 PKN+EnKF。阶段级裂缝半长、缝宽、净压力、六簇模型分配和均衡度属于数字孪生推导结果。")
    add_text(doc, "井底压力的工程换算写为：", first=0, before=2, after=0)
    add_callout(doc, "公式（1）", "P_bhp = P_wh + ρgH − ΔP_pipe − ΔP_perf + b_p，式中 P_wh 为井口压力，ρgH 为静液柱，ΔP_pipe 为管柱摩阻，ΔP_perf 为射孔压降，b_p 为压力偏置。")
    add_text(doc, "压力校正后的值作为观测量 y_t。PKN 前向模型根据排量、累计液量、滤失、黏度、弹性模量、最小水平主应力和断裂韧度计算状态预测 ŷ_t；EnKF 使用观测创新 y_t − ŷ_t 更新参数和状态，并将后验参数重新送入 PKN 进行展示。")
    add_bullet(doc, "对应场景注册：DT-Crack/data_fusion/scenario.py")
    add_bullet(doc, "压力换算：DT-Crack/data_fusion/pressure_schedule_adapter.py")
    add_bullet(doc, "压力-only 同化：DT-Crack/inversion/pressure_only_enkf.py")
    add_bullet(doc, "APP 展示：App/dt_pressure_only_model.py")

    add_heading(doc, "3. 实施例二：有 DAS/FracMonitor 分簇观测", 1)
    add_text(doc, "有 DAS 场景在压力观测之外接入六簇解释结果。观测进入 EnKF 之前执行时间戳、重复时间、时间间隔、六簇完整性、累计量单调性、非负性、总量一致性和覆盖率检查。无效样本被排除并写入质量报告，避免把缺失簇静默当成零。")
    add_text(doc, "对六簇液量份额 q_i 归一化后，均衡度可用归一化熵表示：", first=0, before=2, after=0)
    add_callout(doc, "公式（2）", "B_entropy = −Σᵢ p_i ln(p_i) / ln(N)，其中 p_i = q_i / Σⱼq_j，N 为有效簇数。Gini 指数用于表达份额离散程度，数值越低通常表示越均匀。")
    add_text(doc, "对于具有三维响应位置的样本，响应效率采用井段到响应事件的三维距离除以首次响应时累计注入液量；时间响应速率则由响应距离与响应时间差构造。系统同时保存指标来源，区分真实分簇观测和模型推导分配。")
    add_bullet(doc, "对应代码：DT-Crack/data_fusion/observation_quality.py")
    add_bullet(doc, "分簇适配：DT-Crack/data_fusion/fiber_api_adapter.py")
    add_bullet(doc, "观测算子：DT-Crack/data_fusion/fiber_observation_operator.py")
    add_bullet(doc, "响应指标：DT-Crack/inversion/segment_response_metrics.py")

    fig1 = PATENT_FIGURES / "figure1_flow.png"
    if fig1.exists():
        add_picture(doc, fig1, width=15.0, caption="图 1  双场景观测、知识引导同化与阶段液量反馈实施流程",
                    alt_text="双场景观测、知识引导同化与阶段液量反馈流程图")

    add_heading(doc, "4. 实施例三：KG-EnKF 先验桥接", 1)
    add_text(doc, "KG-EnKF 不把知识图谱直接当作观测值，而是将可解释规则转换成 EnKF 可消费的统计量。规则信号包括压力上升、压力斜率、高砂比、排量下降及其组合。系统根据规则强度和观测质量，分别调整先验均值、先验不确定性、参数协方差和压力观测噪声。")
    add_table(doc, ["模式", "计算作用", "是否改变先验均值", "是否引入相关性"], [
        ["off", "普通 PKN+EnKF 基线", "否", "否"],
        ["uncertainty_only", "扩大先验与过程噪声", "否", "否"],
        ["soft_prior", "在不确定性基础上施加小幅均值偏移", "是", "否"],
        ["soft_correlated", "均值、协方差、观测置信度联合调节", "是", "是"],
    ], [3.2, 6.4, 3.7, 3.3], font_size=9.8)
    add_text(doc, "协方差桥接可表示为：", first=0, before=2, after=0)
    add_callout(doc, "公式（3）", "m₀′ = m₀ + α·s_rule；P₀′ = D(s)·R_rule·D(s)；R_obs′ = R_obs / c_obs，其中 α 为软偏移强度，s_rule 为规则信号，D(s) 为不确定性缩放矩阵，R_rule 为规则相关结构，c_obs 为观测置信度。")
    add_text(doc, "在每个更新点，系统保存先验快照，完成集合预测、观测创新、Kalman 增益计算和后验重算。输出同时包含先验、后验、更新误差、计算时间和审计信息。若输入或预测出现非有限值，则恢复最近一次有效快照。")
    add_bullet(doc, "核心实现：DT-Crack/inversion/knowledge_guided_enkf.py")
    add_bullet(doc, "模式对照：DT-Crack/inversion/compare_knowledge_guided_modes.py")
    add_bullet(doc, "单元测试：DT-Crack/tests/test_knowledge_guided_enkf.py")
    add_bullet(doc, "设计说明：DT-Crack/docs/知识图谱增强EnKF实现说明.md")

    add_heading(doc, "5. 实施例四：14 维参数化分簇正演", 1)
    add_text(doc, "在假设六簇的条件下，模型状态扩展为 14 维。前 5 项描述压力和裂缝响应，接着 6 项描述簇进液能力，最后 3 项描述应力干扰、边界效应和液量分配指数。模型只能由总排量和参数推导簇间分配，不能把总排量建议表述为向某一簇直接注入指定体积。")
    add_table(doc, ["参数组", "维数", "示例参数", "对计算的作用"], [
        ["压力/裂缝", "5", "E′、C_L、μ、σ_min、K_IC", "控制压力、滤失与裂缝几何响应"],
        ["簇进液能力", "6", "κ_c1 … κ_c6", "表达六簇相对入流倾向"],
        ["交互/分配", "3", "γ_sh、γ_e、α_a", "表达应力阴影、边界释放和分配非线性"],
    ], [3.0, 2.0, 5.2, 6.4], font_size=9.8)
    add_text(doc, "示意性的分配权重可写为：", first=0, before=2, after=0)
    add_callout(doc, "公式（4）", "w_i = κ_ci · exp(−γ_sh·I_i) · (1 + γ_e·B_i)，q_i = Q_total · w_i^α_a / Σⱼw_j^α_a。其中 I_i 表示邻簇干扰，B_i 表示边界效应，Q_total 为当前阶段总排量或总液量。")
    add_text(doc, "上述形式用于说明参数在模型中的可解释位置；实际计算由工程配置和代码实现确定，参数后验属于模型状态估计，不等同于现场直接测量。")

    fig2 = PATENT_FIGURES / "figure2_cluster_feedback.png"
    if fig2.exists():
        add_picture(doc, fig2, width=15.0, caption="图 2  六簇响应、均衡度评价与阶段总液量建议",
                    alt_text="六簇响应份额、均衡目标与阶段总液量反馈示意图")

    add_heading(doc, "6. 实施例五：Piggy-Bank 阶段总液量账本", 1)
    add_text(doc, "Piggy-Bank 的调控对象是下一时间窗或下一阶段的总液量计划，而不是某一簇的独立泵注量。当某阶段响应偏快时，尚未泵入的计划液量形成释放记录；当后续阶段响应偏慢且满足安全、储备和数据质量条件时，释放量可形成支取建议。建议、批准、实际执行和账本提交分开记录。")
    add_callout(doc, "公式（5）", "Bₛ₊₁ = Bₛ + ΔVₛ^release − ΔVₛ^draw；Vₛ^recommend = Vₛ^plan − ΔVₛ^release + ΔVₛ^draw。总量守恒要求：ΣV_actual = ΣV_plan + ΣV_release − ΣV_draw。")
    add_table(doc, ["验证内容", "输入", "结果", "证据性质"], [
        ["阶段 1 释放", "合成阶段响应过快", "10 m³ 进入待支取余额", "账本逻辑验证"],
        ["阶段 2 支取", "合成阶段响应偏慢且安全", "支取 10 m³", "账本逻辑验证"],
        ["最终余额", "释放与支取完成", "0 m³", "守恒检查"],
        ["守恒残差", "计划、释放、支取、执行量", "0 m³", "合成数据，不代表现场因果"],
    ], [4.0, 5.0, 4.0, 4.0], font_size=9.8)
    add_bullet(doc, "调度器：DT-Crack/inversion/piggy_bank_allocator.py")
    add_bullet(doc, "账本：DT-Crack/inversion/stage_liquid_ledger.py")
    add_bullet(doc, "顺序验证：DT-Crack/inversion/validate_piggy_bank_sequence.py")
    add_bullet(doc, "测试：DT-Crack/tests/test_piggy_bank.py")
    fig3 = PATENT_FIGURES / "figure3_ledger.png"
    if fig3.exists():
        add_picture(doc, fig3, width=15.0, caption="图 3  多阶段 Piggy-Bank 建议、确认、执行与账本提交",
                    alt_text="多阶段 Piggy-Bank 建议、确认、执行与账本提交关系图")

    add_heading(doc, "7. 实施例六：智能体建议与高保真模型接口", 1)
    add_text(doc, "智能体以数字孪生状态、施工工况、压力风险、分簇均衡度和成本为输入，输出排量与砂比建议。建议经过安全门禁、模型有效性检查和人工确认后才能进入现场操作流程。HMI-KE 中保留 PPO、SAC 和 TD3 候选策略及统一奖励接口，当前定位为离线建议式决策模块。")
    add_bullet(doc, "风险决策与奖励：HMI-KE/decision_engine/decision_engine.py、integrated_reward.py")
    add_bullet(doc, "约束与环境：HMI-KE/decision_engine/pump_schedule_constraints.py、HMI-KE/rl/fracturing_env.py")
    add_bullet(doc, "训练入口：HMI-KE/train_rl_control_agent.py")
    add_text(doc, "PyFrac 适配器、代理模型和收敛验证入口可以作为离线高保真参考或教师样本来源。由于旧版连续前缘重构和状态重启仍需持续验证，本材料仅写明可选接口和验证路径，不把 PyFrac 长时动态验收写成已完成的默认在线能力。")

    add_heading(doc, "8. 代码—证据矩阵", 1)
    add_table(doc, ["技术环节", "代码模块", "当前证据", "材料口径"], [
        ["双场景注册与压力换算", "data_fusion/scenario.py；pressure_schedule_adapter.py", "流程测试、配置检查", "可作为实现依据"],
        ["KG-EnKF 先验桥接", "inversion/knowledge_guided_enkf.py", "单段离线留出对照", "写明数据范围，不外推"],
        ["六簇分配与均衡度", "inversion/physics.py；segment_response_metrics.py", "单元测试与合成示例", "模型推导与观测分别标注"],
        ["Piggy-Bank 账本", "piggy_bank_allocator.py；stage_liquid_ledger.py", "五阶段合成守恒", "写账本机制，不写现场因果"],
        ["智能体建议", "HMI-KE/rl；decision_engine", "离线候选策略对比", "建议式、需审核"],
        ["PyFrac 高保真接口", "forward_models/pyfrac_adapter.py", "适配与收敛入口", "可选路径，持续验证"],
    ], [4.1, 6.0, 4.0, 3.0], font_size=9.0)

    add_heading(doc, "9. 小结", 1)
    add_text(doc, "本实施材料将主交底书中的方法拆解为可复现的场景输入、压力换算、PKN 正演、KG-EnKF 更新、分簇均衡度计算、阶段总液量账本和建议式决策步骤，并将每一步映射到当前代码。实验结果仅按已有数据和验证类型表述，不把合成守恒、单段离线对照或模型推导指标扩大为现场闭环效果。")

    outputs = [PATENT_DIR / "04_实施例与实验依据.docx", PATENT_PACKAGE / "04_实施例与实验依据.docx"]
    return save_and_sync(doc, outputs)


def build_patent07() -> Path:
    doc = Document()
    set_a4(doc.sections[0], left=2.35, right=2.35, top=2.05, bottom=2.0)
    add_patent_header_footer(doc, "专利申报信息")
    set_core(doc, "专利申报信息模板", "专利行政信息填报模板")

    add_text(doc, "专利申报信息", size=13, bold=True, color=BLUE, align=WD_ALIGN_PARAGRAPH.LEFT,
             first=0, before=2, after=1)
    add_text(doc, "专利申报信息模板", size=25, bold=True, color=NAVY, align=WD_ALIGN_PARAGRAPH.LEFT,
             first=0, after=2, font="黑体")
    add_text(doc, "申请主体、发明人和联系人信息待补表", size=13, color=MUTED,
             align=WD_ALIGN_PARAGRAPH.LEFT, first=0, after=8)
    add_cover_meta(
        doc,
        [
            ("材料版本", "V1.0 · 2026-08-30"),
            ("材料性质", "专利申请行政信息填报模板"),
            ("数据边界", "仅含技术名称和待填写字段"),
        ],
        "递交定位",
        "本页仅提供待填写字段，不代填申请人、发明人、日期和盖章信息。",
    )

    add_heading(doc, "1. 待填写行政信息", 1)
    add_table(doc, ["项目", "待填写内容"], [
        ["申请人", ""],
        ["发明名称", "一种基于知识引导参数同化与分簇响应反馈的多簇压裂阶段液量均衡调控方法、系统、设备及存储介质"],
        ["第一发明人", ""],
        ["其他发明人", ""],
        ["联系人", ""],
        ["联系电话/邮箱", ""],
        ["优先权/申请日期", ""],
        ["是否职务发明", "由单位确认"],
    ], [4.2, 12.4], font_size=10.5)
    add_heading(doc, "2. 发明人贡献确认要点", 1)
    add_text(doc, "发明人确认时，应根据实际贡献填写并由单位及代理师核验。建议逐项确认以下技术环节：")
    for text in [
        "双场景数据注册、无 DAS 井口—井底压力换算和压力偏置校正；",
        "PKN 状态建模、KG-EnKF 先验均值/协方差/观测置信度桥接；",
        "六簇进液能力、应力干扰、边界效应和液量分配参数化；",
        "分簇响应效率、均衡度、Gini、熵和 Piggy-Bank 阶段账本；",
        "智能体安全建议、人工确认、回滚和审计记录；",
        "代码实现、实验设计和可复现性材料。",
    ]:
        add_bullet(doc, text)
    add_callout(doc, "说明", "如果某一环节仅为后续接口、工程设想或尚未完成现场验证，应由代理师决定是否作为可选实施方式写入，不应填写为已经完成的现场功能。")

    outputs = [PATENT_DIR / "07_申报信息模板.docx", PATENT_PACKAGE / "07_申报信息模板.docx"]
    return save_and_sync(doc, outputs)


def _load_font(size=28, bold=False):
    candidates = [r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc",
                  r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\arial.ttf"]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def make_manual_figures() -> list[Path]:
    paths = []
    # A compact diagram in the same visual spirit as the reference manual screenshots.
    img = Image.new("RGB", (1800, 700), "#F7F9FB")
    d = ImageDraw.Draw(img)
    title_font = _load_font(40, True)
    box_font = _load_font(27, True)
    small_font = _load_font(23)
    d.text((70, 35), "双场景数据到可视化结果", fill="#173553", font=title_font)
    boxes = [(80, 200, 330, 455, "施工数据", "压力 / 排量\n砂比 / 液量"),
             (410, 200, 660, 455, "场景识别", "无 DAS\n有 DAS"),
             (740, 200, 1030, 455, "PKN + KG-EnKF", "压力状态\n分簇参数"),
             (1110, 200, 1410, 455, "响应评价", "半缝长 / 缝宽\n均衡度 / Gini"),
             (1490, 200, 1720, 455, "建议输出", "液量建议\n动作审计")]
    for x1, y1, x2, y2, head, body in boxes:
        d.rounded_rectangle((x1, y1, x2, y2), 18, fill="#E8F0F7", outline="#2F75B5", width=4)
        d.text((x1 + 20, y1 + 28), head, fill="#2F75B5", font=box_font)
        d.multiline_text((x1 + 20, y1 + 105), body, fill="#203B59", font=small_font, spacing=18)
    for x in [350, 680, 1050, 1430]:
        d.line((x, 327, x + 55, 327), fill="#2F75B5", width=6)
        d.polygon([(x + 55, 327), (x + 38, 315), (x + 38, 339)], fill="#2F75B5")
    p = ASSET_DIR / "manual_scenario_flow.png"
    img.save(p)
    paths.append(p)

    img = Image.new("RGB", (1800, 920), "#142633")
    d = ImageDraw.Draw(img)
    title_font = _load_font(36, True)
    lab_font = _load_font(24, True)
    text_font = _load_font(22)
    d.text((50, 30), "双场景数字孪生 · 示例结果面板", fill="#25C8C8", font=title_font)
    # pressure chart
    d.rounded_rectangle((55, 120, 1120, 565), 12, fill="#1C3040", outline="#3D5B6F", width=3)
    d.text((80, 145), "压力对比：观测 / PKN 先验 / EnKF 后验", fill="#EAF2F7", font=lab_font)
    for y in [230, 320, 410, 500]:
        d.line((95, y, 1080, y), fill="#3D5B6F", width=2)
    pts = [(100, 470), (180, 445), (260, 452), (350, 390), (450, 300), (560, 330), (660, 270), (760, 315), (880, 255), (1000, 280), (1060, 250)]
    d.line(pts, fill="#4FA3E3", width=6)
    pts2 = [(100, 480), (180, 455), (260, 445), (350, 410), (450, 350), (560, 325), (660, 315), (760, 305), (880, 295), (1000, 288), (1060, 282)]
    d.line(pts2, fill="#FFB52F", width=5)
    pts3 = [(100, 475), (180, 450), (260, 448), (350, 402), (450, 335), (560, 330), (660, 295), (760, 310), (880, 280), (1000, 288), (1060, 270)]
    d.line(pts3, fill="#21C6C4", width=5)
    for x, color, label in [(1200, "#4FA3E3", "观测"), (1350, "#FFB52F", "PKN先验"), (1500, "#21C6C4", "EnKF后验")]:
        d.line((x, 175, x + 45, 175), fill=color, width=6); d.text((x + 60, 160), label, fill="#EAF2F7", font=text_font)
    # cluster bars
    d.rounded_rectangle((1160, 250, 1745, 565), 12, fill="#1C3040", outline="#3D5B6F", width=3)
    d.text((1190, 275), "六簇分配与均衡基准", fill="#EAF2F7", font=lab_font)
    base_y = 520
    vals = [95, 135, 116, 145, 125, 102]
    for i, v in enumerate(vals):
        x = 1200 + i * 82
        d.rectangle((x, base_y - v, x + 42, base_y), fill="#21C6C4")
        d.text((x + 2, base_y + 12), f"簇{i+1}", fill="#EAF2F7", font=text_font)
    d.line((1185, 405, 1720, 405), fill="#FFB52F", width=4)
    d.text((1190, 370), "均衡基准", fill="#FFB52F", font=text_font)
    # bottom cards
    for i, (x, head, value, color) in enumerate([(55, "半缝长", "阶段演化", "#4FA3E3"), (390, "缝宽", "阶段演化", "#21C6C4"), (725, "均衡度", "观测/推导", "#FFB52F"), (1060, "建议动作", "保持/调整", "#E9855B"), (1395, "审计", "可追溯", "#9B88D9")]):
        d.rounded_rectangle((x, 650, x + 285, 825), 12, fill="#1C3040", outline="#3D5B6F", width=3)
        d.text((x + 22, 680), head, fill=color, font=lab_font)
        d.text((x + 22, 750), value, fill="#EAF2F7", font=text_font)
    p = ASSET_DIR / "manual_visual_dashboard.png"
    img.save(p)
    paths.append(p)
    return paths


def build_manual() -> Path:
    figures = make_manual_figures()
    doc = Document()
    set_a4(doc.sections[0], left=2.35, right=2.35, top=2.2, bottom=2.0)
    add_manual_header_footer(doc)
    set_core(doc, "智能压裂双场景数字孪生与安全建议软件 V1.0 用户手册", "软件著作权用户手册")

    # Cover page, following the example manual's title/meta/callout hierarchy.
    add_text(doc, "用户手册", size=13, bold=True, color=BLUE, align=WD_ALIGN_PARAGRAPH.LEFT,
             first=0, before=2, after=3)
    add_text(doc, "智能压裂双场景数字孪生与安全建议软件", size=24, bold=True, color=NAVY,
             align=WD_ALIGN_PARAGRAPH.LEFT, first=0, after=0, font="黑体")
    add_text(doc, "V1.0", size=24, bold=True, color=NAVY, align=WD_ALIGN_PARAGRAPH.LEFT,
             first=0, after=4, font="黑体")
    add_text(doc, "软件著作权用户手册正式整理版", size=13, color=MUTED, align=WD_ALIGN_PARAGRAPH.LEFT,
             first=0, after=8)
    add_cover_meta(
        doc,
        [
            ("材料版本", "V1.0 · 2026-08-30"),
            ("材料性质", "软件著作权用户手册正式整理版"),
            ("数据边界", "仅含脱敏代码、合成样例和公开技术描述"),
        ],
        "递交定位",
        "本手册说明软件的启动、场景切换、图表查看、结果导出和常见问题处理。最终申报前可补入脱敏界面截图。",
    )
    add_heading(doc, "1. 关于本手册", 1)
    add_text(doc, "本手册面向软件使用、演示和材料整理人员，说明智能压裂双场景数字孪生与安全建议软件的功能边界、操作顺序和输出含义。软件按三部分组织：工况识别与风险预测、双场景数字孪生、智能决策与人机协同。图表优先表达观测、模型和后验之间的时间关系，数值用于补充当前点状态。")
    add_heading(doc, "2. 软件概览", 1)
    add_text(doc, "软件读取施工压力、排量、砂比、累计液量、阶段信息、井轨迹以及可选的 DAS/FracMonitor 解释结果；根据数据条件注册无 DAS 压力场景或有 DAS 分簇观测场景；通过 PKN 正演和 KG-EnKF 参数更新输出压力、裂缝几何、分簇响应、均衡度和建议式动作。")
    add_picture(doc, figures[0], width=15.4, caption="图 1  软件从数据输入到可视化结果的主要路径",
                alt_text="软件数据输入、场景识别、PKN与KG-EnKF、响应评价和建议输出流程图")
    doc.add_page_break()

    # Static TOC mirrors the example's dedicated TOC page.
    add_text(doc, "目 录", size=22, bold=True, color=NAVY, align=WD_ALIGN_PARAGRAPH.LEFT,
             first=0, after=12, font="黑体")
    toc = [
        ("1 关于本手册", "3"), ("2 软件概览", "3"), ("3 启动与预检", "4"),
        ("4 第一部分：工况识别与风险预测", "5"), ("5 第二部分：双场景数字孪生", "7"),
        ("6 查看 PKN 与 KG-EnKF", "10"), ("7 分簇响应、均衡度与 Piggy-Bank", "12"),
        ("8 第三部分：智能决策", "14"), ("9 结果导出与常见问题", "16"),
    ]
    for label, page in toc:
        p = doc.add_paragraph()
        set_para(p, align=WD_ALIGN_PARAGRAPH.LEFT, first_indent=0, line=1.35, after=2)
        r = p.add_run(label)
        set_run(r, size=12.5, color=TEXT)
        r = p.add_run(" " + "." * max(8, 62 - len(label) * 2) + " ")
        set_run(r, size=11, color="A0A0A0")
        r = p.add_run(page)
        set_run(r, size=12.5, color=TEXT)
    doc.add_page_break()

    add_heading(doc, "3. 启动与预检", 1)
    add_text(doc, "在项目根目录启动软件：")
    add_callout(doc, "PowerShell", "python App\\run_app.py")
    add_text(doc, "首次运行或更换环境时，可先执行环境预检和无界面检查。预检用于确认解释器、依赖、场景注册表、配置文件和输出目录可以读取。")
    add_callout(doc, "PowerShell", "python App\\run_app.py --preflight\npython App\\run_app.py --no-gui")
    add_heading(doc, "3.1 启动后的检查", 2)
    for text in [
        "确认窗口右上角状态信息开始刷新，时间轴和当前场景名称正常显示；",
        "确认数据目录、配置文件和结果输出目录被正确识别；",
        "确认图表有有效时间范围，不把综合数据段误当成单井段；",
        "切换场景或井段后，等待当前场景的缓存和图表同步刷新。",
    ]:
        add_bullet(doc, text)

    add_heading(doc, "4. 第一部分：工况识别与风险预测", 1)
    add_heading(doc, "4.1 选择井段", 2)
    add_text(doc, "在井段选择区域选择井号、阶段或数据文件。页面显示数据来源、时间覆盖和当前井段。综合数据与单井数据分开管理，不把综合数据作为可切换的独立井段展示。切换完成后，施工曲线、风险结果和时间轴应同步刷新。")
    add_heading(doc, "4.2 观察施工曲线", 2)
    add_text(doc, "施工曲线同时展示施工压力、排量、砂比、累计液量和工况区间。工况区间使用 WORKING_TYPE、DRAW_TIME 和 DRAW_TIME_END 标注，事件边界用于定位压力突变、排量变化和砂比调整。")
    add_heading(doc, "4.3 查看风险结果", 2)
    add_text(doc, "风险结果包括风险概率、风险等级、事件类型和对应时间区间。图表优先显示风险概率随时间变化，指标卡仅用于补充当前点；没有独立现场事件标签时，结果应作为当前规则或模型输出查看。")

    add_heading(doc, "5. 第二部分：双场景数字孪生", 1)
    add_heading(doc, "5.1 无 DAS：压力在线校正", 2)
    add_text(doc, "选择无 DAS 场景后，系统读取施工曲线和井轨迹，按照压力换算配置得到井底压力，并运行阶段级 PKN 裂缝演化。主要图表包括压力对比、半缝长、缝宽、裂缝体积、假设簇数下的分配结果和均衡度。无 DAS 场景不加载 DAS 分簇观测。")
    add_callout(doc, "压力换算", "P_bhp = P_wh + ρgH − ΔP_pipe − ΔP_perf + b_p")
    add_heading(doc, "5.2 有 DAS：压力与分簇观测校验", 2)
    add_text(doc, "选择有 DAS 场景后，系统在压力链路基础上读取 FracMonitor/DAS 解释后的分簇结果。页面可查看六簇液量/砂量份额、首次响应、响应效率、均衡度、Gini 和熵。分簇样本在进入同化前经过质量控制，无效样本记录剔除原因。")
    add_picture(doc, figures[1], width=15.4, caption="图 2  双场景页面中的压力、分簇和状态结果示意",
                alt_text="双场景数字孪生示例结果面板")

    add_heading(doc, "6. 查看 PKN 与 KG-EnKF", 1)
    add_heading(doc, "6.1 压力曲线", 2)
    add_text(doc, "压力图建议同时查看 PKN 先验、观测压力和 EnKF 后验。横轴使用当前场景的真实覆盖时间；后验是同化后的模型状态，不是对观测曲线的简单复制。")
    add_heading(doc, "6.2 参数更新", 2)
    add_text(doc, "参数区按参数名、先验、后验和变化量/状态分列显示。参数分为压力/裂缝参数、六簇进液能力参数和交互/分配参数。KG-EnKF 支持 off、uncertainty_only、soft_prior 和 soft_correlated 四种模式。")
    add_table(doc, ["参数组", "主要内容", "页面展示"], [
        ["压力/裂缝", "E′、C_L、μ、σ_min、K_IC", "先验、后验、变化量"],
        ["簇进液能力", "κ_c1 … κ_c6", "六簇参数列"],
        ["交互/分配", "γ_sh、γ_e、α_a", "应力、边界、分配参数"],
    ], [4.0, 7.2, 5.2], font_size=10.2)
    add_heading(doc, "6.3 裂缝几何结果", 2)
    add_text(doc, "阶段级 PKN 根据压力校正结果更新半缝长、缝宽和裂缝体积。缺少真实六簇几何时，页面展示阶段级结果和模型分配，并在指标来源中区分观测、轨迹推导和模型推导。")

    add_heading(doc, "7. 分簇响应、均衡度与 Piggy-Bank", 1)
    add_heading(doc, "7.1 响应指标", 2)
    add_text(doc, "系统记录每个簇的首次响应时间、首次响应累计液量、时间响应速率和液量归一化响应效率。具有三维响应位置时，响应效率使用井段到响应事件的三维距离除以首次响应时累计注入液量。")
    add_heading(doc, "7.2 阶段总液量建议", 2)
    add_text(doc, "Piggy-Bank 只改变下一时间窗或下一阶段的总液量计划，不宣称可以直接向某一个簇注入指定体积。释放量、支取量、建议量、实际执行量和账本余额分开显示；没有现场执行量时，账本表示模型建议状态。")
    add_bullet(doc, "释放：响应过快阶段的计划液量减少建议形成待释放记录。")
    add_bullet(doc, "支取：响应偏慢阶段在安全和储备条件允许时提出增加建议。")
    add_bullet(doc, "审计：记录建议、批准、实际执行、守恒残差和回滚原因。")

    add_heading(doc, "8. 第三部分：智能决策", 1)
    add_text(doc, "智能体根据数字孪生状态、施工工况、风险、均衡度和成本输出排量/砂比建议。页面同时显示当前排量、推荐排量、当前砂比、推荐砂比、动作差值、风险等级和人工确认状态。")
    add_heading(doc, "8.1 安全建议", 2)
    add_text(doc, "当观测无效、压力风险升高、模型输出为 NaN、代理模型超出适用范围或需要人工确认时，系统回退到保持或保守规则动作，并在动作审计中记录原因。")
    add_heading(doc, "8.2 结果查看", 2)
    add_text(doc, "建议结合排量/砂比时间曲线、风险概率曲线、均衡度变化和动作审计查看，不只查看单个奖励值。模型建议属于建议式输出，现场操作仍需按实际审批流程执行。")

    add_heading(doc, "9. 结果导出与常见问题", 1)
    add_heading(doc, "9.1 结果导出", 2)
    for text in [
        "场景配置、压力换算配置、数据清单与质量报告；",
        "PKN/EnKF 参数历史、后验状态和压力对比；",
        "裂缝几何、分簇响应、均衡度和 Piggy-Bank 账本；",
        "智能体动作审计、回退原因、人工确认记录和实验配置。",
    ]:
        add_bullet(doc, text)
    add_heading(doc, "9.2 页面切换后无响应", 2)
    add_text(doc, "先检查是否重复触发了大文件读取或全流程计算，再查看日志中的数据读取和绘图异常。场景切换应使用统一缓存，避免在 GUI 线程重复执行耗时计算。")
    add_heading(doc, "9.3 无 DAS 没有簇级观测", 2)
    add_text(doc, "这是数据条件的正常结果。无 DAS 仍可进行压力换算和阶段级 PKN 演化；分簇观测、观测覆盖率等字段按未接入或不适用显示。")
    add_heading(doc, "9.4 EnKF 后验与观测存在差异", 2)
    add_text(doc, "后验由状态转移、观测误差、先验协方差和参数可观测性共同决定。应结合压力换算参数、观测质量和参数敏感性查看，而不能只根据单个时间点判断。")
    add_heading(doc, "9.5 文件归档", 2)
    add_text(doc, "导出或对外提供材料前，应检查文件中不包含未经授权的原始施工数据、原始 DAS 数据、真实井轨迹、个人信息和本机路径。")

    outputs = [COPYRIGHT_DIR / "02_用户手册.docx", COPYRIGHT_PACKAGE / "02_用户手册.docx"]
    return save_and_sync(doc, outputs)


def build_source_doc(source: Path, outputs: list[Path], segment: str) -> Path:
    source_doc = Document(str(source))
    paras = [p.text for p in source_doc.paragraphs]
    pages: list[list[str]] = []
    current: list[str] = []
    for text in paras:
        if text.startswith("源代码登记页"):
            if current:
                pages.append(current)
            current = []
        elif text.strip() or current is not None:
            current.append(text)
    if current:
        pages.append(current)
    pages = pages[:30]

    doc = Document()
    set_a4(doc.sections[0], left=2.54, right=2.54, top=1.9, bottom=1.9)
    set_core(doc, f"源代码提交件（{segment}30页）", "软件著作权源代码提交材料")
    # Match the reference source-code document: no header/footer, large white margins and dark code blocks.
    for page_index, lines in enumerate(pages, start=1):
        if page_index > 1:
            doc.add_page_break()
        table = doc.add_table(rows=1, cols=1)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = False
        cell = table.cell(0, 0)
        set_cell_shading(cell, CODE_BG)
        set_cell_margins(cell, top=125, start=150, bottom=125, end=150)
        table.columns[0].width = Cm(15.9)
        set_row_cant_split(table.rows[0])
        set_row_header(table.rows[0])
        for li, line in enumerate(lines):
            p = cell.paragraphs[0] if li == 0 else cell.add_paragraph()
            set_para(p, align=WD_ALIGN_PARAGRAPH.LEFT, line=1.0, first_indent=0, after=0)
            p.paragraph_format.keep_together = True
            r = p.add_run(line)
            # The reference uses syntax-colored code; a consistent readable light code face is safer
            # for arbitrary Python/JSON/UTF-8 snippets than a partial tokenizer.
            set_run(r, font="Consolas", size=8.5, color=CODE_TEXT)
    return save_and_sync(doc, outputs)


def normalize_existing_package_tables() -> None:
    """Add accessible header-row metadata to the already-formatted admin attachments."""
    existing = [
        PATENT_PACKAGE / "02_权利要求书初稿.docx",
        PATENT_PACKAGE / "03_说明书附图与绘图说明.docx",
        PATENT_PACKAGE / "06_申请边界与代理师审查清单.docx",
        PATENT_PACKAGE / "08_专利补充材料_阶段级总液量均衡调控.docx",
        COPYRIGHT_PACKAGE / "01_软件功能说明书.docx",
        COPYRIGHT_PACKAGE / "08_申报信息模板.docx",
        COPYRIGHT_PACKAGE / "09_登记审批表填报模板.docx",
        COPYRIGHT_PACKAGE / "10_软件著作权申请表填报模板.docx",
        COPYRIGHT_PACKAGE / "11_附件1_申报书填报模板.docx",
        COPYRIGHT_PACKAGE / "12_审批单填报模板.docx",
    ]
    for path in existing:
        if not path.exists():
            continue
        doc = Document(str(path))
        for table in doc.tables:
            if table.rows:
                set_row_header(table.rows[0])
        doc.save(path)


def main() -> None:
    # Keep the previously rebuilt main disclosure identical in both delivery locations.
    # Otherwise a recipient opening the comprehensive ``word_submission`` package could
    # still see the older five-page version while the top-level delivery folder has the
    # reference-aligned version.
    main_disclosure = PATENT_DIR / "01_专利技术交底书.docx"
    duplicate_disclosure = PATENT_PACKAGE / "01_专利技术交底书.docx"
    if main_disclosure.exists():
        duplicate_disclosure.parent.mkdir(parents=True, exist_ok=True)
        duplicate_disclosure.write_bytes(main_disclosure.read_bytes())
    build_patent04()
    build_patent07()
    build_manual()
    build_source_doc(
        COPYRIGHT_SOURCE_ARCHIVE / "前30页源代码.docx",
        [COPYRIGHT_DIR / "前30页源代码.docx", COPYRIGHT_PACKAGE / "前30页源代码.docx"],
        "前",
    )
    build_source_doc(
        COPYRIGHT_SOURCE_ARCHIVE / "中30页源代码.docx",
        [COPYRIGHT_DIR / "中30页源代码.docx", COPYRIGHT_PACKAGE / "中30页源代码.docx"],
        "中",
    )
    build_source_doc(
        COPYRIGHT_SOURCE_ARCHIVE / "后30页源代码.docx",
        [COPYRIGHT_DIR / "后30页源代码.docx", COPYRIGHT_PACKAGE / "后30页源代码.docx"],
        "后",
    )
    normalize_existing_package_tables()
    print("Rebuilt patent and software-copyright submission documents.")


if __name__ == "__main__":
    main()
