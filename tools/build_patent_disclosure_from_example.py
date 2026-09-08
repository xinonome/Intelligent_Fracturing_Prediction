from __future__ import annotations

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
OUT = ROOT / "专利与软著" / "deliverables" / "专利" / "01_专利技术交底书.docx"
FIG_DIR = ROOT / ".docx_qa_20260830" / "patent_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def font_prop() -> FontProperties:
    return FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc")


CN = r"C:\Windows\Fonts\msyh.ttc"


def pil_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = r"C:\Windows\Fonts\msyhbd.ttc" if bold else CN
    return ImageFont.truetype(path, size=size, index=0)


def centered_text(draw: ImageDraw.ImageDraw, xy, text: str, *, size=24, bold=False, fill="#18324D", anchor="mm"):
    draw.text(xy, text, font=pil_font(size, bold), fill=fill, anchor=anchor, align="center", spacing=4)


def rounded_box(draw: ImageDraw.ImageDraw, box, text: str, fill: str, *, size=22, outline="#4B6480"):
    draw.rounded_rectangle(box, radius=18, fill=fill, outline=outline, width=3)
    x0, y0, x1, y1 = box
    centered_text(draw, ((x0 + x1) / 2, (y0 + y1) / 2), text, size=size)


def arrow(draw: ImageDraw.ImageDraw, start, end, fill="#55738F", width=4):
    draw.line([start, end], fill=fill, width=width)
    x1, y1 = start
    x2, y2 = end
    if abs(x2 - x1) >= abs(y2 - y1):
        if x2 >= x1:
            pts = [(x2, y2), (x2 - 18, y2 - 10), (x2 - 18, y2 + 10)]
        else:
            pts = [(x2, y2), (x2 + 18, y2 - 10), (x2 + 18, y2 + 10)]
    else:
        if y2 >= y1:
            pts = [(x2, y2), (x2 - 10, y2 - 18), (x2 + 10, y2 - 18)]
        else:
            pts = [(x2, y2), (x2 - 10, y2 + 18), (x2 + 10, y2 + 18)]
    draw.polygon(pts, fill=fill)


def save_flow_figure(path: Path) -> None:
    image = Image.new("RGB", (2400, 1160), "white")
    draw = ImageDraw.Draw(image)
    colors = {"input": "#E8F0F8", "model": "#DCEBFA", "update": "#E5F4EE", "action": "#FFF0D8"}
    rounded_box(draw, (50, 440, 360, 700), "施工压力\n排量·砂比·液量\n井轨迹·阶段信息", colors["input"], size=25)
    arrow(draw, (360, 570), (470, 570))
    rounded_box(draw, (470, 440, 730, 700), "场景注册\n数据质量检查", colors["input"], size=25)
    arrow(draw, (730, 520), (860, 310))
    arrow(draw, (730, 620), (860, 850))
    rounded_box(draw, (860, 170, 1240, 450), "无 DAS\n压力换算与偏置校正", colors["input"], size=23)
    rounded_box(draw, (860, 760, 1240, 1040), "有 DAS\n压力 + 分簇响应观测", colors["input"], size=23)
    arrow(draw, (1240, 310), (1370, 570))
    arrow(draw, (1240, 900), (1370, 570))
    rounded_box(draw, (1370, 440, 1670, 700), "PKN 正演\n14维参数化状态", colors["model"], size=23)
    arrow(draw, (1670, 570), (1780, 570))
    rounded_box(draw, (1780, 410, 2070, 730), "KG-EnKF\n软先验·协方差\n观测置信度", colors["update"], size=21)
    arrow(draw, (2070, 570), (2180, 570))
    rounded_box(draw, (2180, 440, 2350, 700), "后验\n裂缝状态", colors["model"], size=21)
    rounded_box(draw, (1370, 840, 1690, 1060), "首次响应·效率\n均衡度·Gini·熵", colors["update"], size=21)
    arrow(draw, (1525, 700), (1525, 840))
    rounded_box(draw, (1810, 840, 2180, 1060), "Piggy-Bank\n阶段总液量建议", colors["action"], size=21)
    arrow(draw, (1690, 950), (1810, 950))
    image.save(path)


def save_cluster_figure(path: Path) -> None:
    image = Image.new("RGB", (2400, 920), "white")
    draw = ImageDraw.Draw(image)
    centered_text(draw, (540, 52), "分簇响应与目标份额", size=30, bold=True)
    centered_text(draw, (1800, 52), "阶段总液量反馈", size=30, bold=True)
    left = (70, 130, 1100, 770)
    draw.rectangle(left, outline="#B8C6D4", width=3)
    base_x, base_y, bar_w, gap, chart_h = 160, 680, 105, 40, 450
    share = [0.10, 0.14, 0.18, 0.24, 0.19, 0.15]
    bar_colors = ["#5B8FF9", "#61DDAA", "#65789B", "#F6BD16", "#7262FD", "#78D3F8"]
    target_y = base_y - chart_h * (1 / 6) / 0.32
    draw.line((120, target_y, 1040, target_y), fill="#D94841", width=4)
    for i, val in enumerate(share):
        x = base_x + i * (bar_w + gap)
        top = base_y - int(chart_h * val / 0.32)
        draw.rectangle((x, top, x + bar_w, base_y), fill=bar_colors[i], outline="white", width=2)
        centered_text(draw, (x + bar_w / 2, base_y + 38), f"簇{i + 1}", size=22)
        centered_text(draw, (x + bar_w / 2, top - 22), f"{val:.2f}", size=18)
    centered_text(draw, (580, 825), "柱高：观测或模型推导的分簇液量份额；虚线：均衡目标 1/6", size=20, fill="#506070")

    rounded_box(draw, (1250, 150, 1660, 340), "响应过快 / 偏高\n形成释放量", "#FDE4E2", size=24)
    rounded_box(draw, (1250, 560, 1660, 750), "响应过慢 / 可调\n形成支取需求", "#E5F4EE", size=24)
    rounded_box(draw, (1770, 340, 2150, 560), "阶段级储备账本\nB(s+1)=B(s)+释放−支取", "#FFF0D8", size=22)
    rounded_box(draw, (2210, 340, 2370, 560), "下一时间窗\n总液量建议", "#E8F0F8", size=20)
    arrow(draw, (1660, 245), (1770, 400))
    arrow(draw, (1660, 655), (1770, 500))
    arrow(draw, (2150, 450), (2210, 450))
    centered_text(draw, (1820, 115), "只能调节阶段总量，不直接指定单簇进液", size=23, bold=True)
    centered_text(draw, (1820, 840), "释放量先入账；支取量不超过可用储备；执行量确认后才提交账本", size=19, fill="#506070")
    image.save(path)


def save_ledger_figure(path: Path) -> None:
    image = Image.new("RGB", (2400, 820), "white")
    draw = ImageDraw.Draw(image)
    nodes = [
        (80, "阶段 s\n计划量 V_s"),
        (470, "响应评价\n快 / 慢 / 中性"),
        (860, "建议生成\n释放或支取"),
        (1250, "人工/策略\n确认"),
        (1640, "泵注执行\n实际量"),
        (2030, "账本提交\nB_(s+1)"),
    ]
    fills = ["#E8F0F8", "#E5F4EE", "#FFF0D8", "#F4ECFA", "#E8F0F8", "#FFF0D8"]
    for (x, text), fc in zip(nodes, fills):
        rounded_box(draw, (x, 250, x + 290, 470), text, fc, size=24)
    for (x, _), (nx, _) in zip(nodes[:-1], nodes[1:]):
        arrow(draw, (x + 290, 360), (nx, 360))
    centered_text(draw, (580, 570), "fast：V_s 减少，差额形成释放量", size=20, fill="#A94442")
    centered_text(draw, (1430, 570), "slow：仅从已有储备支取，不能凭空增加液量", size=20, fill="#2C7A5A")
    centered_text(draw, (1200, 730), "shadow/open-loop 只记录建议；field_observed/simulated 按实际执行量提交", size=20, fill="#506070")
    image.save(path)


def set_run_font(run, name="Times New Roman", size=12, bold=None, italic=None, color=None):
    run.font.name = name
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "宋体")
    run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if color is not None:
        run.font.color.rgb = RGBColor(*color)


def set_paragraph_format(paragraph, *, indent=True, first=True, align=WD_ALIGN_PARAGRAPH.JUSTIFY, line=1.5, before=0, after=0):
    pf = paragraph.paragraph_format
    pf.line_spacing = line
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.alignment = align
    if indent:
        pf.left_indent = Cm(0.53)
        pf.first_line_indent = Cm(0.80 if first else 0)
    else:
        pf.left_indent = Cm(0)
        pf.first_line_indent = Cm(0)


def add_para(doc, text="", *, role="body", bold=False, align=None, first=True, indent=True, size=12, before=0, after=0):
    p = doc.add_paragraph()
    if role == "heading":
        set_paragraph_format(p, indent=False, first=False, align=align or WD_ALIGN_PARAGRAPH.LEFT, line=1.5, before=0, after=0)
        r = p.add_run(text)
        set_run_font(r, size=12, bold=True)
    elif role == "cover_marker":
        set_paragraph_format(p, indent=False, first=False, align=WD_ALIGN_PARAGRAPH.LEFT, line=1.0)
        r = p.add_run(text)
        set_run_font(r, size=16, bold=True)
    elif role == "cover_title":
        set_paragraph_format(p, indent=False, first=False, align=WD_ALIGN_PARAGRAPH.CENTER, line=1.0, before=0, after=0)
        r = p.add_run(text)
        set_run_font(r, size=16)
    elif role == "metadata":
        set_paragraph_format(p, indent=False, first=False, align=WD_ALIGN_PARAGRAPH.LEFT, line=1.5, before=0, after=0)
        r = p.add_run(text)
        set_run_font(r, size=12, bold=bold)
    elif role == "subhead":
        set_paragraph_format(p, indent=True, first=True, align=WD_ALIGN_PARAGRAPH.LEFT, line=1.5)
        r = p.add_run(text)
        set_run_font(r, size=12, bold=True)
    elif role == "formula":
        set_paragraph_format(p, indent=False, first=False, align=WD_ALIGN_PARAGRAPH.CENTER, line=1.0, before=0, after=0)
        r = p.add_run(text)
        set_run_font(r, name="Cambria Math", size=12)
    elif role == "caption":
        set_paragraph_format(p, indent=False, first=False, align=WD_ALIGN_PARAGRAPH.CENTER, line=1.0, before=0, after=0)
        p.paragraph_format.keep_with_next = True
        r = p.add_run(text)
        set_run_font(r, name="仿宋", size=10.5, bold=True)
    else:
        set_paragraph_format(p, indent=indent, first=first, align=align or WD_ALIGN_PARAGRAPH.JUSTIFY, line=line if (line := 1.5) else 1.5, before=before, after=after)
        r = p.add_run(text)
        set_run_font(r, size=size, bold=bold)
    return p


def add_list_para(doc, text: str, *, bullet=False):
    style = "List Bullet" if bullet else "List Number"
    p = doc.add_paragraph(style=style)
    set_paragraph_format(p, indent=False, first=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY, line=1.5, before=0, after=0)
    p.paragraph_format.left_indent = Cm(0.95)
    p.paragraph_format.first_line_indent = Cm(-0.55)
    r = p.add_run(text)
    set_run_font(r, size=12)
    return p


def shade_cell(cell, fill: str):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_text(cell, text: str, *, header=False, size=10.5):
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER if header else WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    r = p.add_run(text)
    set_run_font(r, name="宋体", size=size, bold=header)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def set_table_widths(table, widths: Iterable[float]):
    widths = list(widths)
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl_pr = table._tbl.tblPr
    tbl_layout = tbl_pr.find(qn("w:tblLayout"))
    if tbl_layout is None:
        tbl_layout = OxmlElement("w:tblLayout")
        tbl_pr.append(tbl_layout)
    tbl_layout.set(qn("w:type"), "fixed")
    grid = table._tbl.tblGrid
    for child in list(grid):
        grid.remove(child)
    for width in widths:
        col = OxmlElement("w:gridCol")
        col.set(qn("w:w"), str(int(width * 1440)))
        grid.append(col)
    for row in table.rows:
        for idx, cell in enumerate(row.cells):
            width = widths[min(idx, len(widths) - 1)]
            cell.width = Inches(width)
            tc_pr = cell._tc.get_or_add_tcPr()
            tc_w = tc_pr.find(qn("w:tcW"))
            if tc_w is None:
                tc_w = OxmlElement("w:tcW")
                tc_pr.append(tc_w)
            tc_w.set(qn("w:w"), str(int(width * 1440)))
            tc_w.set(qn("w:type"), "dxa")
            tc_mar = tc_pr.find(qn("w:tcMar"))
            if tc_mar is None:
                tc_mar = OxmlElement("w:tcMar")
                tc_pr.append(tc_mar)
            for side in ["top", "left", "bottom", "right"]:
                el = tc_mar.find(qn(f"w:{side}"))
                if el is None:
                    el = OxmlElement(f"w:{side}")
                    tc_mar.append(el)
                el.set(qn("w:w"), "90")
                el.set(qn("w:type"), "dxa")


def repeat_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def prevent_row_split(row):
    """Keep a table row together when Word paginates the document."""
    tr_pr = row._tr.get_or_add_trPr()
    cant_split = tr_pr.find(qn("w:cantSplit"))
    if cant_split is None:
        cant_split = OxmlElement("w:cantSplit")
        tr_pr.append(cant_split)
    cant_split.set(qn("w:val"), "true")


def add_table(doc, headers, rows, widths):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    set_table_widths(table, widths)
    for i, h in enumerate(headers):
        set_cell_text(table.rows[0].cells[i], h, header=True)
        shade_cell(table.rows[0].cells[i], "D9E2F3")
    repeat_header(table.rows[0])
    prevent_row_split(table.rows[0])
    for row in rows:
        cells = table.add_row().cells
        for i, value in enumerate(row):
            set_cell_text(cells[i], str(value), size=10)
        prevent_row_split(table.rows[-1])
    set_table_widths(table, widths)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)
    return table


def add_picture(doc, path: Path, width=6.0, alt_text=""):
    p = doc.add_paragraph()
    set_paragraph_format(p, indent=False, first=False, align=WD_ALIGN_PARAGRAPH.CENTER, line=1.0, before=0, after=0)
    p.paragraph_format.keep_with_next = True
    run = p.add_run()
    shape = run.add_picture(str(path), width=Inches(width))
    if alt_text:
        shape._inline.docPr.set("title", alt_text)
        shape._inline.docPr.set("descr", alt_text)
    return p


def add_footer(doc):
    section = doc.sections[0]
    footer = section.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p.text = ""
    r = p.add_run("—")
    set_run_font(r, size=12)
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
    r._r.append(fld_begin)
    r._r.append(instr)
    r._r.append(fld_sep)
    r._r.append(text)
    r._r.append(fld_end)
    r2 = p.add_run("—")
    set_run_font(r2, size=12)


def build_document() -> None:
    flow = FIG_DIR / "figure1_flow.png"
    cluster = FIG_DIR / "figure2_cluster_feedback.png"
    ledger = FIG_DIR / "figure3_ledger.png"
    save_flow_figure(flow)
    save_cluster_figure(cluster)
    save_ledger_figure(ledger)

    doc = Document()
    section = doc.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(3.53)
    section.bottom_margin = Cm(3.34)
    section.left_margin = Cm(2.67)
    section.right_margin = Cm(2.48)
    section.header_distance = Cm(1.43)
    section.footer_distance = Cm(2.00)
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    normal.font.size = Pt(12)
    normal.paragraph_format.line_spacing = 1.5
    add_footer(doc)

    add_para(doc, "附件10.3", role="cover_marker")
    add_para(doc, "", role="cover_title")
    add_para(doc, "申请专利技术交底书", role="cover_title")
    add_para(doc, "", role="metadata")
    add_para(doc, "联系人信息", role="metadata", bold=True)
    add_para(doc, "姓名：________    电话：________    邮箱：____________________________", role="metadata")
    add_para(doc, "第一发明人：____________________________", role="metadata")
    add_para(doc, "第一发明人身份证号码：____________________", role="metadata")

    add_para(doc, "1．发明（或实用新型）的名称", role="heading")
    add_para(doc, "一种基于知识引导参数同化与分簇响应反馈的多簇压裂阶段液量均衡调控方法、系统、设备及存储介质。", first=True, indent=True)
    add_para(doc, "2．所属技术领域", role="heading")
    add_para(doc, "本发明属于油气井压裂施工数字化、数字孪生、数据同化和智能调控技术领域，具体涉及一种在有分簇观测或无分簇观测条件下，利用施工压力、井口—井底压力换算、井轨迹、分簇响应和专家知识，对压裂裂缝状态进行在线估计，并根据分簇响应差异和均衡度反馈生成阶段总液量调控建议的方法、系统、设备及存储介质。")

    add_para(doc, "3．现有技术及需要解决的问题", role="heading")
    add_para(doc, "当前压裂施工过程中，对砂堵、压力异常等关键事件的识别主要依赖人工观察施工曲线及现场经验判断。现场工程师通过实时监测施工压力、排量、砂比等参数的变化趋势，凭经验判断是否发生砂堵或压力异常，并在施工日志中手动记录事件起止时间。该方式存在以下核心局限：")
    for text in [
        "人工标注效率低：一个典型的压裂段通常包含数千至数万个秒点数据，人工逐段查看曲线、判断事件起止点并记录，平均每段耗时10～15分钟。对于一口水平井（通常20～40段），仅标注工作就需要一至两个工作日，且无法批量处理多口井。随着非常规油气藏规模化开发，压裂段数呈指数级增长，人工标注已无法满足规模化数据处理的需求。",
        "标注标准主观性强，结果一致性差：不同工程师对同一事件的判断标准存在显著差异。例如，对于“压力上升多少算异常”“排量下降多少算砂堵”等问题，缺乏统一的量化阈值。同一段数据由不同人员标注，事件起止点偏差可达数十秒甚至上百秒，这导致历史数据无法形成标准化的标签库，难以用于监督学习模型的训练。",
        "缺乏连续递减的量化检测能力，误判漏判率高：人工判断“排量是否在下降”“砂比是否在下降”时，难以准确识别连续多个点的下降趋势（尤其存在噪声或小幅波动时）。例如，排量可能在3～5个点内轻微波动但整体呈下降趋势，人眼容易忽略；反之，单点骤降可能被误判为下降趋势。现有技术中尚无针对压裂时序数据的连续递减检测算法，导致标注结果可靠性不足。",
        "缺乏直观的可视化标注输出：传统标注结果以文本表格或日志形式存储，工程师需对照原始施工曲线逐一核对，审核效率低。同时，文本记录无法直观展示事件在曲线上的位置、长度及类型，不利于施工复盘、汇报及归档。缺乏带色块标注的图表输出，使得标注结果与原始数据的关联性弱。",
        "无法批量处理多段、多井数据：现有方法依赖人工逐段操作，无法一次性处理一个区块内数十口井、数百个压裂段的数据。对于需要构建大规模标注数据集以训练智能决策模型的应用场景，人工方式几乎不可行。",
    ]:
        add_para(doc, text)
    add_para(doc, "鉴于此，现有技术普遍存在“依赖人工、标准不一、缺乏可视化输出”的显著缺点，难以满足压裂施工数字化转型对高质量、大规模、标准化标注数据的需求。因此，亟需一种能够自动、客观、精细地标注压裂施工数据中的砂堵与压力异常事件的方法，以取代现有的人工标注模式，为压裂施工智能决策提供可靠的数据基础。")

    add_para(doc, "4．发明的目的", role="heading")
    add_para(doc, "本发明的目的在于解决现有多簇压裂数字孪生与调控方法中观测条件不一致、物理状态估计偏差、规则难以进入参数更新以及只能调总量而难以进行跨阶段液量再分配的问题，具体包括：")
    for text in [
        "在无 DAS 和有 DAS 条件下使用同一数字孪生框架，但根据数据可用性注册不同观测向量和输出语义。",
        "将井口压力、液柱、管柱摩阻、射孔摩阻和偏置纳入井口—井底压力换算，支持压力偏置在线校正。",
        "将专家规则转化为 EnKF 可审计的先验均值、先验协方差和观测置信度调节，而不是直接覆盖观测或裂缝状态。",
        "通过分簇首次响应、三维距离和首次响应时累计注入液量计算响应效率，并结合份额、Gini 系数和归一化熵评价分簇均衡度。",
        "在总液量基本守恒的前提下，通过阶段级 Piggy-Bank 账本生成释放、保持和支取建议；明确建议量、批准量和实际执行量之间的区别。",
        "使智能体建议同时受到压力安全、异常风险、均衡度、裂缝几何效果、成本、数据质量和模型置信度约束，并为开环记录、人工确认和后续闭环执行提供统一接口。",
    ]:
        add_list_para(doc, text, bullet=True)

    add_para(doc, "5．发明的内容", role="heading")
    add_para(doc, "本发明构建一种面向多簇压裂施工的双场景数字孪生与阶段总液量调控方法。其核心不是改变现场设备能够控制的对象，而是先对可观测数据和模型状态进行统一解释，再把分簇响应差异转换为下一时间窗或下一阶段的总液量建议。方法包括数据场景注册、井口—井底压力换算、PKN 参数化正演、知识引导 EnKF、分簇响应评价、Piggy-Bank 账本和安全门禁等步骤。整体流程如图1所示。")
    add_para(doc, "（1）输入数据和场景注册", role="subhead")
    add_para(doc, "系统接收施工压力、累计液量、排量、砂比、井轨迹、阶段信息以及可选的 DAS/FracMonitor 解释后的分簇液量、砂量、响应时间和几何位置。根据数据可用性将数据注册为两个场景：")
    add_para(doc, "no_das_pressure_only\n    观测向量：井底压力或压力校正结果\n    主要输出：阶段级裂缝状态、压力校正结果和模型推导的分簇响应", role="formula")
    add_para(doc, "das_cluster_observation\n    观测向量：井底压力 + 分簇液量/砂量份额或响应事件\n    主要输出：阶段级裂缝状态、分簇响应和均衡度", role="formula")
    add_para(doc, "对于没有分簇观测的井段，系统不将缺失值填充为0，也不把模型推导的簇级量伪装成现场观测；模型推导量在数据结构中记录来源，只有真实接入的分簇观测才进入观测置信度计算。场景、来源、井号、段号和时间覆盖范围写入场景注册表。")

    add_para(doc, "（2）井口—井底压力换算与偏置校正", role="subhead")
    add_para(doc, "对施工压力进行井口—井底换算，得到用于数字孪生状态更新的压力观测。设井口压力为 P_wellhead(t)，静液柱压力为 P_hydrostatic(t)，管柱摩阻和射孔摩阻分别为 P_pipe_friction(t)、P_perforation_friction(t)，压力偏置为 P_bias(t)，则：")
    add_para(doc, "P_bhp(t) = P_wellhead(t) + P_hydrostatic(t) - P_pipe_friction(t) - P_perforation_friction(t) + P_bias(t)", role="formula")
    add_para(doc, "净压力同时保留原始值和用于显示或约束的非负值：")
    add_para(doc, "P_net,raw(t) = P_bhp(t) - σ_min(t)", role="formula")
    add_para(doc, "P_net,display(t) = max(P_net,raw(t), 0)", role="formula")
    add_para(doc, "垂深、液体或砂浆密度、摩阻参数、射孔参数、最小水平主应力、压力偏置及偏置边界均作为配置输入。压力偏置可通过连续压力观测与模型预测残差进行在线更新；当现场换算参数缺失时，可以运行工程估算，但保留参数来源和校准状态。")

    add_para(doc, "（3）PKN 状态和六簇参数化正演", role="subhead")
    add_para(doc, "系统以 PKN 作为在线快速正演模型，将全局压力/裂缝参数与分簇分配参数组合成参数向量：")
    add_para(doc, "θ_t = [log E′, log C_L, log μ, σ_min, log K_IC, κ_1, κ_2, κ_3, κ_4, κ_5, κ_6, γ_sh, γ_e, α_q]^T", role="formula")
    add_para(doc, "其中，E′为平面应变弹性模量，C_L为滤失系数，μ为流体黏度，σ_min为最小水平主应力，K_IC为断裂韧度；κ_i为第i簇进液能力系数，γ_sh为簇间应力干扰系数，γ_e为边界效应系数，α_q为分配指数。")
    add_para(doc, "给定阶段总排量 Q(t) 后，模型先根据进液能力、应力阴影、边界效应、井筒/射孔摩阻和分配指数计算六簇相对权重，再归一化得到簇级液量份额。每簇有效进液量、压力响应、半缝长、最大缝宽和改造体积由相应簇的有效进液量和全局物性参数计算。无 DAS 场景使用参数化分配模型得到模型推导的分簇响应；有 DAS 场景允许观测份额作为同化观测，但不将观测份额同时当作自由参数。")
    add_para(doc, "表 1  参数化状态与计算用途", role="caption")
    add_table(doc,
              ["参数类别", "参数或变量", "计算用途", "数据来源/约束"],
              [
                  ("压力/裂缝参数", "E′、C_L、μ、σ_min、K_IC", "压力、净压力、缝长、缝宽与滤失响应", "先验配置 + EnKF 更新；保留物理域检查"),
                  ("簇进液能力", "κ_1—κ_6", "计算六簇相对进液能力和液量份额", "先验或分簇观测同化"),
                  ("簇间影响", "γ_sh", "表示应力阴影对相邻簇进液竞争的影响", "规则/观测协方差关联"),
                  ("边界效应", "γ_e", "表示内外侧簇的释放或受抑制趋势", "几何配置 + EnKF 更新"),
                  ("分配指数", "α_q", "控制总排量变化下的份额响应强度", "参数化分配模型"),
              ],
              [1.25, 1.70, 2.35, 1.15])

    add_para(doc, "（4）知识引导 EnKF 参数更新", role="subhead")
    add_para(doc, "知识图谱或规则模块从压力上升、压力斜率、排量变化、砂比变化、阶段工况和风险标签中生成可审计信号。规则不直接指定裂缝长度、簇级现场液量或最终控制动作，而是通过软先验影响参数集合。设第m个集合成员的先验参数为 θ_t,prior^(m)，则：")
    add_para(doc, "θ_0^(m) ~ N(μ_KG, P_KG)", role="formula")
    add_para(doc, "P_KG = diag(d) · C_KG · diag(d)", role="formula")
    add_para(doc, "K_t = P_θy · (P_yy + R_t)^−1", role="formula")
    add_para(doc, "θ_t,post^(m) = θ_t,prior^(m) + K_t · [y_t + ε_t^(m) − ŷ_t^(m)]", role="formula")
    add_para(doc, "其中 μ_KG 为知识规则产生的先验均值；d 为根据不确定性规则调整的尺度向量；C_KG 为参数相关性矩阵；R_t 为根据观测质量和压力置信度调节的观测误差；y_t 为当前观测，ŷ_t^(m) 为第m个成员的 PKN 预测观测，ε_t^(m) 为观测扰动。更新完成后重新运行 PKN，得到后验压力、缝长、缝宽、分簇份额和均衡度。")
    add_para(doc, "系统支持以下四种知识引导模式：")
    add_para(doc, "表 2  知识引导 EnKF 模式", role="caption")
    add_table(doc,
              ["模式", "技术作用", "是否改变先验均值", "是否使用参数相关性/观测置信度"],
              [
                  ("off", "不使用知识规则，作为普通 PKN+EnKF 对照", "否", "否"),
                  ("uncertainty_only", "只扩大参数先验和过程噪声的不确定性", "否", "不改变参数相关性"),
                  ("soft_prior", "在不确定性调整基础上施加小幅先验均值偏移", "是", "部分使用"),
                  ("soft_correlated", "同时使用先验均值、参数协方差、观测置信度调节和参数域检查", "是", "是"),
              ],
              [1.30, 2.65, 1.05, 1.15])
    add_para(doc, "四种模式用于对比规则信息进入同化链路的程度。无论采用哪种模式，最终后验仍由 PKN 预测、观测误差协方差和 EnKF 更新共同决定；知识规则不能绕过观测直接写入裂缝状态。")

    add_para(doc, "（5）分簇响应、响应效率和均衡度计算", role="subhead")
    add_para(doc, "对于能够识别分簇响应的场景，系统记录第i簇首次响应事件时间 t_i^first、首次响应时累计注入液量 V_i^first，以及井段位置到响应事件位置的三维距离 d_i。关键响应效率定义为：")
    add_para(doc, "η_i = d_i / V_i^first", role="formula")
    add_para(doc, "η_i 表示单位累计注入液量对应的三维响应距离，可用于识别相对扩展过快或过慢的簇。对每个评价窗口，计算液量份额 q_i = V_i / Σ_j V_j，并以 N 个簇的均衡目标 q_i* = 1/N 为基准。归一化熵均衡度为：")
    add_para(doc, "B_H = −Σ_i q_i ln(q_i) / ln(N)", role="formula")
    add_para(doc, "同时计算 Gini 系数：")
    add_para(doc, "G = Σ_i Σ_j |q_i − q_j| / (2NΣ_i q_i)", role="formula")
    add_para(doc, "B_H 越接近1表示份额越均衡，G 越接近0表示份额越均衡。若输入直接提供 FracMonitor 的 balance_degree，系统优先使用并记录来源；若只有分簇份额，则按上述公式推导；无 DAS 时也可以用 PKN 参数化几何和分配结果推导，但该值标记为模型推导，不等同于现场分簇观测。")

    add_para(doc, "（6）Piggy-Bank 阶段级总液量调控", role="subhead")
    add_para(doc, "本发明不假设能够直接向第i簇注入指定液量。Piggy-Bank 的“储备”来自已经计划但尚未泵入的阶段液量：当某一阶段响应过快或分配偏高时，减少该阶段下一时间窗或阶段计划总液量，减少的部分作为可释放量进入账本；当后续阶段响应较慢且满足安全、数据质量和储备条件时，从账本支取部分液量，增加后续阶段总液量建议。液量经过总排量进入井筒后，六簇实际份额仍由井筒、射孔和储层竞争决定。")
    add_para(doc, "设第s阶段计划总液量为 V_s^plan，阶段建议相对于计划量的调整量为 δ_s，阶段级储备余额为 B_s。若阶段被判定为响应过快，则：")
    add_para(doc, "ΔV_s^release = min(max(−δ_s, 0), V_s^plan)", role="formula")
    add_para(doc, "若阶段被判定为响应过慢且具备可用储备，则：")
    add_para(doc, "ΔV_s^draw = min(max(δ_s, 0), B_s, α_max V_s^plan)", role="formula")
    add_para(doc, "账本和阶段实际建议量分别为：")
    add_para(doc, "B_(s+1) = B_s + ΔV_s^release − ΔV_s^draw", role="formula")
    add_para(doc, "V_s^recommend = V_s^plan − ΔV_s^release + ΔV_s^draw", role="formula")
    add_para(doc, "其中 α_max 为单阶段调整上限。系统将 propose、approve、commit 三个事件分开记录：propose 只生成建议；approve 记录人工或策略批准；commit 只有在取得现场实际泵注量，或在明确标记为 simulated 的实验模式下，才更新储备余额。shadow/open-loop 模式只记录建议，不把虚拟储备称为已发生的现场液量。具体账本关系如图3所示。")
    add_para(doc, "表 3  Piggy-Bank 阶段液量决策规则", role="caption")
    add_table(doc,
              ["状态", "触发条件", "动作", "账本处理"],
              [
                  ("fast", "响应效率偏高或份额偏高，且安全检查通过", "减少下一时间窗/阶段总液量", "减少量作为 release 入账"),
                  ("slow", "响应效率偏低，且有可用储备、数据有效、风险可接受", "增加下一时间窗/阶段总液量", "增加量作为 draw 出账"),
                  ("neutral", "无明确方向或信号冲突", "保持原计划", "不改变余额"),
                  ("hold", "高风险、高不确定性、数据无效或人工冻结", "保持或回退安全动作", "不提交建议账本"),
                  ("rollback", "发现执行记录或输入错误，且为最近一次提交", "撤销最近一次提交", "恢复提交前余额并写入审计记录"),
              ],
              [1.0, 2.25, 1.85, 1.05])

    add_para(doc, "（7）智能体建议与安全门禁", role="subhead")
    add_para(doc, "智能体以 PKN+EnKF 输出、工况标签、压力风险、分簇均衡度、裂缝长度、缝宽和施工成本为状态，输出阶段总排量和砂比的建议。改造效果项优先使用均衡指数、裂缝长度和裂缝宽度等直观指标，不将不同算法计算差异较大的产量指标作为当前主要展示指标。奖励或评价函数可写为：")
    add_para(doc, "R = w_p R_pressure + w_b R_balance + w_l R_length + w_w R_width − w_r C_risk − w_c C_cost − w_a C_action", role="formula")
    add_para(doc, "其中压力安全和异常风险为高优先级约束，均衡度改善量定义为当前评价点均衡度减去上一决策点均衡度。智能体输出经过安全门禁、数据有效性检查、模型置信度检查和人工确认流程；发生高风险、压力接近上限、观测无效、代理模型超出训练分布、输出为 NaN/Inf 或关键状态缺失时，默认保持或回退到安全动作。安全规则优先于任何模型建议。")

    add_para(doc, "6．发明的效果", role="heading")
    add_para(doc, "本发明通过双场景观测注册、压力换算、知识引导参数同化和阶段级总液量反馈，将分簇响应信息转化为可解释、可审计的施工建议，具体效果如下：")
    for text in [
        "能够适应有 DAS 和无 DAS 两种现场数据条件，明确区分真实分簇观测、模型推导量和缺失数据，减少观测语义混用。",
        "将井口压力换算为井底压力并保留偏置校正接口，降低直接使用井口压力造成的数字孪生状态偏差。",
        "将知识规则以先验均值、协方差和观测置信度的形式进入 EnKF，在利用专家信息的同时保留观测对后验参数的决定作用。",
        "通过响应效率、液量份额、均衡指数、Gini 系数和归一化熵，对六簇响应差异进行统一量化，为后续液量调控和智能体奖励提供状态量。",
        "在只能调节阶段总液量的现实条件下，将响应过快阶段的未泵入液量形成释放，将可用储备支取给响应较慢阶段，避免宣称能够直接控制单簇进液。",
        "通过建议、批准、实际执行、守恒校验和回滚审计分离，支持开环验证、人工确认和未来闭环执行的逐步演进。",
        "将压力安全、异常风险、改造效果、均衡度、施工成本和数据质量纳入统一的评价与门禁体系，提高调控结果的可解释性和工程可追溯性。",
    ]:
        add_list_para(doc, text)

    add_para(doc, "附图及附图的简要说明", role="heading")
    add_para(doc, "图1为本发明双场景观测、知识引导参数同化和阶段总液量反馈的整体流程图。施工压力、排量、砂比、液量、井轨迹和阶段信息首先进入场景注册；无 DAS 场景经井口—井底压力换算后形成压力观测，有 DAS 场景在压力观测基础上增加分簇响应观测；两种场景均由 PKN 正演和 KG-EnKF 参数更新得到裂缝状态，随后计算分簇响应效率和均衡度，并由 Piggy-Bank 生成阶段总液量建议。")
    add_picture(doc, flow, width=5.0, alt_text="双场景观测、知识引导同化与阶段液量反馈流程图")
    add_para(doc, "图 1  双场景观测、知识引导同化与阶段液量反馈流程", role="caption")
    add_para(doc, "图2为六簇响应评价与阶段总液量反馈示意图。左侧以六簇液量份额与均衡目标对比表示分簇响应差异，右侧表示响应过快阶段形成释放量、响应过慢阶段在可用储备内形成支取需求，最终只改变下一时间窗或下一阶段的总液量，不对已完成阶段或某一簇的实际进液量进行事后改写。")
    add_picture(doc, cluster, width=6.0, alt_text="六簇响应份额、均衡目标与阶段总液量反馈示意图")
    add_para(doc, "图 2  六簇响应评价与阶段总液量反馈示意", role="caption")

    add_para(doc, "实施例", role="heading")
    add_para(doc, "本实施例以多簇压裂施工的连续时间序列为对象，验证本发明在有 DAS 和无 DAS 条件下的计算流程。实施例不要求有 DAS 和无 DAS 使用同一文件，而是根据场景注册结果选择观测向量。对于无 DAS 井段，输入施工压力、排量、砂比、累计液量和井轨迹，通过压力换算与偏置校正得到井底压力，再由六簇参数化 PKN 计算模型推导各簇液量份额、裂缝半长、缝宽和均衡度；对于有 DAS 井段，在相同压力链路上进一步读入分簇液量份额或响应事件，用于校正六簇进液能力、应力干扰、边界效应和分配指数。")
    add_para(doc, "在每一个更新时刻，系统先保留当前先验参数和状态快照，再以当前观测构造集合成员的 PKN 预测。KG-EnKF 根据当前模式调整先验均值、参数协方差和观测噪声，完成参数更新后重新正演，并将后验压力与分簇响应写入时间序列。若更新过程出现输入缺失、输出非有限、集合成员失效或守恒检查失败，则恢复到最近一次有效快照，采用上一有效状态或安全规则建议，并在审计记录中写入回滚原因。")
    add_para(doc, "在阶段液量建议环节，系统首先计算各簇响应效率和均衡度变化。若某阶段响应过快且安全检查通过，则对下一时间窗的阶段总液量给出减少建议，减少量进入 Piggy-Bank 的待确认释放记录；若后续阶段响应较慢且储备余额、数据质量和风险状态允许，则给出增加阶段总液量的支取建议。只有在人工或策略确认后，且取得实际泵注量时，才把释放或支取写入已提交账本。因而，该实施例能够在不假设单簇直接控制的前提下，实现跨阶段总液量的可审计再分配。")
    add_picture(doc, ledger, width=6.0, alt_text="多阶段 Piggy-Bank 建议、确认、执行与账本提交关系图")
    add_para(doc, "图 3  多阶段 Piggy-Bank 建议、确认、执行与账本提交关系", role="caption")
    add_para(doc, "本实施例中，均衡度可以直接读取分簇监测结果，也可以由六簇液量份额通过归一化熵或 Gini 系数推导。若数据中没有真实分簇观测，则输出应标记为模型推导结果。对外展示重点为压力对比、参数后验、分簇均衡度、裂缝长度和缝宽等指标；任何现场闭环执行均需在开环记录和独立验证可靠后再接入。")

    add_para(doc, "本发明的摘要", role="heading")
    add_para(doc, "本发明公开了一种基于知识引导参数同化与分簇响应反馈的多簇压裂阶段液量均衡调控方法、系统、设备及存储介质，属于油气井压裂施工数字化、数字孪生、数据同化和智能调控技术领域。该方法根据施工压力、累计液量、排量、砂比、井轨迹、阶段信息以及可选的 DAS/FracMonitor 分簇观测注册无 DAS 压力场景或有 DAS 分簇观测场景；通过井口—井底压力换算和压力偏置校正获得压力观测；利用 PKN 对弹性、滤失、黏度、应力、断裂韧度、六簇进液能力、簇间应力干扰、边界效应和分配指数进行参数化正演；将知识规则转化为 EnKF 的先验均值、协方差和观测置信度调节，获得后验压力、裂缝长度、缝宽、分簇份额和均衡度；根据分簇首次响应、三维距离与首次响应时累计注入液量计算响应效率，并结合均衡度、Gini 系数和归一化熵评价分簇差异；在只能调节阶段总液量的条件下，通过 Piggy-Bank 账本将响应过快阶段尚未泵入的液量形成释放记录，并在安全、数据质量和储备条件允许时向后续响应较慢阶段提出支取建议。该方法不假设直接控制单簇进液，通过建议、批准、执行、守恒和回滚记录实现开环可审计调控，并为后续人工确认和闭环执行保留接口。")
    add_para(doc, "", role="metadata")
    doc.core_properties.title = "一种基于知识引导参数同化与分簇响应反馈的多簇压裂阶段液量均衡调控方法"
    doc.core_properties.subject = "专利技术交底书"
    doc.core_properties.author = ""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    build_document()
