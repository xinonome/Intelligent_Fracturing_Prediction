"""Build the three acceptance documents from Markdown sources.

This is intentionally a small deterministic Markdown subset renderer for the
project's long-form Chinese documents.  It keeps headings, real Word lists,
fixed-width tables, code blocks, headers and footers stable across runs.
"""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "Reports"
OUT = REPORTS / "验收材料_正式文档"

BLUE = RGBColor(46, 116, 181)
DARK_BLUE = RGBColor(31, 77, 120)
INK = RGBColor(32, 42, 56)
MUTED = RGBColor(98, 108, 120)
TABLE_HEADER = "E8EEF5"
TABLE_BORDER = "B8C4D0"
CODE_FILL = "F4F6F9"


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=80, start=120, bottom=80, end=120) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for key, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{key}"))
        if node is None:
            node = OxmlElement(f"w:{key}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_table_geometry(table, widths_dxa: list[int], indent_dxa=120) -> None:
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    tbl = table._tbl
    tbl_pr = tbl.tblPr
    tbl_w = tbl_pr.find(qn("w:tblW"))
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), str(sum(widths_dxa)))
    tbl_w.set(qn("w:type"), "dxa")
    tbl_ind = tbl_pr.find(qn("w:tblInd"))
    if tbl_ind is None:
        tbl_ind = OxmlElement("w:tblInd")
        tbl_pr.append(tbl_ind)
    tbl_ind.set(qn("w:w"), str(indent_dxa))
    tbl_ind.set(qn("w:type"), "dxa")
    grid = tbl.tblGrid
    for child in list(grid):
        grid.remove(child)
    for width in widths_dxa:
        col = OxmlElement("w:gridCol")
        col.set(qn("w:w"), str(width))
        grid.append(col)
    for row in table.rows:
        for index, cell in enumerate(row.cells):
            width = widths_dxa[min(index, len(widths_dxa) - 1)]
            tc_pr = cell._tc.get_or_add_tcPr()
            tc_w = tc_pr.find(qn("w:tcW"))
            if tc_w is None:
                tc_w = OxmlElement("w:tcW")
                tc_pr.append(tc_w)
            tc_w.set(qn("w:w"), str(width))
            tc_w.set(qn("w:type"), "dxa")
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_margins(cell)


def set_row_cant_split(row, header: bool = False) -> None:
    """Keep a table row together and optionally repeat it as a header."""

    tr_pr = row._tr.get_or_add_trPr()
    if tr_pr.find(qn("w:cantSplit")) is None:
        tr_pr.append(OxmlElement("w:cantSplit"))
    if header and tr_pr.find(qn("w:tblHeader")) is None:
        tr_pr.append(OxmlElement("w:tblHeader"))


def new_numbering_instance(doc: Document, style_name: str) -> int:
    """Create a fresh numbering instance so each Markdown list restarts."""

    numbering = doc.part.numbering_part.element
    # The default template does not always attach numPr to the list styles.
    # Use the standard abstract definitions shipped with python-docx instead.
    abstract_id = "4" if style_name == "List Bullet" else "7"
    existing = [int(node.get(qn("w:numId"))) for node in numbering.findall(qn("w:num"))]
    new_id = max(existing or [0]) + 1
    num = OxmlElement("w:num")
    num.set(qn("w:numId"), str(new_id))
    abstract = OxmlElement("w:abstractNumId")
    abstract.set(qn("w:val"), str(abstract_id))
    num.append(abstract)
    numbering.append(num)
    return new_id


def apply_numbering(paragraph, num_id: int) -> None:
    p_pr = paragraph._p.get_or_add_pPr()
    num_pr = p_pr.find(qn("w:numPr"))
    if num_pr is None:
        num_pr = OxmlElement("w:numPr")
        p_pr.append(num_pr)
    ilvl = num_pr.find(qn("w:ilvl"))
    if ilvl is None:
        ilvl = OxmlElement("w:ilvl")
        num_pr.append(ilvl)
    ilvl.set(qn("w:val"), "0")
    num = num_pr.find(qn("w:numId"))
    if num is None:
        num = OxmlElement("w:numId")
        num_pr.append(num)
    num.set(qn("w:val"), str(num_id))


def set_run_font(run, name="Calibri", size=11, color=INK, bold=None, italic=None) -> None:
    run.font.name = name
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    run.font.size = Pt(size)
    run.font.color.rgb = color
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic


def set_style_font(style, size, color=INK, bold=False, line_spacing=1.1, after=6, before=0) -> None:
    style.font.name = "Calibri"
    style._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), "Calibri")
    style._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), "Calibri")
    style._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    style.font.size = Pt(size)
    style.font.color.rgb = color
    style.font.bold = bold
    style.paragraph_format.space_before = Pt(before)
    style.paragraph_format.space_after = Pt(after)
    style.paragraph_format.line_spacing = line_spacing


def add_page_field(paragraph) -> None:
    run = paragraph.add_run()
    fld_char1 = OxmlElement("w:fldChar")
    fld_char1.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    fld_char2 = OxmlElement("w:fldChar")
    fld_char2.set(qn("w:fldCharType"), "end")
    run._r.append(fld_char1)
    run._r.append(instr)
    run._r.append(fld_char2)
    set_run_font(run, size=9, color=MUTED)


def configure_doc(doc: Document, manual: bool) -> None:
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(0.85)
    section.bottom_margin = Inches(0.75)
    section.left_margin = Inches(0.85)
    section.right_margin = Inches(0.85)
    section.header_distance = Inches(0.35)
    section.footer_distance = Inches(0.35)

    styles = doc.styles
    set_style_font(styles["Normal"], 10.5, line_spacing=1.1 if not manual else 1.2, after=5)
    set_style_font(styles["Heading 1"], 16, BLUE, True, 1.05, 8, 16)
    set_style_font(styles["Heading 2"], 13, BLUE, True, 1.05, 6, 12)
    set_style_font(styles["Heading 3"], 11.5, DARK_BLUE, True, 1.05, 4, 8)
    if "Heading 4" in styles:
        heading4 = styles["Heading 4"]
    else:
        heading4 = styles.add_style("Heading 4", WD_STYLE_TYPE.PARAGRAPH)
    set_style_font(heading4, 10.8, DARK_BLUE, True, 1.05, 3, 6)
    for style_name in ("List Bullet", "List Number"):
        style = styles[style_name]
        set_style_font(style, 10.5, line_spacing=1.1 if not manual else 1.2, after=4)
        style.paragraph_format.left_indent = Inches(0.38)
        style.paragraph_format.first_line_indent = Inches(-0.19)

    header = section.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.LEFT
    hr = header.add_run("智能压裂预测平台  |  验收技术材料")
    set_run_font(hr, size=8.5, color=MUTED, bold=True)
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    fr = footer.add_run("Intelligent Fracturing Prediction  ·  ")
    set_run_font(fr, size=8.5, color=MUTED)
    add_page_field(footer)


def add_title_block(doc: Document, title: str, manual: bool) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(24 if not manual else 80)
    p.paragraph_format.space_after = Pt(8)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER if manual else WD_ALIGN_PARAGRAPH.LEFT
    run = p.add_run(title)
    set_run_font(run, size=25 if manual else 23, color=INK, bold=True)
    sub = doc.add_paragraph()
    sub.alignment = p.alignment
    sub.paragraph_format.space_after = Pt(18)
    r = sub.add_run("Intelligent Fracturing Prediction  ·  版本 V1.0  ·  2026-08-26")
    set_run_font(r, size=10.5, color=MUTED)
    if manual:
        note = doc.add_paragraph()
        note.alignment = WD_ALIGN_PARAGRAPH.CENTER
        note.paragraph_format.space_after = Pt(54)
        n = note.add_run("面向验收展示、算法复核和授权数据本地运行的操作指南")
        set_run_font(n, size=12, color=DARK_BLUE, italic=True)


def inline_runs(paragraph, text: str, size=10.5) -> None:
    # Keep emphasis readable without trying to implement a full Markdown AST.
    parts = re.split(r"(\*\*.*?\*\*|`.*?`)", text)
    for part in parts:
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            run = paragraph.add_run(part[2:-2])
            set_run_font(run, size=size, color=INK, bold=True)
        elif part.startswith("`") and part.endswith("`"):
            run = paragraph.add_run(part[1:-1])
            set_run_font(run, name="Consolas", size=size - 0.5, color=DARK_BLUE)
        else:
            run = paragraph.add_run(part)
            set_run_font(run, size=size, color=INK)


def parse_table(lines: list[str]) -> tuple[list[str], list[list[str]]]:
    rows = []
    for line in lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and not all(re.fullmatch(r":?-+:?", cell) for cell in cells):
            rows.append(cells)
    if not rows:
        return [], []
    header = rows[0]
    body = rows[1:]
    return header, body


def add_table(doc: Document, header: list[str], body: list[list[str]]) -> None:
    ncols = max(len(header), max((len(row) for row in body), default=0))
    if ncols == 0:
        return
    table = doc.add_table(rows=1, cols=ncols)
    widths = [9360 // ncols] * ncols
    widths[-1] += 9360 - sum(widths)
    set_table_geometry(table, widths)
    for index in range(ncols):
        cell = table.rows[0].cells[index]
        cell.text = ""
        p = cell.paragraphs[0]
        inline_runs(p, header[index] if index < len(header) else "", size=9.5)
        for run in p.runs:
            run.bold = True
        set_cell_shading(cell, TABLE_HEADER)
    set_row_cant_split(table.rows[0], header=True)
    for row_data in body:
        row = table.add_row()
        for index in range(ncols):
            cell = row.cells[index]
            cell.text = ""
            p = cell.paragraphs[0]
            inline_runs(p, row_data[index] if index < len(row_data) else "", size=9.2)
        set_row_cant_split(row)
    set_table_geometry(table, widths)
    for row in table.rows:
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.space_after = Pt(2)
                paragraph.paragraph_format.line_spacing = 1.05
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def render_markdown(doc: Document, markdown: str, manual: bool) -> None:
    lines = markdown.splitlines()
    in_code = False
    code_lines: list[str] = []
    paragraph_lines: list[str] = []
    table_lines: list[str] = []
    active_list_kind: str | None = None
    active_list_num_id: int | None = None

    def flush_paragraph() -> None:
        nonlocal paragraph_lines, active_list_kind, active_list_num_id
        if paragraph_lines:
            p = doc.add_paragraph()
            inline_runs(p, " ".join(line.strip() for line in paragraph_lines))
            paragraph_lines = []
            active_list_kind = None
            active_list_num_id = None

    def flush_table() -> None:
        nonlocal table_lines
        if table_lines:
            header, body = parse_table(table_lines)
            add_table(doc, header, body)
            table_lines = []

    def flush_code() -> None:
        nonlocal code_lines
        if code_lines:
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.15)
            p.paragraph_format.right_indent = Inches(0.15)
            p.paragraph_format.space_before = Pt(3)
            p.paragraph_format.space_after = Pt(6)
            p.paragraph_format.line_spacing = 1.0
            p_pr = p._p.get_or_add_pPr()
            shd = OxmlElement("w:shd")
            shd.set(qn("w:fill"), CODE_FILL)
            p_pr.append(shd)
            run = p.add_run("\n".join(code_lines))
            set_run_font(run, name="Consolas", size=8.8, color=DARK_BLUE)
            code_lines = []

    for raw in lines:
        line = raw.rstrip()
        if line.startswith("```"):
            flush_paragraph()
            flush_table()
            if in_code:
                flush_code()
                in_code = False
            else:
                in_code = True
            continue
        if in_code:
            code_lines.append(line)
            continue
        if not line.strip():
            flush_paragraph()
            flush_table()
            continue
        if line.startswith("|") and line.endswith("|"):
            flush_paragraph()
            table_lines.append(line)
            continue
        if table_lines:
            flush_table()
        heading = re.match(r"^(#{1,4})\s+(.*)$", line)
        if heading:
            flush_paragraph()
            active_list_kind = None
            active_list_num_id = None
            level = len(heading.group(1))
            p = doc.add_paragraph(style=f"Heading {level}")
            inline_runs(p, heading.group(2), size={1: 15, 2: 12.5, 3: 11.5, 4: 10.8}[level])
            for run in p.runs:
                run.bold = True
            continue
        bullet = re.match(r"^\s*[-*]\s+(.*)$", line)
        if bullet:
            flush_paragraph()
            p = doc.add_paragraph(style="List Bullet")
            if active_list_kind != "bullet" or active_list_num_id is None:
                active_list_num_id = new_numbering_instance(doc, "List Bullet")
            apply_numbering(p, active_list_num_id)
            active_list_kind = "bullet"
            inline_runs(p, bullet.group(1), size=10.3)
            continue
        numbered = re.match(r"^\s*(\d+)\.\s+(.*)$", line)
        if numbered:
            flush_paragraph()
            # Keep the source numbering literal.  Word's built-in decimal
            # list definitions can continue numbering across separated lists
            # after pagination; literal numbering is deterministic and keeps
            # the document faithful to the Markdown source.
            p = doc.add_paragraph(style="Normal")
            p.paragraph_format.left_indent = Inches(0.38)
            p.paragraph_format.first_line_indent = Inches(-0.19)
            inline_runs(p, f"{numbered.group(1)}. {numbered.group(2)}", size=10.3)
            active_list_kind = "number"
            continue
        paragraph_lines.append(line)
    flush_paragraph()
    flush_table()
    flush_code()


def build_one(source_name: str, output_name: str, manual: bool) -> Path:
    source = REPORTS / source_name
    doc = Document()
    configure_doc(doc, manual)
    add_title_block(doc, source.stem.replace("_", " "), manual)
    content = source.read_text(encoding="utf-8")
    # Remove the Markdown title and metadata from the body; they are in the
    # dedicated title block above.
    content = re.sub(r"^# .*?\n\n", "", content, count=1, flags=re.S)
    content = re.sub(r"^版本：.*?\n\n", "", content, count=1, flags=re.S)
    render_markdown(doc, content, manual)
    out = OUT / output_name
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out)
    return out


def main() -> None:
    outputs = [
        build_one("产品设计方案_智能压裂预测平台.md", "产品设计方案_智能压裂预测平台.docx", False),
        build_one("概念设计方案_智能压裂预测平台.md", "概念设计方案_智能压裂预测平台.docx", False),
        build_one("用户手册_智能压裂预测平台.md", "用户手册_智能压裂预测平台.docx", True),
        build_one("技术实验附录_结果与边界.md", "技术实验附录_结果与边界.docx", False),
    ]
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
