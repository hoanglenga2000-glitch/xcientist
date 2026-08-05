"""Generate EvoMind new-user DOCX guides from the UTF-8 Markdown sources.

The previous generator embedded long Chinese strings directly in this file, and
that made the DOCX vulnerable to Windows console/source-encoding corruption.
This version keeps the generator ASCII-only and treats the Markdown guides as
the single source of truth.
"""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
SOURCE_GLOB = "EvoMind*20260707.md"
FONT = "Microsoft YaHei"


def set_run_font(run, *, size: float = 10.5, bold: bool | None = None, mono: bool = False) -> None:
    font_name = "Consolas" if mono else FONT
    run.font.name = font_name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    run.font.size = Pt(size)
    run.font.color.rgb = RGBColor(0, 0, 0)
    if bold is not None:
        run.bold = bold


def set_document_styles(doc: Document) -> None:
    for style_name, size, bold in [
        ("Normal", 10.5, False),
        ("Title", 20, True),
        ("Heading 1", 15, True),
        ("Heading 2", 12.5, True),
        ("Heading 3", 11.5, True),
        ("List Bullet", 10.5, False),
    ]:
        try:
            style = doc.styles[style_name]
        except KeyError:
            continue
        style.font.name = FONT
        style._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
        style.font.size = Pt(size)
        style.font.bold = bold
        style.font.color.rgb = RGBColor(0, 0, 0)


def shade_cell(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def add_text_run(paragraph, text: str, *, size: float = 10.5, bold: bool = False, mono: bool = False) -> None:
    run = paragraph.add_run(text)
    set_run_font(run, size=size, bold=bold, mono=mono)


def add_paragraph(doc: Document, text: str) -> None:
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(4)
    add_text_run(paragraph, text)


def add_bullet(doc: Document, text: str) -> None:
    try:
        paragraph = doc.add_paragraph(style="List Bullet")
    except KeyError:
        paragraph = doc.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(2)
    add_text_run(paragraph, text)


def add_code(doc: Document, text: str) -> None:
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.left_indent = Inches(0.24)
    paragraph.paragraph_format.space_before = Pt(3)
    paragraph.paragraph_format.space_after = Pt(7)
    for line in text.rstrip("\n").splitlines():
        add_text_run(paragraph, line + "\n", size=9, mono=True)


def add_table(doc: Document, rows: list[list[str]]) -> None:
    if not rows:
        return
    max_cols = max(len(row) for row in rows)
    table = doc.add_table(rows=1, cols=max_cols)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for row_index, values in enumerate(rows):
        cells = table.rows[row_index].cells if row_index == 0 else table.add_row().cells
        for col_index in range(max_cols):
            text = values[col_index] if col_index < len(values) else ""
            cell = cells[col_index]
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            if row_index == 0:
                shade_cell(cell, "D9EAF7")
            cell.text = ""
            paragraph = cell.paragraphs[0]
            paragraph.paragraph_format.space_after = Pt(0)
            add_text_run(paragraph, text, size=9, bold=(row_index == 0))
    doc.add_paragraph()


def split_table_row(line: str) -> list[str]:
    stripped = line.strip().strip("|")
    return [cell.strip().replace("<br>", "\n") for cell in stripped.split("|")]


def is_table_separator(line: str) -> bool:
    compact = line.strip().replace("|", "").replace(":", "").replace("-", "").replace(" ", "")
    return compact == ""


def render_markdown(doc: Document, markdown: str) -> None:
    lines = markdown.splitlines()
    i = 0
    in_code = False
    code_lines: list[str] = []
    table_lines: list[str] = []

    def flush_table() -> None:
        nonlocal table_lines
        if not table_lines:
            return
        rows = [split_table_row(line) for line in table_lines if not is_table_separator(line)]
        add_table(doc, rows)
        table_lines = []

    while i < len(lines):
        line = lines[i]

        if line.startswith("```"):
            if in_code:
                add_code(doc, "\n".join(code_lines))
                code_lines = []
                in_code = False
            else:
                flush_table()
                in_code = True
            i += 1
            continue

        if in_code:
            code_lines.append(line)
            i += 1
            continue

        if line.strip().startswith("|") and line.strip().endswith("|"):
            table_lines.append(line)
            i += 1
            continue

        flush_table()

        stripped = line.strip()
        if not stripped:
            i += 1
            continue

        if stripped.startswith("# "):
            paragraph = doc.add_paragraph()
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_text_run(paragraph, stripped[2:].strip(), size=20, bold=True)
        elif stripped.startswith("## "):
            doc.add_heading(stripped[3:].strip(), level=1)
        elif stripped.startswith("### "):
            doc.add_heading(stripped[4:].strip(), level=2)
        elif stripped.startswith("- "):
            add_bullet(doc, stripped[2:].strip())
        else:
            add_paragraph(doc, stripped)
        i += 1

    if in_code:
        add_code(doc, "\n".join(code_lines))
    flush_table()


def build_doc(markdown_path: Path) -> Document:
    doc = Document()
    set_document_styles(doc)
    section = doc.sections[0]
    section.top_margin = Inches(0.75)
    section.bottom_margin = Inches(0.75)
    section.left_margin = Inches(0.8)
    section.right_margin = Inches(0.8)

    markdown = markdown_path.read_text(encoding="utf-8")
    render_markdown(doc, markdown)

    for paragraph in doc.paragraphs:
        paragraph.paragraph_format.line_spacing = 1.15
        if paragraph.style.name.startswith("Heading"):
            paragraph.paragraph_format.space_before = Pt(10)
            paragraph.paragraph_format.space_after = Pt(5)

    footer = doc.sections[0].footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_text_run(footer, "EvoMind New User Setup Guide | 2026-07-07 | XCIENTIST AI Research Workstation", size=8)
    return doc


def validate_docx(path: Path) -> dict[str, object]:
    loaded = Document(str(path))
    text = "\n".join(paragraph.text for paragraph in loaded.paragraphs)
    required = [
        "git clone https://github.com/hoanglenga2000-glitch/xcientist.git EvoMind",
        "http://127.0.0.1:8088/?page=control",
        "evomind setup",
        "Kaggle API",
    ]
    missing = [item for item in required if item not in text]
    bad_download_placeholders = [
        item for item in ["<your-repo", "your-repo-url", "git clone <", "仓库地址待填"] if item in text
    ]
    return {
        "paragraphs": len(loaded.paragraphs),
        "tables": len(loaded.tables),
        "missing": missing,
        "bad_download_placeholders": bad_download_placeholders,
        "question_marks": text.count("?"),
        "chinese_chars": sum(1 for char in text if "\u4e00" <= char <= "\u9fff"),
    }


def main() -> None:
    sources = sorted(DOCS.glob(SOURCE_GLOB))
    if not sources:
        raise SystemExit(f"No Markdown sources matched {DOCS / SOURCE_GLOB}")

    for source in sources:
        target = source.with_suffix(".docx")
        document = build_doc(source)
        document.save(target)
        stats = validate_docx(target)
        print(f"DOCX_OUT={target}")
        print(
            "paragraphs={paragraphs} tables={tables} chinese_chars={chinese_chars} "
            "question_marks={question_marks}".format(**stats)
        )
        print(f"missing={stats['missing']!r}")
        print(f"bad_download_placeholders={stats['bad_download_placeholders']!r}")
        if stats["missing"] or stats["bad_download_placeholders"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
