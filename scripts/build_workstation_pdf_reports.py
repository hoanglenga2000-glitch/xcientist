from __future__ import annotations

import base64
import html
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "pdf"
REPORT_DIR = ROOT / "reports"
UI_SHOTS = ROOT / "docs" / "ui-verification-20260627-clickable-11-pages"
KAGGLE_MD = REPORT_DIR / "Kaggle铜牌自动化训练系统汇报.md"

CHROME_CANDIDATES = [
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def image_data_uri(path: Path) -> str:
    data = path.read_bytes()
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def four_layer_architecture_svg() -> str:
    return """
<section class="diagram">
  <h2>系统四层自进化架构图</h2>
  <svg viewBox="0 0 940 430" role="img" aria-label="四层自进化架构">
    <defs>
      <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
        <path d="M0,0 L8,3 L0,6 Z" fill="#000"/>
      </marker>
    </defs>
    <rect x="20" y="20" width="900" height="390" fill="#fff" stroke="#000" stroke-width="1.6"/>
    <g class="layer">
      <rect x="60" y="48" width="820" height="72" fill="#fff" stroke="#000"/>
      <text x="76" y="76" class="diagram-title">Layer 4  Island Model 并行探索层</text>
      <text x="76" y="101">特征工程岛屿 / 模型多样性岛屿 / 集成融合岛屿，负责停滞后的多路线并行探索。</text>
    </g>
    <g class="layer">
      <rect x="60" y="142" width="820" height="72" fill="#fff" stroke="#000"/>
      <text x="76" y="170" class="diagram-title">Layer 3  XCIENTIST Research Harness 科研审计层</text>
      <text x="76" y="195">IdeaContract、ValidationContract、ClaimAudit，约束假设、指标、消融和结论边界。</text>
    </g>
    <g class="layer">
      <rect x="60" y="236" width="820" height="72" fill="#fff" stroke="#000"/>
      <text x="76" y="264" class="diagram-title">Layer 2  MLEvolve Search + Retrospective Memory 搜索层</text>
      <text x="76" y="289">MCGS 搜索、分支选择、历史经验检索、Base / Stepwise / Diff 代码生成策略。</text>
    </g>
    <g class="layer">
      <rect x="60" y="330" width="820" height="56" fill="#fff" stroke="#000"/>
      <text x="76" y="358" class="diagram-title">Layer 1  Workstation Execution 工作站执行层</text>
      <text x="76" y="378">数据审计、代码实现、GPU/HPC 训练、OOF、Gate、Kaggle 提交和报告归档。</text>
    </g>
    <line x1="470" y1="120" x2="470" y2="142" stroke="#000" stroke-width="1.4" marker-end="url(#arrow)"/>
    <line x1="470" y1="214" x2="470" y2="236" stroke="#000" stroke-width="1.4" marker-end="url(#arrow)"/>
    <line x1="470" y1="308" x2="470" y2="330" stroke="#000" stroke-width="1.4" marker-end="url(#arrow)"/>
    <path d="M880 358 C920 300, 920 90, 880 78" fill="none" stroke="#000" stroke-width="1.4" marker-end="url(#arrow)"/>
    <text x="718" y="232" class="diagram-note">分数、失败原因、记忆回流</text>
  </svg>
</section>
"""


def closed_loop_svg() -> str:
    steps = [
        ("任务解析", "Task Spec"),
        ("Agent 分派", "Context Split"),
        ("代码生成", "Code Agent"),
        ("GPU/HPC 训练", "Remote Job"),
        ("指标与 OOF", "Metrics"),
        ("Gate 审计", "Validation"),
        ("提交候选", "Submission"),
        ("报告归档", "Report"),
    ]
    boxes = []
    arrows = []
    x0, y0 = 40, 80
    w, h, gap = 95, 58, 18
    for i, (cn, en) in enumerate(steps):
        x = x0 + (i % 4) * (w + gap)
        y = y0 + (i // 4) * 128
        boxes.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="#fff" stroke="#000"/>'
            f'<text x="{x + w/2}" y="{y + 24}" text-anchor="middle" class="diagram-title">{cn}</text>'
            f'<text x="{x + w/2}" y="{y + 44}" text-anchor="middle">{en}</text>'
        )
        if i in [0, 1, 2, 4, 5, 6]:
            x1 = x + w
            y1 = y + h / 2
            x2 = x + w + gap - 4
            arrows.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y1}" stroke="#000" marker-end="url(#arrow2)"/>')
    arrows.append('<line x1="209" y1="109" x2="322" y2="109" stroke="#000" marker-end="url(#arrow2)"/>')
    arrows.append('<line x1="322" y1="109" x2="435" y2="109" stroke="#000" marker-end="url(#arrow2)"/>')
    arrows.append('<line x1="491" y1="138" x2="491" y2="208" stroke="#000" marker-end="url(#arrow2)"/>')
    arrows.append('<line x1="435" y1="237" x2="322" y2="237" stroke="#000" marker-end="url(#arrow2)"/>')
    arrows.append('<line x1="322" y1="237" x2="209" y2="237" stroke="#000" marker-end="url(#arrow2)"/>')
    arrows.append('<line x1="153" y1="208" x2="153" y2="138" stroke="#000" marker-end="url(#arrow2)"/>')
    return f"""
<section class="diagram">
  <h2>工作站闭环执行流程图</h2>
  <svg viewBox="0 0 540 330" role="img" aria-label="工作站闭环执行流程">
    <defs>
      <marker id="arrow2" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
        <path d="M0,0 L8,3 L0,6 Z" fill="#000"/>
      </marker>
    </defs>
    <rect x="20" y="20" width="500" height="290" fill="#fff" stroke="#000" stroke-width="1.5"/>
    <text x="270" y="52" text-anchor="middle" class="diagram-heading">Artifact-based Workflow with Human Gate</text>
    {''.join(boxes)}
    {''.join(arrows)}
    <text x="270" y="294" text-anchor="middle" class="diagram-note">所有阶段产物均写入 Evidence Ledger，并由 Gate 控制训练、提交和结论声明。</text>
  </svg>
</section>
"""


def ui_architecture_svg() -> str:
    return """
<section class="diagram">
  <h2>前端页面与后端接口对接架构图</h2>
  <svg viewBox="0 0 940 420" role="img" aria-label="UI 和后端对接架构">
    <defs>
      <marker id="arrow3" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
        <path d="M0,0 L8,3 L0,6 Z" fill="#000"/>
      </marker>
    </defs>
    <rect x="20" y="20" width="900" height="380" fill="#fff" stroke="#000" stroke-width="1.6"/>
    <rect x="60" y="64" width="220" height="250" fill="#fff" stroke="#000"/>
    <text x="170" y="94" text-anchor="middle" class="diagram-title">React / Tailwind 前端</text>
    <text x="80" y="126">10 个核心页面</text>
    <text x="80" y="154">data-ui-action</text>
    <text x="80" y="182">导航、按钮、表格、Gate</text>
    <text x="80" y="210">状态显示与人工审批入口</text>
    <rect x="360" y="64" width="220" height="250" fill="#fff" stroke="#000"/>
    <text x="470" y="94" text-anchor="middle" class="diagram-title">Workstation API</text>
    <text x="380" y="126">/api/workstation-actions</text>
    <text x="380" y="154">ui_component_click</text>
    <text x="380" y="182">Gate approval</text>
    <text x="380" y="210">Artifact manifest</text>
    <text x="380" y="238">Run registry</text>
    <rect x="660" y="64" width="220" height="250" fill="#fff" stroke="#000"/>
    <text x="770" y="94" text-anchor="middle" class="diagram-title">科研工作站后端</text>
    <text x="680" y="126">Agent Orchestrator</text>
    <text x="680" y="154">GPU / HPC Job</text>
    <text x="680" y="182">Kaggle Connector</text>
    <text x="680" y="210">Evidence Ledger</text>
    <text x="680" y="238">Report Studio</text>
    <line x1="280" y1="188" x2="356" y2="188" stroke="#000" stroke-width="1.4" marker-end="url(#arrow3)"/>
    <line x1="580" y1="188" x2="656" y2="188" stroke="#000" stroke-width="1.4" marker-end="url(#arrow3)"/>
    <path d="M770 314 C700 372, 240 372, 170 314" fill="none" stroke="#000" stroke-width="1.4" marker-end="url(#arrow3)"/>
    <text x="470" y="360" text-anchor="middle" class="diagram-note">后端状态、证据和 Gate 结果回流前端；前端不绕过训练、提交和凭据控制。</text>
  </svg>
</section>
"""


def markdown_inline(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"✅", '<span class="ok">通过</span>', text)
    text = re.sub(r"🥉", '<span class="bronze">铜牌</span>', text)
    text = re.sub(r"⚠️", '<span class="warn">接近</span>', text)
    text = re.sub(r"❌", '<span class="bad">未达</span>', text)
    text = re.sub(r"🔧", '<span class="warn">修复中</span>', text)
    text = re.sub(r"🚫", '<span class="muted">异常</span>', text)
    return text


def table_to_html(lines: list[str]) -> str:
    rows = []
    for line in lines:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if all(re.match(r"^:?-{3,}:?$", c) for c in cells):
            continue
        rows.append(cells)
    if not rows:
        return ""
    head, body = rows[0], rows[1:]
    out = ["<table><thead><tr>"]
    out.extend(f"<th>{markdown_inline(c)}</th>" for c in head)
    out.append("</tr></thead><tbody>")
    for row in body:
        out.append("<tr>")
        out.extend(f"<td>{markdown_inline(c)}</td>" for c in row)
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def markdown_to_html(md: str) -> str:
    lines = md.splitlines()
    out: list[str] = []
    i = 0
    in_code = False
    code_lines: list[str] = []
    in_ul = False
    in_ol = False
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            if not in_code:
                in_code = True
                code_lines = []
            else:
                out.append("<pre><code>" + html.escape("\n".join(code_lines)) + "</code></pre>")
                in_code = False
            i += 1
            continue
        if in_code:
            code_lines.append(line)
            i += 1
            continue

        if stripped == "":
            if in_ul:
                out.append("</ul>")
                in_ul = False
            if in_ol:
                out.append("</ol>")
                in_ol = False
            i += 1
            continue

        if stripped == "---":
            out.append("<hr/>")
            i += 1
            continue

        if stripped.startswith("|") and "|" in stripped[1:]:
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            out.append(table_to_html(table_lines))
            continue

        heading = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if heading:
            if in_ul:
                out.append("</ul>")
                in_ul = False
            if in_ol:
                out.append("</ol>")
                in_ol = False
            level = len(heading.group(1))
            out.append(f"<h{level}>{markdown_inline(heading.group(2))}</h{level}>")
            i += 1
            continue

        if stripped.startswith(">"):
            out.append(f"<blockquote>{markdown_inline(stripped.lstrip('> ').strip())}</blockquote>")
            i += 1
            continue

        if re.match(r"^[-*]\s+", stripped):
            if not in_ul:
                out.append("<ul>")
                in_ul = True
            item_text = re.sub(r"^[-*]\s+", "", stripped)
            out.append(f"<li>{markdown_inline(item_text)}</li>")
            i += 1
            continue

        if re.match(r"^\d+\.\s+", stripped):
            if not in_ol:
                out.append("<ol>")
                in_ol = True
            ol_text = re.sub(r"^\d+\.\s+", "", stripped)
            out.append(f"<li>{markdown_inline(ol_text)}</li>")
            i += 1
            continue

        if in_ul:
            out.append("</ul>")
            in_ul = False
        if in_ol:
            out.append("</ol>")
            in_ol = False
        out.append(f"<p>{markdown_inline(stripped)}</p>")
        i += 1

    if in_ul:
        out.append("</ul>")
    if in_ol:
        out.append("</ol>")
    return "\n".join(out)


STYLE = """
@page { size: A4; margin: 14mm 13mm 15mm; }
* { box-sizing: border-box; }
body {
  margin: 0;
  background: #fff;
  color: #000;
  font-family: "Noto Sans SC", "Microsoft YaHei", "DengXian", Arial, sans-serif;
  font-size: 12px;
  line-height: 1.62;
}
.page {
  max-width: 960px;
  margin: 0 auto;
  background: #fff;
}
.cover {
  min-height: 255mm;
  padding: 28mm 18mm 20mm;
  background: #fff;
  color: #000;
  border: 1px solid #000;
  page-break-after: always;
}
.eyebrow {
  display: inline-block;
  padding: 5px 10px;
  border: 1px solid #000;
  border-radius: 999px;
  color: #000;
  font-size: 11px;
  letter-spacing: .08em;
  text-transform: uppercase;
}
.cover h1 {
  margin: 20mm 0 8mm;
  font-size: 31px;
  line-height: 1.22;
  letter-spacing: .01em;
  color: #000;
}
.cover .subtitle {
  max-width: 760px;
  font-size: 15px;
  color: #000;
}
.cover-grid {
  margin-top: 18mm;
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 8px;
}
.cover-card {
  min-height: 74px;
  padding: 10px;
  border: 1px solid #000;
  border-radius: 0;
  background: #fff;
}
.cover-card strong { display: block; font-size: 20px; color: #000; }
.cover-card span { display: block; margin-top: 3px; color: #000; }
.cover-footer {
  position: absolute;
  bottom: 18mm;
  left: 18mm;
  right: 18mm;
  color: #000;
  display: flex;
  justify-content: space-between;
  border-top: 1px solid #000;
  padding-top: 8mm;
}
.content { padding: 0 2mm; }
.executive {
  display: grid;
  grid-template-columns: 1.1fr .9fr;
  gap: 10px;
  margin: 0 0 14px;
}
.panel {
  border: 1px solid #000;
  border-radius: 0;
  background: #fff;
  padding: 12px 14px;
  page-break-inside: avoid;
}
.panel h2, .panel h3 { margin-top: 0; }
h1, h2, h3, h4 {
  color: #000;
  line-height: 1.35;
  page-break-after: avoid;
}
h1 { font-size: 24px; margin: 0 0 16px; }
h2 {
  font-size: 18px;
  margin: 22px 0 8px;
  padding-top: 7px;
  border-top: 1px solid #000;
}
h3 { font-size: 14px; margin: 16px 0 7px; color: #000; }
h4 { font-size: 13px; margin: 12px 0 6px; }
p { margin: 6px 0; }
blockquote {
  margin: 8px 0 14px;
  padding: 8px 10px;
  border-left: 4px solid #000;
  background: #fff;
  color: #000;
  border-radius: 0;
}
table {
  width: 100%;
  border-collapse: collapse;
  margin: 9px 0 12px;
  font-size: 10.3px;
  page-break-inside: avoid;
}
th, td {
  border: 1px solid #000;
  padding: 5px 6px;
  vertical-align: top;
  color: #000;
}
th {
  background: #f2f2f2;
  color: #000;
  font-weight: 700;
}
tr:nth-child(even) td { background: #fff; }
pre {
  white-space: pre-wrap;
  word-break: break-word;
  background: #fff;
  color: #000;
  border: 1px solid #000;
  border-radius: 0;
  padding: 10px;
  font-family: "Source Code Pro", Consolas, monospace;
  font-size: 8.8px;
  line-height: 1.35;
  page-break-inside: avoid;
}
code {
  font-family: "Source Code Pro", Consolas, monospace;
  background: #fff;
  border: 1px solid #000;
  border-radius: 0;
  padding: 0 4px;
  color: #000;
}
pre code { background: transparent; border: 0; padding: 0; color: inherit; }
ul, ol { margin: 6px 0 10px 18px; padding: 0; }
li { margin: 3px 0; }
.ok, .bronze, .warn, .bad, .muted {
  display: inline-block;
  border-radius: 0;
  padding: 1px 6px;
  font-weight: 700;
  font-size: 10px;
  border: 1px solid #000;
  background: #fff;
  color: #000;
}
.ok, .bronze, .warn, .bad, .muted { background: #fff; color: #000; }
.shot {
  page-break-inside: avoid;
  margin: 14px 0 18px;
  border: 1px solid #000;
  border-radius: 0;
  overflow: hidden;
  background: #fff;
}
.shot h3 {
  margin: 0;
  padding: 9px 12px;
  color: #000;
  background: #fff;
  border-bottom: 1px solid #000;
}
.shot img {
  width: 100%;
  display: block;
  filter: grayscale(1) contrast(1.08);
}
.shot p {
  padding: 8px 12px 10px;
  margin: 0;
  color: #000;
}
.feature-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 9px;
  margin: 10px 0;
}
.feature-card {
  border: 1px solid #000;
  border-radius: 0;
  padding: 10px;
  background: #fff;
  page-break-inside: avoid;
}
.feature-card strong {
  display: block;
  margin-bottom: 4px;
  color: #000;
}
.diagram {
  page-break-inside: avoid;
  margin: 12px 0 18px;
}
.diagram svg {
  display: block;
  width: 100%;
  border: 1px solid #000;
  background: #fff;
}
.diagram text {
  fill: #000;
  font-family: "Noto Sans SC", "Microsoft YaHei", "DengXian", Arial, sans-serif;
  font-size: 13px;
}
.diagram-title {
  font-weight: 700;
  font-size: 15px;
}
.diagram-heading {
  font-weight: 700;
  font-size: 16px;
}
.diagram-note {
  font-size: 12px;
}
.page-break { page-break-before: always; }
.footer-note { color: #000; font-size: 10px; margin-top: 12px; }
"""

UI_DESIGN_STYLE = STYLE + """
body {
  background: #f6f8fb;
  color: #172033;
}
.cover {
  background:
    linear-gradient(135deg, rgba(37,99,235,.96), rgba(6,19,38,.98)),
    radial-gradient(circle at 80% 10%, rgba(56,189,248,.25), transparent 30%);
  border: 0;
  color: #fff;
}
.cover h1 { color: #fff; }
.cover .subtitle { color: #dbeafe; }
.eyebrow {
  border-color: rgba(255,255,255,.28);
  color: #dbeafe;
}
.cover-card {
  border-color: rgba(255,255,255,.24);
  border-radius: 8px;
  background: rgba(255,255,255,.08);
}
.cover-card strong { color: #fff; }
.cover-card span { color: #bfdbfe; }
.cover-footer {
  color: #dbeafe;
  border-top-color: rgba(255,255,255,.22);
}
.panel, .feature-card, .shot {
  border-color: #dbe3ef;
  border-radius: 8px;
}
.panel { background: #fff; }
h1, h2, h3, h4 { color: #0f172a; }
h2 { border-top-color: #e2e8f0; }
blockquote {
  border-left-color: #2563eb;
  background: #eff6ff;
  color: #1e3a8a;
  border-radius: 6px;
}
th, td { border-color: #d9e2ef; color: #172033; }
th { background: #f1f5f9; color: #0f172a; }
tr:nth-child(even) td { background: #fbfdff; }
code {
  background: #f1f5f9;
  border-color: #e2e8f0;
  color: #172033;
  border-radius: 4px;
}
pre {
  background: #08111f;
  color: #e5eefb;
  border: 0;
  border-radius: 8px;
}
.ok, .bronze, .warn, .bad, .muted {
  border: 0;
  border-radius: 999px;
}
.ok { background: #dcfce7; color: #166534; }
.bronze { background: #ffedd5; color: #9a3412; }
.warn { background: #fef3c7; color: #92400e; }
.bad { background: #fee2e2; color: #991b1b; }
.muted { background: #e2e8f0; color: #475569; }
.shot h3 {
  background: #f8fafc;
  border-bottom-color: #e2e8f0;
  color: #0f172a;
}
.shot img {
  filter: none;
}
.shot p { color: #475569; }
.feature-card {
  background: #fbfdff;
}
.diagram svg {
  border-color: #dbe3ef;
}
.diagram text {
  fill: #0f172a;
}
"""


def html_doc(title: str, cover_subtitle: str, cards: list[tuple[str, str]], body: str, style: str = STYLE) -> str:
    cards_html = "\n".join(
        f'<div class="cover-card"><strong>{html.escape(value)}</strong><span>{html.escape(label)}</span></div>'
        for value, label in cards
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>{html.escape(title)}</title>
  <style>{style}</style>
</head>
<body>
  <section class="cover">
    <div class="eyebrow">Academic Research OS · Kaggle / MLE-Bench Workstation</div>
    <h1>{html.escape(title)}</h1>
    <div class="subtitle">{html.escape(cover_subtitle)}</div>
    <div class="cover-grid">{cards_html}</div>
    <div class="cover-footer">
      <span>负责人：景浩伟</span>
      <span>生成日期：2026-06-27</span>
    </div>
  </section>
  <main class="page content">{body}</main>
</body>
</html>"""


def build_kaggle_html() -> str:
    md = read_text(KAGGLE_MD)
    body = f"""
<section class="executive">
  <div class="panel">
    <h2>汇报重点</h2>
    <p>本报告突出 AI 科研工作站的实际工程成果：四层自进化架构、工作站执行闭环、GPU/HPC 调度、Kaggle 提交与分数追踪、XCIENTIST 风格审计和后续提分路线。</p>
    <p>现阶段成果不是单一训练脚本，而是一套可持续扩展到 Kaggle / MLE-Bench 任务的自动化科研工程系统。</p>
  </div>
  <div class="panel">
    <h2>当前结果快照</h2>
    <table>
      <tr><th>指标</th><th>结果</th></tr>
      <tr><td>有效参赛</td><td>27 场结构化表格竞赛</td></tr>
      <tr><td>铜牌结果</td><td>18 / 27</td></tr>
      <tr><td>铜牌率</td><td>66.7%</td></tr>
      <tr><td>核心能力</td><td>训练、Gate、提交、记忆、审计、报告闭环</td></tr>
    </table>
  </div>
</section>
{four_layer_architecture_svg()}
{closed_loop_svg()}
{markdown_to_html(md)}
"""
    return html_doc(
        "Kaggle 铜牌自动化训练系统成果汇报",
        "面向 Kaggle / MLE-Bench 的自进化 AI 科研工作站：从数据、代码、训练、Gate、提交到报告的闭环成果。",
        [("18/27", "铜牌任务"), ("66.7%", "铜牌率"), ("27", "有效竞赛"), ("4 层", "自进化架构")],
        body,
    )


UI_PAGES = [
    ("科研总览", "overview.png", "展示 Mission Control、运行状态、Benchmark / Medal Board、HPC 资源、证据台账与提交 Gate，是工作站的第一屏指挥台。"),
    ("AI 控制台", "control.png", "用于把研究任务分派给不同 Agent，展示任务拆解、执行队列、人工 Gate 和资源状态。"),
    ("数据 / Kaggle", "data.png", "围绕 Kaggle 数据、competition schema、submission 格式、数据审计和下载状态建立统一入口。"),
    ("报告工作室", "report.png", "根据真实训练结果、metrics、artifact 和 claim audit 自动生成报告，同时保留人工审核和修改入口。"),
    ("代码 Agent IDE", "code.png", "把代码生成做成可审计 IDE：文件树、算法代码、diff、implementation contract、quality gate 和终端记录同屏展示。"),
    ("GPU / HPC", "gpu.png", "展示算力连接、作业队列、资源占用、远程日志和训练产物回传状态，支持后端按 action hook 接入真实调度。"),
    ("证据台账", "evidence.png", "记录 artifact、run、hash、stage、status 和 claim 绑定关系，支撑可追溯科研审计。"),
    ("文献 / RAG", "literature.png", "管理论文、方法库、RAG context 和 Agent 可复用知识，为实验策略和报告生成提供背景证据。"),
    ("Agent 运行时", "runtime.png", "展示每个 Agent 的任务状态、工具调用、失败回退、heartbeat、缓存命中和 runtime trace。"),
    ("系统设置", "settings.png", "包含账号登录、语言切换、浅色/深色模式、连接器、凭据安全、提交策略和设计治理入口。"),
]


def build_ui_report_md() -> str:
    lines = [
        "# AI 科研工作站 UI 设计成果与功能说明",
        "",
        "> 汇报日期：2026-06-27 | 负责人：景浩伟",
        "",
        "## 一、设计目标",
        "",
        "本轮 UI 重设计目标不是做静态展示页，而是把科研工作站界面升级为可真实操作、可审计、可对接后端的 Academic Research OS 控制台。界面围绕 Kaggle / HPC / 多 Agent / Evidence / Gate / Report 闭环组织，不再采用普通后台管理系统的卡片堆叠。",
        "",
        "## 二、设计原则",
        "",
        "- 深色专业侧边栏 + 浅色高密度科研工作区。",
        "- 所有关键控件保留 `data-ui-action`，方便后端按 action id 接管。",
        "- 不使用整页图片覆盖，页面由 React / Tailwind 真实组件构成。",
        "- Gate、证据、资源、Agent 状态必须真实可区分，不能把 blocked 显示成 ready。",
        "- 报告、代码、训练、提交都要能被审计和复现。",
        "",
        "## 三、当前验证结果",
        "",
        "| 项目 | 结果 |",
        "| --- | --- |",
        "| 核心页面 | 10 个唯一页面 |",
        "| 前端技术 | Next.js + React + Tailwind |",
        "| Figma 结构 | 可编辑 frame / component / instance，不是截图覆盖 |",
        "| 后端接口钩子 | `ui_component_click` + `data-ui-action` |",
        "| 构建验证 | `npm run build` 通过 |",
        "| 类型验证 | `npm run typecheck` 通过 |",
        "| 页面访问 | 10 个页面 HTTP 200，CSS/JS 正常加载 |",
        "",
        "## 四、页面功能说明",
        "",
    ]
    for title, filename, desc in UI_PAGES:
        lines.extend([
            f"### {title}",
            "",
            desc,
            "",
            f"截图：`docs/ui-verification-20260627-clickable-11-pages/{filename}`",
            "",
        ])
    lines.extend([
        "## 五、后端对接方式",
        "",
        "前端统一通过 `/api/workstation-actions` 记录 UI 操作。后端可以读取 `metadata.page`、`metadata.component_type`、`metadata.action_id`、`metadata.label` 和 `metadata.disabled`，把普通点击、阻断点击、Gate 审批和资源调度区分开。",
        "",
        "典型 payload：",
        "",
        "```json",
        "{",
        '  "action": "ui_component_click",',
        '  "task_id": "playground_series_s6e6",',
        '  "metadata": {',
        '    "page": "code",',
        '    "component_type": "button",',
        '    "action_id": "blocked_send_to_hpc",',
        '    "label": "Send to HPC",',
        '    "disabled": true',
        "  }",
        "}",
        "```",
        "",
        "## 六、展示价值",
        "",
        "这套 UI 能向老师清楚展示：系统不是单次训练脚本，而是具备任务分派、代码生成、GPU/HPC 调度、证据审计、Gate 控制、报告生成和后续多任务评测扩展能力的科研工作站。",
        "",
        "## 七、下一步规划",
        "",
        "- 接入真实后端 API，把 UI action hook 对应到 workstation run、Agent trace、GPU job、Evidence 和 Report 实体。",
        "- 将 Kaggle / MLE-Bench 结果榜单和 medal gate 做成可更新的正式评测面板。",
        "- 增加实验对比视图、失败归因视图、Retrospective Memory 浏览器和 claim drift 审计页。",
        "- 在正式汇报中区分官方 Kaggle 结果、本地 CV 代理结果和未验证目标，保持证据边界清晰。",
    ])
    return "\n".join(lines)


def build_ui_html(ui_md: str) -> str:
    features = """
<section class="executive">
  <div class="panel">
    <h2>汇报定位</h2>
    <p>这份材料用设计截图解释 AI 科研工作站的功能闭环：多 Agent 控制、代码 IDE、GPU/HPC、Kaggle 数据、证据台账、报告生成、系统设置与后端可接入能力。</p>
  </div>
  <div class="panel">
    <h2>验证快照</h2>
    <table>
      <tr><th>项目</th><th>结果</th></tr>
      <tr><td>核心页面</td><td>10 个</td></tr>
      <tr><td>Figma</td><td>可编辑组件结构</td></tr>
      <tr><td>前端</td><td>React / Tailwind 真实组件</td></tr>
      <tr><td>接口</td><td>data-ui-action 后端钩子</td></tr>
    </table>
  </div>
</section>
<section class="feature-grid">
  <div class="feature-card"><strong>科研 OS 操作台</strong>从总览页即可看到任务、Agent、Gate、HPC、Kaggle、证据和报告状态。</div>
  <div class="feature-card"><strong>可审计 Agent IDE</strong>每段算法代码、diff、质量门禁、终端日志都能被追踪。</div>
  <div class="feature-card"><strong>Evidence-first</strong>所有训练结果和结论都绑定 artifact、run id、hash 和 claim audit。</div>
  <div class="feature-card"><strong>前后端隔离</strong>前端只做操作和审计入口，训练、提交、凭据和资源由后端 Gate 控制。</div>
</section>
"""
    shots = []
    for title, filename, desc in UI_PAGES:
        path = UI_SHOTS / filename
        if path.exists():
            shots.append(
                f'<section class="shot"><h3>{html.escape(title)}</h3>'
                f'<img src="{image_data_uri(path)}" alt="{html.escape(title)}"/>'
                f'<p>{html.escape(desc)}</p></section>'
            )
    figma_path = UI_SHOTS / "figma-current-23-2.png"
    if figma_path.exists():
        shots.insert(
            0,
            f'<section class="shot"><h3>Figma 当前大 Frame</h3>'
            f'<img src="{image_data_uri(figma_path)}" alt="Figma frame"/>'
            f'<p>Figma file key: YRGlARCURv2sKKmSHeNWA6 / node 23:2。该页面用于保留可编辑的整体设计上下文。</p></section>',
        )
    body = features + ui_architecture_svg() + markdown_to_html(ui_md) + '<div class="page-break"></div>' + "\n".join(shots)
    return html_doc(
        "AI 科研工作站 UI 设计成果与功能说明",
        "基于 Figma 设计图和本地页面截图，说明工作站每个功能页、后端接口钩子与科研闭环展示价值。",
        [("10", "核心页面"), ("102+", "页面 action hook"), ("0", "整页截图覆盖"), ("200", "页面访问状态")],
        body,
        UI_DESIGN_STYLE,
    )


def find_chrome() -> Path:
    for path in CHROME_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError("未找到 Chrome 或 Edge，可安装 Chrome 后重试。")


def print_pdf(html_path: Path, pdf_path: Path) -> None:
    chrome = find_chrome()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(chrome),
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_path}",
        html_path.as_uri(),
    ]
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    kaggle_html = build_kaggle_html()
    kaggle_html_path = OUT_DIR / "Kaggle铜牌自动化训练系统成果汇报.html"
    kaggle_pdf_path = OUT_DIR / "Kaggle铜牌自动化训练系统成果汇报.pdf"
    kaggle_bw_pdf_path = OUT_DIR / "Kaggle铜牌自动化训练系统成果汇报_黑白学术版.pdf"
    write_text(kaggle_html_path, kaggle_html)
    print_pdf(kaggle_html_path, kaggle_pdf_path)
    shutil.copyfile(kaggle_pdf_path, kaggle_bw_pdf_path)

    ui_md = build_ui_report_md()
    ui_md_path = REPORT_DIR / "AI科研工作站UI设计成果与功能说明.md"
    ui_html = build_ui_html(ui_md)
    ui_html_path = OUT_DIR / "AI科研工作站UI设计成果与功能说明.html"
    ui_pdf_path = OUT_DIR / "AI科研工作站UI设计成果与功能说明.pdf"
    ui_bw_pdf_path = OUT_DIR / "AI科研工作站UI设计成果与功能说明_黑白学术版.pdf"
    ui_design_pdf_path = OUT_DIR / "AI科研工作站UI设计成果与功能说明_设计图彩色版.pdf"
    write_text(ui_md_path, ui_md)
    write_text(ui_html_path, ui_html)
    print_pdf(ui_html_path, ui_pdf_path)
    shutil.copyfile(ui_pdf_path, ui_design_pdf_path)

    print("generated:")
    for path in [kaggle_pdf_path, kaggle_bw_pdf_path, ui_pdf_path, ui_design_pdf_path, kaggle_html_path, ui_html_path, ui_md_path]:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
