"""Deterministic research-report generation for a completed agent run.

The plan wants an auto-generated report — and, crucially, one whose every claim is
backed by the auditable search graph, not by an LLM's recollection. So this builds
``research_report.md`` DETERMINISTICALLY from the run's own artifacts:
``search_graph.json`` (node lineage + promotion history), ``summary.json`` (best +
counts), and the retrospective memory. No model call, nothing invented.

The report is organized around the plan's evaluation axes (research-process
quality): what was tried, what the gate promoted and why, what failed and how it
was diagnosed, and what lessons were banked — so a reader can audit the run end to
end. It also restates the honest-scope boundary (local CV/proxy; official rank
needs a human-gated Kaggle submission).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


def _load_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


def _fmt_score(v: Any) -> str:
    return f"{v:.6f}" if isinstance(v, (int, float)) else "—"


def build_report(run_dir: str | Path) -> str:
    """Render a markdown research report for a finished run directory.

    Reads whatever artifacts exist; degrades gracefully if some are missing (a run
    that crashed early still gets a truthful, if thin, report)."""
    run_dir = Path(run_dir)
    summary = _load_json(run_dir / "summary.json") or {}
    graph = _load_json(run_dir / "search_graph.json") or {}
    nodes = {n["exp_id"]: n for n in graph.get("nodes", [])}
    promotions = graph.get("promotion_history", []) or summary.get("promotion_history", [])
    task = summary.get("task", graph.get("task_id", "unknown"))
    metric = summary.get("metric", graph.get("metric_name", "cv_score"))
    direction = summary.get("metric_direction", graph.get("metric_direction", "maximize"))

    lines: list[str] = []
    lines.append(f"# 研究报告 — {task}")
    lines.append("")
    lines.append(f"*生成于 {datetime.now().isoformat(timespec='seconds')}  ·  "
                 f"指标 {metric} ({direction})  ·  引擎 research_os agent*")
    lines.append("")

    # ── 1. 结论摘要（全部来自晋升门禁的裁决，非模型自述）────────────────────────
    best_id = summary.get("best_exp_id")
    best = nodes.get(best_id) if best_id else None
    lines.append("## 摘要")
    lines.append("")
    if best is not None:
        lines.append(f"- 最优实验：**{best_id}**，{metric} = **{_fmt_score(best.get('cv_score'))}**")
        lines.append(f"- 假设：{best.get('hypothesis', '(无)')}")
    else:
        lines.append("- 尚无被晋升的最优解（没有实验通过晋升门禁）。")
    lines.append(f"- 实验总数：{summary.get('n_iterations', len(nodes))}　晋升次数："
                 f"{summary.get('n_promotions', 0)}　agent 主动收尾：{summary.get('finished_by_agent', '?')}")
    if summary.get("agent_summary"):
        lines.append(f"- 研究员自述：{summary['agent_summary']}")
    lines.append("")

    # ── 2. 搜索轨迹（节点血缘 + 晋升裁决，可审计）──────────────────────────────
    lines.append("## 搜索轨迹")
    lines.append("")
    if nodes:
        lines.append("| 实验 | 父 | 模式 | " + metric + " | 运行成功 | 裁决 |")
        lines.append("|---|---|---|---|---|---|")
        for n in nodes.values():
            mark = "✅晋升" if n.get("promoted") else (n.get("decision") or "—")
            lines.append(f"| {n['exp_id']} | {n.get('parent_id') or '—'} | {n.get('branch_type','')} "
                         f"| {_fmt_score(n.get('cv_score'))} | {n.get('run_success')} | {mark} |")
    else:
        lines.append("*（无实验节点）*")
    lines.append("")

    # ── 3. 晋升裁决（为什么晋升/保留，逐条 reason）─────────────────────────────
    if promotions:
        lines.append("## 晋升门禁裁决")
        lines.append("")
        for p in promotions:
            verdict = "晋升" if p.get("promoted") else "保留"
            lines.append(f"- **{p.get('candidate_exp_id')}** → {verdict}："
                         f"{p.get('reason','')}（候选={_fmt_score(p.get('candidate_score'))} "
                         f"基线={_fmt_score(p.get('parent_score'))} Δ={_fmt_score(p.get('promotion_delta'))}）")
        lines.append("")

    # ── 4. 失败与归因（诚实记录，不隐藏失败）──────────────────────────────────
    failed = [n for n in nodes.values() if not n.get("run_success")]
    lines.append("## 失败与归因")
    lines.append("")
    if failed:
        for n in failed:
            err_file = run_dir / n["exp_id"] / "run_error.txt"
            hint = ""
            if err_file.exists():
                first = err_file.read_text(encoding="utf-8", errors="replace").strip().splitlines()
                hint = f"　真错误：{first[-1][:160]}" if first else ""
            lines.append(f"- {n['exp_id']}：{(n.get('hypothesis') or '')[:100]}{hint}")
    else:
        lines.append("*（本轮无失败运行）*")
    lines.append("")

    # ── 5. 经验沉淀（跨轮/跨任务可复用的教训）─────────────────────────────────
    memory = _load_json(run_dir.parent / "retrospective_memory.json")
    if isinstance(memory, list) and memory:
        lines.append("## 经验沉淀（retrospective memory）")
        lines.append("")
        for r in memory[-8:]:
            bits = [f"**{r.get('memory_id')}**"]
            if r.get("what_worked"):
                bits.append(f"有效：{r['what_worked']}")
            if r.get("reusable_strategy"):
                bits.append(f"可复用策略：{r['reusable_strategy']}")
            if r.get("failure_pattern"):
                bits.append(f"失败模式：{r['failure_pattern']}")
            lines.append("- " + "　·　".join(bits))
        lines.append("")

    # ── 6. 诚实边界（计划书要求的范围声明）────────────────────────────────────
    lines.append("## 范围与边界")
    lines.append("")
    lines.append("- 本报告的分数均为**本地 CV / proxy 评估**，非 Kaggle 官方排名。")
    lines.append("- 官方提交需经**人工闸门（Human Gate）**确认，agent 不自动提交。")
    lines.append("- 崩溃/超时/OOM 的运行一律**不可晋升**（晋升门禁 run_success 硬前置）。")
    lines.append("")
    return "\n".join(lines)


def write_report(run_dir: str | Path) -> Path:
    """Build and write ``research_report.md`` into the run dir; return its path."""
    run_dir = Path(run_dir)
    report = build_report(run_dir)
    out = run_dir / "research_report.md"
    out.write_text(report, encoding="utf-8")
    return out
