from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]


SOURCE_FILES = [
    ROOT / "web" / "research-agent-workstation" / "src" / "app" / "page.tsx",
    ROOT / "web" / "research-agent-workstation" / "src" / "components" / "workstation" / "AppShell.tsx",
    ROOT / "web" / "research-agent-workstation" / "src" / "components" / "workstation" / "Screens.tsx",
    ROOT / "web" / "research-agent-workstation" / "src" / "components" / "workstation" / "AiControlConsole.tsx",
    ROOT / "web" / "research-agent-workstation" / "src" / "components" / "workstation" / "Common.tsx",
    ROOT / "web" / "research-agent-workstation" / "src" / "components" / "workstation" / "IntegrationStatus.tsx",
    ROOT / "web" / "research-agent-workstation" / "src" / "lib" / "api" / "client.ts",
    ROOT / "web" / "research-agent-workstation" / "src" / "lib" / "api" / "types.ts",
    ROOT / "web" / "research-agent-workstation" / "src" / "app" / "api" / "scientist" / "next-action" / "route.ts",
    ROOT / "web" / "research-agent-workstation" / "src" / "app" / "api" / "scientist" / "continuation-status" / "route.ts",
    ROOT / "web" / "research-agent-workstation" / "src" / "app" / "api" / "scientist" / "continuation-resume" / "route.ts",
    ROOT / "web" / "research-agent-workstation" / "src" / "app" / "api" / "scientist" / "recovery" / "route.ts",
    ROOT / "web" / "research-agent-workstation" / "src" / "app" / "api" / "scientist" / "loop" / "route.ts",
    ROOT / "web" / "research-agent-workstation" / "src" / "app" / "api" / "scientist" / "self-audit" / "route.ts",
    ROOT / "web" / "research-agent-workstation" / "src" / "app" / "api" / "scientist" / "readiness-report" / "route.ts",
    ROOT / "web" / "research-agent-workstation" / "src" / "app" / "api" / "scientist" / "causal-diagnosis" / "route.ts",
    ROOT / "web" / "research-agent-workstation" / "src" / "app" / "api" / "scientist" / "strategy-optimizer" / "route.ts",
    ROOT / "web" / "research-agent-workstation" / "src" / "app" / "api" / "scientist" / "turn" / "route.ts",
    ROOT / "web" / "research-agent-workstation" / "src" / "app" / "api" / "scientist" / "engineering-loop" / "route.ts",
    ROOT / "web" / "research-agent-workstation" / "src" / "app" / "api" / "literature" / "search" / "route.ts",
    ROOT / "web" / "research-agent-workstation" / "src" / "app" / "api" / "tasks" / "[taskId]" / "generate-report-draft" / "route.ts",
]


SOURCE_TERMS = [
    "EvoMind 工作站入口",
    "科学家行动队列",
    "执行安全下一步",
    "科学家工作计划",
    "科学家自我修复计划",
    "科学家执行契约",
    "科学家恢复快照",
    "科学家自主循环",
    "科学家续跑状态",
    "scientist_continuation_status",
    "自动续跑安全工具",
    "scientist_continuation_resume",
    "scientist_self_audit",
    "Scientist Self-Audit",
    "能力趋势",
    "scientist_capability_trend",
    "scientist_readiness_report",
    "Scientist Readiness Report",
    "科学家就绪报告",
    "claim_readiness",
    "capability_readiness",
    "rank_or_medal_claim",
    "Readiness Report",
    "scientist_causal_diagnosis",
    "Scientist Causal Diagnosis",
    "科学家因果诊断",
    "causal_graph",
    "scientist_strategy_optimizer",
    "Scientist Strategy Optimizer",
    "科学家策略优化",
    "intervention_ranking",
    "decision_matrix",
    "scientist_reasoning_synthesis",
    "Evidence-Grounded Scientist Answer",
    "基于证据的科学家回答",
    "reasoning_quality",
    ".xsci/scientist_reasoning_synthesis.json",
    "scientist_engineering_loop",
    "Isolated Engineering Loop",
    "隔离工程执行闭环",
    "review_candidate_before_merge",
    "scientist_loop",
    "科学家回合账本",
    "科学家步骤轨迹",
    "no_training_started",
    "blocked_until_explicit_human_approval",
    "报告工作室",
    "重新生成完整草稿",
    "导出完整 Markdown",
    "导出审计 JSON",
    "source: \"report_studio\"",
    "文献与知识库",
    "文献检索",
    "claim_audit",
    "searchLiterature",
    "系统设置",
    "settings_language_zh_cn",
    "settings_language_en_us",
    "settings_theme_light",
    "settings_theme_dark",
    "setLocale",
    "Code Agent IDE",
    "GPU / HPC",
    "完整性 Gate",
]


SOURCE_GROUPS = {
    "gateway_brand": ["EvoMind Gateway", "EvoMind 工作站入口"],
    "ai_scientist_controls": ["scientist_autopilot", "科学家诊断"],
    "safe_next_api": ["/api/scientist/next-action", "executeSafeNextAction"],
    "scientist_recovery_api": ["/api/scientist/recovery", "scientist_recovery", "科学家恢复快照"],
    "scientist_loop_api": ["/api/scientist/loop", "scientist_loop", "科学家自主循环"],
    "scientist_continuation_status_api": ["/api/scientist/continuation-status", "scientist_continuation_status", "科学家续跑状态"],
    "scientist_continuation_resume_api": ["/api/scientist/continuation-resume", "scientist_continuation_resume", "自动续跑安全工具"],
    "scientist_self_audit_api": ["/api/scientist/self-audit", "scientist_self_audit", "Scientist Self-Audit"],
    "scientist_readiness_report_api": ["/api/scientist/readiness-report", "scientist_readiness_report", "Scientist Readiness Report"],
    "scientist_causal_diagnosis_api": ["/api/scientist/causal-diagnosis", "scientist_causal_diagnosis", "Scientist Causal Diagnosis"],
    "scientist_strategy_optimizer_api": ["/api/scientist/strategy-optimizer", "scientist_strategy_optimizer", "Scientist Strategy Optimizer"],
    "scientist_reasoning_synthesis": ["scientist_reasoning_synthesis", "Evidence-Grounded Scientist Answer", "基于证据的科学家回答"],
    "scientist_engineering_loop": ["scientist_engineering_loop", "Isolated Engineering Loop", "隔离工程执行闭环"],
    "workstation_action_log": ["runWorkstationAction", "metadata: { source:"],
    "report_export": ["exportReport", "exportMarkdown", "导出完整 Markdown"],
    "literature_rag": ["literature_rag", "文献与知识库", "/api/literature/search"],
    "settings_locale_theme": ["language_select", "settings_theme_change", "主题 / Theme"],
    "human_gate_boundary": ["Human Gate", "人工 Gate", "blocked_until_explicit_human_approval"],
}


LIVE_HTML_TERMS = ["EvoMind", "_next/static/css"]
SUMMARY_TERMS = ["Code Agent", "GPU", "Kaggle", "scientist_loop", "scientist_self_audit", "scientist_readiness_report", "scientist_causal_diagnosis", "scientist_strategy_optimizer", "scientist_reasoning_synthesis", "scientist_engineering_loop", "scientist_continuation_status"]
SCIENTIST_API_TERMS = ["no_training_started", "blocked_until_explicit_human_approval", "scientist_next_action"]
SCIENTIST_RECOVERY_API_TERMS = ["no_training_started", "blocked_until_explicit_human_approval", "scientist_recovery"]
SCIENTIST_LOOP_API_TERMS = ["no_training_started", "blocked_until_explicit_human_approval", "scientist_loop", "scientist_loop_lessons"]
SCIENTIST_CONTINUATION_API_TERMS = ["no_training_started", "blocked_until_explicit_human_approval", "scientist_continuation_status", "remaining_safe_tools"]
SCIENTIST_CONTINUATION_RESUME_API_TERMS = ["no_training_started", "blocked_until_explicit_human_approval", "scientist_continuation_resume", "remaining_safe_tools"]
SCIENTIST_SELF_AUDIT_API_TERMS = ["no_training_started", "blocked_until_explicit_human_approval", "scientist_self_audit", "overall_score"]
SCIENTIST_READINESS_REPORT_API_TERMS = ["no_training_started", "blocked_until_explicit_human_approval", "scientist_readiness_report", "claim_readiness", "rank_or_medal_claim"]
SCIENTIST_CAUSAL_DIAGNOSIS_API_TERMS = ["no_training_started", "blocked_until_explicit_human_approval", "scientist_causal_diagnosis", "causal_graph", "root_causes"]
SCIENTIST_STRATEGY_OPTIMIZER_API_TERMS = ["no_training_started", "blocked_until_explicit_human_approval", "scientist_strategy_optimizer", "intervention_ranking", "decision_matrix", "rank_or_medal"]
SCIENTIST_TURN_API_TERMS = ["no_training_started", "blocked_until_explicit_human_approval", "scientist_reasoning_synthesis", "reasoning_quality", "answer_markdown"]
SCIENTIST_ENGINEERING_API_TERMS = ["no_training_started", "blocked_until_explicit_human_approval", "scientist_engineering_loop", "main_worktree_modified", "merge_ready", "review_candidate_before_merge"]


def fail(message: str, evidence: dict) -> None:
    raise SystemExit(json.dumps({"status": "failed", "message": message, "evidence": evidence}, ensure_ascii=False, indent=2))


def read_sources() -> str:
    missing = [str(path.relative_to(ROOT)) for path in SOURCE_FILES if not path.exists()]
    if missing:
        fail("UI source files missing", {"missing_files": missing})
    return "\n".join(path.read_text(encoding="utf-8") for path in SOURCE_FILES)


def missing_terms(terms: Iterable[str], text: str) -> list[str]:
    return [term for term in terms if term not in text]


def missing_groups(groups: dict[str, list[str]], text: str) -> dict[str, list[str]]:
    return {name: terms for name, terms in groups.items() if not any(term in text for term in terms)}


def fetch(url: str, attempts: int = 3) -> str:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=20) as response:
                return response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code < 500 or attempt == attempts - 1:
                raise
        except urllib.error.URLError as error:
            last_error = error
            if attempt == attempts - 1:
                raise
        time.sleep(0.75 * (attempt + 1))
    if last_error:
        raise last_error
    raise RuntimeError(f"Failed to fetch {url}")


def origin_from_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        fail("dashboard URL must include scheme and host", {"url": url})
    return f"{parsed.scheme}://{parsed.netloc}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the current EvoMind UI/API localization contract: gateway labels, "
            "AI Scientist controls, report/literature/settings entries, and safe gates."
        )
    )
    parser.add_argument("--url", default=None, help="Optional running dashboard URL, for example http://127.0.0.1:8088/?page=control")
    args = parser.parse_args()

    source_text = read_sources()
    absent_source_terms = missing_terms(SOURCE_TERMS, source_text)
    absent_source_groups = missing_groups(SOURCE_GROUPS, source_text)
    if absent_source_terms or absent_source_groups:
        fail(
            "required EvoMind UI/API source contract terms are missing",
            {"missing_source_terms": absent_source_terms, "missing_source_groups": absent_source_groups},
        )

    live_result = None
    if args.url:
        origin = origin_from_url(args.url)
        html = fetch(args.url)
        absent_live_terms = missing_terms(LIVE_HTML_TERMS, html)
        if absent_live_terms:
            fail("running dashboard is missing required shell/CSS markers", {"url": args.url, "missing_live_terms": absent_live_terms})

        summary_url = f"{origin}/api/workstation-summary"
        summary_text = fetch(summary_url)
        absent_summary_terms = missing_terms(SUMMARY_TERMS, summary_text)
        if absent_summary_terms:
            fail("workstation summary is missing required integration terms", {"url": summary_url, "missing_summary_terms": absent_summary_terms})
        if "Not Configured" not in summary_text and "Ready" not in summary_text and "fully_ready" not in summary_text and "ready" not in summary_text:
            fail("workstation summary is missing an integration readiness state", {"url": summary_url})

        scientist_url = f"{origin}/api/scientist/next-action"
        scientist_text = fetch(scientist_url)
        absent_scientist_terms = missing_terms(SCIENTIST_API_TERMS, scientist_text)
        if absent_scientist_terms:
            fail("Scientist safe-next API is missing gate/safety markers", {"url": scientist_url, "missing_terms": absent_scientist_terms})

        recovery_url = f"{origin}/api/scientist/recovery"
        recovery_text = fetch(recovery_url)
        absent_recovery_terms = missing_terms(SCIENTIST_RECOVERY_API_TERMS, recovery_text)
        if absent_recovery_terms:
            fail("Scientist recovery API is missing gate/safety markers", {"url": recovery_url, "missing_terms": absent_recovery_terms})

        loop_url = f"{origin}/api/scientist/loop"
        loop_text = fetch(loop_url)
        absent_loop_terms = missing_terms(SCIENTIST_LOOP_API_TERMS, loop_text)
        if absent_loop_terms:
            fail("Scientist loop API is missing gate/safety markers", {"url": loop_url, "missing_terms": absent_loop_terms})

        continuation_url = f"{origin}/api/scientist/continuation-status"
        continuation_text = fetch(continuation_url)
        absent_continuation_terms = missing_terms(SCIENTIST_CONTINUATION_API_TERMS, continuation_text)
        if absent_continuation_terms:
            fail("Scientist continuation-status API is missing gate/safety markers", {"url": continuation_url, "missing_terms": absent_continuation_terms})

        continuation_resume_url = f"{origin}/api/scientist/continuation-resume"
        continuation_resume_text = fetch(continuation_resume_url)
        absent_continuation_resume_terms = missing_terms(SCIENTIST_CONTINUATION_RESUME_API_TERMS, continuation_resume_text)
        if absent_continuation_resume_terms:
            fail("Scientist continuation-resume API is missing gate/safety markers", {"url": continuation_resume_url, "missing_terms": absent_continuation_resume_terms})

        self_audit_url = f"{origin}/api/scientist/self-audit"
        self_audit_text = fetch(self_audit_url)
        absent_self_audit_terms = missing_terms(SCIENTIST_SELF_AUDIT_API_TERMS, self_audit_text)
        if absent_self_audit_terms:
            fail("Scientist self-audit API is missing gate/safety markers", {"url": self_audit_url, "missing_terms": absent_self_audit_terms})

        readiness_report_url = f"{origin}/api/scientist/readiness-report"
        readiness_report_text = fetch(readiness_report_url)
        absent_readiness_report_terms = missing_terms(SCIENTIST_READINESS_REPORT_API_TERMS, readiness_report_text)
        if absent_readiness_report_terms:
            fail("Scientist readiness-report API is missing gate/safety markers", {"url": readiness_report_url, "missing_terms": absent_readiness_report_terms})

        causal_diagnosis_url = f"{origin}/api/scientist/causal-diagnosis"
        causal_diagnosis_text = fetch(causal_diagnosis_url)
        absent_causal_diagnosis_terms = missing_terms(SCIENTIST_CAUSAL_DIAGNOSIS_API_TERMS, causal_diagnosis_text)
        if absent_causal_diagnosis_terms:
            fail("Scientist causal-diagnosis API is missing gate/safety markers", {"url": causal_diagnosis_url, "missing_terms": absent_causal_diagnosis_terms})

        strategy_optimizer_url = f"{origin}/api/scientist/strategy-optimizer"
        strategy_optimizer_text = fetch(strategy_optimizer_url)
        absent_strategy_optimizer_terms = missing_terms(SCIENTIST_STRATEGY_OPTIMIZER_API_TERMS, strategy_optimizer_text)
        if absent_strategy_optimizer_terms:
            fail("Scientist strategy-optimizer API is missing gate/safety markers", {"url": strategy_optimizer_url, "missing_terms": absent_strategy_optimizer_terms})

        scientist_turn_url = f"{origin}/api/scientist/turn"
        scientist_turn_text = fetch(scientist_turn_url)
        absent_scientist_turn_terms = missing_terms(SCIENTIST_TURN_API_TERMS, scientist_turn_text)
        if absent_scientist_turn_terms:
            fail("Scientist turn API is missing reasoning synthesis markers", {"url": scientist_turn_url, "missing_terms": absent_scientist_turn_terms})

        engineering_url = f"{origin}/api/scientist/engineering-loop"
        engineering_text = fetch(engineering_url)
        absent_engineering_terms = missing_terms(SCIENTIST_ENGINEERING_API_TERMS, engineering_text)
        if absent_engineering_terms:
            fail("Scientist engineering-loop API is missing isolation/gate markers", {"url": engineering_url, "missing_terms": absent_engineering_terms})

        live_result = {
            "url": args.url,
            "origin": origin,
            "html_terms": LIVE_HTML_TERMS,
            "summary_terms": SUMMARY_TERMS,
            "scientist_api_terms": SCIENTIST_API_TERMS,
            "scientist_recovery_api_terms": SCIENTIST_RECOVERY_API_TERMS,
            "scientist_loop_api_terms": SCIENTIST_LOOP_API_TERMS,
            "scientist_continuation_api_terms": SCIENTIST_CONTINUATION_API_TERMS,
            "scientist_continuation_resume_api_terms": SCIENTIST_CONTINUATION_RESUME_API_TERMS,
            "scientist_self_audit_api_terms": SCIENTIST_SELF_AUDIT_API_TERMS,
            "scientist_readiness_report_api_terms": SCIENTIST_READINESS_REPORT_API_TERMS,
            "scientist_causal_diagnosis_api_terms": SCIENTIST_CAUSAL_DIAGNOSIS_API_TERMS,
            "scientist_strategy_optimizer_api_terms": SCIENTIST_STRATEGY_OPTIMIZER_API_TERMS,
            "scientist_turn_api_terms": SCIENTIST_TURN_API_TERMS,
            "scientist_engineering_api_terms": SCIENTIST_ENGINEERING_API_TERMS,
        }

    print(json.dumps({
        "status": "passed",
        "source_terms": SOURCE_TERMS,
        "source_groups": SOURCE_GROUPS,
        "live": live_result,
        "conclusion": (
            "EvoMind UI/API localization contract is current: AI Scientist controls, report export, "
            "literature/RAG, settings language/theme, integration readiness, and Human Gate boundaries are protected."
        ),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
