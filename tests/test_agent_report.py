"""Report + ledger tests: graph-backed report, crash-survivable dialogue.

Prove: the report reflects the search graph (best/promotions/failures) and never
invents; the honest-scope boundary is always present; the ledger appends
incrementally, tolerates a truncated trailing line, and rewrites on compaction;
and a finished session auto-writes research_report.md + messages.jsonl.
"""
from __future__ import annotations

import sys

from conftest import LocalSubprocessRunner

from research_os.agent.ledger import MessageLedger
from research_os.agent.report import build_report, write_report
from research_os.agent.session import AgentSession
from research_os.agent.tools import ResearchToolbox
from research_os.variation_generator import TaskContext


def _ctx():
    return TaskContext(task_name="titanic", modality="tabular", task_type="classification",
                       metric="accuracy", metric_direction="maximize")


# ── ledger ──────────────────────────────────────────────────────────────────────
def test_ledger_appends_and_loads(tmp_path):
    led = MessageLedger(tmp_path / "messages.jsonl")
    led.append({"role": "user", "content": "hi"})
    led.append({"role": "assistant", "content": [{"type": "text", "text": "yo"}]})
    loaded = led.load()
    assert len(loaded) == 2
    assert loaded[0]["content"] == "hi"


def test_ledger_tolerates_truncated_trailing_line(tmp_path):
    p = tmp_path / "messages.jsonl"
    p.write_text('{"role": "user", "content": "ok"}\n{"role": "assist', encoding="utf-8")
    assert MessageLedger(p).load() == [{"role": "user", "content": "ok"}]


def test_ledger_rewrite_replaces_all(tmp_path):
    led = MessageLedger(tmp_path / "m.jsonl")
    led.append({"role": "user", "content": "a"})
    led.append({"role": "user", "content": "b"})
    led.rewrite([{"role": "user", "content": "compacted"}])
    assert led.load() == [{"role": "user", "content": "compacted"}]


# ── report from a hand-built run dir ─────────────────────────────────────────────
def _write_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "summary.json").write_text(
        '{"task":"titanic","metric":"accuracy","metric_direction":"maximize",'
        '"best_exp_id":"EXP001","n_iterations":2,"n_promotions":1,'
        '"finished_by_agent":true,"agent_summary":"baseline then improved"}', encoding="utf-8")
    (run / "search_graph.json").write_text(
        '{"task_id":"titanic","metric_name":"accuracy","metric_direction":"maximize",'
        '"best_exp_id":"EXP001","nodes":['
        '{"exp_id":"EXP000","parent_id":null,"branch_type":"Base","cv_score":0.80,'
        '"run_success":true,"promoted":true,"decision":"promote","hypothesis":"gbm baseline"},'
        '{"exp_id":"EXP001","parent_id":"EXP000","branch_type":"Stepwise","cv_score":0.83,'
        '"run_success":true,"promoted":true,"decision":"promote","hypothesis":"add features"}],'
        '"promotion_history":[{"candidate_exp_id":"EXP001","promoted":true,"reason":"improves best-so-far",'
        '"candidate_score":0.83,"parent_score":0.80,"promotion_delta":0.03}]}', encoding="utf-8")
    return run


def test_report_reflects_graph(tmp_path):
    run = _write_run(tmp_path)
    md = build_report(run)
    assert "研究报告 — titanic" in md
    assert "EXP001" in md and "0.830000" in md          # best, real score
    assert "improves best-so-far" in md                  # promotion reason
    assert "add features" in md                          # hypothesis
    assert "baseline then improved" in md                # agent summary
    # honest-scope boundary always present
    assert "Human Gate" in md
    assert "本地 CV / proxy" in md


def test_report_lists_failures_with_real_error(tmp_path):
    run = tmp_path / "run"
    (run / "EXP000").mkdir(parents=True)
    (run / "summary.json").write_text('{"task":"t","best_exp_id":null,"n_iterations":1,"n_promotions":0}',
                                      encoding="utf-8")
    (run / "search_graph.json").write_text(
        '{"nodes":[{"exp_id":"EXP000","parent_id":null,"branch_type":"Base","cv_score":null,'
        '"run_success":false,"promoted":false,"decision":"hold","hypothesis":"risky idea"}]}',
        encoding="utf-8")
    (run / "EXP000" / "run_error.txt").write_text("Traceback...\nValueError: bad shape", encoding="utf-8")
    md = build_report(run)
    assert "失败与归因" in md
    assert "risky idea" in md
    assert "ValueError: bad shape" in md
    assert "尚无被晋升的最优解" in md


def test_write_report_creates_file(tmp_path):
    run = _write_run(tmp_path)
    out = write_report(run)
    assert out.exists() and out.name == "research_report.md"


# ── session auto-writes report + ledger ──────────────────────────────────────────
_GOOD = """
import argparse, json, csv, os
p = argparse.ArgumentParser(); p.add_argument("--data-dir"); p.add_argument("--out-dir")
a = p.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
print("CV_SCORE=0.81")
json.dump({"cv_score":0.81,"metric":"accuracy"}, open(os.path.join(a.out_dir,"metrics.json"),"w"))
csv.writer(open(os.path.join(a.out_dir,"submission.csv"),"w",newline="")).writerow(["id","t"])
"""


def test_session_autowrites_report_and_ledger(tmp_path):
    from research_os.agent.messaging import AssistantTurn, ToolCall

    class _Client:
        def __init__(self):
            self._turns = [
                ("plan", [ToolCall("t1", "plan_next_experiment", {})]),
                ("run", [ToolCall("t2", "run_experiment", {"hypothesis": "h", "code": _GOOD})]),
                ("promote", [ToolCall("t3", "evaluate_promotion", {"exp_id": "EXP000"})]),
                ("done", [ToolCall("t4", "finish", {"summary": "baseline cv=0.81"})]),
            ]
            self.i = 0
        def is_available(self):
            return True
        def send(self, messages, *, system, tools, max_tokens=0, temperature=0.0):
            text, calls = self._turns[self.i]
            self.i += 1
            return AssistantTurn(text=text, tool_calls=calls, stop_reason="tool_use",
                                 raw_content=[{"type": "text", "text": text}], model="m")

    exp_dir = tmp_path / "exp"
    tb = ResearchToolbox(_ctx(), data_dir=str(tmp_path / "d"), work_dir=exp_dir,
                         runner=LocalSubprocessRunner(exp_dir / "runs", timeout=120, python_exe=sys.executable))
    session = AgentSession(context=_ctx(), toolbox=tb, exp_dir=exp_dir, client=_Client())
    session.run("baseline")
    assert (exp_dir / "research_report.md").exists()
    assert (exp_dir / "messages.jsonl").exists()
    report = (exp_dir / "research_report.md").read_text(encoding="utf-8")
    assert "EXP000" in report and "Human Gate" in report
    # ledger captured the seed + turns
    assert len(MessageLedger(exp_dir / "messages.jsonl").load()) >= 4
