"""Layered persistent memory tests (plan's 'experience reuse' thesis).

Prove: lessons persist across runs at the project level; the compact index digest
aggregates correctly for the opening prompt; on-demand retrieval filters by task
type and failure pattern; and record_lesson writes through to the shared store.
"""
from __future__ import annotations

from pathlib import Path

from research_os.agent.memory_library import MemoryLibrary
from research_os.agent.tools import ResearchToolbox
from research_os.evolution_loop import RunResult
from research_os.retrospective_memory import MemoryRecord, RetrospectiveMemoryStore
from research_os.variation_generator import TaskContext


def _rec(mid, task_type, *, worked="", failed="", strategy="", pattern="") -> MemoryRecord:
    return MemoryRecord(
        memory_id=mid, task_type=task_type, dataset_profile={}, method="agent",
        what_worked=worked, what_failed=failed, metric_delta=None,
        reusable_strategy=strategy, failure_pattern=pattern, linked_exp_ids=[mid],
    )


def _ctx(task_type="classification") -> TaskContext:
    return TaskContext(task_name="t", modality="tabular", task_type=task_type,
                       metric="accuracy", metric_direction="maximize")


def test_index_digest_empty(tmp_path):
    lib = MemoryLibrary(RetrospectiveMemoryStore(tmp_path / "m.json"))
    assert "empty" in lib.index_digest("classification")


def test_index_digest_aggregates(tmp_path):
    store = RetrospectiveMemoryStore(tmp_path / "m.json")
    lib = MemoryLibrary(store)
    lib.add(_rec("a", "classification", strategy="oof_stacking"))
    lib.add(_rec("b", "classification", strategy="oof_stacking"))
    lib.add(_rec("c", "classification", failed="OOM", pattern="oom"))
    lib.add(_rec("d", "regression", strategy="log1p_target"))
    digest = lib.index_digest("classification")
    assert "4 lessons" in digest
    assert "oof_stacking×2" in digest       # top strategy counted
    assert "oom×1" in digest                 # failure pattern counted
    assert "task_type=classification: 3 lessons" in digest


def test_retrieve_filters_by_task_type_and_pattern(tmp_path):
    store = RetrospectiveMemoryStore(tmp_path / "m.json")
    lib = MemoryLibrary(store)
    lib.add(_rec("a", "classification", pattern="oom"))
    lib.add(_rec("b", "classification", pattern="timeout"))
    lib.add(_rec("c", "regression", pattern="oom"))
    # task_type filter
    cls = lib.retrieve("classification")
    assert {r["memory_id"] for r in cls} == {"a", "b"}
    # cross-task (task_type=None)
    allt = lib.retrieve(None)
    assert len(allt) == 3
    # failure-pattern filter
    ooms = lib.retrieve(None, failure_pattern="oom")
    assert {r["memory_id"] for r in ooms} == {"a", "c"}


def test_memory_persists_across_toolbox_instances(tmp_path):
    """A lesson recorded by one agent session is visible to a later session that
    points at the same project-level store — cross-run experience persistence."""
    mem_path = tmp_path / "experiments" / "evolution" / "retrospective_memory.json"
    store1 = RetrospectiveMemoryStore(mem_path)

    class _Runner:
        def run(self, code, *, data_dir, out_dir, exp_id):
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            return RunResult(success=True, cv_score=0.9, exit_code=0,
                             artifacts=["/x/metrics.json", "/x/submission.csv"])

    tb1 = ResearchToolbox(_ctx(), data_dir=str(tmp_path / "d"),
                          work_dir=tmp_path / "exp1", runner=_Runner(), memory=store1)
    tb1.dispatch("plan_next_experiment", {})
    tb1.dispatch("run_experiment", {"hypothesis": "h", "code": "print(1)"})
    tb1.dispatch("record_lesson", {"exp_id": "EXP000", "what_worked": "GBM baseline",
                                   "reusable_strategy": "gbm"})

    # a fresh toolbox (new session) on the SAME store sees the lesson
    store2 = RetrospectiveMemoryStore(mem_path)
    tb2 = ResearchToolbox(_ctx(), data_dir=str(tmp_path / "d"),
                          work_dir=tmp_path / "exp2", runner=_Runner(), memory=store2)
    out = tb2.dispatch("read_memory", {})
    assert "gbm" in out.content
    assert "GBM baseline" in out.content
    # and the opening digest reflects it
    assert "1 lessons" in tb2.library.index_digest("classification")


def test_read_memory_all_tasks_and_pattern_filter_via_tool(tmp_path):
    store = RetrospectiveMemoryStore(tmp_path / "m.json")
    store.add_memory(_rec("rec_cls", "classification", pattern="oom"))
    store.add_memory(_rec("rec_reg", "regression", pattern="timeout"))

    class _Runner:
        def run(self, *a, **k):
            return RunResult(success=True, cv_score=0.9, exit_code=0)

    tb = ResearchToolbox(_ctx("classification"), data_dir=str(tmp_path / "d"),
                         work_dir=tmp_path / "exp", runner=_Runner(), memory=store)
    # default: only this task_type
    assert "rec_reg" not in tb.dispatch("read_memory", {}).content
    # all_tasks: sees the regression lesson too
    assert "rec_reg" in tb.dispatch("read_memory", {"all_tasks": True}).content
    # pattern filter across tasks
    only_timeout = tb.dispatch("read_memory", {"all_tasks": True, "failure_pattern": "timeout"})
    assert "rec_reg" in only_timeout.content and "rec_cls" not in only_timeout.content
