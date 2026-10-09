"""Release-local scheduler ownership, reviewed-source parity, and pause evidence.

The runtime snapshot is derived only from the active scheduler plus the reviewed
cooperative-pause patch in endurance-payload-v8, not the dirty research_os tree.
Raw source SHA256: b7d21314a279a6e90f0b602727981f2fa15a20aae4f9e62d5d3baa27655f5597.
Only line endings are normalized in the runtime-owned copy.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from evomind_runtime import managed_scheduler as scheduler
from evomind_runtime.run_control import RunPaused


REVIEWED_SOURCE_LF_SHA256 = "d54d6605801da63896ef5961e83b7f3fa64e002594bcd07d00cc14bbe2890956"


def test_managed_scheduler_matches_reviewed_snapshot():
    source = Path(scheduler.__file__).read_text(encoding="utf-8")
    assert hashlib.sha256(source.encode("utf-8")).hexdigest() == REVIEWED_SOURCE_LF_SHA256


def test_scheduler_has_only_explicit_standard_library_dependencies():
    source = Path(scheduler.__file__).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] in sys.stdlib_module_names for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            assert node.module.split(".")[0] in sys.stdlib_module_names
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id != "__import__"


def test_scheduler_imports_and_pauses_without_shared_research_tree(tmp_path):
    # A fresh isolated interpreter gets just the two managed files and a package
    # marker. The real runtime's other research_os dependencies are not replaced.
    package = tmp_path / "evomind_runtime"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    shutil.copyfile(scheduler.__file__, package / "managed_scheduler.py")
    shutil.copyfile(Path(scheduler.__file__).with_name("run_control.py"), package / "run_control.py")
    probe = r'''
import importlib.abc
import json
import sys
from pathlib import Path

class RejectSharedTree(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "research_os" or fullname.startswith("research_os."):
            raise AssertionError("managed scheduler attempted shared-tree import")
        return None

sys.meta_path.insert(0, RejectSharedTree())
sys.path.insert(0, sys.argv[1])
from evomind_runtime import managed_scheduler as m
from evomind_runtime.run_control import RunPaused

def pause(*args):
    raise RunPaused()

run = m.create_run(run_id="isolated", objective="offline package check",
    tasks=[m.AgentTask("task", "pause before work", "worker")],
    roles=[m.AgentRoleSpec("worker", ("fixture",))])
store = m.MultiAgentStore(Path(sys.argv[1]) / "run")
result = m.MultiAgentSupervisor(run, store, {"worker": pause}).run_until_blocked()
assert result.status == "paused"
assert result.tasks["task"].status == "ready"
assert not result.idempotency_results
assert not any(name == "research_os" or name.startswith("research_os.") for name in sys.modules)
print(json.dumps({"status": result.status, "module": m.__name__, "shared_tree_imported": False}))
'''
    result = subprocess.run(
        [sys.executable, "-I", "-c", probe, str(tmp_path)],
        cwd=tmp_path, capture_output=True, text=True, timeout=15, check=True,
    )
    assert json.loads(result.stdout) == {
        "status": "paused", "module": "evomind_runtime.managed_scheduler", "shared_tree_imported": False,
    }


def test_aibuild_uses_managed_scheduler_types():
    from evomind_runtime import aibuild_engine

    for name in ("AgentResult", "AgentRoleSpec", "AgentTask", "MultiAgentStore", "MultiAgentSupervisor", "create_run"):
        assert getattr(aibuild_engine, name) is getattr(scheduler, name)
    imports = [node for node in ast.walk(ast.parse(Path(aibuild_engine.__file__).read_text(encoding="utf-8")))
               if isinstance(node, ast.ImportFrom)]
    assert not any(node.module == "research_os.agent.multi_agent" for node in imports)
    # Migration is intentionally scheduler-only: provider/search/claim adapters
    # still resolve through the pinned release's explicit research_os modules.
    assert {node.module for node in imports if node.module.startswith("research_os.")} == {
        "research_os.agent.messaging", "research_os.mcgs_selector", "research_os.search_graph", "research_os.claim_audit",
    }


def test_model_recovery_uses_the_same_managed_scheduler():
    from evomind_runtime import aibuild_model_recovery

    assert aibuild_model_recovery.MultiAgentStore is scheduler.MultiAgentStore
    assert aibuild_model_recovery.MultiAgentSupervisor is scheduler.MultiAgentSupervisor


def test_reviewed_shared_v1_state_remains_compatible_without_rewrite(tmp_path):
    # Data compatibility does not require importing the current dirty shared
    # module. This is the existing scheduler's durable v1 wire format.
    fixture = {
        "schema": "evomind.multi_agent.run.v1", "run_id": "legacy-fixture",
        "objective": "legacy fixture", "status": "needs_continuation", "seq": 12,
        "max_concurrency": 1, "created_at": "2026-09-07T00:00:00+00:00",
        "updated_at": "2026-09-07T00:01:00+00:00",
        "roles": {"SetupAgent": {"role": "SetupAgent", "capabilities": ["fixture"]}},
        "tasks": {"setup": {
            "task_id": "setup", "goal": "prepare", "role": "SetupAgent",
            "status": "failed", "attempts": 1, "max_retries": 0,
            "idempotency_key": "legacy-fixed-key", "error": "ModelTransportError: http_503",
        }},
        "idempotency_results": {},
    }
    path = tmp_path / "run.json"
    before = json.dumps(fixture, sort_keys=True).encode("utf-8")
    path.write_bytes(before)
    store = scheduler.MultiAgentStore(tmp_path)
    run = store.load()
    assert path.read_bytes() == before
    assert run.run_id == "legacy-fixture" and run.seq == 12
    assert run.tasks["setup"].idempotency_key == "legacy-fixed-key"
    assert run.tasks["setup"].attempts == 1
    assert run.tasks["setup"].status == "failed"


def _graph(run_id):
    return scheduler.create_run(
        run_id=run_id, objective="offline pause evidence",
        tasks=[
            scheduler.AgentTask("prepare", "write once", "worker"),
            scheduler.AgentTask("work", "cooperative boundary", "worker", dependencies=("prepare",)),
            scheduler.AgentTask("finish", "finish after resume", "worker", dependencies=("work",)),
        ],
        roles=[scheduler.AgentRoleSpec("worker", ("fixture",))], max_concurrency=1,
    )


def test_pause_reload_resume_keeps_durable_completed_result(tmp_path):
    store = scheduler.MultiAgentStore(tmp_path / "run")
    completed = []
    pause_requested = True
    marker = tmp_path / "prepared-once.txt"

    def execute(task, _handoff, _run):
        if task.task_id == "prepare":
            with marker.open("x", encoding="utf-8") as handle:
                handle.write("fixture side effect must execute once")
        if task.task_id == "work" and pause_requested:
            raise RunPaused()
        completed.append(task.task_id)
        return scheduler.AgentResult(task.task_id, "fixture complete")

    result = scheduler.MultiAgentSupervisor(_graph("reload"), store, {"worker": execute}).run_until_blocked()
    assert result.status == "paused"
    assert {name: task.status for name, task in result.tasks.items()} == {
        "prepare": "completed", "work": "ready", "finish": "pending",
    }
    assert completed == ["prepare"]
    assert len(result.idempotency_results) == 1
    assert not (store.run_dir / "failure").exists()
    first_result = store.run_dir / result.tasks["prepare"].result_ref
    first_bytes, first_mtime = first_result.read_bytes(), first_result.stat().st_mtime_ns
    marker_mtime = marker.stat().st_mtime_ns

    # Reopen serialized v1 state rather than reuse Python task/result identities.
    reopened_store = scheduler.MultiAgentStore(store.run_dir)
    reopened = reopened_store.load()
    assert reopened.to_dict()["schema"] == "evomind.multi_agent.run.v1"
    pause_requested = False
    supervisor = scheduler.MultiAgentSupervisor(reopened, reopened_store, {"worker": execute})
    supervisor.resume()
    assert supervisor.run_until_blocked().status == "completed"
    assert completed == ["prepare", "work", "finish"]
    assert first_result.read_bytes() == first_bytes
    assert first_result.stat().st_mtime_ns == first_mtime
    assert marker.stat().st_mtime_ns == marker_mtime
    assert len(reopened.idempotency_results) == 3
    events = [json.loads(line) for line in (store.run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    paused_event = next(index for index, event in enumerate(events) if event["schema"] == "evomind.multi_agent.run.paused.v1")
    assert not any(event["status"] == "completed" and event["task_id"] in {"work", "finish"}
                   for event in events[:paused_event + 1])
    assert [event["seq"] for event in events] == list(range(1, len(events) + 1))


def test_pause_control_finishes_active_node_before_stopping_dispatch(tmp_path):
    store = scheduler.MultiAgentStore(tmp_path)
    calls = []

    def execute(task, _handoff, _run):
        calls.append(task.task_id)
        if task.task_id == "prepare":
            (tmp_path / "control.json").write_text(json.dumps({"action": "pause"}), encoding="utf-8")
        return scheduler.AgentResult(task.task_id, "safe boundary reached")

    graph = scheduler.MultiAgentSupervisor(_graph("control"), store, {"worker": execute}).run_until_blocked()
    assert graph.status == "paused"
    assert calls == ["prepare"]
    assert graph.tasks["prepare"].status == "completed"
    assert graph.tasks["work"].status == "pending"
    assert graph.tasks["finish"].status == "pending"
    assert store.load().status == "paused"
    assert (tmp_path / graph.tasks["prepare"].result_ref).is_file()
