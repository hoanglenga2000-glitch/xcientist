"""Explicit, bounded recovery of failed research model calls before side effects."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .managed_scheduler import MultiAgentStore, MultiAgentSupervisor

from .execution_progress import resource_lease


SAFE_HISTORY = frozenset({"file_list", "file_search", "file_read", "directory_list", "directory_read", "directory_stat", "directory_hash", "attachment_list", "attachment_read", "artifact_list", "artifact_preview", "memory_search", "capability_discover", "runtime_health", "model_protocols", "training_route"})


def reconcile_model_failure(runtime, run_id: str, instruction: str, request_key: str = "") -> dict | None:
    session = runtime.store.get_session(run_id)
    if not session or (session.get("metadata") or {}).get("research_orchestrator") != "aibuildai2":
        return None
    root = runtime.runtime_root / "aibuild_runs" / run_id
    if not (root / "run.json").is_file():
        return None
    with resource_lease(runtime.runtime_root / "resource_leases", f"aibuild:{run_id}"):
        store = MultiAgentStore(root)
        graph = store.load()
        if graph.status not in {"failed", "needs_continuation"}:
            return None
        if any(task.status == "running" for task in graph.tasks.values()) or any(root.glob("*.pending.json")):
            raise ValueError("research_worker_or_approval_requires_settlement")
        failed = [task for task in graph.tasks.values() if task.status == "failed"]
        if len(failed) != 1 or not failed[0].error.startswith(("LLMError:", "ModelTransportError:")):
            return None
        if failed[0].role != "SetupAgent":
            raise ValueError("research_model_retry_requires_role_review")
        calls = runtime.store.list_tool_calls(run_id)
        if any(call["tool_name"] not in SAFE_HISTORY or call["status"] not in {"completed", "failed"} for call in calls):
            raise ValueError("research_model_retry_requires_tool_settlement")
        budget_path = root / "budget.json"
        budget_raw = budget_path.read_bytes() if budget_path.is_file() else b"{}"
        budget = json.loads(budget_raw)
        if int(budget.get("calls", 0)) >= 48 or int(budget.get("tokens", 0)) >= 200_000:
            raise ValueError("research_budget_exhausted")
        graph_path = root / "run.json"
        before = graph_path.read_bytes()
        fingerprint = hashlib.sha256(json.dumps({"run": run_id, "task": failed[0].task_id,
                                                "graph_sha256": hashlib.sha256(before).hexdigest(),
                                                "instruction": instruction}, sort_keys=True).encode()).hexdigest()
        identifier = hashlib.sha256(request_key.encode()).hexdigest() if request_key else fingerprint
        # Keep the Windows journal path short; retain and compare full digests
        # inside the receipt so a shortened directory name cannot alias a request.
        journal = root / "model-reconciliations" / identifier[:24]
        if journal.exists():
            receipt = json.loads((journal / "receipt.json").read_text())
            if receipt["fingerprint"] != fingerprint or receipt["request_id"] != identifier:
                raise ValueError("research_resume_request_conflict")
            return {**receipt, "replayed": True}
        journal.mkdir(parents=True, exist_ok=False)
        (journal / "before-run.json").write_bytes(before)
        (journal / "before-budget.json").write_bytes(budget_raw)
        attempts = {task.task_id: task.attempts for task in graph.tasks.values()}
        supervisor = MultiAgentSupervisor(graph, store, {})
        supervisor.resume(retry_failed=True)
        for task_id, count in attempts.items():
            graph.tasks[task_id].attempts = count
        task = graph.tasks[failed[0].task_id]
        task.payload["user_continuation"] = instruction
        task.payload["continuation_request_id"] = identifier
        store.save(graph)
        if budget_path.is_file() and budget_path.read_bytes() != budget_raw:
            raise RuntimeError("research_recovery_changed_budget")
        receipt = {"schema": "evomind.research_model_reconciliation.v1", "run_id": run_id,
                   "task_id": task.task_id, "fingerprint": fingerprint, "request_id": identifier,
                   "before_graph_sha256": hashlib.sha256(before).hexdigest(),
                   "after_graph_sha256": hashlib.sha256(graph_path.read_bytes()).hexdigest(),
                   "budget_sha256": hashlib.sha256(budget_raw).hexdigest(), "budget_reset": False,
                   "previous_attempts": attempts[task.task_id], "training_started": False}
        (journal / "receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2), encoding="utf-8")
        runtime.store.append_event(run_id, "research_model_reconciled", receipt)
        return receipt
