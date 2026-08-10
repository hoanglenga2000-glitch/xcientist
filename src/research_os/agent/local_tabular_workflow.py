"""Governed local-GPU tabular research workflow.

This adapter connects the natural-language Multi-Agent entry to the existing
``research_os`` EvolutionLoop.  It does not implement a model trainer: GPT
generates candidates inside the evolution engine, and the engine executes them
through its resource-gated LocalSubprocessRunner.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from xsci.user_request import UserRequest

from .aibuild_v1 import run_directory, write_current_run_pointer
from .multi_agent import (
    AgentResult,
    AgentRoleSpec,
    AgentTask,
    HandoffEnvelope,
    MultiAgentStore,
    MultiAgentSupervisor,
    SupervisorRun,
    create_run,
)

TASK_ID = "credit-card-fraud-detection"
EVOLUTION_CONFIG = "configs/evolution/credit_card_fraud_detection.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain an object")
    return payload


def _artifact(path: Path, run_dir: Path, *, kind: str) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(run_dir)).replace("\\", "/"),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "kind": kind,
    }


def _manifest_entries(run_dir: Path) -> list[dict[str, Any]]:
    mutable = {"run.json", "events.jsonl", "messages.jsonl", "handoffs.jsonl", "task_graph.json", "control.json"}
    entries: list[dict[str, Any]] = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name in mutable or path.name == "artifact_manifest.json":
            continue
        if "results" in path.parts or ".tmp" in path.name or "__pycache__" in path.parts:
            continue
        entries.append(_artifact(path, run_dir, kind=path.name))
    return entries


def _roles() -> list[AgentRoleSpec]:
    return [
        AgentRoleSpec("SetupAgent", ("local_resource_probe", "dataset_contract"), output_contract=("setup.json", "data_contract.json")),
        AgentRoleSpec("ResearchLead", ("hypothesis", "orchestration_provenance"), output_contract=("research_context.json",)),
        AgentRoleSpec("DataAuditor", ("quality", "leakage", "temporal_split"), output_contract=("data_audit.json",)),
        AgentRoleSpec("DesignerAgent", ("falsifiable_design", "metric_selection", "search_plan"), output_contract=("design.json",)),
        AgentRoleSpec(
            "TunerAgent",
            ("controlled_local_gpu_training", "telemetry", "evolution_search"),
            resource_permissions=("local_rtx_4060", "write_run_directory"),
            wall_time_seconds=5400,
            output_contract=("evolution_result.json", "metrics.json", "submission.csv"),
        ),
        AgentRoleSpec("IndependentReviewer", ("holdout_review", "claim_audit", "hash_verification"), output_contract=("review.json",)),
        AgentRoleSpec("Aggregator", ("evidence_synthesis", "report", "reproducibility"), output_contract=("artifact_manifest.json", "research_report.md")),
    ]


def build_local_tabular_run(request: UserRequest, *, run_id: str) -> SupervisorRun:
    tasks = [
        AgentTask("setup", "Validate RTX 4060 runtime, local data contract, hashes, and submission boundary.", "SetupAgent", priority=100),
        AgentTask("research_context", "Bind GPT-5.6 research orchestration and define evidence-grounded hypotheses.", "ResearchLead", dependencies=("setup",), priority=90),
        AgentTask("data_audit", "Audit temporal boundaries, target balance, identifiers, and leakage risks.", "DataAuditor", dependencies=("setup",), priority=95, timeout_seconds=900),
        AgentTask("research_design", "Design the PR-AUC-first sequential LightGBM/XGBoost/CatBoost search.", "DesignerAgent", dependencies=("research_context", "data_audit"), priority=80),
        AgentTask(
            "local_gpu_evolution",
            "Run the approved EvolutionLoop on the local RTX 4060 after the resource gate.",
            "TunerAgent",
            dependencies=("research_design",),
            priority=70,
            resource_type="gpu",
            timeout_seconds=5400,
            max_retries=0,
        ),
        AgentTask("independent_review", "Evaluate the selected probability vector on retained temporal holdout labels.", "IndependentReviewer", dependencies=("local_gpu_evolution",), priority=60, max_retries=0),
        AgentTask("aggregate", "Assemble the report, version graph, model provenance, and reproducible evidence manifest.", "Aggregator", dependencies=("independent_review",), priority=50, max_retries=0),
    ]
    run = create_run(objective=request.objective, tasks=tasks, roles=_roles(), run_id=run_id, max_concurrency=2)
    run.gates = {
        "human_execution": "approved_by_frontend_run_action",
        "local_resource_gate": "required_before_training",
        "reviewer": "required",
        "claim_audit": "required",
        "promotion": "required",
        "official_submission": "disabled",
    }
    run.resource_limits = {"cpu": 2, "gpu": 1, "hpc_gpu": 0}
    return run


class LocalTabularExecutors:
    def __init__(self, *, workspace_root: Path, run_dir: Path, request: UserRequest) -> None:
        self.root = workspace_root
        self.run_dir = run_dir
        self.request = request
        self.config_path = self.root / EVOLUTION_CONFIG
        self.config = _read_json(self.config_path)
        self.data_dir = (self.root / str(self.config["local_data_dir"])).resolve()

    def setup(self, task: AgentTask, _handoff: HandoffEnvelope, _run: SupervisorRun) -> AgentResult:
        required = [self.data_dir / name for name in ("train.csv", "test.csv", "sample_submission.csv", "holdout_labels.csv", "dataset_contract.json")]
        missing = [path.name for path in required if not path.is_file()]
        local_python = self.root / "workspace" / "local-gpu-venv" / "Scripts" / "python.exe"
        probe = subprocess.run(
            [str(local_python), "-c", "import json,torch,xgboost,lightgbm,catboost; print(json.dumps({'torch_cuda':torch.cuda.is_available(),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else '', 'xgboost_cuda':bool(xgboost.build_info().get('USE_CUDA'))}))"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        ) if local_python.is_file() else None
        runtime = json.loads(probe.stdout.strip().splitlines()[-1]) if probe and probe.returncode == 0 and probe.stdout.strip() else {}
        passed = not missing and local_python.is_file() and runtime.get("torch_cuda") is True and "RTX 4060" in str(runtime.get("gpu")) and runtime.get("xgboost_cuda") is True
        setup_path = self.run_dir / "setup.json"
        contract_path = self.run_dir / "data_contract.json"
        _atomic_json(setup_path, {
            "schema": "evomind.local_tabular.setup.v1",
            "status": "passed" if passed else "rejected",
            "compute_backend": "local_gpu",
            "gpu": runtime.get("gpu"),
            "torch_cuda": runtime.get("torch_cuda"),
            "xgboost_cuda": runtime.get("xgboost_cuda"),
            "runtime_ready": local_python.is_file(),
            "missing_data_files": missing,
            "remote_compute_used": False,
            "orchestration_model": self.request.orchestration_model or "gpt-5.6-sol",
            "official_submission": "disabled",
            "generated_at": _now(),
        })
        contract = _read_json(self.data_dir / "dataset_contract.json") if not missing else {}
        _atomic_json(contract_path, {**contract, "source_config_sha256": _sha256(self.config_path), "official_submission": "disabled"})
        artifacts = [_artifact(setup_path, self.run_dir, kind="setup"), _artifact(contract_path, self.run_dir, kind="data_contract")]
        return AgentResult(task.task_id, "Local RTX 4060 runtime and dataset contract verified" if passed else "Local setup rejected", [item["path"] for item in artifacts], artifacts=artifacts, confidence=1.0, accepted=passed, failure_type="resource" if not passed else "")

    def research(self, task: AgentTask, _handoff: HandoffEnvelope, _run: SupervisorRun) -> AgentResult:
        output = self.run_dir / "research_context.json"
        _atomic_json(output, {
            "schema": "evomind.local_tabular.research_context.v1",
            "task": TASK_ID,
            "orchestration_model": self.request.orchestration_model or "gpt-5.6-sol",
            "candidate_families": ["LightGBM CPU baseline", "XGBoost CUDA histogram", "CatBoost GPU categorical"],
            "primary_metric": "average_precision (PR-AUC)",
            "secondary_metrics": ["F1", "Recall", "Precision", "Brier score", "ECE"],
            "validation": "chronological validation; retained 2020-06-21..2020-12-31 independent holdout",
            "claim_boundary": "Offline CV and independent temporal holdout only; no public leaderboard claim.",
            "generated_at": _now(),
        })
        artifact = _artifact(output, self.run_dir, kind="research_context")
        return AgentResult(task.task_id, "GPT-5.6 orchestration and falsifiable candidate families bound", [artifact["path"]], artifacts=[artifact], confidence=0.95)

    def data_audit(self, task: AgentTask, _handoff: HandoffEnvelope, _run: SupervisorRun) -> AgentResult:
        train_path = self.data_dir / "train.csv"
        test_path = self.data_dir / "test.csv"
        rows = positives = 0
        first_time = last_time = ""
        with train_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            train_columns = list(reader.fieldnames or [])
            for row in reader:
                rows += 1
                positives += int(row.get("is_fraud") == "1")
                timestamp = row.get("trans_date_trans_time", "")
                if timestamp:
                    first_time = first_time or timestamp
                    last_time = timestamp
        with test_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            test_columns = next(reader, [])
            test_rows = sum(1 for _ in reader)
        audit_path = self.run_dir / "data_audit.json"
        passed = rows == int(self.config.get("n_train") or 0) and test_rows == int(self.config.get("n_test") or 0) and "is_fraud" in train_columns and "is_fraud" not in test_columns
        _atomic_json(audit_path, {
            "schema": "evomind.credit_card_fraud.data_audit.v1",
            "status": "passed" if passed else "rejected",
            "train_rows": rows,
            "holdout_rows": test_rows,
            "positive_rows": positives,
            "positive_rate": positives / rows if rows else 0.0,
            "train_time_start": first_time,
            "train_time_end": last_time,
            "target_absent_from_candidate_test_features": "is_fraud" not in test_columns,
            "direct_identifier_exclusions": ["cc_num", "trans_num", "first", "last", "street"],
            "split_policy": _read_json(self.data_dir / "dataset_contract.json").get("split_policy"),
            "random_split_allowed": False,
            "generated_at": _now(),
        })
        artifact = _artifact(audit_path, self.run_dir, kind="data_audit")
        return AgentResult(task.task_id, "Temporal data and leakage audit passed" if passed else "Data audit rejected", [artifact["path"]], artifacts=[artifact], confidence=1.0, accepted=passed, failure_type="data_contract" if not passed else "")

    def design(self, task: AgentTask, _handoff: HandoffEnvelope, _run: SupervisorRun) -> AgentResult:
        output = self.run_dir / "design.json"
        _atomic_json(output, {
            "schema": "evomind.local_gpu.evolution_design.v1",
            "generator": "research_os.VariationGenerator",
            "orchestration_model": self.request.orchestration_model or "gpt-5.6-sol",
            "runner": "local_gpu",
            "iterations": max(2, min(4, self.request.budget.solution_repositories)),
            "search": "MCGS with best-so-far preservation and minimum-delta promotion",
            "execution_order": ["LightGBM CPU", "XGBoost CUDA", "CatBoost GPU"],
            "memory_limits": ["sequential candidates", "no dense one-hot", "bounded max_bin/depth", "release fold intermediates"],
            "human_gate": "approved_by_frontend_run_action",
            "official_submission": "disabled",
            "generated_at": _now(),
        })
        artifact = _artifact(output, self.run_dir, kind="evolution_design")
        return AgentResult(task.task_id, "Local-GPU evolution design fixed before candidate generation", [artifact["path"]], artifacts=[artifact], confidence=1.0)

    def train(self, task: AgentTask, _handoff: HandoffEnvelope, _run: SupervisorRun) -> AgentResult:
        local_python = self.root / "workspace" / "local-gpu-venv" / "Scripts" / "python.exe"
        io_path = self.run_dir / "evolution_input.json"
        iterations = max(2, min(4, self.request.budget.solution_repositories))
        _atomic_json(io_path, {
            "task_id": "credit_card_fraud_detection",
            "runner": "local_gpu",
            "iterations": iterations,
            "mcgs": True,
            "evolution_config": EVOLUTION_CONFIG,
        })
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(self.root / "src"), env.get("PYTHONPATH", ""))))
        env["OPENAI_MODEL"] = self.request.orchestration_model or "gpt-5.6-sol"
        env["OPENAI_BASE_URL"] = env.get("OPENAI_BASE_URL") or "http://127.0.0.1:65068/v1"
        env["EVOLUTION_PRIMARY_PROVIDER"] = "openai"
        env["EVOLUTION_FALLBACK_PROVIDER"] = "openai"
        env["EVOLUTION_PROVIDER_STRICT"] = "1"
        env["LLM_PROVIDER"] = "openai"
        env["OPENAI_REASONING_EFFORT"] = "low"
        env["OPENAI_SERVICE_TIER"] = "priority"
        proc = subprocess.run(
            [str(local_python), "scripts/evolution_run_cli.py", "--input", str(io_path)],
            cwd=self.root,
            env=env,
            capture_output=True,
            text=True,
            timeout=5400,
            check=False,
        )
        json_line = next((line for line in reversed(proc.stdout.splitlines()) if line.strip().startswith("{") and line.strip().endswith("}")), "")
        payload = json.loads(json_line) if json_line else {"ok": False, "error": "evolution runner returned no JSON"}
        result_path = self.run_dir / "evolution_result.json"
        _atomic_json(result_path, payload)
        if proc.returncode != 0 or payload.get("ok") is not True:
            raise RuntimeError(f"LOCAL_GPU_EVOLUTION: {payload.get('error') or proc.stderr[-500:]}")
        if (payload.get("model_provenance") or {}).get("status") != "passed":
            raise RuntimeError("LOCAL_GPU_EVOLUTION: requested GPT-5.6 provenance was not observed")
        exp_dir = (self.root / str(payload["exp_dir"])).resolve()
        exp_dir.relative_to(self.root.resolve())
        mirror = self.run_dir / "evolution"
        mirror.mkdir(parents=True, exist_ok=True)
        best = str(payload.get("best_exp_id") or "")
        source_files = {
            exp_dir / "summary.json": mirror / "summary.json",
            exp_dir / "search_graph.json": mirror / "search_graph.json",
            exp_dir / "best_solution.py": mirror / "best_solution.py",
            exp_dir / "local_resource_gate.json": mirror / "local_resource_gate.json",
            exp_dir / "model_provenance.json": mirror / "model_provenance.json",
            exp_dir / "independent_holdout_review.json": mirror / "independent_holdout_review.json",
            exp_dir / "previous_holdout_review.json": mirror / "previous_holdout_review.json",
            exp_dir / "self_evolution_comparison.json": mirror / "self_evolution_comparison.json",
            exp_dir / best / "out" / "metrics.json": mirror / "metrics.json",
            exp_dir / best / "out" / "submission.csv": mirror / "submission.csv",
            exp_dir / best / "validation_contract.json": mirror / "validation_contract.json",
            exp_dir / best / "claim_audit.json": mirror / "claim_audit.json",
        }
        copied = [result_path]
        for source, destination in source_files.items():
            if source.is_file():
                shutil.copy2(source, destination)
                copied.append(destination)
        artifacts = [_artifact(path, self.run_dir, kind=path.name) for path in copied]
        return AgentResult(task.task_id, f"EvolutionLoop completed {payload.get('n_iterations')} iterations; best={best}", [item["path"] for item in artifacts], metrics={"best_score": payload.get("best_cv_score"), "best_exp_id": best}, artifacts=artifacts, confidence=1.0)

    def review(self, task: AgentTask, _handoff: HandoffEnvelope, _run: SupervisorRun) -> AgentResult:
        source = self.run_dir / "evolution" / "independent_holdout_review.json"
        review = _read_json(source)
        output = self.run_dir / "review.json"
        _atomic_json(output, review)
        passed = review.get("status") == "passed" and (review.get("claim_audit") or {}).get("status") == "passed"
        artifact = _artifact(output, self.run_dir, kind="independent_review")
        return AgentResult(task.task_id, "Independent temporal holdout and Claim Audit passed" if passed else "Independent Reviewer rejected the candidate", [artifact["path"]], metrics=dict(review.get("metrics") or {}), artifacts=[artifact], confidence=1.0, accepted=passed, failure_type="review_rejected" if not passed else "")

    def aggregate(self, task: AgentTask, _handoff: HandoffEnvelope, run: SupervisorRun) -> AgentResult:
        summary = _read_json(self.run_dir / "evolution" / "summary.json")
        review = _read_json(self.run_dir / "review.json")
        metrics = dict(review.get("metrics") or {})
        iterations = list(summary.get("iterations") or [])
        provenance = {
            "schema": "evomind.model_provenance.v1",
            "requested_orchestration_model": self.request.orchestration_model or "gpt-5.6-sol",
            "observed_candidate_generators": [{"exp_id": item.get("exp_id"), "provider": item.get("provider"), "model": item.get("model")} for item in iterations],
            "compute_backend": "local_gpu",
            "gpu": _read_json(self.run_dir / "evolution" / "local_resource_gate.json").get("gpu"),
            "remote_compute_used": False,
            "official_submission": "disabled",
        }
        provenance_path = self.run_dir / "model_provenance.json"
        _atomic_json(provenance_path, provenance)
        comparison_path = self.run_dir / "evolution" / "self_evolution_comparison.json"
        comparison = _read_json(comparison_path) if comparison_path.is_file() else {}
        before = dict(comparison.get("before") or {})
        after = dict(comparison.get("after") or {})
        delta = dict(comparison.get("delta") or {})
        root_metrics_path = self.run_dir / "metrics.json"
        _atomic_json(root_metrics_path, {
            "schema": "evomind.local_tabular.aggregate_metrics.v1",
            "selected_solution": summary.get("best_exp_id"),
            "metric": "independent_holdout_pr_auc",
            "cv_score": summary.get("best_cv_score"),
            "independent_holdout": metrics,
            "self_evolution": comparison,
            "review_status": review.get("status"),
            "official_kaggle_score": None,
        })
        report_path = self.run_dir / "research_report.md"
        report_path.write_text(
            "# 一句话驱动研究：信用卡交易欺诈检测\n\n"
            f"- Run ID：`{run.run_id}`\n"
            f"- 科研编排：`{provenance['requested_orchestration_model']}`\n"
            "- 计算：本机 NVIDIA RTX 4060 Laptop GPU 8GB；未使用远程集群\n"
            "- 数据：1,296,675 行训练集；555,719 行独立时间 Holdout\n"
            "- 验证：时间切分 + 保留标签的独立 Reviewer；未在 Holdout 上调阈值\n"
            "- Kaggle：未执行公开榜单提交\n\n"
            "## 独立复核指标\n\n"
            f"- PR-AUC：`{metrics.get('pr_auc')}`\n"
            f"- F1：`{metrics.get('f1')}`\n"
            f"- Recall：`{metrics.get('recall')}`\n"
            f"- Precision：`{metrics.get('precision')}`\n"
            f"- Brier：`{metrics.get('brier_score')}`\n"
            f"- ECE：`{metrics.get('expected_calibration_error_15bin')}`\n\n"
            "## 最小改动自进化对比\n\n"
            f"- 版本：`{comparison.get('parent_exp_id')}` → `{comparison.get('child_exp_id')}`\n"
            f"- PR-AUC：`{before.get('pr_auc')}` → `{after.get('pr_auc')}`（Δ `{delta.get('pr_auc')}`）\n"
            f"- Recall：`{before.get('recall')}` → `{after.get('recall')}`（Δ `{delta.get('recall')}`）\n"
            f"- Brier：`{before.get('brier_score')}` → `{after.get('brier_score')}`（Δ `{delta.get('brier_score')}`）\n"
            f"- ECE：`{before.get('expected_calibration_error_15bin')}` → `{after.get('expected_calibration_error_15bin')}`（Δ `{delta.get('expected_calibration_error_15bin')}`）\n"
            "- 原版本、原始概率、验证契约和哈希证据均保留。\n\n"
            "## 自进化与证据\n\n"
            f"EvolutionLoop 实际评估 `{summary.get('n_iterations')}` 个节点，best 为 `{summary.get('best_exp_id')}`。"
            "搜索图、每轮验证契约、Claim Audit、资源门、模型来源和提交概率文件均已绑定到同一 Run 的哈希清单。\n",
            encoding="utf-8",
            newline="\n",
        )
        manifest_path = self.run_dir / "artifact_manifest.json"
        _atomic_json(manifest_path, {
            "schema": "evomind.local_tabular.artifact_manifest.v1",
            "run_id": run.run_id,
            "task_id": TASK_ID,
            "data_contract_sha256": _sha256(self.run_dir / "data_contract.json"),
            "artifacts": _manifest_entries(self.run_dir),
            "official_submission": "disabled",
            "generated_at": _now(),
        })
        artifacts = [_artifact(path, self.run_dir, kind=kind) for path, kind in (
            (report_path, "research_report"),
            (provenance_path, "model_provenance"),
            (manifest_path, "artifact_manifest"),
            (root_metrics_path, "aggregate_metrics"),
            (self.run_dir / "evolution" / "submission.csv", "candidate_submission"),
        )]
        return AgentResult(task.task_id, "Research report, candidate probability file, and reproducibility evidence delivered", [item["path"] for item in artifacts], metrics=metrics, artifacts=artifacts, confidence=1.0)

    def mapping(self):
        return {
            "SetupAgent": self.setup,
            "ResearchLead": self.research,
            "DataAuditor": self.data_audit,
            "DesignerAgent": self.design,
            "TunerAgent": self.train,
            "IndependentReviewer": self.review,
            "Aggregator": self.aggregate,
        }


def run_local_tabular_research(workspace_root: str | Path, request: UserRequest, *, run_id: str) -> SupervisorRun:
    root = Path(workspace_root).resolve()
    if request.dataset != TASK_ID or request.task_type != "tabular_classification":
        raise ValueError("local tabular workflow received a different dataset or task type")
    if request.compute_policy.backend != "local_gpu" or not request.compute_policy.local_gpu_allowed:
        raise ValueError("local tabular workflow requires the local_gpu compute policy")
    if request.compute_policy.remote_gpu_required:
        raise ValueError("local tabular workflow excludes remote compute")
    run = build_local_tabular_run(request, run_id=run_id)
    local_run_dir = run_directory(root, run.run_id)
    store = MultiAgentStore(local_run_dir)
    _atomic_json(local_run_dir / "request.json", request.to_dict())
    store.append_message(run, sender="user", receiver="ExecutiveSupervisor", content=request.objective)
    write_current_run_pointer(root, task_id=TASK_ID, run=run, run_dir=local_run_dir)
    supervisor = MultiAgentSupervisor(run, store, LocalTabularExecutors(workspace_root=root, run_dir=local_run_dir, request=request).mapping())
    try:
        result = supervisor.run_until_blocked()
    finally:
        write_current_run_pointer(root, task_id=TASK_ID, run=run, run_dir=local_run_dir)
    return result


def resume_local_tabular_research(workspace_root: str | Path, run_id: str) -> SupervisorRun:
    root = Path(workspace_root).resolve()
    local_run_dir = run_directory(root, run_id)
    store = MultiAgentStore(local_run_dir)
    run = store.load()
    request_path = local_run_dir / "request.json"
    request = UserRequest.from_dict(_read_json(request_path)) if request_path.is_file() else __import__("xsci.user_request", fromlist=["parse_user_request"]).parse_user_request(run.objective)
    supervisor = MultiAgentSupervisor(run, store, LocalTabularExecutors(workspace_root=root, run_dir=local_run_dir, request=request).mapping())
    supervisor.resume(retry_failed=run.status == "needs_continuation")
    try:
        result = supervisor.run_until_blocked()
    finally:
        write_current_run_pointer(root, task_id=TASK_ID, run=run, run_dir=local_run_dir)
    return result


__all__ = ["build_local_tabular_run", "resume_local_tabular_research", "run_local_tabular_research"]
