from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from research_agent_workstation.server.agents import (
    DataAgent,
    TaskReaderAgent,
)
from research_agent_workstation.server.core import AgentRuntime, GateEngine, TaskState, TaskStateMachine
from research_agent_workstation.server.core.json_utils import write_json
from research_agent_workstation.server.memory import ExperimentMemory
from research_agent_workstation.server.schemas.agent import AgentInput
from research_agent_workstation.server.strategy import StrategyRegistry
from research_agent_workstation.server.training import (
    EnsembleTemplateRegistry,
    JobManifestBuilder,
)
from research_os.claim_audit import audit_claim
from research_os.hpc_policy import HPCPolicyError, require_remote_workspace
from research_os.mlevolve_controller import (
    TOP30_TARGET_PERCENTILE,
    build_benchmark_claim_gate,
    build_search_controller_decision,
    classify_workstation_status,
    evaluate_rank_gate,
)
from research_os.search_graph import ExperimentNode as ResearchOSExperimentNode
from research_os.search_graph import SearchGraph as ResearchOSSearchGraph
from research_os.validation_contract import check_required_artifacts, create_contract, evaluate_acceptance

from ..adapters import (
    DisabledKaggleAdapter,
    LocalMockGPUAdapter,
    LocalStorageAdapter,
    LocalTemplateCodeAgentAdapter,
    RuleBasedLLMAdapter,
)
from ..schemas.connector import ConnectorStatus, ProviderStatus
from .evidence_service import EvidenceService
from .experiment_service import ExperimentService
from .gate_service import GateService
from .report_service import ReportService
from .task_service import TaskService

ORCHESTRATOR_STAGES = [
    "task_understanding",
    "literature_context",
    "experiment_planning",
    "human_plan_gate",
    "eda",
    "code_generation",
    "code_review",
    "training",
    "validation_review",
    "submission_check",
    "human_submission_gate",
    "report_generation",
    "human_final_gate",
    "reflection",
]


class AgentOrchestrator:
    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root
        self.storage = LocalStorageAdapter(workspace_root)
        self.task_service = TaskService(workspace_root)
        self.evidence_service = EvidenceService(self.storage)
        self.experiment_service = ExperimentService(self.storage)
        self.gate_service = GateService(self.storage)
        self.code_agent = LocalTemplateCodeAgentAdapter()
        self.llm = RuleBasedLLMAdapter()
        self.report_service = ReportService(self.llm)
        self.gpu = LocalMockGPUAdapter()
        self.kaggle = DisabledKaggleAdapter()

    def connector_status(self) -> ConnectorStatus:
        env_keys = {
            "CODE_AGENT_PROVIDER": os.getenv("CODE_AGENT_PROVIDER", self.code_agent.provider),
            "PYTHON_RUNNER": os.getenv("PYTHON_RUNNER", "disabled_hpc_only"),
            "GPU_PROVIDER": os.getenv("GPU_PROVIDER", self.gpu.provider),
            "KAGGLE_ENABLED": os.getenv("KAGGLE_ENABLED", "false"),
            "LLM_PROVIDER": os.getenv("LLM_PROVIDER", self.llm.provider),
        }
        return ConnectorStatus(
            code_agent=ProviderStatus("Code Agent", env_keys["CODE_AGENT_PROVIDER"], True, "External code agents use export/import patch flow."),
            python_runner=ProviderStatus(
                "Python Runner",
                env_keys["PYTHON_RUNNER"],
                False,
                "Local subprocess training is disabled; registered training templates queue through the HPC/GPU gateway.",
            ),
            gpu=ProviderStatus("GPU", env_keys["GPU_PROVIDER"], self.gpu.provider != "mock", "Mock adapter reserves SSH/Slurm/Docker/Kubernetes/Cloud GPU slots."),
            kaggle=ProviderStatus("Kaggle", "Configured" if env_keys["KAGGLE_ENABLED"].lower() == "true" else "Not Configured", env_keys["KAGGLE_ENABLED"].lower() == "true", "Kaggle submission requires credentials and Human Gate."),
            llm=ProviderStatus("LLM", env_keys["LLM_PROVIDER"], True, "No API key required for rule-based mode; API adapters can replace it later."),
            storage=ProviderStatus("Storage", self.storage.provider, True, "S3/OSS/MinIO reserved."),
            env_keys=env_keys,
        )

    def run_local_tabular_closed_loop(
        self,
        config_path: Path,
        output_base: Path | None = None,
        random_state: int = 42,
    ) -> dict[str, Any]:
        """Reject the retired workstation-local training path."""
        raise HPCPolicyError(
            "blocked_local_training_disabled: local tabular training is disabled by the HPC-only release policy; "
            "queue a registered template through run_ensemble_closed_loop instead."
        )

    def run_ensemble_closed_loop(
        self,
        config_path: Path,
        template_id: str = "exp007_style_lgb_xgb_cat_blend",
        output_base: Path | None = None,
        random_state: int = 42,
        fast_mode: bool = False,
        sample_rows: int = 20000,
        n_folds: int | None = None,
        seeds: list[int] | None = None,
        training_timeout_seconds: int = 3600,
        branch_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Prepare a governed HPC job manifest and stop before training starts."""
        task_profile = self.task_service.import_from_config(config_path)
        output_base = output_base or self.workspace_root / "experiments"
        runtime = AgentRuntime(task_profile.task_id)
        state_machine = TaskStateMachine(task_profile.task_id)
        gate_engine = GateEngine(task_profile.task_id)
        experiment_memory = ExperimentMemory(self.workspace_root)
        stage_records: list[dict[str, Any]] = []
        branch_metadata = branch_metadata or {}
        branch_id = str(branch_metadata.get("branch_id") or "")

        template = EnsembleTemplateRegistry.get(template_id)
        if template is None:
            raise ValueError(f"Unknown ensemble template: {template_id}. Available: {EnsembleTemplateRegistry.template_ids()}")
        if not template.approved:
            raise RuntimeError(f"Template {template_id} is not approved for workstation dispatch.")
        if not template.hpc_required:
            raise HPCPolicyError("Local ensemble templates are disabled by the HPC-only release policy.")
        remote_workspace = require_remote_workspace()

        strategies = StrategyRegistry.recommend(task_profile)
        strategy_summary = [
            {"id": s.strategy_id, "primary": s.primary_templates, "hpc_required": s.hpc_required}
            for s in strategies
        ]
        memory_summary = experiment_memory.summary_for_agent_context(task_profile.task_id)

        def stage(name: str, status: str, details: dict | None = None) -> None:
            stage_records.append({"stage": name, "status": status, "details": details or {}, "at": datetime.now().isoformat(timespec="seconds")})

        def agent_input(stage_name: str, run_id: str | None = None, artifacts: list[dict[str, Any]] | None = None) -> dict[str, Any]:
            return {
                "task_id": task_profile.task_id,
                "run_id": run_id,
                "stage": stage_name,
                "research_question": f"Can a reproducible {template.name} ensemble improve {task_profile.metric} for {task_profile.name}?",
                "task_profile": self._relative_payload(asdict(task_profile)),
                "current_artifacts": artifacts or [],
                "previous_runs": [memory_summary],
                "memory_context": [
                    {"type": "strategy_recommendations", "items": strategy_summary},
                    {"type": "search_branch", "branch_metadata": branch_metadata},
                ],
                "gate_status": {"gates": [asdict(gate) for gate in gate_engine.gates]},
                "user_constraints": [
                    "ensemble must follow registered template",
                    "no Kaggle submission without token",
                    "hpc_required" if template.hpc_required else "local_only",
                    f"search_branch={branch_id or 'default'}",
                ],
            }

        # Stage 1: Task understanding + strategy recommendation
        state_machine.transition(TaskState.IMPORTED, "Task config imported with ensemble strategy.")
        state_machine.transition(TaskState.UNDERSTANDING, "TaskReaderAgent starts task understanding.")
        runtime.execute(TaskReaderAgent(), AgentInput(**agent_input("task_understanding")))
        state_machine.transition(TaskState.UNDERSTOOD, "Task understood with ensemble strategy.")
        stage("task_understanding", "passed", {
            "template": template_id,
            "model_family": template.model_family,
            "hpc_required": template.hpc_required,
            "strategies_recommended": len(strategies),
            "historical_best_public": memory_summary.get("best_public_score"),
        })

        # Stage 2: Data check
        state_machine.transition(TaskState.EDA_RUNNING, "DataAgent checks data before ensemble training.")
        data_output = runtime.execute(DataAgent(), AgentInput(**agent_input("eda")))
        state_machine.transition(TaskState.EDA_DONE, data_output.summary)
        stage("eda", "passed" if data_output.status == "success" else data_output.status)

        # Stage 3: Plan + Gate
        state_machine.transition(TaskState.PLANNING, "PlannerAgent creates ensemble scaffold.")
        scaffold = {
            "template_id": template_id,
            "template_name": template.name,
            "model_family": template.model_family,
            "validation_strategy": template.validation_strategy,
            "risk_level": template.risk_level,
            "seeds": template.seeds,
            "params": template.params,
            "expected_outputs": template.expected_outputs,
            "branch_metadata": branch_metadata,
        }
        stage("experiment_planning", "passed", scaffold)
        plan_gate = gate_engine.create_gate(
            "PLAN_APPROVAL",
            triggered_by="PlannerAgent",
            reason=f"Ensemble template {template_id} scaffold must be reviewed.",
            required_evidence=["strategy_recommendation.json", "experiment_memory.json"],
            risk_level=template.risk_level,
        )
        state_machine.transition(TaskState.PLAN_WAITING_APPROVAL, "PLAN_APPROVAL gate created.")
        stage("human_plan_gate", "pending", {"template": template_id, "gate": plan_gate.gate_id})

        # Prepare a draft manifest. Only the external gateway can attach a dispatch
        # receipt and remote job id, then transition the run to TRAINING_QUEUED.
        run_id = f"wr_{datetime.now().isoformat().replace(':', '-')}_{uuid4().hex[:8]}"
        job_builder = JobManifestBuilder(self.workspace_root)
        job_manifest = job_builder.build(
            task_id=task_profile.task_id,
            run_id=run_id,
            agent_id="workstation_orchestrator",
            template_id=template_id,
            command_template=template.command_template,
            remote_workspace=remote_workspace,
            timeout=training_timeout_seconds,
        )
        output_dir = output_base / task_profile.task_id / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        job_builder.write(job_manifest, output_dir)
        state_machine.transition(
            TaskState.MANIFEST_PREPARED,
            f"HPC manifest prepared for {template_id}; awaiting approved dispatch.",
        )
        # This Python workflow prepares a governed manifest only. The separate
        # verified GPU gateway must execute it and attach fresh runtime evidence.
        summary = {
            "output_dir": str(output_dir),
            "accepted": False,
            "status": "manifest_prepared_awaiting_dispatch",
            "best_model": None,
            "best_metrics": {},
            "hpc_job_queued": False,
            "manifest_prepared": True,
            "remote_job_id": None,
            "dispatch_receipt": None,
            "job_manifest": str(output_dir / "job_manifest.json"),
            "training_started": False,
            "queue_metadata": {
                "random_state": random_state,
                "fast_mode": fast_mode,
                "sample_rows": sample_rows if fast_mode else None,
                "n_folds": n_folds,
                "seeds": seeds or template.seeds,
                "timeout_seconds": training_timeout_seconds,
                "branch_metadata": branch_metadata,
            },
        }
        stage("training", "awaiting_dispatch", summary)
        gate_engine.run_id = run_id
        for gate in gate_engine.gates:
            if gate.run_id is None:
                gate.run_id = run_id
        gate_engine.write(output_dir)
        runtime.flush(output_dir)
        write_json(output_dir / "task_state_machine.json", state_machine.snapshot())
        orchestrator_summary = {
            "task": asdict(task_profile),
            "run": summary,
            "connector_status": asdict(self.connector_status()),
            "stages": stage_records,
            "task_state": state_machine.snapshot(),
            "agent_trace": [asdict(trace) for trace in runtime.traces],
            "strategy_recommendations": strategy_summary,
            "experiment_memory_summary": memory_summary,
            "template": template_id,
            "status": "manifest_prepared_awaiting_dispatch",
            "claim_boundary": "A draft manifest is prepared, not queued. No training result exists until the verified GPU gateway returns a dispatch receipt, remote job id, and runtime artifacts.",
            "pending_gates": [asdict(gate) for gate in gate_engine.gates if gate.status == "pending"],
        }
        self.storage.write_json(output_dir / "orchestrator_run.json", orchestrator_summary)
        self.write_workstation_summary(orchestrator_summary)
        return orchestrator_summary

    def _write_experiment_governance_artifacts(
        self,
        *,
        task_profile: Any,
        run_id: str,
        output_dir: Path,
        template_id: str,
        template: Any,
        metric_name: str,
        metric_direction: str,
        candidate_score: float | None,
        score_promotion_gate: dict[str, Any],
        ensemble_metrics: dict[str, Any],
        submission_audit: dict[str, Any],
        fast_mode: bool,
        sample_rows: int | None,
        n_folds: int | None,
        seeds: list[int] | None,
        training_timeout_seconds: int,
        branch_metadata: dict[str, Any] | None = None,
    ) -> list[Path]:
        branch_metadata = branch_metadata or {}
        decision = score_promotion_gate.get("decision", {}) if isinstance(score_promotion_gate, dict) else {}
        parent_exp_id = decision.get("parent_exp_id")
        parent_score = decision.get("parent_score")
        promoted = bool(decision.get("promoted"))
        candidate_exp_id = f"EXP_{run_id}"
        branch_id = str(branch_metadata.get("branch_id") or "")
        branch_type = str(branch_metadata.get("branch_type") or "") or ("ensemble_exploitation" if parent_exp_id else "robust_baseline")
        branch_hypothesis = str(branch_metadata.get("hypothesis") or "")
        code_generation_mode_override = str(branch_metadata.get("code_generation_mode") or "")
        selected_stage = "exploitation" if parent_exp_id else "exploration"
        recent_failures = self._recent_failure_records(task_profile.task_id, limit=5)
        failure_count = len(recent_failures)
        branch_stagnant = len([item for item in recent_failures if item.get("failure_type") == "hold"]) >= 2
        cross_branch_references = branch_metadata.get("cross_branch_references")
        if not isinstance(cross_branch_references, list):
            cross_branch_references = []
        if parent_exp_id:
            cross_branch_references.append({
                "source": parent_exp_id,
                "target": candidate_exp_id,
                "reference_type": "best_so_far_reference",
                "reason": "Candidate should learn from current best-so-far and preserve it unless validated improvement occurs.",
            })
        memory_reuse_records = [
            {
                "memory_type": item.get("failure_type", "unknown"),
                "linked_run_id": item.get("run_id"),
                "artifact": item.get("artifact"),
                "reuse_policy": "avoid repeating failed route; prefer robust baseline before aggressive leaderboard optimization",
            }
            for item in recent_failures
        ]

        search_controller_decision = build_search_controller_decision(
            task_id=task_profile.task_id,
            run_id=run_id,
            selected_branch=branch_type,
            exploration_stage=selected_stage,
            metric=metric_name,
            metric_direction=metric_direction,
            has_parent=bool(parent_exp_id),
            branch_stagnant=branch_stagnant,
            global_stagnant=failure_count >= 4,
            failure_count=failure_count,
            cross_branch_references=cross_branch_references,
            memory_reuse_records=memory_reuse_records,
        )
        search_controller_decision.update({
            "branch_type": branch_type,
            "branch_id": branch_id,
            "template_id": template_id,
            "model_family": list(template.model_family),
            "candidate_score": candidate_score,
            "parent_score": parent_score,
            "score_promotion_decision": decision.get("decision"),
            "code_generation_mode_requested": code_generation_mode_override,
            "timeout_budget": training_timeout_seconds,
            "resource_mode": "local_cpu" if not template.hpc_required else "hpc_manifest",
            "fast_mode": fast_mode,
            "sample_rows": sample_rows,
            "n_folds": n_folds,
            "seeds": seeds or template.seeds,
            "official_submission_policy": "blocked until Human Gate approval; conservative budget <=2 official submits per task batch",
        })
        search_path = write_json(output_dir / "search_controller_decision.json", search_controller_decision)

        submission_audit.update({
            "rank_target_percentile": TOP30_TARGET_PERCENTILE,
            "official_submit_budget": search_controller_decision["official_submit_budget"],
            "official_rank_status": "proxy_only",
            "top30_reached": False,
        })
        submission_path = write_json(output_dir / "submission_audit.json", submission_audit)

        hypothesis = branch_hypothesis or "Workstation ensemble candidate should improve best-so-far before promotion."
        acceptance_criteria: dict[str, Any] = {"submission_audit_passed": {"equals": True}}
        if parent_score is not None:
            if metric_direction.lower() in {"minimize", "lower", "lower_is_better"}:
                acceptance_criteria[metric_name] = {"max": parent_score}
            else:
                acceptance_criteria[metric_name] = {"min": parent_score}

        required_artifacts = [
            "metrics.json",
            "oof_predictions.csv",
            "submission.csv",
            "artifact_manifest.json",
            "score_promotion_gate.json",
            "submission_audit.json",
        ]
        contract = create_contract(
            contract_id=f"vc_{run_id}",
            exp_id=candidate_exp_id,
            claim=hypothesis,
            hypothesis=hypothesis,
            implementation_requirement=(
                "Run must be launched through AgentOrchestrator and produce metrics, OOF, "
                "submission, artifact manifest, score promotion gate, and submission audit."
            ),
            metric=metric_name,
            baseline_exp_id=str(parent_exp_id or "none"),
            acceptance_criteria=acceptance_criteria,
            ablation_plan=[
                "Compare against imported best-so-far under declared metric direction.",
                "Record hold decision as negative evidence when candidate fails to improve.",
            ],
            risk_checklist=[
                {"risk": "data_leakage", "status": "checked_proxy", "evidence": "shared train/test features only"},
                {"risk": "cv_public_gap", "status": "not_officially_verified", "evidence": "local CV/proxy only"},
                {"risk": "submission_schema", "status": "checked", "evidence": "submission_audit.json"},
                {"risk": "leaderboard_overclaim", "status": "blocked", "evidence": "official submission requires Human Gate"},
            ],
            conclusion_boundary="Only local CV/proxy conclusions are allowed until Kaggle response artifacts exist; top-30% claims require rank_promotion_gate.json with top30_reached=true.",
            required_artifacts=required_artifacts,
        )
        available_artifacts = [path.name for path in output_dir.iterdir() if path.is_file()]
        contract_payload = asdict(contract)
        contract_payload["artifact_check"] = check_required_artifacts(contract, available_artifacts)
        contract_payload["acceptance_check"] = evaluate_acceptance(
            contract,
            {
                metric_name: candidate_score,
                "submission_audit_passed": bool(submission_audit.get("submission_audit_passed")),
            },
        )
        contract_path = write_json(output_dir / "validation_contract.json", contract_payload)

        missing_evidence = list(contract_payload["artifact_check"].get("missing_artifacts", []))
        completed_ablations = ["submission_schema_audit"]
        if (output_dir / "score_promotion_gate.json").exists():
            completed_ablations.append("best_so_far_promotion_gate")
        claim_text = (
            f"{hypothesis} Local CV/proxy gate decision is {decision.get('decision')} "
            f"for {metric_name}={candidate_score}; official Kaggle ranking and top-30% status are not claimed."
        )
        claim_audit = audit_claim(
            claim_id=f"claim_{run_id}",
            claim_text=claim_text,
            related_exp_ids=[candidate_exp_id],
            contract=contract_payload,
            supporting_metrics={
                metric_name: candidate_score,
                "parent_score": parent_score,
                "promotion_delta": decision.get("promotion_delta"),
                "promoted": promoted,
            },
            required_ablations=["best_so_far_promotion_gate", "submission_schema_audit"],
            completed_ablations=completed_ablations,
            evidence={
                "has_required_experiments": (output_dir / "score_promotion_gate.json").exists(),
                "has_mechanistic_evidence": False,
                "missing_evidence": missing_evidence,
            },
        )
        claim_path = write_json(output_dir / "claim_audit.json", claim_audit)

        return [search_path, submission_path, contract_path, claim_path]

    def _write_run_tracking_artifacts(
        self,
        *,
        task_profile: Any,
        run_id: str,
        output_dir: Path,
        template_id: str,
        metric_name: str,
        metric_direction: str,
        candidate_score: float | None,
        score_promotion_gate: dict[str, Any],
        runtime: AgentRuntime,
        gate_engine: GateEngine,
        branch_metadata: dict[str, Any] | None = None,
    ) -> list[Path]:
        branch_metadata = branch_metadata or {}
        decision = score_promotion_gate.get("decision", {}) if isinstance(score_promotion_gate, dict) else {}
        parent_run = None
        historical_best = score_promotion_gate.get("historical_best") if isinstance(score_promotion_gate, dict) else None
        if isinstance(historical_best, dict):
            parent_run = historical_best.get("run_id")
        agent_ids = sorted({trace.agent for trace in runtime.traces if getattr(trace, "agent", None)})
        claim_audit_payload = self._read_json(output_dir / "claim_audit.json") or {}
        claim_status = claim_audit_payload.get("audit_result", "unknown")
        gate_status = "promoted" if decision.get("promoted") else decision.get("decision", "needs_review")
        branch_id = str(branch_metadata.get("branch_id") or "")
        branch_type = str(branch_metadata.get("branch_type") or "") or ("ensemble_exploitation" if parent_run else "robust_baseline")
        official_submission = self._read_json(output_dir / "kaggle_official_submission.json")
        rank_gate = evaluate_rank_gate(
            task_id=task_profile.task_id,
            run_id=run_id,
            official_submission=official_submission if isinstance(official_submission, dict) else None,
        )
        workstation_status = classify_workstation_status(
            rank_gate=None if rank_gate.get("status") == "blocked_by_gate" else rank_gate,
            has_official_response=rank_gate.get("status") == "official_submitted",
        )
        benchmark_claim_gate = build_benchmark_claim_gate(
            evaluated_tasks=1 if candidate_score is not None else 0,
            medal_rate=None,
        )
        rank_gate_path = write_json(output_dir / "rank_promotion_gate.json", rank_gate)
        benchmark_claim_gate_path = write_json(output_dir / "benchmark_claim_gate.json", benchmark_claim_gate)
        run_registry_entry = {
            "schema": "academic_research_os.workstation_run_registry.v1",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "task_id": task_profile.task_id,
            "run_id": run_id,
            "agent_ids": agent_ids,
            "branch_id": branch_id,
            "branch_type": branch_type,
            "parent_run": parent_run,
            "metric": metric_name,
            "metric_direction": metric_direction,
            "candidate_score": candidate_score,
            "resource_mode": "local_cpu",
            "template_id": template_id,
            "branch_metadata": branch_metadata,
            "gate_status": gate_status,
            "claim_status": claim_status,
            "workstation_status": workstation_status,
            "rank_target_percentile": TOP30_TARGET_PERCENTILE,
            "rank_gate_status": rank_gate.get("decision"),
            "top30_reached": bool(rank_gate.get("top30_reached")),
            "official_rank": rank_gate.get("official_rank"),
            "leaderboard_team_count": rank_gate.get("leaderboard_team_count"),
            "official_submission_ref": rank_gate.get("official_submission_ref"),
            "official_submit_allowed": False,
            "official_submit_gate": "needs_human_gate",
            "artifact_path": str(output_dir),
        }
        run_registry_path = write_json(output_dir / "workstation_run_registry.json", run_registry_entry)

        workspace_registry = self.workspace_root / "workspace" / "workstation_run_registry.json"
        existing_registry = self._read_json(workspace_registry) or {"schema": "academic_research_os.workstation_run_registry_index.v1", "runs": []}
        existing_runs = [
            item for item in existing_registry.get("runs", [])
            if not (item.get("task_id") == task_profile.task_id and item.get("run_id") == run_id)
        ]
        existing_runs.append(run_registry_entry)
        existing_registry["runs"] = existing_runs[-500:]
        write_json(workspace_registry, existing_registry)

        recent_failures = self._recent_failure_records(task_profile.task_id, limit=5)
        task_state = {
            "schema": "academic_research_os.task_benchmark_state.v1",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "task_id": task_profile.task_id,
            "last_run_id": run_id,
            "last_branch_id": branch_id,
            "last_branch_type": branch_type,
            "best_so_far": {
                "run_id": run_id if decision.get("promoted") else parent_run,
                "score": candidate_score if decision.get("promoted") else decision.get("parent_score"),
                "metric": metric_name,
                "metric_direction": metric_direction,
            },
            "latest_candidate": {
                "run_id": run_id,
                "score": candidate_score,
                "decision": decision.get("decision"),
                "promotion_delta": decision.get("promotion_delta"),
            },
            "recent_failures": recent_failures,
            "exploration_stage": "exploitation" if parent_run else "exploration",
            "stagnation_detected": len([item for item in recent_failures if item.get("failure_type") == "hold"]) >= 3,
            "rank_target_percentile": TOP30_TARGET_PERCENTILE,
            "rank_gate": rank_gate,
            "benchmark_claim_gate": benchmark_claim_gate,
            "workstation_status": workstation_status,
            "official_submit_candidate": False,
            "official_submit_allowed": False,
            "official_submit_blocker": "Human Gate approval and Kaggle response artifact are required before any rank/medal claim.",
        }
        state_path = write_json(output_dir / "task_benchmark_state.json", task_state)
        workspace_state_dir = self.workspace_root / "workspace" / "task_benchmark_states"
        write_json(workspace_state_dir / f"{task_profile.task_id}.json", task_state)
        return [run_registry_path, state_path, rank_gate_path, benchmark_claim_gate_path]

    def _recent_failure_records(self, task_id: str, limit: int = 5) -> list[dict[str, Any]]:
        root = self.workspace_root / "experiments" / task_id
        if not root.exists():
            return []
        failures: list[dict[str, Any]] = []
        for run_dir in sorted([path for path in root.iterdir() if path.is_dir()], key=lambda path: path.stat().st_mtime, reverse=True):
            failure_review = self._read_json(run_dir / "failure_review.json")
            timeout_manifest = self._read_json(run_dir / "timeout_manifest.json")
            gate = self._read_json(run_dir / "score_promotion_gate.json")
            rank_gate = self._read_json(run_dir / "rank_promotion_gate.json")
            decision = gate.get("decision", {}) if isinstance(gate, dict) else {}
            rank_decision = rank_gate.get("decision") if isinstance(rank_gate, dict) else None
            if timeout_manifest:
                failures.append({"run_id": run_dir.name, "failure_type": "timeout", "artifact": str(run_dir / "timeout_manifest.json")})
            elif failure_review:
                failures.append({"run_id": run_dir.name, "failure_type": "failed", "artifact": str(run_dir / "failure_review.json")})
            elif rank_decision == "top30_failed":
                failures.append({"run_id": run_dir.name, "failure_type": "top30_failed", "artifact": str(run_dir / "rank_promotion_gate.json")})
            elif decision.get("decision") == "hold":
                failures.append({
                    "run_id": run_dir.name,
                    "failure_type": "hold",
                    "candidate_score": decision.get("candidate_score"),
                    "parent_score": decision.get("parent_score"),
                    "reason": decision.get("reason"),
                })
            if len(failures) >= limit:
                break
        return failures

    def _read_json(self, path: Path) -> Any:
        try:
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return None


    def _historical_best_from_experiments(self, task_id: str, metric_name: str, metric_direction: str, exclude_run_id: str | None = None) -> dict[str, Any] | None:
        def canonical_metric(value: Any) -> str:
            normalized = str(value or "").lower().replace("-", "_")
            aliases = {
                "auc": "roc_auc",
                "gini": "normalized_gini",
            }
            return aliases.get(normalized, normalized)

        wanted_metric = canonical_metric(metric_name)
        candidates: list[dict[str, Any]] = []
        progress_path = self.workspace_root / "workspace" / "kaggle_10_self_evolution_progress_20260623.json"
        if progress_path.exists():
            try:
                progress = json.loads(progress_path.read_text(encoding="utf-8"))
                aliases = {
                    "house_prices": "house_prices_advanced_regression_techniques",
                }
                wanted_ids = {task_id, aliases.get(task_id, task_id)}
                for item in progress.get("results", []):
                    if canonical_metric(item.get("metric")) != wanted_metric:
                        continue
                    if item.get("task_id") in wanted_ids and isinstance(item.get("best_score"), (int, float)):
                        candidates.append({
                            "run_id": f"progress_best_{item.get('task_id')}",
                            "score": float(item["best_score"]),
                            "source": str(progress_path),
                            "path": str(progress_path),
                            "source_type": "kaggle_10_progress",
                        })
            except Exception:
                pass
        root = self.workspace_root / "experiments" / task_id
        if not root.exists():
            root_candidates = []
        else:
            root_candidates = [path for path in root.iterdir() if path.is_dir()]
        for run_dir in root_candidates:
            if exclude_run_id and run_dir.name == exclude_run_id:
                continue
            score: float | None = None
            source = None
            metrics_path = run_dir / "metrics.json"
            if metrics_path.exists():
                try:
                    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
                    payload_metric = payload.get("metric") or payload.get("ensemble", {}).get("selection_metric")
                    if payload_metric and canonical_metric(payload_metric) != wanted_metric:
                        continue
                    raw = payload.get("ensemble", {}).get("best_validation_score")
                    if isinstance(raw, (int, float)):
                        score = float(raw)
                        source = str(metrics_path)
                except Exception:
                    pass
            record_path = run_dir / "experiment_record.json"
            if score is None and record_path.exists():
                try:
                    payload = json.loads(record_path.read_text(encoding="utf-8"))
                    metric_payload = payload.get("metric", {})
                    for key in (metric_name, f"cv_{metric_name}_mean", "cv_rmsle_mean", "cv_accuracy_mean", "holdout_rmsle"):
                        raw = metric_payload.get(key)
                        if isinstance(raw, (int, float)):
                            score = float(raw)
                            source = str(record_path)
                            break
                except Exception:
                    pass
            if score is not None:
                candidates.append({"run_id": run_dir.name, "score": score, "source": source, "path": str(run_dir)})
        if not candidates:
            return None
        reverse = metric_direction.lower() not in {"minimize", "lower", "lower_is_better"}
        return sorted(candidates, key=lambda item: item["score"], reverse=reverse)[0]

    def _write_score_promotion_gate(
        self,
        task_id: str,
        run_id: str,
        output_dir: Path,
        metric_name: str,
        metric_direction: str,
        candidate_score: float | None,
        artifacts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        historical_best = self._historical_best_from_experiments(task_id, metric_name, metric_direction, exclude_run_id=run_id)
        parent_exp_id = f"BEST_{historical_best['run_id']}" if historical_best else None
        graph = ResearchOSSearchGraph(
            task_id=task_id,
            root_exp_id=parent_exp_id or f"EXP_{run_id}",
            metric_name=metric_name,
            metric_direction=metric_direction,
            best_exp_id=parent_exp_id,
        )
        if historical_best:
            graph.add_node(ResearchOSExperimentNode(
                exp_id=parent_exp_id or f"BEST_{historical_best['run_id']}",
                parent_id=None,
                branch_type="baseline",
                task_name=task_id,
                hypothesis="Historical best-so-far node imported for promotion comparison.",
                implementation_summary="Existing workstation artifact selected as parent best.",
                code_path="historical_artifact",
                artifacts=[{"artifact_type": "historical_best", "path": historical_best.get("source") or historical_best.get("path")}],
                metrics={metric_name: historical_best["score"]},
                cv_score=historical_best["score"],
                decision="promote",
                created_at=datetime.now().isoformat(timespec="seconds"),
                metric_name=metric_name,
                metric_direction=metric_direction,
                promoted=True,
            ))
        graph.add_node(ResearchOSExperimentNode(
            exp_id=f"EXP_{run_id}",
            parent_id=parent_exp_id,
            branch_type="ensemble",
            task_name=task_id,
            hypothesis="Workstation ensemble candidate should improve best-so-far before promotion.",
            implementation_summary="Candidate generated by AgentOrchestrator.run_ensemble_closed_loop.",
            code_path="registered_hpc_template",
            artifacts=artifacts,
            metrics={metric_name: candidate_score} if candidate_score is not None else {},
            cv_score=candidate_score,
            decision="needs_review",
            created_at=datetime.now().isoformat(timespec="seconds"),
            metric_name=metric_name,
            metric_direction=metric_direction,
        ))
        if parent_exp_id:
            graph.add_edge(parent_exp_id, f"EXP_{run_id}", "workstation_ensemble_candidate")
        decision = graph.decide_promotion(
            f"EXP_{run_id}",
            parent_exp_id=parent_exp_id,
            metric=metric_name,
            direction=metric_direction,
            required_artifacts=["metrics.json", "submission.csv"],
        )
        payload = {
            "schema": "academic_research_os.score_promotion_gate.v1",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "task_id": task_id,
            "run_id": run_id,
            "historical_best": historical_best,
            "decision": decision,
            "invariant": "best-so-far is promoted only when candidate improves under declared metric direction and required artifacts exist",
        }
        write_json(output_dir / "score_promotion_gate.json", payload)
        graph.export_json(output_dir / "research_os_search_graph.json")
        return payload

    def write_workstation_summary(self, payload: dict[str, Any]) -> Path:
        target = self.workspace_root / "workspace" / "workstation_summary.json"
        return self.storage.write_json(target, self._relative_payload(payload))

    def summarize_latest_existing(self, task_ids: list[str]) -> dict[str, Any]:
        runs = []
        for task_id in task_ids:
            latest = self.experiment_service.latest_experiment_dir(task_id)
            if latest:
                runs.append({"task_id": task_id, **self.experiment_service.summarize_existing_run(latest)})
        payload = {
            "connector_status": asdict(self.connector_status()),
            "runs": runs,
            "stages": [{"stage": stage, "status": "reserved"} for stage in ORCHESTRATOR_STAGES],
        }
        self.write_workstation_summary(payload)
        return payload

    def _relative_payload(self, value: Any) -> Any:
        if isinstance(value, Path):
            try:
                return str(value.relative_to(self.workspace_root))
            except ValueError:
                return str(value)
        if isinstance(value, str):
            try:
                path_value = Path(value)
                if path_value.is_absolute():
                    return str(path_value.relative_to(self.workspace_root))
            except (OSError, ValueError):
                return value
            return value
        if isinstance(value, list):
            return [self._relative_payload(item) for item in value]
        if isinstance(value, dict):
            return {key: self._relative_payload(item) for key, item in value.items()}
        return value
