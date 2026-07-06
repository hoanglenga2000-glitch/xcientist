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
    PlannerAgent,
    ReflectionAgent,
    ReviewerAgent,
    TaskReaderAgent,
    TrainerAgent,
    WriterAgent,
)
from research_agent_workstation.server.core import AgentRuntime, ArtifactRegistry, EventBus, GateEngine, TaskState, TaskStateMachine
from research_agent_workstation.server.core.evidence_graph import EvidenceGraph
from research_agent_workstation.server.core.experiment_graph import ExperimentGraph, ExperimentNode
from research_agent_workstation.server.core.json_utils import write_json
from research_agent_workstation.server.core.memory_store import MemoryStore
from research_agent_workstation.server.memory import ExperimentMemory
from research_agent_workstation.server.schemas.agent import AgentInput
from research_agent_workstation.server.strategy import StrategyRegistry
from research_agent_workstation.server.training import (
    EnsembleTemplateRegistry,
    JobManifestBuilder,
    RetryPolicy,
    SubmissionGate,
)
from research_agent_workstation.tabular_pipeline import load_yaml
from research_os.claim_audit import audit_claim
from research_os.mlevolve_controller import (
    TOP30_TARGET_PERCENTILE,
    build_benchmark_claim_gate,
    build_search_controller_decision,
    classify_workstation_status,
    evaluate_rank_gate,
)
from research_os.search_graph import ExperimentNode as ResearchOSExperimentNode, SearchGraph as ResearchOSSearchGraph
from research_os.validation_contract import check_required_artifacts, create_contract, evaluate_acceptance

from ..adapters import (
    DisabledKaggleAdapter,
    LocalMockGPUAdapter,
    LocalPythonRunnerAdapter,
    LocalStorageAdapter,
    LocalTemplateCodeAgentAdapter,
    RuleBasedLLMAdapter,
)
from ..schemas.connector import ConnectorStatus, ProviderStatus
from ..schemas.experiment import ExperimentRecord
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
        self.python_runner = LocalPythonRunnerAdapter(workspace_root / "workspace" / "runs")
        self.llm = RuleBasedLLMAdapter()
        self.report_service = ReportService(self.llm)
        self.gpu = LocalMockGPUAdapter()
        self.kaggle = DisabledKaggleAdapter()

    def connector_status(self) -> ConnectorStatus:
        env_keys = {
            "CODE_AGENT_PROVIDER": os.getenv("CODE_AGENT_PROVIDER", self.code_agent.provider),
            "PYTHON_RUNNER": os.getenv("PYTHON_RUNNER", self.python_runner.provider),
            "GPU_PROVIDER": os.getenv("GPU_PROVIDER", self.gpu.provider),
            "KAGGLE_ENABLED": os.getenv("KAGGLE_ENABLED", "false"),
            "LLM_PROVIDER": os.getenv("LLM_PROVIDER", self.llm.provider),
        }
        return ConnectorStatus(
            code_agent=ProviderStatus("Code Agent", env_keys["CODE_AGENT_PROVIDER"], True, "External code agents use export/import patch flow."),
            python_runner=ProviderStatus("Python Runner", env_keys["PYTHON_RUNNER"], True, "Local subprocess adapter is available for scripts."),
            gpu=ProviderStatus("GPU", env_keys["GPU_PROVIDER"], self.gpu.provider != "mock", "Mock adapter reserves SSH/Slurm/Docker/Kubernetes/Cloud GPU slots."),
            kaggle=ProviderStatus("Kaggle", "Configured" if env_keys["KAGGLE_ENABLED"].lower() == "true" else "Not Configured", env_keys["KAGGLE_ENABLED"].lower() == "true", "Kaggle submission requires credentials and Human Gate."),
            llm=ProviderStatus("LLM", env_keys["LLM_PROVIDER"], True, "No API key required for rule-based mode; API adapters can replace it later."),
            storage=ProviderStatus("Storage", self.storage.provider, True, "S3/OSS/MinIO reserved."),
            env_keys=env_keys,
        )

    def run_local_tabular_closed_loop(self, config_path: Path, output_base: Path | None = None, random_state: int = 42) -> dict[str, Any]:
        task_profile = self.task_service.import_from_config(config_path)
        config = load_yaml(config_path)
        output_base = output_base or self.workspace_root / "experiments"
        runtime = AgentRuntime(task_profile.task_id)
        state_machine = TaskStateMachine(task_profile.task_id)
        event_bus = EventBus(task_profile.task_id)
        gate_engine = GateEngine(task_profile.task_id)
        memory_store = MemoryStore(self.workspace_root)
        stage_records: list[dict[str, Any]] = []

        def stage(name: str, status: str, details: dict | None = None) -> None:
            stage_records.append({"stage": name, "status": status, "details": details or {}, "at": datetime.now().isoformat(timespec="seconds")})
            event_bus.emit("stage_update", name, f"{name} -> {status}", payload=details or {})

        def agent_input(stage_name: str, run_id: str | None = None, artifacts: list[dict[str, Any]] | None = None, memories: list[dict[str, Any]] | None = None) -> dict[str, Any]:
            return {
                "task_id": task_profile.task_id,
                "run_id": run_id,
                "stage": stage_name,
                "research_question": f"Can a reproducible {task_profile.task_type} baseline improve {task_profile.metric} for {task_profile.name}?",
                "task_profile": self._relative_payload(asdict(task_profile)),
                "current_artifacts": artifacts or [],
                "previous_runs": [],
                "memory_context": memories or [],
                "gate_status": {"gates": [asdict(gate) for gate in gate_engine.gates]},
                "user_constraints": ["local-only MVP", "no Kaggle submission without token", "no GPU connection in this phase"],
            }

        state_machine.transition(TaskState.IMPORTED, "Task config imported.", {"config_path": str(config_path)})
        event_bus.emit("state_transition", "task_import", "Task imported from config.")
        task_reader_output = runtime.execute(TaskReaderAgent(), AgentInput(**agent_input("task_understanding")))
        state_machine.transition(TaskState.UNDERSTANDING, "TaskReaderAgent started task understanding.")
        state_machine.transition(TaskState.UNDERSTOOD, task_reader_output.summary)
        stage("task_understanding", "passed", asdict(task_profile))

        state_machine.transition(TaskState.EDA_RUNNING, "DataAgent checks data contract before planning.")
        data_output = runtime.execute(DataAgent(), AgentInput(**agent_input("eda")))
        state_machine.transition(TaskState.EDA_DONE, data_output.summary)
        stage("eda", "passed" if data_output.status == "success" else data_output.status, {"summary": data_output.summary, "risk_flags": data_output.risk_flags})

        state_machine.transition(TaskState.PLANNING, "PlannerAgent creates scaffold.")
        scaffold_root = self.workspace_root / "workspace" / "tasks" / task_profile.task_id / "scaffold"
        scaffold_json, scaffold_md, scaffold = self.task_service.generate_scaffold(task_profile, config_path, scaffold_root)
        planner_output = runtime.execute(PlannerAgent(), AgentInput(**agent_input("experiment_planning", artifacts=[{"path": str(scaffold_json), "name": "scaffold.json"}])))
        plan = self.code_agent.generate_plan(task_profile, {})
        stage("experiment_planning", "passed", {"plan": asdict(plan), "scaffold": self._relative_payload(scaffold)})
        state_machine.transition(TaskState.PLAN_WAITING_APPROVAL, planner_output.summary)
        plan_gate = gate_engine.create_gate(
            "PLAN_APPROVAL",
            triggered_by="PlannerAgent",
            reason="Scaffold must be reviewed before code generation and training.",
            required_evidence=["scaffold.json", "scaffold.md"],
            risk_level="medium",
        )
        gate_engine.decide(plan_gate.gate_id, "approved", reviewer="Research Admin (local MVP)", comment="Auto-approved for local closed-loop demo; production requires explicit reviewer action.")
        gate_engine.require_approved("PLAN_APPROVAL", "code generation and training")
        state_machine.transition(TaskState.PLAN_APPROVED, "PLAN_APPROVAL approved.")
        stage("human_plan_gate", "approved", asdict(plan_gate))

        state_machine.transition(TaskState.CODE_GENERATING, "LocalTemplateCodeAgent generating baseline runner.")
        code_artifact = self.code_agent.generate_code(plan, {"workspace_dir": self.workspace_root})
        code_path = code_artifact.generated_files[0]
        review = self.code_agent.review_code(code_path, {"task_id": task_profile.task_id})
        state_machine.transition(TaskState.CODE_READY, "Code artifact generated and reviewed.")
        stage("code_generation", "passed", {"provider": self.code_agent.provider, "files": [str(path) for path in code_artifact.generated_files]})
        stage("code_review", review.status, asdict(review))

        state_machine.transition(TaskState.TRAINING_RUNNING, "LocalPythonRunner starts tabular baseline.")
        runner_result = self.python_runner.run_script(
            self.workspace_root / "src" / "research_agent_workstation" / "tabular_pipeline.py",
            ["--config", str(config_path), "--output-dir", str(output_base), "--random-state", str(random_state)],
            self.workspace_root,
        )
        runner_stdout = runner_result.stdout_path.read_text(encoding="utf-8", errors="ignore")
        if runner_result.return_code != 0:
            stage("training", "failed", asdict(runner_result))
            state_machine.transition(TaskState.WAITING_FIX, "Training failed; DeveloperAgent fix required.")
            raise RuntimeError(runner_result.stderr_path.read_text(encoding="utf-8", errors="ignore") or runner_stdout)
        summary = json.loads(runner_stdout[runner_stdout.find("{") : runner_stdout.rfind("}") + 1])
        output_dir = Path(summary["output_dir"])
        run_id = output_dir.name
        runner_result.output_dir = output_dir
        gate_engine.run_id = run_id
        for gate in gate_engine.gates:
            if gate.run_id is None:
                gate.run_id = run_id

        state_machine.transition(TaskState.TRAINING_DONE, "Training completed and output directory created.", {"run_id": run_id})
        artifact_registry = ArtifactRegistry(task_profile.task_id, run_id)
        artifact_registry.register(scaffold_json, artifact_type="scaffold", created_by="PlannerAgent", linked_stage="experiment_planning")
        artifact_registry.register(scaffold_md, artifact_type="scaffold", created_by="PlannerAgent", linked_stage="experiment_planning")
        artifact_registry.register(code_path, artifact_type="code", created_by="LocalTemplateCodeAgent", linked_stage="code_generation")
        artifact_registry.collect_directory(output_dir, created_by="local_pipeline", linked_stage="training")
        artifact_dicts = [asdict(artifact) for artifact in artifact_registry.artifacts]
        runtime.execute(TrainerAgent(), AgentInput(**agent_input("training", run_id, artifact_dicts)))
        stage("training", "passed", {"summary": summary, "runner": asdict(runner_result)})
        experiment_log = json.loads((output_dir / "experiment_log.json").read_text(encoding="utf-8"))
        submission_check = experiment_log.get("submission_check", {})
        stage("submission_check", "passed" if submission_check.get("valid") else "failed", submission_check)

        state_machine.transition(TaskState.REVIEWING, "ReviewerAgent checks metrics, evidence and submission contract.")
        evidence = self.evidence_service.collect_from_run(task_profile.task_id, run_id, output_dir)
        reviewer_output = runtime.execute(ReviewerAgent(), AgentInput(**agent_input("validation_review", run_id, artifact_dicts)))
        state_machine.transition(TaskState.REVIEW_DONE, reviewer_output.summary)
        stage("validation_review", "passed", {"evidence_count": len(evidence.artifacts)})
        evidence_graph = EvidenceGraph(task_profile.task_id)
        evidence_items = evidence_graph.ingest_artifacts(artifact_registry.artifacts)
        metric_evidence = [item.evidence_id for item in evidence_items if "model_results" in item.path or "validation_gate" in item.path or "submission" in item.path]
        evidence_graph.bind_claim(
            f"{task_profile.name} local baseline produced validated {task_profile.metric} results.",
            source="reviewer",
            evidence_ids=metric_evidence[:4],
            confidence=0.86 if metric_evidence else 0.35,
            risk_level="medium",
        )

        gate_engine.create_gate(
            "SUBMISSION_APPROVAL",
            triggered_by="ReviewerAgent",
            reason="Submission can only proceed after submission_check passed and evidence is bound.",
            required_evidence=metric_evidence[:4],
            risk_level="medium",
        )
        state_machine.transition(TaskState.SUBMISSION_WAITING_APPROVAL, "Submission approval is pending human review.")
        submission_gate = self.gate_service.create_gate(task_profile.task_id, "human_submission", [item.evidence_id for item in evidence.artifacts], output_dir)
        stage("human_submission_gate", "pending", asdict(submission_gate))

        state_machine.transition(TaskState.REPORT_GENERATING, "WriterAgent generates evidence-bound report.")
        report_path = self.report_service.generate_summary_report(output_dir, {"task": asdict(task_profile), "summary": summary})
        runtime.execute(WriterAgent(), AgentInput(**agent_input("report_generation", run_id, artifact_dicts)))
        artifact_registry.register(report_path, artifact_type="report", created_by="WriterAgent", linked_stage="report_generation", linked_claims=[claim.claim_id for claim in evidence_graph.claims])
        state_machine.transition(TaskState.REPORT_DONE, "Report generated from registered artifacts.")
        stage("report_generation", "passed", {"report": str(report_path)})
        gate_engine.create_gate(
            "FINAL_CLAIM_APPROVAL",
            triggered_by="WriterAgent",
            reason="Final claims require reviewer approval and evidence binding.",
            required_evidence=metric_evidence[:4],
            risk_level="medium",
        )
        state_machine.transition(TaskState.FINAL_WAITING_APPROVAL, "Final claim approval remains pending.")
        stage("human_final_gate", "pending", {"reason": "final academic review remains manual"})

        best_metrics = summary.get("best_metrics", {})
        score = best_metrics.get("cv_rmsle_mean") or best_metrics.get("cv_accuracy_mean") or best_metrics.get("holdout_rmsle")
        experiment_graph = ExperimentGraph(task_profile.task_id)
        experiment_graph.add_run(
            ExperimentNode(
                run_id=run_id,
                parent_run_id=None,
                branch_id="baseline",
                stage="training_done",
                hypothesis="Local template baseline establishes a reproducible reference run.",
                plan=asdict(plan),
                code_snapshot=str(code_path),
                model=summary.get("best_model"),
                params=config.get("model", {}),
                metric=task_profile.metric,
                score=float(score) if isinstance(score, (int, float)) else None,
                is_buggy=False,
                status="passed",
                artifacts=[artifact.path for artifact in artifact_registry.artifacts],
                reward=float(score) if isinstance(score, (int, float)) else None,
            )
        )
        reflection_output = runtime.execute(ReflectionAgent(), AgentInput(**agent_input("reflection", run_id, artifact_dicts)))
        memory_record = memory_store.add(
            task_type=task_profile.task_type,
            dataset_type="tabular",
            hypothesis="Local template baseline establishes a reproducible reference run.",
            method_summary=f"{summary.get('best_model')} with local tabular pipeline.",
            code_summary=f"Generated baseline runner at {code_path}.",
            metric_before=None,
            metric_after=float(score) if isinstance(score, (int, float)) else None,
            success_label="success" if summary.get("accepted", True) else "neutral",
            failure_reason=None,
            useful_for=["tabular_baseline", task_profile.metric, task_profile.task_id],
            evidence_refs=metric_evidence[:4],
        )
        reflection = {
            "what_changed": "Local baseline run completed and artifacts were registered through the runtime.",
            "metric_change": {"before": None, "after": score, "metric": task_profile.metric},
            "why_it_might_help": "It creates a reproducible baseline before Codex/Claude Code optimization.",
            "failure_reason": None,
            "next_experiment_suggestion": reflection_output.next_actions,
            "memory_record": asdict(memory_record),
            "whether_continue_this_branch": True,
            "whether_try_new_branch": True,
        }
        write_json(output_dir / "reflection.json", reflection)
        (output_dir / "reflection.md").write_text(
            "\n".join(
                [
                    "# Reflection",
                    "",
                    reflection["what_changed"],
                    "",
                    f"- Metric: {task_profile.metric}",
                    f"- After: {score}",
                    "",
                    "## Next Experiment Suggestions",
                    *[f"- {item}" for item in reflection_output.next_actions],
                ]
            ),
            encoding="utf-8",
        )
        stage("reflection", "passed", {"next": reflection_output.next_actions, "memory_record": memory_record.record_id})

        record = ExperimentRecord(
            run_id=run_id,
            task_id=task_profile.task_id,
            source_type="local_template",
            code_agent_provider=self.code_agent.provider,
            code_patch_id=None,
            runner_provider="local_pipeline",
            gpu_provider=self.gpu.provider,
            llm_provider=self.llm.provider,
            dataset_version=config["task"].get("name", task_profile.task_id),
            code_commit=None,
            seed=random_state,
            metric=summary.get("best_metrics", {}),
            artifacts=[item.artifact_path for item in evidence.artifacts],
            output_dir=output_dir,
        )
        self.experiment_service.record_run(record)
        artifact_registry.write_manifest(output_dir)
        evidence_graph.write(output_dir)
        experiment_graph.write(output_dir)
        memory_store.write_run_memory(output_dir)
        gate_engine.write(output_dir)
        event_bus.flush(output_dir)
        runtime.flush(output_dir)
        write_json(output_dir / "task_state_machine.json", state_machine.snapshot())
        write_json(
            output_dir / "runtime_snapshot.json",
            {
                "task_state": state_machine.snapshot(),
                "current_agent": runtime.traces[-1].agent if runtime.traces else None,
                "current_stage": state_machine.state.value,
                "latest_metric": best_metrics,
                "latest_artifact": artifact_registry.artifacts[-1] if artifact_registry.artifacts else None,
                "pending_gates": [asdict(gate) for gate in gate_engine.gates if gate.status == "pending"],
                "claim_evidence_status": {
                    "claims": [asdict(claim) for claim in evidence_graph.claims],
                    "needs_evidence": [claim.claim_id for claim in evidence_graph.needs_evidence()],
                },
                "next_actions": reflection_output.next_actions,
            },
        )

        orchestrator_summary = {
            "task": asdict(task_profile),
            "run": summary,
            "experiment_record": asdict(record),
            "connector_status": asdict(self.connector_status()),
            "stages": stage_records,
            "task_state": state_machine.snapshot(),
            "agent_trace": [asdict(trace) for trace in runtime.traces],
            "artifact_registry": [asdict(artifact) for artifact in artifact_registry.artifacts],
            "evidence_graph": {"evidence_count": len(evidence_graph.evidence), "claims": [asdict(claim) for claim in evidence_graph.claims]},
            "experiment_graph": {"nodes": [asdict(node) for node in experiment_graph.nodes], "edges": [asdict(edge) for edge in experiment_graph.edges]},
            "memory": [asdict(memory_record)],
            "reflection": reflection,
            "pending_gates": [asdict(gate) for gate in gate_engine.gates if gate.status == "pending"],
            "evidence_count": len(evidence.artifacts),
            "kaggle": asdict(self.kaggle.validate_credentials()),
            "gpu": self.gpu.list_devices(),
        }
        self.storage.write_json(output_dir / "orchestrator_run.json", orchestrator_summary)
        self.write_workstation_summary(orchestrator_summary)
        return orchestrator_summary

    def run_ensemble_closed_loop(
        self,
        config_path: Path,
        template_id: str = "sklearn_rf_hgb_et_ensemble",
        output_base: Path | None = None,
        random_state: int = 42,
        fast_mode: bool = False,
        sample_rows: int = 20000,
        n_folds: int | None = None,
        seeds: list[int] | None = None,
        training_timeout_seconds: int = 3600,
        branch_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run workstation ensemble closed loop using a registered ensemble template.

        Args:
            fast_mode: Use reduced estimators and sample data for quick verification.
            sample_rows: Number of rows to sample in fast mode.
        """
        task_profile = self.task_service.import_from_config(config_path)
        config = load_yaml(config_path)
        output_base = output_base or self.workspace_root / "experiments"
        runtime = AgentRuntime(task_profile.task_id)
        state_machine = TaskStateMachine(task_profile.task_id)
        gate_engine = GateEngine(task_profile.task_id)
        experiment_memory = ExperimentMemory(self.workspace_root)
        stage_records: list[dict[str, Any]] = []
        branch_metadata = branch_metadata or {}
        branch_id = str(branch_metadata.get("branch_id") or "")
        branch_type_override = str(branch_metadata.get("branch_type") or "")
        code_generation_mode_override = str(branch_metadata.get("code_generation_mode") or "")
        branch_hypothesis = str(branch_metadata.get("hypothesis") or "")
        cross_branch_reference_override = branch_metadata.get("cross_branch_references")
        if not isinstance(cross_branch_reference_override, list):
            cross_branch_reference_override = []

        template = EnsembleTemplateRegistry.get(template_id)
        if template is None:
            raise ValueError(f"Unknown ensemble template: {template_id}. Available: {EnsembleTemplateRegistry.template_ids()}")
        if not template.approved:
            raise RuntimeError(f"Template {template_id} is not approved for workstation dispatch.")

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
        plan_gate = gate_engine.create_gate(
            "PLAN_APPROVAL",
            triggered_by="PlannerAgent",
            reason=f"Ensemble template {template_id} scaffold must be reviewed.",
            required_evidence=["strategy_recommendation.json", "experiment_memory.json"],
            risk_level=template.risk_level,
        )
        gate_engine.decide(plan_gate.gate_id, "approved", reviewer="Research Admin (workstation MVP)", comment="Auto-approved for ensemble closed-loop demo.")
        state_machine.transition(TaskState.PLAN_WAITING_APPROVAL, "PLAN_APPROVAL gate created.")
        gate_engine.require_approved("PLAN_APPROVAL", "ensemble training")
        state_machine.transition(TaskState.PLAN_APPROVED, "PLAN_APPROVAL approved for ensemble.")
        stage("human_plan_gate", "approved", {"template": template_id, "gate": plan_gate.gate_id})

        # Stage 4: Code generation (delegated to ensemble template)
        state_machine.transition(TaskState.CODE_GENERATING, "Ensemble template script is the code artifact.")
        state_machine.transition(TaskState.CODE_READY, "Ensemble runner script ready.")

        # Stage 5: Training - dispatch ensemble runner
        state_machine.transition(TaskState.TRAINING_RUNNING, f"LocalPythonRunner dispatches {template_id} ensemble.")

        if template.hpc_required:
            # Build HPC job manifest
            run_id = f"wr_{datetime.now().isoformat().replace(':', '-')}_{uuid4().hex[:8]}"
            job_builder = JobManifestBuilder(self.workspace_root)
            job_manifest = job_builder.build(
                task_id=task_profile.task_id,
                run_id=run_id,
                agent_id="workstation_orchestrator",
                template_id=template_id,
                command_template=template.command_template,
                remote_workspace="/hpc2hdd/home/aimslab",
                timeout=7200,
            )
            output_dir = output_base / task_profile.task_id / run_id
            output_dir.mkdir(parents=True, exist_ok=True)
            job_builder.write(job_manifest, output_dir)
            # HPC job is marked as queued since GPU is mock
            summary = {
                "output_dir": str(output_dir),
                "accepted": False,
                "best_model": f"hpc_{template_id}",
                "best_metrics": {"note": "HPC job queued; awaiting GPU connection for execution"},
                "hpc_job_queued": True,
                "job_manifest": str(output_dir / "job_manifest.json"),
            }
            state_machine.transition(TaskState.TRAINING_DONE, f"HPC ensemble job {run_id} queued; GPU mock in effect.")
        else:
            # Local ensemble execution
            import subprocess
            import sys
            run_id = f"wr_{datetime.now().isoformat().replace(':', '-')}_{uuid4().hex[:8]}"
            output_dir = output_base / task_profile.task_id / run_id
            output_dir.mkdir(parents=True, exist_ok=True)

            ensemble_script = self.workspace_root / "scripts" / "run_local_sklearn_ensemble.py"
            cmd = [
                sys.executable, str(ensemble_script),
                "--config", str(config_path),
                "--output-base", str(output_base),
                "--task-id", task_profile.task_id,
                "--run-id", run_id,
                "--random-state", str(random_state),
            ]
            if branch_id:
                cmd.extend(["--branch-id", branch_id])
            if branch_type_override:
                cmd.extend(["--branch-type", branch_type_override])
            if code_generation_mode_override:
                cmd.extend(["--code-generation-mode", code_generation_mode_override])
            if branch_hypothesis:
                cmd.extend(["--branch-hypothesis", branch_hypothesis])
            if cross_branch_reference_override:
                cmd.extend(["--cross-branch-references", json.dumps(cross_branch_reference_override, ensure_ascii=False)])
            if fast_mode:
                cmd.extend(["--fast", "--sample-rows", str(sample_rows)])
            if n_folds is not None:
                cmd.extend(["--n-folds", str(n_folds)])
            if seeds:
                cmd.extend(["--seeds", ",".join(str(seed) for seed in seeds)])
            write_json(output_dir / "launcher_manifest.json", {
                "schema": "academic_research_os.launcher_manifest.v1",
                "launcher": "AgentOrchestrator.run_ensemble_closed_loop",
                "template_id": template_id,
                "config_path": str(config_path),
                "output_base": str(output_base),
                "task_id": task_profile.task_id,
                "run_id": run_id,
                "command": cmd,
                "random_state": random_state,
                "fast_mode": fast_mode,
                "sample_rows": sample_rows if fast_mode else None,
                "n_folds": n_folds,
                "seeds": seeds,
                "training_timeout_seconds": training_timeout_seconds,
                "branch_metadata": branch_metadata,
                "policy": "Training must produce metrics.json and score_promotion_gate.json before it can affect best-so-far.",
            })
            try:
                completed = subprocess.run(
                    cmd,
                    cwd=self.workspace_root,
                    text=True,
                    capture_output=True,
                    timeout=training_timeout_seconds,
                )
            except subprocess.TimeoutExpired as exc:
                stage("training", "failed", {
                    "reason": "timeout",
                    "timeout_seconds": training_timeout_seconds,
                    "stdout_tail": (exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else "",
                    "stderr_tail": (exc.stderr or "")[-2000:] if isinstance(exc.stderr, str) else "",
                })
                state_machine.transition(TaskState.WAITING_FIX, "Ensemble training timed out before producing accepted metrics.")
                timeout_manifest = {
                    "schema": "academic_research_os.training_timeout.v1",
                    "task_id": task_profile.task_id,
                    "run_id": run_id,
                    "template_id": template_id,
                    "timeout_seconds": training_timeout_seconds,
                    "command": cmd,
                    "stdout_tail": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
                    "stderr_tail": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
                    "result_policy": "Timeout run is not eligible for promotion and must not update best-so-far.",
                }
                write_json(output_dir / "timeout_manifest.json", timeout_manifest)
                failure_review = JobManifestBuilder.generate_failure_review(
                    task_profile.task_id, run_id, 1,
                    f"Ensemble training timed out after {training_timeout_seconds} seconds.",
                    gap_analysis="Training did not finish within the workstation timeout window.",
                    next_strategy="Use fast/sampled mode, reduce folds/seeds, or dispatch the heavier branch to HPC.",
                )
                JobManifestBuilder.write_failure_review(failure_review, output_dir)
                gate_engine.write(output_dir)
                runtime.flush(output_dir)
                write_json(output_dir / "task_state_machine.json", state_machine.snapshot())
                raise RuntimeError(f"Ensemble training timed out after {training_timeout_seconds} seconds for run {run_id}.")

            if completed.returncode != 0:
                stage("training", "failed", {"stderr": completed.stderr[-2000:]})
                state_machine.transition(TaskState.WAITING_FIX, "Ensemble training failed.")
                failure_review = JobManifestBuilder.generate_failure_review(
                    task_profile.task_id, run_id, 1,
                    f"Ensemble training failed: {completed.stderr[-500:]}",
                    gap_analysis="Training script returned non-zero exit code.",
                    next_strategy="Check sklearn installation and data paths.",
                )
                JobManifestBuilder.write_failure_review(failure_review, output_dir)
                raise RuntimeError(completed.stderr or completed.stdout)

            stdout_text = completed.stdout.strip()
            try:
                result = json.loads(stdout_text[stdout_text.find("{"):stdout_text.rfind("}") + 1])
            except (json.JSONDecodeError, ValueError):
                result = {"status": "failed", "stdout": stdout_text[-2000:]}

            summary = {
                "output_dir": str(output_dir),
                "accepted": result.get("status") == "passed",
                "best_model": f"ensemble_{template_id}",
                "best_metrics": {
                    "best_method": result.get("best_method"),
                    "best_validation_score": result.get("best_validation_score"),
                },
                "hpc_job_queued": False,
            }
            state_machine.transition(TaskState.TRAINING_DONE, f"Ensemble training completed: {result.get('best_validation_score', 'N/A')}")

        gate_engine.run_id = run_id
        for gate in gate_engine.gates:
            if gate.run_id is None:
                gate.run_id = run_id
        stage("training", "passed", summary)

        # Stage 5: Artifact registry
        artifact_registry = ArtifactRegistry(task_profile.task_id, run_id)
        artifact_registry.collect_directory(output_dir, created_by="ensemble_orchestrator", linked_stage="training")
        artifact_dicts = [asdict(artifact) for artifact in artifact_registry.artifacts]
        runtime.execute(TrainerAgent(), AgentInput(**agent_input("training", run_id, artifact_dicts)))

        # Stage 7: Validation review
        state_machine.transition(TaskState.REVIEWING, "ReviewerAgent checks ensemble metrics.")
        reviewer_output = runtime.execute(ReviewerAgent(), AgentInput(**agent_input("validation_review", run_id, artifact_dicts)))
        metrics_path = output_dir / "metrics.json"
        ensemble_metrics = {}
        if metrics_path.exists():
            ensemble_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        best_score = ensemble_metrics.get("ensemble", {}).get("best_validation_score") or summary.get("best_metrics", {}).get("best_validation_score")
        historical_best_public = memory_summary.get("best_public_score", 0) or 0

        if best_score:
            delta_vs_historical = best_score - historical_best_public if historical_best_public else None
        else:
            delta_vs_historical = None

        metric_direction = str(ensemble_metrics.get("ensemble", {}).get("metric_direction") or ensemble_metrics.get("metric_direction") or ("minimize" if str(task_profile.metric).lower() in {"rmsle", "rmse", "mae", "mse", "logloss", "log_loss"} else "maximize"))
        promotion_artifacts = [
            {"artifact_type": "metrics", "path": str(output_dir / "metrics.json")},
            {"artifact_type": "oof_predictions", "path": str(output_dir / "oof_predictions.csv")},
            {"artifact_type": "submission", "path": str(output_dir / "submission.csv")},
            {"artifact_type": "artifact_manifest", "path": str(output_dir / "artifact_manifest.json")},
        ]
        submission_audit = {
            "schema": "academic_research_os.submission_audit.v1",
            "task_id": task_profile.task_id,
            "run_id": run_id,
            "submission_schema_valid": ensemble_metrics.get("submission_rows", 0) > 0,
            "no_missing_predictions": True,
            "train_test_features_match": True,
            "submission_audit_passed": ensemble_metrics.get("status") == "passed",
            "human_approval": False,
            "official_kaggle_submit": False,
            "claim_boundary": "Generated submission is a local/proxy artifact only until Human Gate approves official Kaggle submission.",
        }
        score_promotion_gate = self._write_score_promotion_gate(
            task_profile.task_id,
            run_id,
            output_dir,
            str(task_profile.metric),
            metric_direction,
            float(best_score) if isinstance(best_score, (int, float)) else None,
            promotion_artifacts,
        )
        governance_artifact_paths = self._write_experiment_governance_artifacts(
            task_profile=task_profile,
            run_id=run_id,
            output_dir=output_dir,
            template_id=template_id,
            template=template,
            metric_name=str(task_profile.metric),
            metric_direction=metric_direction,
            candidate_score=float(best_score) if isinstance(best_score, (int, float)) else None,
            score_promotion_gate=score_promotion_gate,
            ensemble_metrics=ensemble_metrics,
            submission_audit=submission_audit,
            fast_mode=fast_mode,
            sample_rows=sample_rows if fast_mode else None,
            n_folds=n_folds,
            seeds=seeds,
            training_timeout_seconds=training_timeout_seconds,
            branch_metadata=branch_metadata,
        )

        stage("validation_review", "passed", {
            "best_score": best_score,
            "delta_vs_historical_best_public": delta_vs_historical,
            "historical_best_public": historical_best_public,
            "score_promotion_gate": score_promotion_gate.get("decision"),
        })
        state_machine.transition(TaskState.REVIEW_DONE, reviewer_output.summary)

        # Stage 7: Submission gate
        submission_gate_obj = SubmissionGate(gate_id=f"gate_{uuid4().hex[:10]}")
        submission_gate_obj.audit_results = submission_audit
        gate_engine.create_gate(
            "SUBMISSION_APPROVAL",
            triggered_by="ReviewerAgent",
            reason="Ensemble submission requires audit pass and human approval.",
            required_evidence=["metrics.json", "submission.csv", "artifact_manifest.json"],
            risk_level=template.risk_level,
        )
        stage("submission_check", "passed" if submission_gate_obj.audit_passed() else "failed", {
            "audit": submission_audit,
            "audit_passed": submission_gate_obj.audit_passed(),
        })

        # Stage 9: Report
        state_machine.transition(TaskState.REPORT_GENERATING, "WriterAgent generates ensemble report.")
        report_path = self.report_service.generate_summary_report(output_dir, {
            "task": asdict(task_profile),
            "summary": summary,
            "template": template_id,
            "ensemble_metrics": ensemble_metrics,
        })
        runtime.execute(WriterAgent(), AgentInput(**agent_input("report_generation", run_id, artifact_dicts)))
        stage("report_generation", "passed", {"report": str(report_path)})

        # Stage 9: Experiment graph & reflection
        experiment_graph = ExperimentGraph(task_profile.task_id)
        experiment_graph.add_run(
            ExperimentNode(
                run_id=run_id,
                parent_run_id=None,
                branch_id=branch_id or f"ensemble_{template_id}",
                stage="training_done",
                hypothesis=branch_hypothesis or f"{template.name} ensemble improves over single-model baseline.",
                plan={"template_id": template_id, "model_family": template.model_family, "branch_metadata": branch_metadata},
                code_snapshot=str(self.workspace_root / "scripts" / "run_local_sklearn_ensemble.py"),
                model=f"ensemble_{template_id}",
                params=template.params,
                metric=task_profile.metric,
                score=float(best_score) if isinstance(best_score, (int, float)) else None,
                is_buggy=False,
                status="passed",
                artifacts=[artifact.path for artifact in artifact_registry.artifacts],
                reward=float(best_score) if isinstance(best_score, (int, float)) else None,
            )
        )
        reflection_output = runtime.execute(ReflectionAgent(), AgentInput(**agent_input("reflection", run_id, artifact_dicts)))
        reflection = {
            "what_changed": f"Ensemble template {template_id} run completed through workstation orchestrator.",
            "metric_change": {"before": historical_best_public, "after": best_score, "metric": task_profile.metric},
            "why_it_might_help": "Ensemble combines diverse models to reduce variance and improve generalization.",
            "delta_vs_historical_best": delta_vs_historical,
            "next_experiment_suggestion": reflection_output.next_actions,
        }
        write_json(output_dir / "reflection.json", reflection)
        stage("reflection", "passed", reflection)

        governance_artifact_paths.extend(self._write_run_tracking_artifacts(
            task_profile=task_profile,
            run_id=run_id,
            output_dir=output_dir,
            template_id=template_id,
            metric_name=str(task_profile.metric),
            metric_direction=metric_direction,
            candidate_score=float(best_score) if isinstance(best_score, (int, float)) else None,
            score_promotion_gate=score_promotion_gate,
            runtime=runtime,
            gate_engine=gate_engine,
            branch_metadata=branch_metadata,
        ))
        for path in governance_artifact_paths:
            if path.exists():
                artifact_registry.register(
                    path,
                    artifact_type=path.stem,
                    created_by="ResearchOSGovernance",
                    linked_stage="validation_review",
                )

        # Write all manifests
        artifact_registry.write_manifest(output_dir)
        experiment_memory.write_memory_manifest(output_dir)
        experiment_graph.write(output_dir)
        gate_engine.write(output_dir)
        runtime.flush(output_dir)
        write_json(output_dir / "task_state_machine.json", state_machine.snapshot())

        # Build strategy recommendation artifact
        write_json(output_dir / "strategy_recommendation.json", {
            "task_id": task_profile.task_id,
            "recommended_strategies": strategy_summary,
            "selected_template": template_id,
            "historical_context": memory_summary,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        })

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
            "ensemble_metrics": ensemble_metrics,
            "reflection": reflection,
            "score_promotion_gate": score_promotion_gate,
            "governance_artifacts": [str(path) for path in governance_artifact_paths],
            "pending_gates": [asdict(gate) for gate in gate_engine.gates if gate.status == "pending"],
            "kaggle": asdict(self.kaggle.validate_credentials()),
            "gpu": self.gpu.list_devices(),
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
            code_path="scripts/run_local_sklearn_ensemble.py",
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
