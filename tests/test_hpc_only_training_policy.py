from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest

from research_agent_workstation.server.services.agent_orchestrator import AgentOrchestrator
from research_agent_workstation.server.training.ensemble_templates import EnsembleTemplateRegistry
from research_agent_workstation.server.training.job_manifest import JobManifestBuilder
from research_os.hpc_policy import (
    HPCPolicyError,
    require_hpc_compute,
    require_remote_workspace,
    validate_remote_workspace,
)


@pytest.mark.parametrize("value", ["", " ", "/", "/home", "/root", "relative/path", "/safe/../escape"])
def test_remote_workspace_rejects_missing_shared_or_unsafe_paths(value: str) -> None:
    with pytest.raises(HPCPolicyError):
        validate_remote_workspace(value)


@pytest.mark.parametrize("value", ["/srv/evomind", "/workspace/evomind", "~/evomind"])
def test_remote_workspace_accepts_dedicated_generic_paths(value: str) -> None:
    assert validate_remote_workspace(value) == value


def test_remote_workspace_requires_explicit_environment() -> None:
    with pytest.raises(HPCPolicyError, match="configured explicitly"):
        require_remote_workspace({})


def test_training_policy_allows_only_gpu() -> None:
    require_hpc_compute("gpu")
    with pytest.raises(HPCPolicyError, match="Local training is disabled"):
        require_hpc_compute("local")


def test_all_approved_ensemble_templates_require_hpc() -> None:
    approved = EnsembleTemplateRegistry.list_approved()
    assert approved
    assert all(template.hpc_required for template in approved)
    assert EnsembleTemplateRegistry.list_local() == []


def test_job_manifest_rejects_shared_remote_root(tmp_path) -> None:
    builder = JobManifestBuilder(tmp_path)
    with pytest.raises(HPCPolicyError):
        builder.build(
            task_id="task",
            run_id="run",
            agent_id="agent",
            template_id="template",
            remote_workspace="/root",
        )


def test_job_manifest_cannot_be_queued_without_matching_dispatch_receipt(tmp_path) -> None:
    builder = JobManifestBuilder(tmp_path)
    manifest = builder.build(
        task_id="task",
        run_id="run",
        agent_id="agent",
        template_id="template",
        remote_workspace="/srv/evomind/task",
    )

    assert manifest.status == "manifest_prepared"
    assert manifest.job_id is None
    with pytest.raises(ValueError, match="dispatch_receipt job id"):
        builder.mark_queued(
            manifest,
            remote_job_id="remote-1",
            dispatch_receipt={"job_id": "remote-2", "status": "accepted"},
        )

    builder.mark_queued(
        manifest,
        remote_job_id="remote-1",
        dispatch_receipt={"job_id": "remote-1", "status": "accepted"},
    )
    manifest_path = builder.write(manifest, tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["status"] == "queued"
    assert payload["job_id"] == "remote-1"
    assert payload["dispatch_receipt"]["status"] == "accepted"


def test_direct_local_orchestrator_call_fails_before_reading_config(tmp_path) -> None:
    orchestrator = AgentOrchestrator(tmp_path)

    with pytest.raises(HPCPolicyError, match="blocked_local_training_disabled"):
        orchestrator.run_local_tabular_closed_loop(tmp_path / "missing.yaml")

    assert not (tmp_path / "experiments").exists()
    assert not (tmp_path / "workspace").exists()


def test_ensemble_orchestrator_stops_at_manifest_prepared(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "task.yaml"
    config_path.write_text(
        """
task:
  name: queue_contract
  competition: Queue Contract
  type: classification
  target: target
  metric: accuracy
data:
  task_dir: tasks/queue_contract
  train: tasks/queue_contract/data/train.csv
  test: tasks/queue_contract/data/test.csv
  sample_submission: tasks/queue_contract/data/sample_submission.csv
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("EVOMIND_HPC_REMOTE_WORKSPACE", "/srv/evomind/queue-contract")

    def reject_subprocess(*args, **kwargs):
        pytest.fail(f"local subprocess training was invoked: args={args!r}, kwargs={kwargs!r}")

    monkeypatch.setattr(subprocess, "run", reject_subprocess)
    output_base = tmp_path / "experiments"
    summary = AgentOrchestrator(tmp_path).run_ensemble_closed_loop(
        config_path,
        output_base=output_base,
        training_timeout_seconds=1234,
    )

    run = summary["run"]
    output_dir = Path(run["output_dir"])
    manifest = json.loads(Path(run["job_manifest"]).read_text(encoding="utf-8"))
    states = [item["to_state"] for item in summary["task_state"]["history"]]

    assert summary["status"] == "manifest_prepared_awaiting_dispatch"
    assert summary["task_state"]["state"] == "MANIFEST_PREPARED"
    assert run["training_started"] is False
    assert run["accepted"] is False
    assert run["hpc_job_queued"] is False
    assert run["manifest_prepared"] is True
    assert run["remote_job_id"] is None
    assert run["dispatch_receipt"] is None
    assert run["best_model"] is None
    assert run["best_metrics"] == {}
    assert manifest["remote_workspace"] == "/srv/evomind/queue-contract"
    assert manifest["timeout"] == 1234
    assert manifest["status"] == "manifest_prepared"
    assert manifest["job_id"] is None
    assert manifest["dispatch_receipt"] is None
    assert "TRAINING_QUEUED" not in states
    assert "TRAINING_RUNNING" not in states
    assert "TRAINING_DONE" not in states
    assert not (output_dir / "metrics.json").exists()
    assert not (output_dir / "submission.csv").exists()
    assert not (output_dir / "launcher_manifest.json").exists()
    pending_gate = summary["pending_gates"][0]
    assert pending_gate["status"] == "pending"
    assert pending_gate["reviewer"] is None
    assert pending_gate["decided_at"] is None


def test_ensemble_orchestrator_contains_no_local_subprocess_branch() -> None:
    source = inspect.getsource(AgentOrchestrator.run_ensemble_closed_loop)

    assert "subprocess" not in source
    assert "run_local_sklearn_ensemble" not in source
    assert "Research Admin" not in source
    assert "state_machine.transition(TaskState.TRAINING_QUEUED" not in source
    assert "state_machine.transition(TaskState.TRAINING_RUNNING" not in source
    assert "state_machine.transition(TaskState.TRAINING_DONE" not in source


def test_retired_local_runner_cli_reports_blocked(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "run_workstation_orchestrator.py"),
            "--config",
            str(tmp_path / "missing.yaml"),
            "--output-base",
            str(tmp_path / "experiments"),
        ],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )

    payload = json.loads(completed.stdout)
    assert completed.returncode == 2
    assert payload["status"] == "blocked_local_training_disabled"
    assert payload["training_started"] is False
    assert payload["hpc_queue_command"][1:3] == [
        "scripts/run_workstation_ensemble.py",
        "--config",
    ]
    assert not (tmp_path / "experiments").exists()


def test_xsci_public_parser_is_gpu_only_and_defaults_gpu() -> None:
    from xsci.__main__ import _build_parser

    parser = _build_parser()
    assert parser.parse_args(["init"]).compute == "gpu"
    assert parser.parse_args(["run", "task"]).compute == "gpu"
    assert parser.parse_args(["agent", "task"]).compute == "gpu"
    for argv in (
        ["init", "--compute", "local"],
        ["run", "task", "--compute", "local"],
        ["agent", "task", "--compute", "local"],
    ):
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(argv)
        assert exc_info.value.code == 2


def test_project_init_rejects_local_before_writing(tmp_path: Path) -> None:
    from xsci.project import run_init

    with pytest.raises(HPCPolicyError, match="Local training is disabled"):
        run_init(tmp_path, compute="local")

    assert list(tmp_path.iterdir()) == []


def test_packaged_tabular_pipeline_public_run_fails_closed(tmp_path: Path) -> None:
    from research_agent_workstation.tabular_pipeline import run

    with pytest.raises(HPCPolicyError, match="blocked_local_training_disabled"):
        run({}, tmp_path / "output", 42)

    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    ("relative", "arguments"),
    [
        (
            "scripts/run_local_sklearn_ensemble.py",
            ["--config", "missing.yaml", "--output-base", "never-created"],
        ),
        (
            "src/research_agent_workstation/tabular_pipeline.py",
            ["--config", "missing.yaml", "--output-dir", "never-created"],
        ),
    ],
)
def test_source_local_training_commands_fail_closed(
    tmp_path: Path,
    relative: str,
    arguments: list[str],
) -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / relative), *arguments],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    payload = json.loads(completed.stdout)
    assert completed.returncode == 2
    assert payload["status"] == "blocked_local_training_disabled"
    assert payload["training_started"] is False
    assert not (tmp_path / "never-created").exists()


@pytest.mark.parametrize(
    "relative",
    [
        "scripts/verify_launch_resource_readiness.py",
        "scripts/verify_training_optimization_readiness.py",
        "scripts/verify_final_two_resource_blockers.py",
    ],
)
def test_legacy_local_readiness_commands_cannot_pass_release_gate(relative: str) -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / relative)],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )

    payload = json.loads(completed.stdout)
    assert completed.returncode == 1
    assert payload["overall_status"] == "blocked_hpc_runtime_verification_required"
    assert payload["legacy_local_evidence"] == "not_release_readiness"


def test_built_wheel_public_surfaces_cannot_execute_local_training(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    dist = tmp_path / "dist"
    build = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            ".",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(dist),
        ],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    wheel = next(dist.glob("xcientist-*.whl"))
    probe = f"""
import pathlib, sys
sys.path.insert(0, {str(wheel)!r})
from research_os.evolution_loop import LocalSubprocessRunner
from research_os.hpc_policy import HPCPolicyError
from research_agent_workstation.tabular_pipeline import run
from xsci.__main__ import _build_parser

root = pathlib.Path({str(tmp_path / 'probe')!r})
for call in (
    lambda: LocalSubprocessRunner(root / 'work').run('print(1)', data_dir='x', out_dir=str(root / 'out'), exp_id='e'),
    lambda: run({{}}, root / 'tabular', 42),
):
    try:
        call()
    except HPCPolicyError:
        pass
    else:
        raise SystemExit('local training call unexpectedly succeeded')

parser = _build_parser()
assert parser.parse_args(['init']).compute == 'gpu'
for argv in (['init', '--compute', 'local'], ['run', 't', '--compute', 'local'], ['agent', 't', '--compute', 'local']):
    try:
        parser.parse_args(argv)
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise SystemExit('local CLI option unexpectedly accepted')
assert not root.exists()
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", probe],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
