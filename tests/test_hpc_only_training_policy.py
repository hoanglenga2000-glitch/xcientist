from __future__ import annotations

import inspect
import importlib.util
import json
import os
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


def test_training_readiness_reports_local_evidence_without_claiming_release() -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / "scripts/verify_training_optimization_readiness.py")],
        cwd=root,
        env={**os.environ, "PYTHONIOENCODING": "cp1252"},
        text=True,
        capture_output=True,
        check=False,
    )

    payload = json.loads(completed.stdout)
    assert completed.returncode == 0
    assert payload["overall_status"] == "passed"
    assert payload["completion_rate_percent"] == 100.0
    assert payload["ready_task_count"] == payload["required_task_count"] == 3
    assert "release" not in payload


def test_final_two_resource_gate_accepts_strict_hpc_evidence() -> None:
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "verify_final_two_resource_blockers_contract",
        root / "scripts/verify_final_two_resource_blockers.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    launch = {
        "external_resources": {
            "code_agent": {"configured": True, "missing_keys": []},
            "gpu_ssh_gateway": {"configured": False, "missing_keys": []},
            "hpc_gpu_strict_runtime": {
                "configured": True,
                "state": "strict_hpc_runtime_verified",
                "missing_keys": [],
            },
            "kaggle_official_api_optional": {"missing_keys": []},
        }
    }
    assert module.blocker_groups_from_launch(launch) == []


def _load_launch_resource_module(root: Path):
    spec = importlib.util.spec_from_file_location(
        "verify_launch_resource_readiness",
        root / "scripts" / "verify_launch_resource_readiness.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_job90948_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, tamper_smoke: bool = False, direct_gateway: bool = False) -> Path:
    root = tmp_path / "repo"
    hpc = root / "workspace" / "hpc"
    hpc.mkdir(parents=True)
    appdata = tmp_path / "appdata"
    profile_dir = appdata / "ResearchAgentWorkstation" / "profiles" / "job90948"
    profile_dir.mkdir(parents=True)
    monkeypatch.setenv("APPDATA", str(appdata))
    monkeypatch.setenv("EVOMIND_HPC_CREDENTIAL_PROFILE", "job90948")

    metadata = {
        "schema": "evomind.hpc.dpapi_profile.v2",
        "credential_profile": "job90948",
        "profile_state": "active",
        "job_id": 90948,
        "host": "10.120.18.240" if direct_gateway else "100.85.169.63",
        "port": 6988 if direct_gateway else 1235,
        "socks_host": "127.0.0.1",
        "socks_port": 7890,
        "remote_workspace": "/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra",
        "expected_host_uuid": "c5a13e3d9f534a35b7964f28501a404f",
        "expected_gpu_uuid": "GPU-acca073d-419d-3cc8-d15f-9d8fd72e6a96",
        "container_binding_sha256": "fixture-binding",
        "allocation_inner_endpoint": "10.120.18.240:6988",
    }
    (profile_dir / "hpc_ssh_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    readiness = {
        "schema": "evomind.hpc.profile_readiness.v1",
        "profile": "job90948",
        "status": "ready",
        "failed_checks": [],
        "details": {
            "profile": "job90948",
            "job_id": 90948,
            "gateway_host": "100.85.169.63",
            "gateway_port": 1235,
            "socks_host": "127.0.0.1",
            "socks_port": 7890,
            "remote_workspace": "/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra",
        },
    }
    (hpc / "job90948_profile_readiness_current.json").write_text(json.dumps(readiness), encoding="utf-8")

    evidence = tmp_path / "evidence"
    evidence.mkdir()
    collection = {
        "profile": "job90948",
        "job_id": 90948,
        "run_id": "job90948-driver-test",
        "remote_run_dir": "/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/evomind_job90948_smoke/job90948-driver-test",
        "identity_gate": {
            "job_id": 90948,
            "credential_profile": "job90948",
            "host_uuid": "c5a13e3d9f534a35b7964f28501a404f",
            "gpu_uuids": ["GPU-acca073d-419d-3cc8-d15f-9d8fd72e6a96"],
            "gpu_name": "NVIDIA A800-SXM4-80GB",
            "gpu_memory_total_mib": 81920,
            "remote_root": "/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra",
            "designated_proxy_path_verified": True,
            "pinned_host_key_verified": True,
            "job_container_verified": True,
        },
        "gpu_smoke": {
            "status": "passed",
            "job_id": 90948,
            "credential_profile": "job90948",
            "device_name": "NVIDIA A800-SXM4-80GB",
            "gpu_uuid": "GPU-acca073d-419d-3cc8-d15f-9d8fd72e6a96",
            "remote_write_boundary_ok": True,
            "residual_running_processes": 0,
            "training_started": tamper_smoke,
            "signals_sent": 0,
            "other_processes_modified": False,
            "run_dir": "/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/evomind_job90948_smoke/job90948-driver-test",
        },
        "signals_sent": 0,
        "other_processes_modified": False,
    }
    collection_path = evidence / "collection.json"
    collection_path.write_text(json.dumps(collection), encoding="utf-8")
    for name in ("gpu_smoke.json", "container_identity.json"):
        (evidence / name).write_text(json.dumps({"ok": True}), encoding="utf-8")
    (evidence / "nvidia_smi_samples.jsonl").write_text("{}\n{}\n{}\n{}\n{}\n", encoding="utf-8")

    import hashlib

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    pointer = {
        "schema": "evomind.hpc.bounded_smoke_pointer.v1",
        "profile": "job90948",
        "job_id": 90948,
        "evidence_root": str(evidence),
        "collection_path": str(collection_path),
        "collection_sha256": digest(collection_path),
        "required_files": {
            "gpu_smoke.json": digest(evidence / "gpu_smoke.json"),
            "nvidia_smi_samples.jsonl": digest(evidence / "nvidia_smi_samples.jsonl"),
            "container_identity.json": digest(evidence / "container_identity.json"),
        },
    }
    (hpc / "job90948_bounded_smoke_current.json").write_text(json.dumps(pointer), encoding="utf-8")
    return root


def test_launch_resource_strict_hpc_runtime_accepts_profile_probe_and_bounded_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_launch_resource_module(repo_root)
    fixture_root = _write_job90948_fixture(tmp_path, monkeypatch)

    status = module.strict_hpc_runtime_status(
        fixture_root,
        live_probe=lambda profile, job_id, samples: {
            "ok": True,
            "status": "job_container_verified",
            "job_container_verified": True,
            "samples_passed": 5,
            "signals_sent": 0,
            "other_processes_modified": False,
        },
    )

    assert status["configured"] is True
    assert status["bounded_smoke"]["status"] == "passed"


@pytest.mark.parametrize("case", ["tampered_smoke", "direct_gateway", "legacy_only"])
def test_launch_resource_strict_hpc_runtime_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_launch_resource_module(repo_root)
    if case == "legacy_only":
        fixture_root = tmp_path / "repo"
        (fixture_root / "workspace" / "hpc").mkdir(parents=True)
        (fixture_root / "workspace" / "hpc" / "web_terminal_probe.txt").write_text("NVIDIA A800 " * 4, encoding="utf-8")
        monkeypatch.setenv("APPDATA", str(tmp_path / "missing-appdata"))
    else:
        fixture_root = _write_job90948_fixture(
            tmp_path,
            monkeypatch,
            tamper_smoke=case == "tampered_smoke",
            direct_gateway=case == "direct_gateway",
        )

    status = module.strict_hpc_runtime_status(
        fixture_root,
        live_probe=lambda profile, job_id, samples: {
            "ok": True,
            "status": "job_container_verified",
            "job_container_verified": True,
            "samples_passed": 5,
            "signals_sent": 0,
            "other_processes_modified": False,
        },
    )

    assert status["configured"] is False
    assert status["state"] == "blocked_hpc_runtime_verification_required" or status["state"] == "no_active_named_dpapi_profile"
    assert status.get("secrets_returned") is not True


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
