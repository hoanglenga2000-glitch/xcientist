from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from xsci import multi_agent_cli

LLM_REQUEST = (
    "请用本地 EvoMind 文档，让一个成熟的 7B 中文大模型更懂科研工作流；"
    "在远程 A40 上完成训练和评测，不要使用本地 GPU，也不要发布模型。"
)

LOCAL_GPU_REQUEST = (
    "请创建信用卡交易欺诈检测研究任务，使用 fraudTrain.csv 和 fraudTest.csv，"
    "科研编排使用 GPT-5.6，训练仅使用本机 RTX 4060，不使用远程集群，不提交公开榜单。"
)

SIIM_REQUEST = (
    "请使用 SIIM-ISIC Melanoma Classification 数据，在 HPC A800 上完成患者分组训练、"
    "独立复核并生成报告和证据包；不使用本机 GPU，不提交公开榜单。"
)


def _completed(run_id: str):
    return SimpleNamespace(
        status="completed",
        run_id=run_id,
        seq=9,
        open_requirements=[],
    )


def test_run_dispatches_llm_request_to_finetune_workflow(tmp_path, monkeypatch):
    request_path = tmp_path / "request.txt"
    request_path.write_text(LLM_REQUEST, encoding="utf-8")
    calls = []

    monkeypatch.setattr(multi_agent_cli, "active_root", lambda: tmp_path)
    monkeypatch.setattr(
        multi_agent_cli,
        "run_llm_finetune",
        lambda root, request, *, run_id: calls.append((root, request, run_id)) or _completed(run_id),
    )
    monkeypatch.setattr(
        multi_agent_cli,
        "run_titanic_aibuild",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Titanic workflow must not run")),
    )

    exit_code = multi_agent_cli.main([
        "run",
        "--request-file",
        str(request_path),
        "--run-id",
        "evomind_run_llm_dispatch",
        "--hpc-job-id",
        "91051",
        "--hpc-credential-profile",
        "job91051",
        "--hpc-resource-profile",
        "aimslab_a800_80gb",
        "--execution-backend",
        "hpc",
    ])

    assert exit_code == 0
    assert calls[0][0] == tmp_path
    assert calls[0][1].task_type == "llm_finetune"
    assert calls[0][1].compute_policy.job_id == 91051
    assert calls[0][1].compute_policy.credential_profile == "job91051"
    assert calls[0][2] == "evomind_run_llm_dispatch"


def test_resume_dispatches_existing_llm_run_to_finetune_workflow(tmp_path, monkeypatch):
    run_id = "evomind_run_llm_resume"
    run_dir = tmp_path / "workspace" / "evomind_runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "request.json").write_text(
        json.dumps({"task_type": "llm_finetune"}),
        encoding="utf-8",
    )
    calls = []

    monkeypatch.setattr(multi_agent_cli, "active_root", lambda: tmp_path)
    monkeypatch.setattr(
        multi_agent_cli,
        "resume_llm_finetune",
        lambda root, selected_run_id: calls.append((root, selected_run_id)) or _completed(selected_run_id),
    )
    monkeypatch.setattr(
        multi_agent_cli,
        "resume_titanic_aibuild",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Titanic workflow must not resume")),
    )

    exit_code = multi_agent_cli.main(["resume", "--run-id", run_id])

    assert exit_code == 0
    assert calls == [(tmp_path, run_id)]


def test_run_dispatches_credit_card_request_to_local_gpu_workflow(tmp_path, monkeypatch):
    request_path = tmp_path / "request.txt"
    request_path.write_text(LOCAL_GPU_REQUEST, encoding="utf-8")
    calls = []

    monkeypatch.setattr(multi_agent_cli, "active_root", lambda: tmp_path)
    monkeypatch.setattr(
        multi_agent_cli,
        "run_local_tabular_research",
        lambda root, request, *, run_id: calls.append((root, request, run_id)) or _completed(run_id),
    )
    monkeypatch.setattr(
        multi_agent_cli,
        "run_titanic_aibuild",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Titanic workflow must not run")),
    )

    exit_code = multi_agent_cli.main([
        "run",
        "--request-file",
        str(request_path),
        "--run-id",
        "evomind_run_local4060_dispatch",
    ])

    assert exit_code == 0
    assert calls[0][0] == tmp_path
    assert calls[0][1].task_type == "tabular_classification"
    assert calls[0][1].compute_policy.backend == "local_gpu"
    assert calls[0][2] == "evomind_run_local4060_dispatch"


def test_run_persists_explicit_siim_allocation_before_dispatch(tmp_path, monkeypatch):
    request_path = tmp_path / "request.txt"
    request_path.write_text(SIIM_REQUEST, encoding="utf-8")
    calls = []

    monkeypatch.delenv("EVOMIND_SIIM_HPC_JOB_ID", raising=False)
    monkeypatch.delenv("EVOMIND_HPC_CREDENTIAL_PROFILE", raising=False)
    monkeypatch.setattr(multi_agent_cli, "active_root", lambda: tmp_path)
    monkeypatch.setattr(
        multi_agent_cli,
        "run_siim_hpc_research",
        lambda root, request, *, run_id: calls.append((root, request, run_id)) or _completed(run_id),
    )

    exit_code = multi_agent_cli.main([
        "run",
        "--request-file",
        str(request_path),
        "--run-id",
        "evomind_siim_job90353_dispatch",
        "--hpc-job-id",
        "90353",
        "--hpc-credential-profile",
        "job90353",
        "--hpc-resource-profile",
        "aimslab_a800_80gb",
        "--execution-backend",
        "hpc",
    ])

    assert exit_code == 0
    request = calls[0][1]
    assert request.compute_policy.job_id == 90353
    assert request.compute_policy.credential_profile == "job90353"
    assert request.compute_policy.resource_profile == "aimslab_a800_80gb"
    assert request.compute_policy.backend == "hpc"
    assert request.compute_policy.local_gpu_allowed is False
    assert request.compute_policy.remote_gpu_required is True


def test_run_rejects_siim_without_durable_allocation_binding(tmp_path, monkeypatch):
    request_path = tmp_path / "request.txt"
    request_path.write_text(SIIM_REQUEST, encoding="utf-8")
    monkeypatch.delenv("EVOMIND_SIIM_HPC_JOB_ID", raising=False)
    monkeypatch.delenv("EVOMIND_HPC_CREDENTIAL_PROFILE", raising=False)
    monkeypatch.setattr(multi_agent_cli, "active_root", lambda: tmp_path)

    with pytest.raises(ValueError, match="Missing HPC execution contract"):
        multi_agent_cli.main([
            "run",
            "--request-file",
            str(request_path),
            "--run-id",
            "evomind_siim_missing_binding",
        ])


def test_run_rejects_partial_siim_execution_contract(tmp_path, monkeypatch):
    request_path = tmp_path / "request.txt"
    request_path.write_text(SIIM_REQUEST, encoding="utf-8")
    monkeypatch.setattr(multi_agent_cli, "active_root", lambda: tmp_path)

    with pytest.raises(ValueError, match="Missing HPC execution contract"):
        multi_agent_cli.main([
            "run",
            "--request-file",
            str(request_path),
            "--run-id",
            "evomind_siim_partial_binding",
            "--hpc-job-id",
            "90353",
            "--hpc-credential-profile",
            "job90353",
        ])


def test_run_persists_explicit_generic_hpc_allocation_before_titanic_dispatch(tmp_path, monkeypatch):
    request_path = tmp_path / "request.txt"
    request_path.write_text(
        "请使用本地 Titanic 数据，在 HPC 上训练三个候选并生成证据；不要使用本地 GPU，不提交 Kaggle。",
        encoding="utf-8",
    )
    calls = []
    monkeypatch.setattr(multi_agent_cli, "active_root", lambda: tmp_path)
    monkeypatch.setattr(
        multi_agent_cli,
        "run_titanic_aibuild",
        lambda root, request, *, run_id: calls.append((root, request, run_id)) or _completed(run_id),
    )

    exit_code = multi_agent_cli.main([
        "run",
        "--request-file", str(request_path),
        "--run-id", "evomind_titanic_job91051_dispatch",
        "--hpc-job-id", "91051",
        "--hpc-credential-profile", "job91051",
        "--hpc-resource-profile", "aimslab_a800_80gb",
        "--execution-backend", "hpc",
    ])

    assert exit_code == 0
    request = calls[0][1]
    assert request.compute_policy.job_id == 91051
    assert request.compute_policy.credential_profile == "job91051"
    assert request.compute_policy.resource_profile == "aimslab_a800_80gb"
    assert request.compute_policy.backend == "hpc"


def test_run_rejects_generic_hpc_request_without_contract(tmp_path, monkeypatch):
    request_path = tmp_path / "request.txt"
    request_path.write_text(
        "请使用本地 Titanic 数据，在 HPC 上训练三个候选并生成证据；不要使用本地 GPU。",
        encoding="utf-8",
    )
    monkeypatch.setattr(multi_agent_cli, "active_root", lambda: tmp_path)

    with pytest.raises(ValueError, match="Missing HPC execution contract"):
        multi_agent_cli.main([
            "run",
            "--request-file", str(request_path),
            "--run-id", "evomind_titanic_missing_contract",
        ])
