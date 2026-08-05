from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from research_os.agent.aibuild_v1 import (
    build_aibuild_run,
    read_current_run_pointer,
    run_directory,
    write_current_run_pointer,
)
from research_os.agent.llm_finetune_workflow import (
    _training_source,
    build_document_dataset,
    build_llm_finetune_run,
)
from research_os.agent.multi_agent import MultiAgentStore, validate_task_graph
from research_os.agent.titanic_workflow import _manifest_entries, review_solution_evidence
from research_os.hpc_runtime import HpcRuntime
from xsci.user_request import parse_user_request

TITANIC_REQUEST = (
    "请用本地已有的 Titanic 数据完成一个小型二分类模型开发任务：自动检查数据、提出并比较方案、"
    "在 HPC 上训练、独立审核、生成候选 submission 和研究报告；不要使用本地 GPU，也不要提交 Kaggle。"
)
LLM_REQUEST = (
    "请用本地 EvoMind 文档，让一个成熟的 7B 中文大模型更懂我们的科研工作流；自动准备数据，"
    "在远程 A40 上完成训练和评测，并生成可下载的适配器、模型卡和报告。"
    "不要使用本地 GPU，也不要发布模型。"
)
REMOTE_ROOT = "/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _data_hash(data_dir: Path) -> str:
    paths = [data_dir / "train.csv", data_dir / "test.csv", data_dir / "sample_submission.csv"]
    return hashlib.sha256("".join(_sha256(path) for path in paths).encode("ascii")).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_valid_candidate(run_dir: Path, data_dir: Path, *, run_id: str = "run-review-test") -> tuple[str, str]:
    solution_id = "solution_01"
    solution_dir = run_dir / "solutions" / solution_id
    output_dir = solution_dir / "output"
    output_dir.mkdir(parents=True)
    source = solution_dir / "solution.py"
    source.write_text("print('fixture')\n", encoding="utf-8")
    source_hash = _sha256(source)
    combined_hash = _data_hash(data_dir)
    _write_json(solution_dir / "code_manifest.json", {"source_sha256": source_hash})

    train = pd.read_csv(data_dir / "train.csv").sort_values("PassengerId").reset_index(drop=True)
    test = pd.read_csv(data_dir / "test.csv").reset_index(drop=True)
    oof = pd.DataFrame(
        {
            "PassengerId": train["PassengerId"],
            "Survived": train["Survived"],
            "probability": train["Survived"].astype(float),
            "fold": [index % 5 for index in range(len(train))],
        }
    )
    oof.to_csv(output_dir / "oof_predictions.csv", index=False)
    pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": 0}).to_csv(output_dir / "submission.csv", index=False)
    metrics = {
        "run_id": run_id,
        "data_hash": combined_hash,
        "variant": "linear_baseline",
        "cv_score": 1.0,
        "fold_scores": [1.0] * 5,
        "n_train": len(train),
        "n_test": len(test),
        "oof_rows": len(train),
        "uses_cuda": False,
        "gpu_telemetry": [],
        "source_sha256": source_hash,
        "official_kaggle_score": None,
    }
    environment = {
        "run_id": run_id,
        "data_hash": combined_hash,
        "variant": "linear_baseline",
        "uses_cuda": False,
        "gpu_telemetry": [],
        "source_sha256": source_hash,
    }
    _write_json(output_dir / "metrics.json", metrics)
    _write_json(output_dir / "environment.json", environment)
    (output_dir / "training.log").write_text(
        "".join(f"FOLD={fold} ACCURACY=1.00000000\n" for fold in range(5)) + "CV_SCORE=1.00000000\n",
        encoding="utf-8",
    )
    artifact_names = ("metrics.json", "oof_predictions.csv", "submission.csv", "environment.json", "training.log")
    _write_json(
        solution_dir / "hpc_job.json",
        {
            "status": "completed",
            "run_id": run_id,
            "solution_id": solution_id,
            "remote_dir": f"{REMOTE_ROOT}/evomind_runs/{run_id}/solutions/{solution_id}",
            "exit_code": 0,
            "local_artifacts": [
                {"path": str(output_dir / name), "sha256": _sha256(output_dir / name)} for name in artifact_names
            ],
        },
    )
    return source_hash, combined_hash


def test_aibuild_graph_has_three_isolated_solution_pipelines():
    run = build_aibuild_run(parse_user_request(TITANIC_REQUEST), task_id="titanic", run_id="run-graph-test")

    validate_task_graph(run.tasks, run.roles)
    assert len(run.tasks) == 14
    assert {run.tasks[f"solution_{index:02d}_design"].role for index in range(1, 4)} == {"DesignerAgent"}
    for index in range(1, 4):
        solution_id = f"solution_{index:02d}"
        assert run.tasks[f"{solution_id}_code"].dependencies == (f"{solution_id}_design",)
        assert run.tasks[f"{solution_id}_train"].dependencies == (f"{solution_id}_code",)
    assert run.tasks["independent_review"].dependencies[-3:] == tuple(
        f"solution_{index:02d}_train" for index in range(1, 4)
    )
    assert run.tasks["aggregate"].dependencies == ("independent_review",)
    assert run.roles["IndependentReviewer"].resource_permissions == ("read_run_evidence",)


def test_current_run_pointer_rejects_path_traversal_and_mismatched_run(tmp_path):
    request = parse_user_request(TITANIC_REQUEST)
    run = build_aibuild_run(request, task_id="titanic", run_id="run-pointer-test")
    run_dir = run_directory(tmp_path, run.run_id)
    MultiAgentStore(run_dir).save(run)
    pointer_path = write_current_run_pointer(tmp_path, task_id="titanic", run=run, run_dir=run_dir)

    assert read_current_run_pointer(tmp_path)["run_id"] == run.run_id
    payload = json.loads(pointer_path.read_text(encoding="utf-8"))
    payload["run_dir"] = "../outside"
    pointer_path.write_text(json.dumps(payload), encoding="utf-8")
    assert read_current_run_pointer(tmp_path) is None
    with pytest.raises(ValueError, match="does not match"):
        write_current_run_pointer(tmp_path, task_id="titanic", run=run, run_dir=tmp_path / "wrong")


def test_hpc_runtime_rejects_unsafe_run_ids_before_creating_local_state(tmp_path):
    with pytest.raises(ValueError, match="invalid run_id"):
        HpcRuntime(run_id="../escape", local_run_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_llm_finetune_graph_has_nine_review_gated_nodes():
    request = parse_user_request(LLM_REQUEST)
    run = build_llm_finetune_run(request, run_id="run-llm-graph-test")

    validate_task_graph(run.tasks, run.roles)
    assert list(run.tasks) == [
        "setup",
        "data_audit",
        "training_design",
        "engineering",
        "hpc_train",
        "evaluate",
        "independent_review",
        "claim_audit",
        "synthesis",
    ]
    assert run.tasks["hpc_train"].resource_type == "hpc_gpu"
    assert run.tasks["independent_review"].dependencies == ("evaluate",)
    assert run.tasks["claim_audit"].dependencies == ("independent_review",)
    assert run.tasks["synthesis"].dependencies == ("claim_audit",)
    assert run.roles["IndependentReviewer"].resource_permissions == ("read_run_evidence",)
    assert run.gates["model_publication"] == "forbidden"


def test_llm_base_evaluation_does_not_construct_trainer_for_quantized_model():
    source = _training_source()
    before_adapter = source.split("model = prepare_model_for_kbit_training", maxsplit=1)[0]

    assert "DataLoader(dataset" in before_adapter
    assert "with torch.no_grad():" in before_adapter
    assert "before_loss = evaluation_loss(model" in before_adapter
    assert "Trainer(" not in before_adapter


def test_llm_training_masks_user_tokens_and_restores_best_validation_checkpoint():
    source = _training_source()

    assert 'labels = [-100] * prompt_length + input_ids[prompt_length:]' in source
    assert 'load_best_model_at_end=True' in source
    assert 'metric_for_best_model="eval_loss"' in source
    assert 'save_strategy="epoch"' in source
    assert 'generation_eval_samples = int(config.get("generation_eval_samples", 24))' in source
    assert 'np.linspace(0, len(rows) - 1' in source


def test_document_dataset_is_exact_and_source_disjoint(tmp_path):
    root = Path(__file__).resolve().parents[1]
    manifest = build_document_dataset(root, tmp_path / "data")

    assert manifest["counts"] == {"train": 800, "validation": 100, "test": 100}
    assert all(not values for values in manifest["source_overlap"].values())
    assert manifest["grounded_context_contract"] is True
    for split in ("train", "validation", "test"):
        row = json.loads((tmp_path / "data" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()[0])
        assert row["grounded_context"] is True
        assert "<document_context" in row["prompt"]
        assert row["source_file"] in row["prompt"]
    assert manifest["duplicate_record_ids"] is False
    assert manifest["duplicate_content_records"] is False
    assert len(manifest["data_hash"]) == 64
    for split, count in manifest["counts"].items():
        rows = (tmp_path / "data" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(rows) == count
        assert all("证据单元" not in row for row in rows)


def test_hpc_bundle_rejects_path_traversal_before_connecting(tmp_path):
    source = tmp_path / "input.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    runtime = HpcRuntime(
        run_id="run-bundle-validation",
        local_run_dir=tmp_path / "run",
        connector=lambda: (_ for _ in ()).throw(AssertionError("connector must not be called")),
    )

    with pytest.raises(ValueError, match="relative bundle path"):
        runtime.stage_bundle({"../escape.jsonl": source})


def test_hpc_runtime_dependency_probe_exercises_torch_optimizer_in_run_directory(tmp_path):
    class Stream:
        def __init__(self, payload=b"", exit_code=0):
            self.payload = payload
            self.exit_code = exit_code
            self.channel = self

        def read(self):
            return self.payload

        def recv_exit_status(self):
            return self.exit_code

    class Client:
        def __init__(self):
            self.commands = []
            self.closed = False

        def exec_command(self, command, timeout):
            self.commands.append((command, timeout))
            return None, Stream(), Stream()

        def close(self):
            self.closed = True

    client = Client()
    runtime = HpcRuntime(
        run_id="run-dependency-probe",
        local_run_dir=tmp_path,
        connector=lambda: client,
    )

    result = runtime.prepare_python_environment()

    command, timeout = client.commands[0]
    assert result["status"] == "ready"
    assert result["packages"]["sympy"] == "1.13.3"
    assert result["packages"]["mpmath"] == "1.3.0"
    assert "torch.optim.AdamW" in command
    assert "sympy==1.13.3" in command
    assert "mpmath==1.3.0" in command
    assert runtime.remote_python_deps in command
    assert timeout == 300
    assert client.closed is True


def test_llm_environment_uses_run_scoped_target_without_ensurepip(tmp_path):
    class Stream:
        def __init__(self, payload=b"", exit_code=0):
            self.payload = payload
            self.exit_code = exit_code
            self.channel = self

        def read(self):
            return self.payload

        def recv_exit_status(self):
            return self.exit_code

    class Client:
        def __init__(self):
            self.commands = []
            self.closed = False

        def exec_command(self, command, timeout):
            self.commands.append((command, timeout))
            return None, Stream(b"torch transformers peft\n"), Stream()

        def close(self):
            self.closed = True

    client = Client()
    runtime = HpcRuntime(
        run_id="run-llm-target-deps",
        local_run_dir=tmp_path,
        connector=lambda: client,
    )

    result = runtime.prepare_llm_environment()

    command, timeout = client.commands[0]
    assert result["status"] == "ready"
    assert result["python"] == "python3"
    assert result["remote_python_deps"] == runtime.remote_python_deps
    assert "python3 -m pip install" in command
    assert "--target" in command
    assert "bitsandbytes==0.49.2" in command
    assert "MarkupSafe==3.0.2" in command
    assert "sympy==1.13.3" in command
    assert "numpy==1.26.4" in command
    assert "actual==expected" in command
    assert "import accelerate,bitsandbytes,numpy,peft,sympy,torch,transformers" in command
    assert "HOST_USER_SITE=$(python3 -c 'import site; print(site.getusersitepackages())')" in command
    assert f"HOME={runtime.remote_runtime_home}" in command
    assert f"TRITON_CACHE_DIR={runtime.remote_triton_cache}" in command
    assert f"TORCH_EXTENSIONS_DIR={runtime.remote_torch_extensions}" in command
    assert f"CPATH={runtime.remote_python_headers}/usr/include:" in command
    assert "libpython3.10-dev_3.10.12-1~22.04.16_amd64.deb" in command
    assert "dpkg-deb -x" in command
    assert runtime.remote_python_deps in command
    assert "PYTHONPATH=" in command
    assert "python3 -m venv" not in command
    assert "llm_venv" not in command
    assert timeout == 900
    assert client.closed is True


def test_run_environment_probe_uses_target_dependencies_not_venv(tmp_path):
    class Stream:
        def __init__(self, payload=b"", exit_code=0):
            self.payload = payload
            self.exit_code = exit_code
            self.channel = self

        def read(self):
            return self.payload

        def recv_exit_status(self):
            return self.exit_code

    payload = (
        b"user\nhost\n/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra\nPython 3.10.12\n"
        b"0, NVIDIA A40, 46068, 575.57.08\n"
        b'EVOMIND_RUNTIME={"torch":{"version":"2.6.0","cuda_available":true,"device_count":1},'
        b'"packages":{"dill":{"ok":true},"joblib":{"ok":true},"sklearn":{"ok":true}}}\n'
    )

    class Client:
        def __init__(self):
            self.command = ""

        def exec_command(self, command, timeout):
            self.command = command
            return None, Stream(payload), Stream()

        def close(self):
            pass

    client = Client()
    runtime = HpcRuntime(run_id="run-llm-probe", local_run_dir=tmp_path, connector=lambda: client)

    result = runtime.probe(use_run_environment=True)

    assert result.status == "passed"
    assert f"PYTHONPATH={runtime.remote_python_deps}" in client.command
    assert "HOST_USER_SITE=$(python3 -c 'import site; print(site.getusersitepackages())')" in client.command
    assert f"HOME={runtime.remote_runtime_home}" in client.command
    assert f"TRITON_CACHE_DIR={runtime.remote_triton_cache}" in client.command
    assert "python3 - <<'PY'" in client.command
    assert "llm_venv" not in client.command


def test_llm_execution_uses_same_isolated_environment_as_probe(tmp_path):
    class Stream:
        def __init__(self, payload=b"", exit_code=1):
            self.payload = payload
            self.exit_code = exit_code
            self.channel = self

        def read(self):
            return self.payload

        def recv_exit_status(self):
            return self.exit_code

    class Client:
        def __init__(self):
            self.commands = []

        def exec_command(self, command, timeout):
            self.commands.append(command)
            if len(self.commands) == 1:
                return None, Stream(b"EVOMIND_LAUNCH_STATE=started\n", exit_code=0), Stream(exit_code=0)
            return (
                None,
                Stream(
                    b"EVOMIND_JOB_STATE=finished\nEVOMIND_EXIT_CODE=1\n"
                    b"EVOMIND_LOG_TAIL_BEGIN\nfixture failure\nEVOMIND_LOG_TAIL_END\n",
                    exit_code=0,
                ),
                Stream(exit_code=0),
            )

        def close(self):
            pass

    client = Client()
    runtime = HpcRuntime(run_id="run-llm-exec-env", local_run_dir=tmp_path, connector=lambda: client)
    runtime._llm_poll_interval_seconds = 0
    runtime.stage_bundle = lambda *_args, **_kwargs: {"remote_dir": f"{runtime.remote_run_dir}/llm_finetune"}
    runtime._collect_llm_output = lambda active_client, **_kwargs: (active_client, [])
    source = tmp_path / "train.py"
    config = tmp_path / "config.json"
    data = tmp_path / "train.jsonl"
    source.write_text("print('fixture')\n", encoding="utf-8")
    config.write_text("{}\n", encoding="utf-8")
    data.write_text("{}\n", encoding="utf-8")

    result = runtime.execute_llm_finetune(
        script_path=source,
        data_files={"train.jsonl": data},
        config_path=config,
        base_model="fixture/model",
    )

    assert result.status == "failed"
    launch_command = client.commands[0]
    assert "HOST_USER_SITE=$(python3 -c 'import site; print(site.getusersitepackages())')" in launch_command
    assert "timeout 1200 env HOME=" in launch_command
    assert f"HOME={runtime.remote_runtime_home}" in launch_command
    assert f"PYTHONPATH={runtime.remote_python_deps}:$HOST_USER_SITE" in launch_command
    assert f"TRITON_CACHE_DIR={runtime.remote_triton_cache}" in launch_command
    assert f"CPATH={runtime.remote_python_headers}/usr/include:" in launch_command
    assert "nohup setsid bash" in launch_command
    assert "EVOMIND_LAUNCH_STATE=started" in launch_command
    assert "pgrep" not in launch_command


def test_llm_execution_reconnects_poll_without_duplicate_launch(tmp_path):
    class Stream:
        def __init__(self, payload=b"", exit_code=0):
            self.payload = payload
            self.exit_code = exit_code
            self.channel = self

        def read(self):
            return self.payload

        def recv_exit_status(self):
            return self.exit_code

    class Client:
        def __init__(self, responses):
            self.responses = list(responses)
            self.commands = []
            self.closed = False

        def exec_command(self, command, timeout):
            self.commands.append(command)
            response = self.responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response

        def close(self):
            self.closed = True

    ok = Stream(exit_code=0)
    first = Client(
        [
            (None, Stream(b"EVOMIND_LAUNCH_STATE=started\n"), ok),
            OSError("fixture SSH channel disconnected"),
        ]
    )
    second = Client(
        [
            (
                None,
                Stream(
                    b"EVOMIND_JOB_STATE=running\nEVOMIND_LOG_TAIL_BEGIN\n"
                    b"Downloading shards: 1/4\nEVOMIND_LOG_TAIL_END\n"
                ),
                ok,
            ),
            (
                None,
                Stream(
                    b"EVOMIND_JOB_STATE=finished\nEVOMIND_EXIT_CODE=0\n"
                    b"EVOMIND_LOG_TAIL_BEGIN\ntraining complete\nEVOMIND_LOG_TAIL_END\n"
                ),
                ok,
            ),
        ]
    )
    clients = iter((first, second))
    connector_calls = []

    def connect():
        connector_calls.append(True)
        return next(clients)

    runtime = HpcRuntime(run_id="run-llm-reconnect", local_run_dir=tmp_path, connector=connect)
    runtime._llm_poll_interval_seconds = 0
    runtime.stage_bundle = lambda *_args, **_kwargs: {"remote_dir": f"{runtime.remote_run_dir}/llm_finetune"}
    required_paths = [
        "metrics.json",
        "evaluation.json",
        "environment.json",
        "telemetry.jsonl",
        "adapter_reload.json",
        "adapter/adapter_config.json",
        "adapter/adapter_model.safetensors",
    ]

    def collect(active_client, **_kwargs):
        artifacts = [{"path": str(tmp_path / "llm_output" / name)} for name in required_paths]
        return active_client, artifacts

    runtime._collect_llm_output = collect
    source = tmp_path / "train.py"
    config = tmp_path / "config.json"
    data = tmp_path / "train.jsonl"
    source.write_text("print('fixture')\n", encoding="utf-8")
    config.write_text("{}\n", encoding="utf-8")
    data.write_text("{}\n", encoding="utf-8")

    result = runtime.execute_llm_finetune(
        script_path=source,
        data_files={"train.jsonl": data},
        config_path=config,
        base_model="fixture/model",
    )

    assert result.status == "completed"
    assert len(connector_calls) == 2
    assert sum("EVOMIND_LAUNCH_STATE=started" in command for command in first.commands + second.commands) == 1
    assert len(first.commands) == 2
    assert len(second.commands) == 2
    assert all("EVOMIND_JOB_STATE=" in command for command in second.commands)
    assert first.closed is True
    assert second.closed is True


def test_llm_correction_execution_uses_isolated_remote_bundle_job_and_output(tmp_path):
    class Client:
        def close(self):
            pass

    runtime = HpcRuntime(run_id="run-llm-correction", local_run_dir=tmp_path, connector=Client)
    captured = {}
    runtime.stage_bundle = lambda _files, *, remote_subdir: (
        captured.update(remote_subdir=remote_subdir) or {"remote_dir": f"{runtime.remote_run_dir}/{remote_subdir}"}
    )
    runtime._launch_or_attach_llm_job = lambda _client, **kwargs: (
        captured.update(launch=kwargs)
        or (
            "started",
            f"{runtime.remote_runtime_dir}/llm_jobs/correction_01",
            f"{runtime.remote_runtime_dir}/llm_jobs/correction_01/training.log",
        )
    )
    runtime._poll_llm_job = lambda client, **_kwargs: (client, 1, "fixture failure", "")
    runtime._collect_llm_output = lambda client, **kwargs: captured.update(collect=kwargs) or (client, [])
    source = tmp_path / "train.py"
    config = tmp_path / "config.json"
    data = tmp_path / "train.jsonl"
    source.write_text("print('fixture')\n", encoding="utf-8")
    config.write_text("{}\n", encoding="utf-8")
    data.write_text("{}\n", encoding="utf-8")

    result = runtime.execute_llm_finetune(
        script_path=source,
        data_files={"train.jsonl": data},
        config_path=config,
        base_model="fixture/model",
        execution_id="correction_01",
    )

    assert result.solution_id == "qwen7b_qlora_correction_01"
    assert captured["remote_subdir"] == "llm_finetune/correction_01"
    assert captured["launch"]["execution_id"] == "correction_01"
    assert captured["launch"]["remote_output"].endswith("/llm_attempts/correction_01/llm_output")
    assert captured["collect"]["remote_output"] == captured["launch"]["remote_output"]
    assert captured["collect"]["remote_log"].endswith("/llm_jobs/correction_01/training.log")


def test_manifest_hashes_only_immutable_research_artifacts(tmp_path):
    stable = tmp_path / "metrics.json"
    stable.write_text('{"cv_score": 0.8}\n', encoding="utf-8")
    for name in ("events.jsonl", "handoffs.jsonl", "messages.jsonl", "run.json", "task_graph.json"):
        (tmp_path / name).write_text("mutable\n", encoding="utf-8")
    cache = tmp_path / "solutions" / "solution_01" / "__pycache__" / "solution.pyc"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"bytecode")

    entries = _manifest_entries(tmp_path)

    assert [entry["path"] for entry in entries] == ["metrics.json"]
    assert entries[0]["sha256"] == _sha256(stable)
    (tmp_path / "events.jsonl").write_text("changed after aggregation\n", encoding="utf-8")
    assert entries[0]["sha256"] == _sha256(stable)


def test_reviewer_accepts_only_fully_bound_raw_evidence(tmp_path):
    data_dir = Path(__file__).resolve().parents[1] / "tasks" / "titanic" / "data"
    run_id = "run-review-test"
    _, combined_hash = _write_valid_candidate(tmp_path, data_dir, run_id=run_id)

    checks, candidate = review_solution_evidence(
        run_dir=tmp_path,
        data_dir=data_dir,
        run_id=run_id,
        solution_id="solution_01",
        expected_variant="linear_baseline",
        expected_cuda=False,
        data_hash=combined_hash,
        remote_run_dir=f"{REMOTE_ROOT}/evomind_runs/{run_id}",
    )

    assert checks["passed"] is True
    assert candidate is not None
    assert candidate["cv_score"] == 1.0


@pytest.mark.parametrize("injection", ["forged_score", "old_run", "missing_oof", "source_mismatch"])
def test_reviewer_rejects_forged_or_stale_evidence(tmp_path, injection):
    data_dir = Path(__file__).resolve().parents[1] / "tasks" / "titanic" / "data"
    run_id = "run-review-test"
    _, combined_hash = _write_valid_candidate(tmp_path, data_dir, run_id=run_id)
    solution_dir = tmp_path / "solutions" / "solution_01"
    metrics_path = solution_dir / "output" / "metrics.json"
    if injection == "forged_score":
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["cv_score"] = 0.123
        _write_json(metrics_path, metrics)
    elif injection == "old_run":
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["run_id"] = "old-run"
        _write_json(metrics_path, metrics)
    elif injection == "missing_oof":
        (solution_dir / "output" / "oof_predictions.csv").unlink()
    else:
        (solution_dir / "solution.py").write_text("print('changed')\n", encoding="utf-8")

    checks, candidate = review_solution_evidence(
        run_dir=tmp_path,
        data_dir=data_dir,
        run_id=run_id,
        solution_id="solution_01",
        expected_variant="linear_baseline",
        expected_cuda=False,
        data_hash=combined_hash,
        remote_run_dir=f"{REMOTE_ROOT}/evomind_runs/{run_id}",
    )

    assert checks["passed"] is False
    assert candidate is None
