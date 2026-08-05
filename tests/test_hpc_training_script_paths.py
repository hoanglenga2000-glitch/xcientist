from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from uuid import uuid4

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(relative_path: str) -> ModuleType:
    path = ROOT / relative_path
    module_name = f"_evomind_test_{path.stem}_{uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "relative_path",
    ["scripts/gpu_batch_trainer_v1.py", "scripts/mlebench_server_runner.py"],
)
def test_hpc_training_scripts_do_not_initialize_paths_at_import(
    relative_path: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "not-created-at-import"
    monkeypatch.setenv("EVOMIND_HPC_REMOTE_WORKSPACE", str(workspace))

    module = load_script(relative_path)

    assert module.BASE_DIR is None
    assert not workspace.exists()


@pytest.mark.parametrize(
    "relative_path",
    ["scripts/gpu_batch_trainer_v1.py", "scripts/mlebench_server_runner.py"],
)
def test_hpc_training_scripts_require_explicit_absolute_workspace(
    relative_path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script(relative_path)
    monkeypatch.delenv("EVOMIND_HPC_REMOTE_WORKSPACE", raising=False)

    with pytest.raises(RuntimeError, match="configured explicitly"):
        module._runtime_paths()
    with pytest.raises(RuntimeError, match="absolute path"):
        module.configure_runtime_paths("relative/workspace")

    assert module.BASE_DIR is None


def test_gpu_trainer_initializes_on_first_training_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "gpu-runtime"
    monkeypatch.setenv("EVOMIND_HPC_REMOTE_WORKSPACE", str(workspace))
    module = load_script("scripts/gpu_batch_trainer_v1.py")

    result = module.train_single(
        "missing-competition",
        {"type": "classification", "metric": "accuracy"},
    )

    assert result == {"error": "Data missing: missing-competition"}
    assert module.BASE_DIR == workspace
    assert (workspace / "results").is_dir()


def test_mlebench_runner_initializes_checkpoint_and_log_on_first_use(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "mle-runtime"
    monkeypatch.setenv("EVOMIND_HPC_REMOTE_WORKSPACE", str(workspace))
    module = load_script("scripts/mlebench_server_runner.py")

    assert module.load_checkpoint() is None
    module.log("runtime paths ready")

    assert module.BASE_DIR == workspace
    assert (workspace / "mlebench_data").is_dir()
    assert (workspace / "mlebench_results").is_dir()
    assert "runtime paths ready" in (workspace / "mlebench_75_runner.log").read_text(encoding="utf-8")
