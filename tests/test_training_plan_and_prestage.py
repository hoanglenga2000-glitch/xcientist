"""2026-11-03: consent card training plan + environment prepared before queueing for the card."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from evomind_runtime import dependency_lock, training_plan
from evomind_runtime import tools as runtime_tools
from evomind_runtime.hpc_runtime_overlay import DispatchCancelled, HpcRuntime
from evomind_runtime.training_control import digest


def _workspace(tmp_path: Path, config: dict) -> tuple[Path, dict]:
    root = tmp_path / "ws"
    (root / "inputs").mkdir(parents=True)
    lines = ["a,b,c,label"] + [f"{i},{i * 2 % 7},{i % 3},{i % 2}" for i in range(100)]
    (root / "inputs" / "data.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "work").mkdir()
    path = root / "work" / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return root, {"configuration_path": "work/config.json", "configuration_sha256": digest(path)}


def test_plan_fills_sizing_like_the_trainer_and_estimates_on_card_time(tmp_path):
    root, args = _workspace(tmp_path, {"data_path": "inputs/data.csv", "task": "classification", "target_column": "label"})
    plan = training_plan.build(root, args, vram_budget_mb=4096)
    assert plan is not None
    assert plan["rows"] == 100 and plan["features"] == 3 and plan["train_rows"] == 80 and plan["holdout_rows"] == 20
    assert plan["hidden_sizes"] == [64, 32] and plan["epochs"] >= 20 and plan["batch_size"] == 80
    assert plan["task_label"].startswith("分类") and plan["target_column"] == "label"
    assert 0 < plan["estimated_gpu_seconds"]["low"] < plan["estimated_gpu_seconds"]["high"]
    assert "不计费" in plan["billing_note"]


def test_plan_uses_the_approved_values_when_the_configuration_sets_them(tmp_path):
    root, args = _workspace(tmp_path, {"data_path": "inputs/data.csv", "task": "regression", "hidden_sizes": [16], "epochs": 5,
                                       "batch_size": 10, "learning_rate": 0.01})
    plan = training_plan.build(root, args)
    assert plan["hidden_sizes"] == [16] and plan["epochs"] == 5 and plan["batch_size"] == 10 and plan["steps"] == 40
    assert plan["metrics"] == ["RMSE", "MAE", "R²"]


def test_plan_is_absent_when_the_bound_configuration_changed(tmp_path):
    root, args = _workspace(tmp_path, {"data_path": "inputs/data.csv", "task": "classification"})
    assert training_plan.build(root, {**args, "configuration_sha256": "0" * 64}) is None
    assert training_plan.build(root, {"configuration_path": "../outside.json", "configuration_sha256": "0" * 64}) is None


def test_runtime_attaches_the_plan_to_the_training_approval(tmp_path):
    from evomind_runtime.runtime import AgentRuntime
    runtime = AgentRuntime(tmp_path / "rt")
    try:
        root, args = _workspace(tmp_path, {"data_path": "inputs/data.csv", "task": "classification"})
        scope = runtime._training_plan_scope({"workspace_root": str(root), "metadata": {}}, args)
        assert scope["training_plan"]["rows"] == 100
        assert runtime._training_plan_scope({"workspace_root": str(root), "metadata": {}}, {}) == {}
    finally:
        runtime.close()


def test_prestaged_environment_is_not_prepared_again_inside_the_lease(tmp_path):
    runtime = HpcRuntime(run_id="run_prestage", local_run_dir=tmp_path / "hpc", connector=lambda: None)
    calls = []
    runtime.prepare_solution_environment = lambda path: calls.append(path) or {"status": "ready"}  # type: ignore[method-assign]
    assert runtime.prestage_environment("x.py") == {"status": "ready"}
    assert runtime.environment_prestaged == {"status": "ready"}
    runtime.use_shared_environment()
    assert runtime._solution_deps_base.endswith("/scripts/gpu_tra/solution-deps-shared")
    assert calls == ["x.py"]


def test_prestage_failure_falls_back_and_cancel_stops_before_queueing(tmp_path):
    runtime = HpcRuntime(run_id="run_prestage_fail", local_run_dir=tmp_path / "hpc", connector=lambda: None)
    def boom(_path):
        raise RuntimeError("managed dependency lock preparation failed: exit=1")
    runtime.prepare_solution_environment = boom  # type: ignore[method-assign]
    assert runtime.prestage_environment("x.py") is None and runtime.environment_prestaged is None
    runtime.cancel_check = lambda: True
    with pytest.raises(DispatchCancelled):
        runtime.prestage_environment("x.py")


def test_time_share_prepares_before_acquiring_the_card(tmp_path, monkeypatch):
    order: list[str] = []

    class Runtime(HpcRuntime):
        def prepare_solution_environment(self, script_path):
            order.append("prepare")
            return {"status": "ready"}

    runtime = Runtime(run_id="run_order", local_run_dir=tmp_path / "hpc", connector=lambda: None)

    class Gate:
        sched = None
        broker_factory = "absent"
        def __init__(self, *_a, **_k):
            pass
        def acquire(self, **_kwargs):
            order.append("acquire")
            from evomind_runtime.gpu_admission import GpuAdmissionError
            raise GpuAdmissionError("pool_unavailable")
        def close(self):
            pass

    script = Path(runtime_tools.__file__).parent / "managed_tensor_train.py"
    context = SimpleNamespace(metadata={}, session_id="s1", runtime_root=tmp_path, gpu_probe=lambda: None, gpu_gate_factory=Gate,
                              gpu_requested_seconds=60, gpu_vram_mb=None, gpu_broker_factory=None, progress=None,
                              managed_asset_adapter=True, trusted_script_path=script, store=SimpleNamespace(get_session=lambda _id: {"status": "running"}))
    from evomind_runtime import gpu_admission
    monkeypatch.setattr(gpu_admission, "ALLOWED_HPC_JOB_ID", 98270)
    result = runtime_tools._gpu_time_share_execute({}, context, runtime, {"solution_id": "s", "script_path": script}, SimpleNamespace(expected_gpu_uuid=""), 98270, "p")
    assert order == ["prepare", "acquire"]
    assert result.ok is False and runtime.environment_prestaged == {"status": "ready"}
    assert runtime._solution_deps_base.endswith("/solution-deps-shared")


@pytest.mark.skipif(sys.platform == "win32", reason="the shared lock uses fcntl (HPC side is Linux)")
def test_shared_root_that_fails_verification_is_set_aside_and_rebuilt(tmp_path, monkeypatch):
    root = tmp_path / dependency_lock.SHARED_DIR_NAME / "abc"
    root.mkdir(parents=True)
    attempts = []
    def fake(target, pins, store=None):
        attempts.append(target)
        if len(attempts) == 1:
            raise ValueError("installed_dependency_integrity_mismatch")
        return {"reused": False}
    monkeypatch.setattr(dependency_lock, "_prepare_target", fake)
    assert dependency_lock.prepare_target(root, {"x": "1"}) == {"reused": False}
    assert len(attempts) == 2 and any(p.name.startswith("abc.aside-") for p in root.parent.iterdir())


def test_prestaged_bundle_is_reused_by_execute_solution_when_the_fingerprint_matches(tmp_path):
    script = tmp_path / "train.py"
    script.write_text("print(1)\n", encoding="utf-8")
    data = tmp_path / "data"
    data.mkdir()
    (data / "training.npz").write_bytes(b"abc")
    runtime = HpcRuntime(run_id="run_bundle", local_run_dir=tmp_path / "hpc", connector=lambda: None)
    uploads = []
    runtime.stage_bundle = lambda files, remote_subdir: uploads.append(remote_subdir) or {"remote_dir": runtime.remote_run_dir + "/" + remote_subdir}  # type: ignore[method-assign]
    first = runtime.prestage_bundle(solution_id="s1", script_path=script, data_dir=data)
    assert first and first["remote_subdir"].startswith("solutions/s1/bundle_") and len(uploads) == 1
    runtime.environment_prestaged = {"status": "ready"}
    runtime.on_dispatch = lambda: False  # stop right at the dispatch gate: everything before it ran
    runtime._connector = lambda: SimpleNamespace(close=lambda: None)  # type: ignore[assignment]
    result = runtime.execute_solution(solution_id="s1", script_path=script, data_dir=data)
    assert len(uploads) == 1, "execute_solution must not upload the same bundle again"
    assert result.failure_type == "cancelled"
    (data / "training.npz").write_bytes(b"changed")
    runtime.on_dispatch = lambda: False
    runtime.execute_solution(solution_id="s1", script_path=script, data_dir=data)
    assert len(uploads) == 2, "a changed input is uploaded again"


def test_prestage_bundle_refuses_symlinks_and_empty_inputs(tmp_path):
    script = tmp_path / "train.py"
    script.write_text("print(1)\n", encoding="utf-8")
    empty = tmp_path / "empty"
    empty.mkdir()
    runtime = HpcRuntime(run_id="run_bundle2", local_run_dir=tmp_path / "hpc", connector=lambda: None)
    runtime.stage_bundle = lambda *_a, **_k: pytest.fail("must not upload")  # type: ignore[method-assign]
    assert runtime.prestage_bundle(solution_id="s1", script_path=script, data_dir=empty) is None
    assert runtime.prestage_bundle(solution_id="../x", script_path=script, data_dir=empty) is None
