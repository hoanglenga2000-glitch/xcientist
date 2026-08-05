from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "deploy_hpc88240_siim_continuation.py"


def load_module():
    spec = importlib.util.spec_from_file_location("deploy_siim_hpc", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_inventory() -> dict:
    return {
        "file_count": 33_129,
        "total_bytes": 25_765_345_055,
        "train_jpeg_count": 28_984,
        "test_jpeg_count": 4_142,
        "manifest_sha256": "8fef392368fcc433f4532129312f3b2ef7118d76cad04dc70f694246c34eff13",
    }


def test_hpc_plans_preserve_siim_model_contract() -> None:
    module = load_module()
    sources = module.source_records()
    ablation_base = json.loads(module.LOCAL_ABLATION_BASE.read_text(encoding="utf-8-sig"))
    final_base = json.loads(module.LOCAL_FINAL_BASE.read_text(encoding="utf-8-sig"))
    ablation = module.build_ablation_plan(ablation_base, fake_inventory(), sources)
    module.common.write_json_atomic(module.LOCAL_ABLATION_PLAN, ablation)
    final = module.build_final_plan(
        final_base,
        fake_inventory(),
        sources,
        ablation_plan_sha256=module.common.sha256_file(module.LOCAL_ABLATION_PLAN),
    )
    assert ablation["training"]["evaluation_seeds"] == [40, 41, 42]
    assert ablation["training"]["folds"] == 3
    assert final["training"]["outer_folds"] == 5
    assert final["training"]["inner_folds"] == 3
    assert final["training"]["backbone"] == "convnext_small"
    assert final["training"]["secondary_backbone"] == "efficientnet_v2_s"
    assert final["objective"] == final_base["objective"]
    assert final["launch_contract"]["process_signals_allowed"] is False


def test_wrapper_is_serial_non_preemptive_and_human_gated() -> None:
    module = load_module()
    sources = module.source_records()
    wrapper = module.render_wrapper(
        ablation_plan_sha256="a" * 64,
        final_plan_sha256="b" * 64,
        sources=sources,
        inventory=fake_inventory(),
    )
    assert "waiting_for_leaf" in wrapper
    assert "verification_passed|verification_complete_gate_failed" in wrapper
    assert "--candidate-only" in wrapper
    assert "--hold-cuda-lease" in wrapper
    assert "64 32 16" in wrapper
    lowered = wrapper.lower()
    assert "pkill" not in lowered
    assert "killall" not in lowered
    assert "taskkill" not in lowered
    assert "kaggle competitions submit" not in lowered
    assert "official_grader_executed':False" in wrapper
    assert "process_signals_sent':0" in wrapper


def test_connect_with_retry_eventually_returns(monkeypatch) -> None:
    module = load_module()
    calls = {"count": 0}

    def fake_connect(timeout: int):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("temporary socks failure")
        return {"timeout": timeout}

    monkeypatch.setattr(module.common, "connect_ssh", fake_connect)
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)
    assert module.connect_with_retry(attempts=2) == {"timeout": 30}
    assert calls["count"] == 2
