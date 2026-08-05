from __future__ import annotations

import importlib.util
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "deploy_hpc88240_leaf_continuation.py"


def load_module():
    spec = importlib.util.spec_from_file_location("deploy_hpc88240_leaf", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_public() -> dict:
    return {
        "public_root": "/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_official_data/leaf-classification/prepared/public",
        "file_count": 994,
        "total_bytes": 28_617_101,
        "image_count": 990,
        "core_files": {
            name: {"bytes": 1, "sha256": character * 64}
            for name, character in (
                ("train.csv", "a"),
                ("test.csv", "b"),
                ("sample_submission.csv", "c"),
                ("description.md", "d"),
            )
        },
    }


def test_hpc_plan_preserves_model_contract_and_moves_runtime_paths() -> None:
    module = load_module()
    base = json.loads(module.BASE_PLAN.read_text(encoding="utf-8-sig"))
    sources = module.source_records()
    plan = module.build_hpc_plan(base, fake_public(), sources)

    assert plan["schema"] == "evomind.leaf.multibackbone_frozen_plan.v1"
    assert plan["training"]["seeds"] == [40, 41, 42]
    assert plan["training"]["folds"] == 5
    assert plan["training"]["backbones"] == [
        "convnext_small",
        "efficientnet_v2_s",
    ]
    assert plan["objective"] == base["objective"]
    assert plan["operational_revision"][
        "model_data_folds_seeds_backbones_and_promotion_thresholds_changed"
    ] is False
    root = PurePosixPath(module.ALLOWED_GPU_REMOTE_ROOT)
    for key in ("script", "data_root", "output_root", "torch_home"):
        PurePosixPath(plan["training"][key]).relative_to(root)


def test_wrapper_is_non_preemptive_and_human_gated() -> None:
    module = load_module()
    wrapper = module.render_wrapper(
        plan_sha256="e" * 64,
        sources=module.source_records(),
        public_snapshot=fake_public(),
    )

    assert "verification_passed|verification_complete_gate_failed" in wrapper
    assert 'stable="$((stable+1))"' not in wrapper
    assert "stable=$((stable+1))" in wrapper
    assert "nvidia-smi --query-compute-apps=pid" in wrapper
    assert "process_signals_sent':0" in wrapper
    assert "official_grader_executed':False" in wrapper
    assert "kaggle_submission_executed':False" in wrapper
    assert "tmp.write_text(json.dumps(payload,indent=2)+'\\n')" in wrapper
    lowered = wrapper.lower()
    assert "pkill" not in lowered
    assert "killall" not in lowered
    assert "taskkill" not in lowered
    assert "stop-process" not in lowered
    assert "kaggle competitions submit" not in lowered


def test_every_declared_remote_path_is_confined() -> None:
    module = load_module()
    for value in (
        module.REMOTE_BASE,
        module.REMOTE_JIGSAW_STATUS,
        module.REMOTE_SITE_PACKAGES,
        module.REMOTE_DATA_ROOT,
        module.REMOTE_PUBLIC_ROOT,
        module.REMOTE_TORCH_HOME,
        module.REMOTE_OUTPUT_ROOT,
        module.REMOTE_RUN_DIR,
        module.REMOTE_PLAN,
        module.REMOTE_WRAPPER,
        module.REMOTE_STATUS,
    ):
        assert module.ensure_remote_path(value) == value
