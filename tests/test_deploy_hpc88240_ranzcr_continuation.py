from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "deploy_hpc88240_ranzcr_continuation.py"


def load_module():
    spec = importlib.util.spec_from_file_location("deploy_ranzcr_hpc", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_data() -> dict:
    return {
        "file_count": 30_087,
        "total_bytes": 6_919_980_199,
        "train_count": 27_074,
        "test_count": 3_009,
        "core": {
            "train.csv": {"bytes": 1, "sha256": "a" * 64},
            "sample_submission.csv": {"bytes": 1, "sha256": "b" * 64},
        },
    }


def test_plan_is_materially_different_from_baseline() -> None:
    module = load_module()
    plan = module.build_plan(module.source_records(), fake_data())
    assert plan["baseline"]["model_family"].startswith("efficientnet_v2_s_512")
    assert plan["training"]["backbone"] == "convnext_small"
    assert plan["training"]["image_size"] == 768
    assert plan["training"]["epochs"] == 12
    assert plan["training"]["candidate_only"] is True
    assert plan["ensemble"]["method"] == "per_label_fold_crossfit_probability_blend"
    assert plan["launch_contract"]["process_signals_allowed"] is False


def test_wrapper_waits_for_siim_and_never_submits() -> None:
    module = load_module()
    wrapper = module.render_wrapper("c" * 64, module.source_records(), fake_data())
    assert "waiting_for_siim" in wrapper
    assert "--wave2-ranzcr-backbone convnext_small" in wrapper
    assert "--wave2-ranzcr-image-size 768" in wrapper
    assert "--candidate-only" in wrapper
    assert "--grid-step 0.025" in wrapper
    lowered = wrapper.lower()
    assert "pkill" not in lowered
    assert "killall" not in lowered
    assert "kaggle competitions submit" not in lowered
    assert "official_grader_executed':False" in wrapper
