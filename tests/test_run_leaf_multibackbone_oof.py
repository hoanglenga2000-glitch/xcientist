from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "run_leaf_multibackbone_oof.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_leaf_multibackbone_oof", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_backbone_and_seed_contracts() -> None:
    module = load_module()
    assert module.parse_backbones("convnext_small,efficientnet_v2_s") == [
        "convnext_small",
        "efficientnet_v2_s",
    ]
    assert module.parse_csv_integers("40,41,42", minimum_count=3) == [40, 41, 42]
    with pytest.raises(ValueError, match="two distinct"):
        module.parse_backbones("convnext_small")
    with pytest.raises(ValueError, match="distinct seeds"):
        module.parse_csv_integers("42,42,43", minimum_count=3)


def test_embedding_contract_changes_for_material_input() -> None:
    module = load_module()
    base = module.embedding_cache_contract(
        backbone="convnext_small",
        weight_sha256="a" * 64,
        image_manifest_sha256="b" * 64,
        image_size=288,
        tta_count=8,
        source_sha256="c" * 64,
    )
    changed = module.embedding_cache_contract(
        backbone="convnext_small",
        weight_sha256="a" * 64,
        image_manifest_sha256="b" * 64,
        image_size=224,
        tta_count=8,
        source_sha256="c" * 64,
    )
    assert base["cache_key_sha256"] != changed["cache_key_sha256"]
    assert base["private_labels_used"] is False


def test_portable_unicode_arrays_roundtrip_without_pickle(tmp_path: Path) -> None:
    module = load_module()
    target = module.portable_unicode_array(["Acer_Capillipes", "Quercus"])
    component_names = module.portable_unicode_array(["numeric", "multimodal"])
    path = tmp_path / "leaf_strings.npz"
    np.savez_compressed(path, target=target, component_names=component_names)

    assert target.dtype.kind == "U"
    assert component_names.dtype.kind == "U"
    with np.load(path, allow_pickle=False) as archive:
        assert archive["target"].astype(str).tolist() == ["Acer_Capillipes", "Quercus"]
        assert archive["component_names"].astype(str).tolist() == ["numeric", "multimodal"]


def test_general_blend_is_normalized_and_crossfit_covers_every_row() -> None:
    module = load_module()
    truth = np.asarray(["A", "B"] * 12)
    folds = np.asarray([0, 0, 1, 1, 2, 2] * 4)
    classes = ["A", "B"]
    strong = np.asarray(
        [[0.9, 0.1] if value == "A" else [0.1, 0.9] for value in truth],
        dtype=np.float64,
    )
    weak = np.full_like(strong, 0.5)
    reversed_component = strong[:, ::-1]

    prediction, records = module.cross_fit_blend(
        [strong, weak, reversed_component],
        truth,
        classes,
        folds,
    )

    assert prediction.shape == strong.shape
    assert np.isfinite(prediction).all()
    assert np.allclose(prediction.sum(axis=1), 1.0)
    assert len(records) == 3
    assert all(len(record["weights"]) == 3 for record in records)


def test_windows_dataset_is_module_level_and_source_has_no_submission_path() -> None:
    module = load_module()
    assert module.LeafEmbeddingDataset.__qualname__ == "LeafEmbeddingDataset"
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "official_grader_executed\": False" in source
    assert "kaggle_submission_executed\": False" in source
    assert "subprocess" not in source
