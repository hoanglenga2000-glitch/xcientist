from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "diagnose_may2022_oof_meta_stacker.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "diagnose_may2022_oof_meta_stacker", MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_bundle(path: Path, *, duplicate_id: bool = False, nonfinite: bool = False) -> None:
    rng = np.random.default_rng(42)
    rows = 1000
    test_rows = 100
    target = np.tile(np.array([0, 1], dtype=np.int8), rows // 2)
    folds = np.arange(rows, dtype=np.int16) % 5
    latent = target * 2.0 - 1.0 + rng.normal(0.0, 0.8, rows)
    components = [1.0 / (1.0 + np.exp(-(latent + rng.normal(0, 0.3, rows)))) for _ in range(3)]
    test_components = [rng.uniform(0.05, 0.95, test_rows) for _ in range(3)]
    ids = np.arange(rows)
    if duplicate_id:
        ids[-1] = ids[-2]
    if nonfinite:
        components[0][0] = np.nan
    np.savez_compressed(
        path,
        id=ids,
        target=target,
        fold_assignment=folds,
        mlp_oof=components[0],
        xgboost_oof=components[1],
        catboost_oof=components[2],
        oof_probability=np.mean(components, axis=0),
        mlp_test=test_components[0],
        xgboost_test=test_components[1],
        catboost_test=test_components[2],
        test_probability=np.mean(test_components, axis=0),
        test_id=np.arange(test_rows),
    )


def test_candidate_grid_hash_is_deterministic() -> None:
    module = load_module()
    assert module.candidate_grid_sha256() == module.candidate_grid_sha256()
    assert len(module.candidate_grid_sha256()) == 64
    assert len(module.CANDIDATE_SPECS) == 10


@pytest.mark.parametrize(
    ("space", "columns"),
    [("raw", 3), ("logit", 3), ("rank", 3), ("poly2_logit", 9)],
)
def test_feature_spaces_are_finite_and_have_frozen_width(space: str, columns: int) -> None:
    module = load_module()
    values = np.array([[0.1, 0.2, 0.3], [0.8, 0.7, 0.6]], dtype=np.float64)
    transformed = module.transform_features(values, space)
    assert transformed.shape == (2, columns)
    assert np.isfinite(transformed).all()


def test_bundle_validation_rejects_duplicate_ids_and_nonfinite_predictions(tmp_path: Path) -> None:
    module = load_module()
    duplicate = tmp_path / "duplicate.npz"
    invalid = tmp_path / "invalid.npz"
    write_bundle(duplicate, duplicate_id=True)
    write_bundle(invalid, nonfinite=True)

    with pytest.raises(RuntimeError, match="not unique"):
        module.load_and_validate_bundle(duplicate)
    with pytest.raises(RuntimeError, match="non-finite"):
        module.load_and_validate_bundle(invalid)


def test_single_crossfit_candidate_covers_all_rows(tmp_path: Path) -> None:
    module = load_module()
    bundle_path = tmp_path / "bundle.npz"
    write_bundle(bundle_path)
    arrays = module.load_and_validate_bundle(bundle_path)
    values = np.column_stack([arrays[key] for key in module.COMPONENT_KEYS])
    specs = ({"name": "smoke", "space": "logit", "c": 1e-3},)

    prediction, best, records = module.evaluate_candidates(
        values,
        arrays["target"],
        arrays["fold_assignment"],
        specs,
    )

    assert prediction.shape == arrays["target"].shape
    assert np.isfinite(prediction).all()
    assert best["name"] == "smoke"
    assert len(best["folds"]) == 5
    assert records == [best]
