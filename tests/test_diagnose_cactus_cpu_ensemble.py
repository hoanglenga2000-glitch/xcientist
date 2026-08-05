from __future__ import annotations

import numpy as np
import pytest

from scripts import diagnose_cactus_cpu_ensemble as diagnostic


def test_image_features_are_finite_and_deterministic():
    image = np.arange(32 * 32 * 3, dtype=np.uint16).reshape(32, 32, 3).astype(np.uint8)
    first = diagnostic.image_features(image)
    second = diagnostic.image_features(image.copy())

    assert first.shape == (476,)
    assert first.dtype == np.float32
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)


def test_image_features_reject_shape_drift():
    with pytest.raises(ValueError, match="32x32 RGB"):
        diagnostic.image_features(np.zeros((16, 16, 3), dtype=np.uint8))


def test_cli_defaults_freeze_cpu_contract():
    args = diagnostic.parse_args(
        ["--data-root", "/data", "--output-dir", "/output", "--allowed-root", "/"]
    )
    assert args.seed == 42
    assert args.folds == 5
    assert args.trees == 700
    assert args.threads == 48
    assert args.promotion_auc == pytest.approx(0.9997)
