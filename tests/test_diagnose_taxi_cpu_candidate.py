from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.diagnose_taxi_cpu_candidate import (
    TaxiCpuCandidateError,
    attach_route_stats,
    load_split,
    rmse,
    validate_manifest_artifacts,
)


def test_attach_route_stats_preserves_float32_and_alignment() -> None:
    base = np.arange(18, dtype=np.float32).reshape(3, 6)
    route = np.arange(12, dtype=np.float32).reshape(3, 4)
    result = attach_route_stats(base, route)
    assert result.shape == (3, 10)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result[:, :6], base)
    np.testing.assert_array_equal(result[:, 6:], route)


def test_attach_route_stats_rejects_shape_or_nonfinite() -> None:
    with pytest.raises(TaxiCpuCandidateError, match="shape"):
        attach_route_stats(np.ones((3, 2)), np.ones((2, 4)))
    route = np.ones((3, 4), dtype=np.float32)
    route[0, 0] = np.nan
    with pytest.raises(TaxiCpuCandidateError, match="non-finite"):
        attach_route_stats(np.ones((3, 2)), route)


def test_rmse_requires_aligned_finite_vectors() -> None:
    assert rmse(np.array([1.0, 2.0]), np.array([1.0, 4.0])) == pytest.approx(2**0.5)
    with pytest.raises(TaxiCpuCandidateError, match="misaligned"):
        rmse(np.ones(2), np.ones(3))


def test_manifest_artifact_validation_checks_hash(tmp_path: Path) -> None:
    artifact = tmp_path / "x.npy"
    artifact.write_bytes(b"abc")
    import hashlib

    manifest = {
        "artifacts": [
            {
                "relative_path": "x.npy",
                "bytes": 3,
                "sha256": hashlib.sha256(b"abc").hexdigest(),
            }
        ]
    }
    validate_manifest_artifacts(tmp_path, manifest)
    artifact.write_bytes(b"abd")
    with pytest.raises(TaxiCpuCandidateError, match="hash"):
        validate_manifest_artifacts(tmp_path, manifest)


def test_load_split_rejects_missing_or_empty() -> None:
    archive = {"fit": np.array([0, 2], dtype=np.int64), "empty": np.array([], dtype=np.int64)}
    np.testing.assert_array_equal(load_split(archive, "fit"), [0, 2])
    with pytest.raises(TaxiCpuCandidateError, match="missing"):
        load_split(archive, "valid")
    with pytest.raises(TaxiCpuCandidateError, match="invalid"):
        load_split(archive, "empty")
