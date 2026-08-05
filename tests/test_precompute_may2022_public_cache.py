from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import mlebench_medal_recovery_adapters as recovery
from scripts import precompute_may2022_public_cache as cli

COMPETITION = "tabular-playground-series-may-2022"


def _write_public_fixture(root: Path, *, train_rows: int = 100, test_rows: int = 24) -> Path:
    public = root / "data" / COMPETITION / "prepared" / "public"
    private = root / "data" / COMPETITION / "prepared" / "private"
    public.mkdir(parents=True)
    private.mkdir(parents=True)

    def frame(rows: int, *, target: bool) -> pd.DataFrame:
        values: dict[str, object] = {"id": np.arange(rows, dtype=np.int64)}
        for index in range(31):
            if index == 27:
                alphabet = "ABCDEFGHIJKLMNOPQRST"
                values["f_27"] = [
                    "".join(alphabet[(row + position) % len(alphabet)] for position in range(10))
                    for row in range(rows)
                ]
            else:
                values[f"f_{index:02d}"] = (
                    np.arange(rows, dtype=np.float32) * np.float32(0.001)
                    + np.float32(index)
                )
        if target:
            values["target"] = np.arange(rows, dtype=np.int8) % 2
        return pd.DataFrame(values)

    train = frame(train_rows, target=True)
    test = frame(test_rows, target=False)
    sample = pd.DataFrame({"id": test["id"], "target": np.zeros(test_rows)})
    train.to_csv(public / "train.csv", index=False)
    test.to_csv(public / "test.csv", index=False)
    sample.to_csv(public / "sample_submission.csv", index=False)
    # A poison file proves the cache builder does not need the private tree.
    (private / "gold_submission.csv").write_text("POISON_PRIVATE_FILE\n", encoding="utf-8")
    return root / "data"


def _build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    data_root = _write_public_fixture(tmp_path)
    cache = tmp_path / "cache"
    recovery.build_may2022_public_feature_cache(
        data_root=data_root,
        output_dir=cache,
        allowed_root=tmp_path,
        seed=42,
        folds=5,
        source_paths=(Path(cli.__file__),),
    )
    return data_root, cache


def test_public_cache_builds_all_required_artifacts_and_loads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    reads: list[Path] = []
    original = recovery.pd.read_csv

    def tracked_read_csv(path, *args, **kwargs):
        reads.append(Path(path).resolve())
        return original(path, *args, **kwargs)

    monkeypatch.setattr(recovery.pd, "read_csv", tracked_read_csv)
    data_root, cache = _build(tmp_path, monkeypatch)
    assert {path.name for path in reads} == {
        "train.csv",
        "test.csv",
        "sample_submission.csv",
    }
    assert all("private" not in {part.lower() for part in path.parts} for path in reads)
    assert {path.name for path in cache.iterdir()} == {
        *recovery.MAY2022_PUBLIC_CACHE_DATA_ARTIFACTS,
        "cache_manifest.json",
    }

    loaded = recovery.load_may2022_public_feature_cache(
        cache,
        data_root=data_root,
        allowed_root=tmp_path,
        seed=42,
        folds=5,
    )
    assert loaded["train_features"].shape[0] == 100
    assert loaded["test_features"].shape[0] == 24
    assert loaded["train_features"].dtype == np.float32
    assert loaded["manifest"]["contracts"]["private_files_read"] == []
    assert loaded["manifest"]["contracts"]["gpu_used"] is False
    assert np.bincount(loaded["fold_assignment"], minlength=5).tolist() == [20] * 5


def test_public_cache_rejects_artifact_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    data_root, cache = _build(tmp_path, monkeypatch)
    with (cache / "train_id.npy").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(RuntimeError, match="artifact hash failed: train_id.npy"):
        recovery.load_may2022_public_feature_cache(
            cache,
            data_root=data_root,
            allowed_root=tmp_path,
            seed=42,
            folds=5,
        )


def test_public_cache_rejects_builder_source_drift_in_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    data_root, cache = _build(tmp_path, monkeypatch)
    manifest_path = cache / "cache_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    builder = next(item for item in manifest["sources"] if item["role"] == "feature_builder")
    builder["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="feature-builder source drifted"):
        recovery.load_may2022_public_feature_cache(
            cache,
            data_root=data_root,
            allowed_root=tmp_path,
            seed=42,
            folds=5,
        )


def test_public_cache_rejects_public_input_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    data_root, cache = _build(tmp_path, monkeypatch)
    train_path = data_root / COMPETITION / "prepared" / "public" / "train.csv"
    with train_path.open("a", encoding="utf-8") as handle:
        handle.write("drift\n")
    with pytest.raises(RuntimeError, match="public input drifted: train.csv"):
        recovery.load_may2022_public_feature_cache(
            cache,
            data_root=data_root,
            allowed_root=tmp_path,
            seed=42,
            folds=5,
        )


def test_public_cache_requires_empty_cuda_visibility(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    data_root = _write_public_fixture(tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(RuntimeError, match="requires CUDA_VISIBLE_DEVICES to be empty"):
        recovery.build_may2022_public_feature_cache(
            data_root=data_root,
            output_dir=tmp_path / "cache",
            allowed_root=tmp_path,
            seed=42,
            folds=5,
        )


def test_cli_exposes_fail_closed_verify_mode():
    args = cli.parse_args(["--output-dir", "/tmp/cache", "--verify-only"])
    assert args.verify_only is True
    assert args.seed == 42
    assert args.folds == 5
