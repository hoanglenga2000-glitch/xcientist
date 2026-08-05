from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "stage_may2022_public_from_hpc.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "stage_may2022_public_from_hpc", MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_direct_import_bootstraps_src_path() -> None:
    module = load_module()
    assert module.SRC_ROOT == PROJECT_ROOT / "src"
    assert module.ALLOWED_GPU_REMOTE_ROOT in module.REMOTE_PUBLIC_ROOT


@pytest.mark.parametrize(
    "path", ["train.csv", "test.csv", "sample_submission.csv"]
)
def test_only_frozen_may2022_public_files_are_accepted(path: str) -> None:
    module = load_module()
    assert module.validate_relative_path(path) == path


@pytest.mark.parametrize(
    "path",
    [
        "",
        "../private/train.csv",
        "/absolute/train.csv",
        "private/train.csv",
        "test_private.csv",
        "description.md",
        "nested/train.csv",
    ],
)
def test_every_other_may2022_path_is_rejected(path: str) -> None:
    module = load_module()
    with pytest.raises(ValueError):
        module.validate_relative_path(path)


def test_manifest_requires_exact_files_sizes_and_sha256() -> None:
    module = load_module()
    hashes = {
        "train.csv": "a" * 64,
        "test.csv": "b" * 64,
        "sample_submission.csv": "c" * 64,
    }
    output = "\n".join(
        f"{name}\t{size}\t{hashes[name]}"
        for name, size in module.EXPECTED_PUBLIC_FILES.items()
    )

    entries = module.parse_remote_manifest(output)
    inventory = module.build_inventory(entries)

    assert inventory["worker_count"] == 1
    assert inventory["file_count"] == 3
    assert inventory["total_bytes"] == sum(module.EXPECTED_PUBLIC_FILES.values())
    assert inventory["private_paths_requested"] is False
    assert inventory["remote_writes_performed"] is False
    assert inventory["process_signals_sent"] == 0


def test_manifest_rejects_changed_size_invalid_hash_and_missing_file() -> None:
    module = load_module()
    valid = "\n".join(
        f"{name}\t{size}\t{'d' * 64}"
        for name, size in module.EXPECTED_PUBLIC_FILES.items()
    )
    changed_size = valid.replace("283303880", "283303879", 1)
    invalid_hash = valid.replace("d" * 64, "not-a-hash", 1)
    missing = "\n".join(valid.splitlines()[:-1])

    with pytest.raises(RuntimeError, match="size changed"):
        module.parse_remote_manifest(changed_size)
    with pytest.raises(RuntimeError, match="SHA-256"):
        module.parse_remote_manifest(invalid_hash)
    with pytest.raises(RuntimeError, match="incomplete"):
        module.parse_remote_manifest(missing)


def test_local_target_cannot_escape_destination(tmp_path: Path) -> None:
    module = load_module()
    target = module.local_target(tmp_path, "train.csv")
    assert target == tmp_path.resolve() / "train.csv"
    with pytest.raises(ValueError):
        module.local_target(tmp_path, "../../outside")


def test_remote_manifest_command_is_read_only_and_hashes_three_files() -> None:
    module = load_module()
    command = module.build_remote_manifest_command()

    assert "sha256sum" in command
    assert "stat -c %s" in command
    assert "train.csv" in command
    assert "test.csv" in command
    assert "sample_submission.csv" in command
    assert all(token not in command for token in ("rm ", "mv ", "cp ", ">"))
