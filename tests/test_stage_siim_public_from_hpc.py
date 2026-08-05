from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "stage_siim_public_from_hpc.py"


def load_module():
    spec = importlib.util.spec_from_file_location("stage_siim_public_from_hpc", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_direct_import_bootstraps_src_path() -> None:
    module = load_module()
    assert module.SRC_ROOT == PROJECT_ROOT / "src"
    assert module.ALLOWED_GPU_REMOTE_ROOT in module.REMOTE_PUBLIC_ROOT


def test_remote_manifest_command_preserves_train_and_test_components() -> None:
    module = load_module()
    command = module.build_remote_manifest_command()

    assert 'find "$base/jpeg"' in command
    assert 'find "$base/jpeg/train" "$base/jpeg/test"' not in command
    assert '-mindepth 2 -maxdepth 2' in command
    assert '-path "$base/jpeg/train/*.jpg"' in command
    assert '-path "$base/jpeg/test/*.jpg"' in command
    assert "sed -e 's#^#jpeg/#'" in command


def test_parse_remote_manifest_accepts_only_frozen_public_subset() -> None:
    module = load_module()
    output = "\n".join(
        [
            "train.csv\t11",
            "test.csv\t12",
            "sample_submission.csv\t13",
            "jpeg/test/ISIC_2.jpg\t102",
            "jpeg/train/ISIC_1.jpg\t101",
        ]
    )

    entries = module.parse_remote_manifest(output)
    inventory = module.build_inventory(entries)

    assert inventory["file_count"] == 5
    assert inventory["total_bytes"] == 239
    assert inventory["train_jpeg_count"] == 1
    assert inventory["test_jpeg_count"] == 1
    assert inventory["private_paths_requested"] is False
    assert inventory["remote_writes_performed"] is False


@pytest.mark.parametrize(
    "path",
    [
        "",
        "../private/train.csv",
        "/absolute/train.csv",
        "private/train.csv",
        "jpeg/private/ISIC_1.jpg",
        "jpeg/train/nested/ISIC_1.jpg",
        "jpeg/train/ISIC_1.png",
        "jpeg/ISIC_1.jpg",
    ],
)
def test_validate_relative_public_path_rejects_everything_else(path: str) -> None:
    module = load_module()
    with pytest.raises(ValueError):
        module.validate_relative_public_path(path)


def test_parse_remote_manifest_rejects_duplicate_or_invalid_size() -> None:
    module = load_module()
    duplicate = "\n".join(
        [
            "train.csv\t11",
            "test.csv\t12",
            "sample_submission.csv\t13",
            "jpeg/train/ISIC_1.jpg\t101",
            "jpeg/train/ISIC_1.jpg\t101",
            "jpeg/test/ISIC_2.jpg\t102",
        ]
    )
    invalid_size = duplicate.replace("jpeg/train/ISIC_1.jpg\t101", "jpeg/train/ISIC_1.jpg\tbad", 1)

    with pytest.raises(RuntimeError, match="invalid entry"):
        module.parse_remote_manifest(duplicate)
    with pytest.raises(RuntimeError, match="invalid size"):
        module.parse_remote_manifest(invalid_size)


def test_local_target_cannot_escape_destination(tmp_path: Path) -> None:
    module = load_module()
    destination = tmp_path / "public"
    target = module.ensure_local_target(destination, "jpeg/train/ISIC_1.jpg")

    assert target == destination.resolve() / "jpeg" / "train" / "ISIC_1.jpg"
    with pytest.raises(ValueError):
        module.ensure_local_target(destination, "../../outside")
