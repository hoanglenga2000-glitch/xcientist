from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "stage_leaf_public_from_hpc.py"


def load_module():
    spec = importlib.util.spec_from_file_location("stage_leaf_public_from_hpc", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "path",
    [
        "train.csv",
        "test.csv",
        "sample_submission.csv",
        "description.md",
        "images/1.jpg",
        "images/1584.jpg",
    ],
)
def test_frozen_leaf_public_paths_are_accepted(path: str) -> None:
    module = load_module()
    assert module.validate_relative_path(path) == path


@pytest.mark.parametrize(
    "path",
    [
        "",
        "../private/train.csv",
        "/absolute/train.csv",
        "private/train.csv",
        "images/a.jpg",
        "images/1.png",
        "images/nested/1.jpg",
    ],
)
def test_every_other_leaf_path_is_rejected(path: str) -> None:
    module = load_module()
    with pytest.raises(ValueError):
        module.validate_relative_path(path)


def test_local_leaf_target_cannot_escape(tmp_path: Path) -> None:
    module = load_module()
    target = module.local_target(tmp_path, "images/1.jpg")
    assert target == tmp_path.resolve() / "images" / "1.jpg"
    with pytest.raises(ValueError):
        module.local_target(tmp_path, "../../outside")
