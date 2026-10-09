from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_hpc_bounded_smoke as bounded  # noqa: E402


def test_remote_source_is_job_bound_and_preserves_hard_boundaries() -> None:
    source = bounded.build_remote_source(job_id=91051, profile="job91051")

    assert 'PROFILE = "job91051"' in source
    assert "JOB_ID = 91051" in source
    assert bounded.ALLOWED_GPU_REMOTE_ROOT in source
    assert '"training_started": False' in source
    assert '"grader_calls": 0' in source
    assert '"kaggle_submissions": 0' in source
    assert '"signals_sent": 0' in source
    assert '"other_processes_modified": False' in source
    assert "ptx_vector_add" in source


def test_remote_source_rejects_cross_job_profile() -> None:
    try:
        bounded.build_remote_source(job_id=91051, profile="job90948")
    except ValueError as exc:
        assert "one-to-one binding" in str(exc)
    else:  # pragma: no cover - failure assertion
        raise AssertionError("cross-job profile was accepted")


def test_pointer_hashes_collection_and_every_required_file(tmp_path: Path) -> None:
    for name in bounded.REMOTE_REQUIRED_FILES:
        (tmp_path / name).write_text(name + "\n", encoding="utf-8")
    collection = tmp_path / "collection.json"
    collection.write_text(json.dumps({"status": "passed"}) + "\n", encoding="utf-8")

    pointer = bounded.build_pointer(
        profile="job91051",
        job_id=91051,
        evidence_root=tmp_path,
        collection_path=collection,
        created_at="2026-08-09T12:00:00+00:00",
    )

    assert pointer["schema"] == bounded.POINTER_SCHEMA
    assert pointer["profile"] == "job91051"
    assert pointer["job_id"] == 91051
    assert len(pointer["collection_sha256"]) == 64
    assert set(pointer["required_files"]) == set(bounded.REMOTE_REQUIRED_FILES)
    assert all(len(value) == 64 for value in pointer["required_files"].values())
