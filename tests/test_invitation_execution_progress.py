import io
import json
import hashlib
from pathlib import Path

import pytest

from evomind_runtime.execution_progress import ProgressStore, project_progress, resource_lease, safe_text
from evomind_runtime import managed_model_download as downloads
from evomind_runtime.dependency_lock import lock_pip_report, requirements_text
from evomind_runtime.research_knowledge import KnowledgeStore


def test_legacy_running_is_not_training():
    row = project_progress(None, now=100)
    assert row["freshness"] == "unknown"
    assert row["work_kind"] == "executing"
    assert row["percent"] is None


def test_freshness_and_work_kind_require_managed_evidence():
    row = {"heartbeat_at": 100, "work_kind": "training", "source": "executor", "completed_units": 2, "total_units": 10}
    assert project_progress(row, now=101)["work_kind"] == "executing"
    assert project_progress(row, now=146)["freshness"] == "stale"
    row["source"] = "managed_adapter"
    assert project_progress(row, now=101)["percent"] == 20
    assert project_progress(row, status="completed", now=200)["objective_outcome"] == "not_verified"


def test_progress_survives_restart_and_call_changes(tmp_path):
    database = tmp_path / "runtime.sqlite3"
    store = ProgressStore(database)
    store.update("r", {"tool_call_id": "a", "completed_units": 100, "total_units": 100, "source": "managed_adapter", "work_kind": "downloading"})
    assert ProgressStore(database).read("r")["completed_units"] == 100
    store.update("r", {"tool_call_id": "b", "source": "executor"})
    assert store.read("r")["percent"] is None


def test_single_resource_owner(tmp_path):
    with resource_lease(tmp_path, "gpu-1"):
        with pytest.raises((RuntimeError, OSError)):
            with resource_lease(tmp_path, "gpu-1"):
                pytest.fail("duplicate resource lease")
    with resource_lease(tmp_path, "gpu-1"):
        pass


def test_signed_urls_and_passwords_are_redacted():
    assert "secret-value" not in safe_text("password=secret-value https://huggingface.co/file?Signature=secret-value")


class Response(io.BytesIO):
    def __init__(self, body, start, end, total, status=206):
        super().__init__(body)
        self.status = status
        self.headers = {"Content-Range": f"bytes {start}-{end}/{total}"}


def test_short_range_resumes_existing_bytes_and_verifies(tmp_path, monkeypatch):
    monkeypatch.setattr(downloads, "SEGMENT_BYTES", 4)
    data = b"abcdefghij"
    entry = {"name": "tokenizer.json", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    (tmp_path / "tokenizer.json.part").write_bytes(data[:3])
    ranges = []
    def fetch(_url, headers):
        start, end = map(int, headers["Range"].split("=")[1].split("-"))
        ranges.append((start, end))
        return Response(data[start:end+1], start, end, len(data))
    result = downloads.download_file(tmp_path, entry, fetch=fetch)
    assert result["resumed_bytes"] == 3
    assert ranges == [(3, 6), (7, 9)]
    assert (tmp_path / "tokenizer.json").read_bytes() == data


def test_bad_range_does_not_append(tmp_path):
    part = tmp_path / "tokenizer.json.part"
    part.write_bytes(b"abc")
    with pytest.raises(downloads.DownloadRejected, match="content_range"):
        downloads.download_file(tmp_path, {"name": "tokenizer.json", "bytes": 6, "sha256": "0"*64}, fetch=lambda *_: Response(b"def", 0, 2, 6))
    assert part.read_bytes() == b"abc"


def test_404_is_not_retried():
    import urllib.error
    class Opener:
        count = 0
        def open(self, req, timeout):
            self.count += 1
            raise urllib.error.HTTPError(req.full_url, 404, "not found", {}, None)
    opener = Opener()
    with pytest.raises(downloads.DownloadRejected, match="http_404"):
        downloads.request("https://huggingface.co/not-found", {}, opener=opener, sleep=lambda _: None)
    assert opener.count == 1


def test_sources_and_metadata_are_not_guessed():
    assert not downloads.allowed_url("http://huggingface.co/file")
    assert not downloads.allowed_url("https://huggingface.co.attacker.invalid/file")
    with pytest.raises(downloads.DownloadRejected, match="required_model_files"):
        downloads.manifest_entries({"sha": downloads.REVISION, "siblings": []})
    assert "model-00001-of-00003.safetensors" not in downloads.EXPECTED_SHARDS


def test_hash_first_dependency_lock():
    report = {"install": [{"metadata": {"name": "torch", "version": "2.8.0"}, "download_info": {"url": "https://download.pytorch.org/whl/torch.whl", "archive_info": {"hashes": {"sha256": "a"*64}}}}]}
    lock = lock_pip_report(report)
    assert "torch==2.8.0 --hash=sha256:" in requirements_text(lock)
    report["install"][0]["download_info"]["url"] = "https://untrusted.invalid/torch.whl"
    with pytest.raises(ValueError, match="source"):
        lock_pip_report(report)


def test_knowledge_is_project_scoped_and_requires_independent_evidence(tmp_path):
    db = str(tmp_path / "knowledge.sqlite3")
    current = KnowledgeStore(db, "tenant-a", "project-a")
    record = current.propose(category="evaluation", text="Preserve the sample axis.", source={"uri": "run:1", "version": "v1", "license": "project-owned"}, evidence=[{"sha256": "a"*64}], run_id="r1")
    assert current.context("evaluation", "DesignerAgent")["L2"] == []
    assert current.promote(record, lambda _: {"verified": True, "independent": True, "run_id": "r1", "evidence_hashes": ["a"*64]})
    assert len(current.context("evaluation", "DesignerAgent")["L2"]) == 1
    assert KnowledgeStore(db, "tenant-b", "project-a").context("evaluation", "DesignerAgent")["L2"] == []
    assert KnowledgeStore(db, "tenant-a", "project-b").context("evaluation", "DesignerAgent")["L2"] == []
