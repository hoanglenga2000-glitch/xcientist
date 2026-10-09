import base64
import copy
import hashlib
import json
import socket
import ssl
import urllib.error
from pathlib import Path

import pytest

from evomind_runtime import managed_model_download as downloads


def no_network(*_args, **_kwargs):
    raise AssertionError("network_must_not_be_used")


def test_pins_match_independently_captured_official_metadata(monkeypatch):
    metadata = json.loads((Path(__file__).parent / "fixtures" / "gptoss_6cee_official_metadata.json").read_text())
    monkeypatch.setattr(downloads, "request", no_network)
    expected = downloads.manifest_entries(metadata)
    assert downloads.pinned_manifest_entries() == expected
    assert len(expected) == 13
    assert {row["name"] for row in expected if row["name"] in downloads.EXPECTED_SHARDS} == set(downloads.EXPECTED_SHARDS)


def test_pinned_manifest_returns_fresh_objects():
    entries = downloads.pinned_manifest_entries()
    entries[0]["bytes"] = -1
    assert downloads.pinned_manifest_entries()[0]["bytes"] > 0


def test_wrong_pinned_shard_is_rejected(monkeypatch):
    entries = copy.deepcopy(downloads.PINNED_FILE_ENTRIES)
    next(row for row in entries if row["name"] in downloads.EXPECTED_SHARDS)["sha256"] = "0" * 64
    monkeypatch.setattr(downloads, "PINNED_FILE_ENTRIES", entries)
    with pytest.raises(downloads.DownloadRejected, match="pinned_shard"):
        downloads.pinned_manifest_entries()


def special_entry():
    return next(row for row in downloads.pinned_manifest_entries() if row["name"] == "special_tokens_map.json")


def test_tiny_official_file_restored_without_network(tmp_path, monkeypatch):
    monkeypatch.setattr(downloads, "request", no_network)
    result = downloads.prepare_entry(tmp_path, special_entry())
    payload = (tmp_path / "special_tokens_map.json").read_bytes()
    assert len(payload) == 98
    assert hashlib.sha256(payload).hexdigest() == "dd5e191d20c12d2fee1da5bae14ca1db0f5f4215300af691f23cdee97120a293"
    assert result["network_bytes_downloaded"] == 0
    assert result["source"] == "bundled_official_file"


def test_tiny_file_valid_partial_is_resumed(tmp_path, monkeypatch):
    monkeypatch.setattr(downloads, "request", no_network)
    payload = base64.b64decode(downloads.BUNDLED_OFFICIAL_FILES["special_tokens_map.json"][0])
    (tmp_path / "special_tokens_map.json.part").write_bytes(payload[:31])
    result = downloads.prepare_entry(tmp_path, special_entry())
    assert result["resumed_bytes"] == 31
    assert (tmp_path / "special_tokens_map.json").read_bytes() == payload


def test_tiny_file_wrong_partial_preserved(tmp_path):
    partial = tmp_path / "special_tokens_map.json.part"
    partial.write_bytes(b"bad")
    with pytest.raises(downloads.DownloadRejected, match="prefix_mismatch"):
        downloads.prepare_entry(tmp_path, special_entry())
    assert partial.read_bytes() == b"bad"
    assert not (tmp_path / "special_tokens_map.json").exists()


def test_bad_bundled_payload_is_not_written(tmp_path, monkeypatch):
    monkeypatch.setattr(downloads, "BUNDLED_OFFICIAL_FILES", {"special_tokens_map.json": ("YmFk", "0" * 64)})
    with pytest.raises(downloads.DownloadRejected, match="hash_mismatch"):
        downloads.prepare_entry(tmp_path, special_entry())
    assert list(tmp_path.iterdir()) == []


def test_bad_existing_final_is_not_overwritten(tmp_path):
    final = tmp_path / "special_tokens_map.json"
    final.write_bytes(b"bad")
    with pytest.raises(downloads.DownloadRejected):
        downloads.prepare_entry(tmp_path, special_entry())
    assert final.read_bytes() == b"bad"


def test_verified_final_ignores_unrelated_partial_candidates(tmp_path, monkeypatch):
    monkeypatch.setattr(downloads, "request", no_network)
    body = b"verified"
    name = "tokenizer.json"
    digest = hashlib.sha256(body).hexdigest()
    (tmp_path / name).write_bytes(body)
    cache = tmp_path / ".cache/huggingface/download"
    cache.mkdir(parents=True)
    for suffix in ("a", "b"):
        (cache / (name + "." + digest + suffix + ".incomplete")).write_bytes(body[:2])
    result = downloads.prepare_entry(tmp_path, {"name": name, "bytes": len(body), "sha256": digest})
    assert result["sha256"] == digest
    assert len(list(cache.iterdir())) == 2


def test_complete_hub_cache_promoted_without_network(tmp_path, monkeypatch):
    monkeypatch.setattr(downloads, "request", no_network)
    body = b"hash-bound-model"
    name = "model-00001-of-00002.safetensors"
    digest = hashlib.sha256(body).hexdigest()
    cache = tmp_path / ".cache/huggingface/download"
    cache.mkdir(parents=True)
    partial = cache / (name + "." + digest + ".incomplete")
    partial.write_bytes(body)
    result = downloads.prepare_entry(tmp_path, {"name": name, "bytes": len(body), "sha256": digest})
    assert result["resumed_bytes"] == len(body)
    assert (tmp_path / name).read_bytes() == body


@pytest.mark.parametrize("cause,code", [
    (ConnectionResetError("signed-url-secret"), "connection_reset"),
    (ConnectionRefusedError("signed-url-secret"), "connection_refused"),
    (socket.gaierror(-2, "signed-url-secret"), "dns_resolution_failed"),
    (TimeoutError("signed-url-secret"), "network_timeout"),
    (ssl.SSLCertVerificationError("signed-url-secret"), "tls_certificate_verification_failed"),
])
def test_network_failure_is_bounded_specific_and_redacted(cause, code):
    class Opener:
        count = 0
        def open(self, _request, timeout):
            self.count += 1
            raise urllib.error.URLError(cause)
    opener = Opener()
    sleeps = []
    with pytest.raises(downloads.DownloadConnectionError) as error:
        downloads.request("https://huggingface.co/file", {}, opener=opener, sleep=sleeps.append)
    assert error.value.code == code
    assert error.value.attempts == 3
    assert "signed-url-secret" not in str(error.value)
    assert opener.count == 3
    assert sleeps == [1, 2]


def test_content_tree_identity_ignores_transient_progress():
    rows = [{"name": "model", "bytes": 7, "sha256": "a" * 64, "resumed": True}]
    changed = [{**rows[0], "resumed": False, "resumed_bytes": 3, "status": "verified"}]
    assert downloads.model_tree_sha256(rows) == downloads.model_tree_sha256(changed)
    changed[0]["sha256"] = "b" * 64
    assert downloads.model_tree_sha256(rows) != downloads.model_tree_sha256(changed)
