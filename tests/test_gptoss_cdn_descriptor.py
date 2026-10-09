import base64
import hashlib
import json
import ssl
import time
from types import SimpleNamespace
import urllib.parse

import pytest

from evomind_runtime import managed_model_download as downloads
from evomind_runtime import model_descriptor_store as private_store
from evomind_runtime import tools
from evomind_runtime.policy import PolicyEngine, argument_fingerprint


def descriptor(now=None):
    expiry = int(time.time() if now is None else now) + 3600
    resource = downloads.CDN_ORIGIN + downloads.CDN_PATH
    policy = {"Statement": [{"Resource": resource, "Condition": {"DateLessThan": {"EpochTime": expiry}}}]}
    encoded = base64.b64encode(json.dumps(policy).encode()).decode().replace("+", "-").replace("=", "_").replace("/", "~")
    url = resource + "?" + urllib.parse.urlencode({"Expires": expiry, "Policy": encoded, "Signature": "fixture-secret", "Key-Pair-Id": "fixture-key"})
    size, sha = downloads.EXPECTED_SHARDS[downloads.CDN_ASSET]
    return {"schema": "evomind.gptoss_cdn_descriptor.v1", "repo": downloads.REPO, "revision": downloads.REVISION,
            "name": downloads.CDN_ASSET, "bytes": size, "sha256": sha, "url": url}


def encoded(value):
    return json.dumps(value, sort_keys=True).encode()


def test_fixed_descriptor_identity_and_expiry():
    raw = encoded(descriptor(now=1000))
    result = downloads.validate_cdn_descriptor(raw, now=1000)
    assert result["expires_at"] == 4600
    assert result["sha256"] == hashlib.sha256(raw).hexdigest()
    with pytest.raises(downloads.DownloadRejected, match="expired"):
        downloads.validate_cdn_descriptor(raw, now=4590)


@pytest.mark.parametrize("suffix,accepted", [(r"\?response-content-disposition=attachment.*", True), (r"\other-object", False)])
def test_official_policy_escaped_query_delimiter(suffix, accepted):
    data = descriptor(now=1000)
    parsed = urllib.parse.urlsplit(data["url"])
    query = urllib.parse.parse_qs(parsed.query)
    policy = {"Statement": [{"Resource": downloads.CDN_ORIGIN + downloads.CDN_PATH + suffix,
                              "Condition": {"DateLessThan": {"EpochTime": 4600}}}]}
    policy_text = base64.b64encode(json.dumps(policy).encode()).decode().replace("+", "-").replace("=", "_").replace("/", "~")
    query["Policy"] = [policy_text]
    data["url"] = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path,
                                           urllib.parse.urlencode(query, doseq=True), ""))
    if accepted:
        assert downloads.validate_cdn_descriptor(encoded(data), now=1000)["expires_at"] == 4600
    else:
        with pytest.raises(downloads.DownloadRejected):
            downloads.validate_cdn_descriptor(encoded(data), now=1000)


@pytest.mark.parametrize("field,value", [("revision", "a"*40), ("name", "other.bin"), ("sha256", "0"*64), ("bytes", 1), ("repo", "other/private")])
def test_descriptor_cannot_retarget_model(field, value):
    data = descriptor()
    data[field] = value
    with pytest.raises(downloads.DownloadRejected):
        downloads.validate_cdn_descriptor(encoded(data))


@pytest.mark.parametrize("kind", ["http", "host", "path", "userinfo", "duplicate", "fragment"])
def test_descriptor_url_boundaries(kind):
    data = descriptor()
    if kind == "http": data["url"] = data["url"].replace("https:", "http:")
    if kind == "host": data["url"] = data["url"].replace("us.aws.cdn.hf.co", "attacker.invalid")
    if kind == "path": data["url"] = data["url"].replace(downloads.CDN_PATH, "/other-private-object")
    if kind == "userinfo": data["url"] = data["url"].replace("https://", "https://user:password@")
    if kind == "duplicate": data["url"] += "&Expires=9999999999"
    if kind == "fragment": data["url"] += "#hidden"
    with pytest.raises(downloads.DownloadRejected) as error:
        downloads.validate_cdn_descriptor(encoded(data))
    assert "fixture-secret" not in str(error.value)


def test_descriptor_fetch_uses_cdn_without_hub_request(monkeypatch):
    raw = encoded(descriptor())
    parsed = downloads.validate_cdn_descriptor(raw)
    calls = []
    monkeypatch.setattr(downloads, "request", lambda url, headers, **kwargs: calls.append((url, headers, kwargs)))
    canonical = f"https://huggingface.co/{downloads.REPO}/resolve/{downloads.REVISION}/{downloads.CDN_ASSET}"
    downloads.descriptor_fetch(parsed)(canonical, {"Range": "bytes=7-9"})
    assert calls[0][0] == parsed["url"]
    assert calls[0][1] == {"Range": "bytes=7-9"}
    assert calls[0][2] == {"retries": 1}


def test_existing_ca_bundle_does_not_disable_tls():
    opener = downloads.trusted_https_opener()
    handler = next(handler for handler in opener.handlers if hasattr(handler, "_context"))
    assert handler._context.check_hostname
    assert handler._context.verify_mode == ssl.CERT_REQUIRED


def test_hidden_hub_partial_is_reused_without_network(tmp_path, monkeypatch):
    body = b"fixture-model-content"
    digest = hashlib.sha256(body).hexdigest()
    cache = tmp_path / ".cache/huggingface/download"
    cache.mkdir(parents=True)
    partial = cache / ("." + downloads.CDN_ASSET + "." + digest + ".incomplete")
    partial.write_bytes(body)
    monkeypatch.setattr(downloads, "request", lambda *_a, **_k: pytest.fail("network not needed"))
    result = downloads.prepare_entry(tmp_path, {"name": downloads.CDN_ASSET, "bytes": len(body), "sha256": digest})
    assert result["resumed_bytes"] == len(body)
    assert (tmp_path / downloads.CDN_ASSET).read_bytes() == body


def test_private_blob_plaintext_hash_binding(tmp_path, monkeypatch):
    raw = encoded(descriptor())
    digest = hashlib.sha256(raw).hexdigest()
    path = private_store.descriptor_path(tmp_path, digest)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"encrypted-fixture")
    monkeypatch.setattr(private_store, "_unprotect", lambda _path: raw)
    assert private_store.load_descriptor(tmp_path, digest) == raw
    monkeypatch.setattr(private_store, "_unprotect", lambda _path: b"changed")
    with pytest.raises(ValueError, match="hash_mismatch"):
        private_store.load_descriptor(tmp_path, digest)


def test_descriptor_digest_cannot_select_other_paths(tmp_path):
    with pytest.raises(ValueError, match="digest_invalid"):
        private_store.descriptor_path(tmp_path, "../secret")


def test_generic_execution_cannot_inject_private_values(tmp_path):
    result = tools._hpc_execute_solution({}, SimpleNamespace(approval_verified=False, metadata={}), secret_files={downloads.CDN_SECRET_ENV: b"fixture"})
    assert not result.ok and result.error == "managed_secret_approval_required"


def test_descriptor_hash_changes_require_new_approval(tmp_path):
    engine = PolicyEngine()
    args = {"model": downloads.REPO, "revision": downloads.REVISION, "official_cdn_descriptor_sha256": "a"*64}
    pending = engine.evaluate(tool_name="managed_model_prepare", arguments=args, permission_level="full-auto", workspace_root=tmp_path)
    assert pending.requires_approval
    fingerprint = argument_fingerprint("managed_model_prepare", pending.normalized_arguments)
    approved = engine.evaluate(tool_name="managed_model_prepare", arguments=args, permission_level="full-auto", workspace_root=tmp_path, approved_fingerprint=fingerprint)
    assert approved.allowed
    changed = engine.evaluate(tool_name="managed_model_prepare", arguments={**args, "official_cdn_descriptor_sha256": "b"*64}, permission_level="full-auto", workspace_root=tmp_path, approved_fingerprint=fingerprint)
    assert not changed.allowed and changed.requires_approval
