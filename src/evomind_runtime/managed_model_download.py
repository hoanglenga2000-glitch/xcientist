"""Deterministic, bounded download of the approved official GPT-OSS snapshot.

This file is also a standalone managed HPC adapter. It never loads a model or
installs software. Model bytes stay in the persistent managed asset directory.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import http.client
import json
import os
import re
import socket
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable


REPO = "openai/gpt-oss-20b"
REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"
ALLOWED_ROOT = Path("/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra")
SEGMENT_BYTES = 16 * 1024 * 1024
EXPECTED_SHARDS = {
    "model-00000-of-00002.safetensors": (4792272488, "16d0f997dcfc4462089d536bffe51b4bcea2f872f5c430be09ef8ed392312427"),
    "model-00001-of-00002.safetensors": (4798702184, "4fbe328ab445455d6f58dc73852b85873bd626986310abd91cd4d2ce3245eaea"),
    "model-00002-of-00002.safetensors": (4170342232, "a18106b209e9ab35c3406db4f6f12a927364a058b21e9d1373d682e20674b303"),
}
SMALL_FILES = {"config.json", "generation_config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja", "LICENSE", "README.md", "USAGE_POLICY"}
REQUIRED = set(EXPECTED_SHARDS) | {"config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"}

# Source: anonymously retrieved fixed-revision official metadata and file bytes.
# This snapshot is immutable; a live API request is not needed to authenticate it.
OFFICIAL_METADATA_RESPONSE_SHA256 = "09df6638a25c1b1029a2e9c3ef4ff4bddfa2f1ec5f6c924a8e1868e2c505a182"
OFFICIAL_SNAPSHOT_DESCRIPTOR_SHA256 = "d585da52bd617e4516f269a381e2af9034595ff3b1128b26a05d733f6cfcafba"
PINNED_FILE_ENTRIES = [
    {
        "name": "LICENSE",
        "bytes": 11357,
        "sha256": "",
        "git_blob_sha1": "7a4a3ea2424c09fbe48d455aed1eaa94d9124835"
    },
    {
        "name": "README.md",
        "bytes": 7095,
        "sha256": "",
        "git_blob_sha1": "f6e25105129ce88f068afff463e0db36024e9ae8"
    },
    {
        "name": "USAGE_POLICY",
        "bytes": 200,
        "sha256": "",
        "git_blob_sha1": "b030f63aecc61cbaf2316a7b6401254f4312df74"
    },
    {
        "name": "chat_template.jinja",
        "bytes": 16738,
        "sha256": "",
        "git_blob_sha1": "dc7bb11927d29f653ba2740f2db2c688fd77592f"
    },
    {
        "name": "config.json",
        "bytes": 1806,
        "sha256": "",
        "git_blob_sha1": "8fb5a4a03376ab5a12afd94b4ed93da61edf5b1c"
    },
    {
        "name": "generation_config.json",
        "bytes": 177,
        "sha256": "",
        "git_blob_sha1": "86f91466555bd40e3de0b1edee3d5d82f4ccdbfe"
    },
    {
        "name": "model.safetensors.index.json",
        "bytes": 36355,
        "sha256": "",
        "git_blob_sha1": "ae08521471ab125be4af84d0e51ecfc245830119"
    },
    {
        "name": "special_tokens_map.json",
        "bytes": 98,
        "sha256": "",
        "git_blob_sha1": "73bd12e55e2004cdfff088f85092b39dba2ccdd0"
    },
    {
        "name": "tokenizer.json",
        "bytes": 27868174,
        "sha256": "0614fe83cadab421296e664e1f48f4261fa8fef6e03e63bb75c20f38e37d07d3",
        "git_blob_sha1": "6ec3ef1795cbbda6b7cb7d1f114919cbe3fdd647"
    },
    {
        "name": "tokenizer_config.json",
        "bytes": 4200,
        "sha256": "",
        "git_blob_sha1": "c021cddb0a9dd35b1bf83a9f145be2d9b3757891"
    },
    {
        "name": "model-00000-of-00002.safetensors",
        "bytes": 4792272488,
        "sha256": "16d0f997dcfc4462089d536bffe51b4bcea2f872f5c430be09ef8ed392312427",
        "git_blob_sha1": "9fd3b4997a862e8554a21820eeee256ec2c06299"
    },
    {
        "name": "model-00001-of-00002.safetensors",
        "bytes": 4798702184,
        "sha256": "4fbe328ab445455d6f58dc73852b85873bd626986310abd91cd4d2ce3245eaea",
        "git_blob_sha1": "6925c5875f9ff955704ed5900214211d4ac4b910"
    },
    {
        "name": "model-00002-of-00002.safetensors",
        "bytes": 4170342232,
        "sha256": "a18106b209e9ab35c3406db4f6f12a927364a058b21e9d1373d682e20674b303",
        "git_blob_sha1": "fc29009d625182d064ca05401b0494af2b382198"
    }
]
BUNDLED_OFFICIAL_FILES = {
    "special_tokens_map.json": (
        "ewogICJib3NfdG9rZW4iOiAiPHxzdGFydG9mdGV4dHw+IiwKICAiZW9zX3Rva2VuIjogIjx8cmV0dXJufD4iLAogICJwYWRfdG9rZW4iOiAiPHxlbmRvZnRleHR8PiIKfQo=",
        "dd5e191d20c12d2fee1da5bae14ca1db0f5f4215300af691f23cdee97120a293",
    ),
}




class DownloadRejected(RuntimeError):
    """Permanent protocol, source, integrity or permission failure: never retry."""


class DownloadConnectionError(ConnectionError):
    """Safe structured transport failure, without URLs or raw client text."""

    def __init__(self, code: str, attempts: int = 1):
        self.code = code if re.fullmatch(r"[a-z0-9_]{1,100}", code) else "official_source_unreachable"
        self.attempts = int(attempts)
        super().__init__(self.code)


CDN_ASSET = "model-00001-of-00002.safetensors"
CDN_ORIGIN = "https://us.aws.cdn.hf.co"
CDN_PATH = "/xet-bridge-us/68913539bd3d0a833438591d/3f05b8460cc6c36fa6d570fe4b6e74b49a29620f29264c82a02cf4ea5136f10c"
CDN_XET_HASH = "3f05b8460cc6c36fa6d570fe4b6e74b49a29620f29264c82a02cf4ea5136f10c"
CDN_SECRET_ENV = "EVOMIND_SECRET_MODEL_CDN_FILE"
CDN_QUERY_KEYS = {"Expires", "Key-Pair-Id", "Policy", "Signature", "X-Xet-Cas-Uid", "response-content-disposition", "user_id"}


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise DownloadRejected("descriptor_duplicate_key")
        result[key] = value
    return result


def validate_cdn_descriptor(raw: bytes, *, now: float | None = None, minimum_remaining: int = 30) -> dict[str, Any]:
    """Accept only the pinned public object, never arbitrary CDN resources."""
    if not isinstance(raw, bytes) or not raw or len(raw) > 64 * 1024:
        raise DownloadRejected("descriptor_size_invalid")
    try:
        data = json.loads(raw, object_pairs_hook=_unique_json_object)
        if not isinstance(data, dict) or set(data) != {"schema", "repo", "revision", "name", "bytes", "sha256", "url"}:
            raise ValueError("shape")
        if data["schema"] != "evomind.gptoss_cdn_descriptor.v1" or data["repo"] != REPO or data["revision"] != REVISION or data["name"] != CDN_ASSET:
            raise ValueError("identity")
        if (data["bytes"], data["sha256"]) != EXPECTED_SHARDS[CDN_ASSET]:
            raise ValueError("asset")
        url = data["url"]
        if not isinstance(url, str) or len(url) > 16384 or any(character in url for character in "\r\n\x00\\"):
            raise ValueError("url")
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme != "https" or parsed.hostname != "us.aws.cdn.hf.co" or parsed.port not in (None, 443) or parsed.username or parsed.password or parsed.fragment or parsed.path != CDN_PATH:
            raise ValueError("source")
        query = urllib.parse.parse_qs(parsed.query, keep_blank_values=True, strict_parsing=True)
        if not {"Expires", "Key-Pair-Id", "Policy", "Signature"} <= set(query) or set(query) - CDN_QUERY_KEYS:
            raise ValueError("query")
        if any(len(values) != 1 or not values[0] or len(values[0]) > 8192 for values in query.values()):
            raise ValueError("query_values")
        encoded = query["Policy"][0].replace("-", "+").replace("_", "=").replace("~", "/")
        policy = json.loads(base64.b64decode(encoded, validate=True), object_pairs_hook=_unique_json_object)
        statements = policy["Statement"]
        if len(statements) != 1:
            raise ValueError("policy")
        statement = statements[0]
        resource = str(statement["Resource"])
        expected_resource = CDN_ORIGIN + CDN_PATH
        suffix = resource[len(expected_resource):]
        if not resource.startswith(expected_resource) or (suffix[:1] not in ("", "?", "*") and not suffix.startswith("\\?")):
            raise ValueError("policy_resource")
        expiry = statement["Condition"]["DateLessThan"]
        expires = expiry.get("EpochTime", expiry.get("AWS:EpochTime"))
        if type(expires) is not int or expires != int(query["Expires"][0]):
            raise ValueError("expiry")
    except DownloadRejected:
        raise
    except (KeyError, TypeError, ValueError, AttributeError, IndexError, OverflowError):
        raise DownloadRejected("cdn_descriptor_invalid") from None
    current = time.time() if now is None else now
    if not minimum_remaining <= expires - current <= 86400:
        raise DownloadRejected("cdn_descriptor_expired_or_not_fresh")
    return {"url": url, "expires_at": expires, "name": CDN_ASSET, "sha256": hashlib.sha256(raw).hexdigest()}


def trusted_https_opener():
    try:
        import certifi
        context = ssl.create_default_context()
        context.load_verify_locations(cafile=certifi.where())
    except (ImportError, OSError, ssl.SSLError):
        raise DownloadRejected("trusted_ca_bundle_unavailable") from None
    if not context.check_hostname or context.verify_mode != ssl.CERT_REQUIRED:
        raise DownloadRejected("tls_verification_required")
    return urllib.request.build_opener(OfficialRedirects(), urllib.request.HTTPSHandler(context=context))


def descriptor_from_environment() -> dict[str, Any] | None:
    value = os.environ.get(CDN_SECRET_ENV, "")
    if not value:
        return None
    try:
        path = Path(value)
        run = os.environ["EVOMIND_RUN_ID"]
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", run) or path.is_symlink() or not path.is_file():
            raise ValueError("path")
        path.resolve(strict=True).relative_to(ALLOWED_ROOT / "evomind_runs" / run)
        if path.stat().st_mode & 0o077:
            raise ValueError("permissions")
        with path.open("rb") as handle:
            raw = handle.read(64 * 1024 + 1)
    except (OSError, KeyError, ValueError):
        raise DownloadRejected("private_cdn_descriptor_scope_invalid") from None
    return validate_cdn_descriptor(raw)


def descriptor_fetch(descriptor: dict[str, Any]):
    canonical = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{CDN_ASSET}"
    def fetch(url, headers):
        if url == canonical:
            if descriptor["expires_at"] <= time.time() + 5:
                raise DownloadRejected("cdn_descriptor_expired_partial_preserved")
            return request(descriptor["url"], headers, retries=1)
        return request(url, headers, retries=1)
    return fetch


def transport_error_code(error: BaseException) -> str:
    current: BaseException | None = error
    seen: set[int] = set()
    for _ in range(8):
        if not isinstance(current, BaseException) or id(current) in seen:
            break
        seen.add(id(current))
        if isinstance(current, DownloadConnectionError):
            return current.code
        if isinstance(current, ssl.SSLCertVerificationError):
            return "tls_certificate_verification_failed"
        if isinstance(current, ssl.SSLError):
            return "tls_error"
        if isinstance(current, socket.gaierror):
            return "dns_resolution_failed"
        if isinstance(current, ConnectionResetError):
            return "connection_reset"
        if isinstance(current, ConnectionRefusedError):
            return "connection_refused"
        if isinstance(current, TimeoutError):
            return "network_timeout"
        reason = getattr(current, "reason", None)
        current = reason if isinstance(reason, BaseException) else current.__cause__ or current.__context__
    return "official_source_unreachable"


def allowed_url(url: str) -> bool:
    parsed = urllib.parse.urlsplit(url)
    host = (parsed.hostname or "").lower()
    return parsed.scheme == "https" and parsed.port in (None, 443) and not parsed.username and not parsed.password and (host == "huggingface.co" or host.endswith(".hf.co"))


class OfficialRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not allowed_url(newurl):
            raise DownloadRejected("redirect_outside_official_source")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def request(url: str, headers: dict[str, str], *, opener=None, retries: int = 3, sleep: Callable[[float], None] = time.sleep):
    if not allowed_url(url):
        raise DownloadRejected("source_not_allowed")
    client = opener or trusted_https_opener()
    for attempt in range(retries):
        try:
            response = client.open(urllib.request.Request(url, headers={"Accept-Encoding": "identity", **headers}), timeout=120)
            if not allowed_url(response.geturl()):
                response.close()
                raise DownloadRejected("response_source_not_allowed")
            return response
        except urllib.error.HTTPError as exc:
            if exc.code not in {408, 429, 500, 502, 503, 504}:
                raise DownloadRejected(f"http_{exc.code}") from None
            if attempt + 1 == retries:
                raise DownloadConnectionError(f"http_{exc.code}_retry_exhausted", attempt + 1) from None
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            if attempt + 1 == retries:
                raise DownloadConnectionError(transport_error_code(exc), attempt + 1) from None
        sleep(min(2 ** attempt, 8))
    raise DownloadConnectionError("retry_exhausted", retries)


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def regular_path(root: Path, name: str) -> Path:
    if not name or Path(name).name != name or "/" in name or "\\" in name or name in {".", ".."}:
        raise DownloadRejected("unsafe_asset_name")
    path = root / name
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise DownloadRejected("asset_not_regular")
    return path


def atomic_json(path: Path, payload: Any) -> None:
    if path.is_symlink():
        raise DownloadRejected("receipt_symlink")
    temporary = path.with_name(path.name + ".tmp")
    if temporary.is_symlink():
        raise DownloadRejected("receipt_temp_symlink")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, sort_keys=True, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def manifest_entries(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    if metadata.get("sha") != REVISION:
        raise DownloadRejected("revision_mismatch")
    entries = []
    seen: set[str] = set()
    for row in metadata.get("siblings", []):
        name = row.get("rfilename", "")
        if name not in set(EXPECTED_SHARDS) | SMALL_FILES:
            continue
        if name in seen:
            raise DownloadRejected("duplicate_metadata_file")
        seen.add(name)
        lfs = row.get("lfs") or {}
        size = lfs.get("size", row.get("size"))
        digest, blob_id = lfs.get("sha256", ""), row.get("blobId", "")
        if not isinstance(size, int) or size <= 0 or not (re.fullmatch("[a-f0-9]{64}", digest) or re.fullmatch("[a-f0-9]{40}", blob_id)):
            raise DownloadRejected("metadata_integrity_missing")
        if name in EXPECTED_SHARDS and (size, digest) != EXPECTED_SHARDS[name]:
            raise DownloadRejected("pinned_shard_mismatch")
        entries.append({"name": name, "bytes": size, "sha256": digest, "git_blob_sha1": blob_id})
    if not REQUIRED <= seen:
        raise DownloadRejected("required_model_files_missing")
    return sorted(entries, key=lambda item: (item["name"] in EXPECTED_SHARDS, item["name"]))


def verify_file(path: Path, entry: dict[str, Any]) -> str:
    if path.is_symlink() or not path.is_file() or path.stat().st_size != entry["bytes"]:
        raise DownloadRejected("asset_size_or_type_mismatch")
    digest = sha256_file(path)
    if entry.get("sha256"):
        if digest != entry["sha256"]:
            raise DownloadRejected("asset_sha256_mismatch")
    else:
        blob = hashlib.sha1(f"blob {entry['bytes']}\0".encode())
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                blob.update(chunk)
        if blob.hexdigest() != entry["git_blob_sha1"]:
            raise DownloadRejected("asset_git_blob_mismatch")
    return digest


def pinned_manifest_entries() -> list[dict[str, Any]]:
    """Revalidate a fresh copy of the archived official file identities."""
    siblings = []
    for entry in PINNED_FILE_ENTRIES:
        row: dict[str, Any] = {"rfilename": entry["name"], "size": entry["bytes"], "blobId": entry["git_blob_sha1"]}
        if entry["sha256"]:
            row["lfs"] = {"size": entry["bytes"], "sha256": entry["sha256"]}
        siblings.append(row)
    return manifest_entries({"sha": REVISION, "siblings": siblings})


def restore_bundled_official_file(root: Path, entry: dict[str, Any]) -> dict[str, Any] | None:
    bundled = BUNDLED_OFFICIAL_FILES.get(entry["name"])
    final = regular_path(root, entry["name"])
    if bundled is None or final.exists():
        return None
    try:
        payload = base64.b64decode(bundled[0], validate=True)
    except (ValueError, TypeError):
        raise DownloadRejected("bundled_asset_encoding_invalid") from None
    digest = hashlib.sha256(payload).hexdigest()
    blob = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
    if len(payload) != entry["bytes"] or digest != bundled[1] or (entry.get("sha256") and digest != entry["sha256"]) or (entry.get("git_blob_sha1") and blob != entry["git_blob_sha1"]):
        raise DownloadRejected("bundled_asset_hash_mismatch")
    part = select_partial(root, entry)
    start = part.stat().st_size if part.exists() else 0
    if start > len(payload):
        raise DownloadRejected("partial_oversized_preserved")
    if part.exists() and part.read_bytes() != payload[:start]:
        raise DownloadRejected("bundled_partial_prefix_mismatch_preserved")
    with part.open("ab") as handle:
        if handle.tell() != start:
            raise DownloadRejected("partial_changed_by_another_writer")
        handle.write(payload[start:])
        handle.flush()
        os.fsync(handle.fileno())
    digest = verify_file(part, entry)
    if final.exists():
        raise DownloadRejected("final_appeared_during_download")
    os.replace(part, final)
    return {**entry, "sha256": digest, "status": "verified", "source": "bundled_official_file",
            "resumed_bytes": start, "network_bytes_downloaded": 0}


def select_partial(root: Path, entry: dict[str, Any]) -> Path:
    part = regular_path(root, entry["name"] + ".part")
    if part.exists():
        return part
    cache = root / ".cache" / "huggingface" / "download"
    for parent in (root / ".cache", root / ".cache" / "huggingface", cache):
        if parent.is_symlink():
            raise DownloadRejected("cache_symlink")
    candidates = []
    if cache.is_dir():
        hashes = {entry.get("sha256", "")}
        if entry["name"] == CDN_ASSET:
            hashes.add(CDN_XET_HASH)
        paths = set(cache.glob(entry["name"] + ".*.incomplete")) | set(cache.glob("." + entry["name"] + ".*.incomplete"))
        candidates = [p for p in sorted(paths) if p.is_file() and not p.is_symlink() and any(value and value in p.name for value in hashes) and p.stat().st_size <= entry["bytes"]]
    if len(candidates) > 1:
        raise DownloadRejected("ambiguous_partial_requires_reconciliation")
    return candidates[0] if candidates else part


def download_file(root: Path, entry: dict[str, Any], *, fetch=None, progress=lambda _p: None, deadline: float | None = None) -> dict[str, Any]:
    fetch = fetch or (lambda url, headers: request(url, headers, retries=1))
    final = regular_path(root, entry["name"])
    if final.exists():
        return {**entry, "sha256": verify_file(final, entry), "status": "verified", "resumed": True}
    part = select_partial(root, entry)
    start = part.stat().st_size if part.exists() else 0
    if start > entry["bytes"]:
        raise DownloadRejected("partial_oversized_preserved")
    offset = start
    while offset < entry["bytes"]:
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("download_budget_exhausted_partial_preserved")
        end = min(offset + SEGMENT_BYTES, entry["bytes"]) - 1
        for attempt in range(3):
            response = None
            try:
                response = fetch(f"https://huggingface.co/{REPO}/resolve/{REVISION}/{entry['name']}", {"Range": f"bytes={offset}-{end}"})
                status = response.status
                expected_range = f"bytes {offset}-{end}/{entry['bytes']}"
                if status == 206:
                    if response.headers.get("Content-Range") != expected_range:
                        raise DownloadRejected("content_range_mismatch")
                elif not (status == 200 and offset == 0 and end + 1 == entry["bytes"]):
                    raise DownloadRejected("range_not_honored")
                if response.headers.get("Content-Encoding", "identity") not in ("identity", ""):
                    raise DownloadRejected("encoded_range_rejected")
                block = response.read(end - offset + 2)
                if len(block) != end - offset + 1:
                    raise ConnectionError("segment_incomplete_partial_preserved")
                break
            except DownloadRejected:
                raise
            except (OSError, http.client.HTTPException) as exc:
                if attempt == 2:
                    code = transport_error_code(exc)
                    raise DownloadConnectionError(code, attempt + 1) from None
                progress({"phase": "retry_wait", "detail": entry["name"], "retry": attempt + 1})
                time.sleep(2 ** attempt)
            finally:
                if response is not None:
                    response.close()
        # Append only a completely validated segment; partial HTTP bodies never
        # become a durable prefix. The caller holds the shared model-directory lock.
        with part.open("ab") as handle:
            if handle.tell() != offset:
                raise DownloadRejected("partial_changed_by_another_writer")
            handle.write(block)
            handle.flush()
            os.fsync(handle.fileno())
        offset = end + 1
        progress({"name": entry["name"], "completed_units": offset, "total_units": entry["bytes"], "unit": "bytes"})
    digest = verify_file(part, entry)
    if final.exists():
        raise DownloadRejected("final_appeared_during_download")
    os.replace(part, final)
    return {**entry, "sha256": digest, "status": "verified", "resumed_bytes": start}


def prepare_entry(root: Path, entry: dict[str, Any], *, progress=lambda _p: None, deadline: float | None = None, fetch=None) -> dict[str, Any]:
    final = regular_path(root, entry["name"])
    if final.exists():
        progress({"status": "verifying_cached_asset", "current_file": entry["name"],
                  "completed_units": None, "total_units": entry["bytes"], "unit": "bytes"})
        return download_file(root, entry, progress=progress, deadline=deadline, fetch=fetch)
    if entry["name"] in BUNDLED_OFFICIAL_FILES:
        progress({"status": "restoring_bundled_official_asset", "current_file": entry["name"],
                  "completed_units": 0, "total_units": entry["bytes"], "unit": "bytes"})
        restored = restore_bundled_official_file(root, entry)
        if restored is not None:
            return restored
    partial = select_partial(root, entry)
    start = partial.stat().st_size if partial.exists() else 0
    progress({"status": "verifying_cached_asset" if start == entry["bytes"] else "downloading",
              "current_file": entry["name"], "completed_units": start, "total_units": entry["bytes"],
              "unit": "bytes", "resume_source": partial.name, "remaining_bytes": max(0, entry["bytes"]-start)})
    return download_file(root, entry, progress=progress, deadline=deadline, fetch=fetch)


def model_tree_sha256(files: list[dict[str, Any]]) -> str:
    identity = sorted(
        ({"name": row["name"], "bytes": row["bytes"], "sha256": row["sha256"]} for row in files),
        key=lambda row: row["name"],
    )
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@contextmanager
def model_lock(root: Path):
    import fcntl
    path = regular_path(root, ".download.lock")
    with path.open("ab") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise DownloadRejected("model_download_worker_already_active") from None
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    data = Path(args.data_dir)
    if data.is_symlink() or data.resolve() != ALLOWED_ROOT / "competition_data" / "cure_bench":
        raise DownloadRejected("managed_dataset_root_mismatch")
    persistent = data / "managed_runtime" / "gpt-oss-20b-r1"
    root = persistent / "model"
    for path in (data / "managed_runtime", persistent, root):
        if path.is_symlink():
            raise DownloadRejected("managed_asset_root_symlink")
        path.mkdir(exist_ok=True)
    out = Path(args.out_dir)
    out.resolve().relative_to(ALLOWED_ROOT)
    out.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {"schema": "evomind.model_download.v1", "repo": REPO, "revision": REVISION, "status": "preparing", "files": [], "signals_sent": 0, "other_processes_modified": False}
    def report(update: dict[str, Any]) -> None:
        state.update(update)
        state["updated_at"] = time.time()
        atomic_json(persistent / "download-progress.json", state)
        atomic_json(out / "download-progress.json", state)
        work_kind = "preparing" if state["status"] in {"loading_frozen_official_manifest", "verifying_cached_asset", "restoring_bundled_official_asset"} else "downloading"
        print("EVOMIND_PROGRESS " + json.dumps({"work_kind": work_kind, "phase": state["status"], "detail": state.get("current_file", ""), **update}), flush=True)
    try:
        with model_lock(persistent):
            deadline = time.monotonic() + 1100
            descriptor = descriptor_from_environment()
            fetch = descriptor_fetch(descriptor) if descriptor else None
            if descriptor:
                report({"status": "preparing", "cdn_descriptor_sha256": descriptor["sha256"],
                        "cdn_origin": CDN_ORIGIN, "cdn_descriptor_expires_at": descriptor["expires_at"],
                        "tls_hostname_verification": True, "tls_certificate_verification": "required"})
            report({"status": "loading_frozen_official_manifest", "metadata_source": "bundled_official_snapshot"})
            entries = pinned_manifest_entries()
            atomic_json(persistent / "official-model-lock.json", {
                "schema": "evomind.official_model_lock.v2", "repo": REPO, "revision": REVISION, "files": entries,
                "source_metadata_sha256": OFFICIAL_METADATA_RESPONSE_SHA256,
                "snapshot_descriptor_sha256": OFFICIAL_SNAPSHOT_DESCRIPTOR_SHA256,
                "metadata_source": "bundled_official_snapshot",
            })
            for entry in entries:
                if time.monotonic() >= deadline:
                    raise TimeoutError("download_budget_exhausted_partial_preserved")
                state["files"].append(prepare_entry(root, entry, progress=report, deadline=deadline, fetch=fetch))
                report({"verified_file_count": len(state["files"]), "expected_file_count": len(entries)})
            state["status"] = "completed"
            state["tree_identity_schema"] = "evomind.model_content_tree.v1"
            state["tree_sha256"] = model_tree_sha256(state["files"])
            atomic_json(persistent / "model-manifest.json", state)
            atomic_json(out / "model-manifest.json", state)
            report({"status": "completed"})
        return 0
    except Exception as exc:
        # Exception strings from HTTP clients may include signed URLs. Emit only
        # stable internal error codes and preserve all partial data.
        error = str(exc) if isinstance(exc, DownloadRejected) else transport_error_code(exc) if isinstance(exc, (OSError, urllib.error.URLError)) else type(exc).__name__
        report({"status": "failed" if isinstance(exc, DownloadRejected) else "resumable", "failure_class": error,
                "failure_type": type(exc).__name__, "attempts": getattr(exc, "attempts", None)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
