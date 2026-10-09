"""Bounded CURE-only asset and official HTTPS diagnostics; no preparation or fit."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import ipaddress
import json
import os
import re
import shutil
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path("/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra")
REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"
REPO = "openai/gpt-oss-20b"
SHARDS = {
    "model-00000-of-00002.safetensors": (4792272488, "16d0f997dcfc4462089d536bffe51b4bcea2f872f5c430be09ef8ed392312427"),
    "model-00001-of-00002.safetensors": (4798702184, "4fbe328ab445455d6f58dc73852b85873bd626986310abd91cd4d2ce3245eaea"),
    "model-00002-of-00002.safetensors": (4170342232, "a18106b209e9ab35c3406db4f6f12a927364a058b21e9d1373d682e20674b303"),
}
SMALL = ("config.json", "generation_config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
PACKAGES = ("torch", "transformers", "safetensors", "accelerate", "triton", "huggingface_hub", "numpy", "requests", "PySocks")


def within(path: Path) -> Path:
    path.resolve().relative_to(ROOT.resolve(strict=True))
    if path.is_symlink():
        raise ValueError("symlink_not_read")
    return path


def digest(path: Path, deadline: float) -> str:
    value = hashlib.sha256()
    with within(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4*1024*1024), b""):
            if time.monotonic() >= deadline:
                raise TimeoutError("bounded_hash_budget_exhausted")
            value.update(block)
    return value.hexdigest()


def stat(path: Path) -> dict:
    if path.is_symlink():
        return {"exists": True, "symlink": True, "contents_read": False}
    if not path.exists():
        return {"exists": False}
    within(path)
    return {"exists": True, "directory": path.is_dir(), "bytes": path.stat().st_size}


def inventory(root: Path) -> list:
    if root.is_symlink() or not root.is_dir():
        return []
    rows = []
    for path in sorted(within(root).iterdir()):
        if len(rows) >= 64:
            break
        if re.search(r"(?i)^(token|credentials|password|cookies?)(\.|$)", path.name):
            continue
        rows.append({"name": path.name, **stat(path)})
    return rows


def exception_detail(error: BaseException) -> dict:
    classes, texts, visited = [], [], set()
    current = error
    for _ in range(8):
        if not isinstance(current, BaseException) or id(current) in visited:
            break
        visited.add(id(current))
        classes.append(type(current).__name__)
        texts.append(str(current).lower())
        current = getattr(current, "reason", None) or current.__cause__ or current.__context__
    combined = " ".join(texts)
    code = "transport_error"
    for needle, label in (("407", "proxy_authentication_required"), ("certificate_verify_failed", "tls_certificate_verification_failed"),
                          ("name or service not known", "dns_resolution_failed"), ("name resolution", "dns_resolution_failed"),
                          ("connection refused", "connection_refused"), ("network is unreachable", "network_unreachable"),
                          ("unknown url type", "unsupported_proxy_or_url_scheme"), ("timed out", "network_timeout"),
                          ("connection reset", "connection_reset"), ("unexpected eof", "tls_unexpected_eof"),
                          ("remote end closed", "remote_closed_without_response")):
        if needle in combined:
            code = label
            break
    status = getattr(error, "code", None)
    return {"error_code": code, "exception_classes": classes,
            "http_status": status if isinstance(status, int) else None,
            "raw_exception_text_emitted": False}


class OfficialRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urllib.parse.urlsplit(newurl)
        host = (parsed.hostname or "").lower()
        if parsed.scheme != "https" or parsed.port not in (None, 443) or parsed.username or parsed.password or not (host == "huggingface.co" or host.endswith(".hf.co")):
            raise ValueError("redirect_outside_official_source")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def official_https() -> dict:
    proxies = {}
    for name, value in urllib.request.getproxies().items():
        if name not in ("http", "https", "all"):
            continue
        parsed = urllib.parse.urlsplit(value)
        proxies[name] = {"configured": True, "scheme": parsed.scheme,
                         "credentials_present": bool(parsed.username or parsed.password),
                         "loopback": parsed.hostname in ("localhost", "127.0.0.1", "::1")}
    result = {"proxy_configuration": proxies, "proxy_configuration_changed": False,
              "hf_endpoint_override_present": bool(os.environ.get("HF_ENDPOINT")), "requests_attempted": 0}
    try:
        addresses = {row[4][0] for row in socket.getaddrinfo("huggingface.co", 443, type=socket.SOCK_STREAM)}
        public = [ipaddress.ip_address(address).is_global for address in addresses]
        result["dns"] = {"resolved": True, "address_count": len(addresses), "all_addresses_public": all(public)}
        if not public or not all(public):
            result["status"] = "non_public_dns_target_not_requested"
            return result
    except Exception as error:
        result.update(status="dns_failed", **exception_detail(error))
        return result
    url = "https://huggingface.co/api/models/"+REPO+"/revision/"+REVISION+"?blobs=true"
    started = time.monotonic()
    try:
        result["requests_attempted"] = 1
        opener = urllib.request.build_opener(OfficialRedirects())
        with opener.open(urllib.request.Request(url, headers={"Accept-Encoding": "identity"}), timeout=15) as response:
            raw = response.read(2*1024*1024+1)
            if len(raw) > 2*1024*1024:
                raise ValueError("official_metadata_size_limit")
            metadata = json.loads(raw)
            if metadata.get("sha") != REVISION or metadata.get("id") != REPO:
                raise ValueError("official_snapshot_identity_mismatch")
            result.update(status="reachable", http_status=response.status,
                          metadata_sha256=hashlib.sha256(raw).hexdigest(), sibling_count=len(metadata.get("siblings", [])))
    except Exception as error:
        result.update(status="failed", **exception_detail(error))
    result["elapsed_seconds"] = time.monotonic()-started
    return result


def environment(prefix: Path | None = None) -> dict:
    if prefix is None:
        result = {}
        for name in PACKAGES:
            try:
                result[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                result[name] = None
        return {"scope": "executor_base_python", "packages": result}
    if prefix.is_symlink() or not prefix.is_dir():
        return {"exists": False}
    sites = [within(path) for path in prefix.glob("lib/python*/site-packages") if path.is_dir() and not path.is_symlink()]
    result = {}
    for distribution in importlib.metadata.distributions(path=[str(path) for path in sites]):
        name = distribution.metadata.get("Name", "")
        if name.lower().replace("-", "_") in {item.lower().replace("-", "_") for item in PACKAGES}:
            result[name] = distribution.version
    return {"exists": True, "scope": "persistent_venv_metadata_only", "packages": result,
            "model_imported": False, "python_entrypoint": stat(prefix/"bin/python")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    data = within(args.data_dir)
    if data.resolve() != ROOT/"competition_data/cure_bench":
        raise ValueError("cure_dataset_root_required")
    output = within(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    persistent = within(data/"managed_runtime/gpt-oss-20b-r1")
    model = within(persistent/"model")
    result = {"schema": "evomind.cure.managed_runtime_diagnostic.v1", "status": "observation_completed",
              "repo": REPO, "revision": REVISION, "persistent_children": inventory(persistent),
              "model_children": inventory(model), "shards": [], "training_executed": False,
              "inference_executed": False, "model_preparation_completed": False,
              "signals_sent": 0, "other_processes_modified": False, "model_bytes_downloaded": 0}
    import fcntl
    lock_path = persistent/".download.lock"
    deadline = time.monotonic()+130
    if lock_path.is_file() and not lock_path.is_symlink():
        with lock_path.open("rb") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_SH|fcntl.LOCK_NB)
                result["asset_read_lock"] = "acquired"
                for name, (size, expected) in SHARDS.items():
                    row = {"name": name, "expected_bytes": size, "final": stat(model/name), "partial": stat(model/(name+".part"))}
                    if row["final"].get("bytes") == size and not row["final"].get("symlink"):
                        try:
                            row["sha256"] = digest(model/name, deadline)
                            row["matches_pinned_sha256"] = row["sha256"] == expected
                        except TimeoutError:
                            row["hash_status"] = "budget_exhausted_not_verified"
                    result["shards"].append(row)
            except BlockingIOError:
                result["asset_read_lock"] = "busy_existing_downloader_not_interrupted"
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
    else:
        result["asset_read_lock"] = "missing_no_lock_created"
    result["small_files"] = {name: stat(model/name) for name in SMALL}
    result["cache_children"] = {name: inventory(persistent/name) for name in ("cache", "hf-cache", ".cache")}
    result["environment"] = [environment(), *[{"name": name, **environment(persistent/name)} for name in ("venv", ".venv")]]
    manifest = data/".evomind/data-manifest.json"
    result["dataset_manifest_sha256"] = digest(manifest, time.monotonic()+10)
    if result["dataset_manifest_sha256"] != "b2af0ade5f012bae9722192c1c7575815f788228b8f8de509be6506438b90197":
        raise ValueError("dataset_manifest_mismatch")
    validation = within(data/"data/curebench_valset_pharse1.jsonl")
    rows = [json.loads(line) for line in validation.read_text().splitlines() if line.strip()]
    result["public_validation"] = {"rows": len(rows), "keys": sorted({key for row in rows for key in row}),
                                   "sha256": digest(validation, time.monotonic()+10), "record_values_emitted": False}
    result["official_https"] = official_https()
    result["source_sha256"] = digest(Path(__file__), time.monotonic()+10)
    (output/"cure-runtime-diagnostic.json").write_text(json.dumps(result, ensure_ascii=True, indent=2)+"\n")
    shutil.copyfile(Path(__file__), output/"inspect_cure_managed_runtime_v1.py")
    print(json.dumps(result, ensure_ascii=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
