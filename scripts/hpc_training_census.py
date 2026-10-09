"""Read-only census of every training/evaluation artifact under /hpc2hdd/home.

Runs on the application host under the service account
(start_hpc_training_census.ps1) so the named DPAPI HPC profile can be used.
It walks directories, hashes result files and extracts scalar metrics; it never
writes to the HPC, never trains, never signals a process and never reads
answers or credentials.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BASE = Path("C:/ProgramData/EvoMind")
STAGE = BASE / "staging/siim-mlebench-calibration-20260908/runtime-extension"
OUTPUT = STAGE / "service-output"
POLICY_PATH = BASE / "config/official-calibration/siim-mlebench-20260908.json"
RELEASE_SRC = "C:/EvoMind/releases/664a636ddd419a66f73cc10820c0a16429784866/src"

WORKER = r'''
import hashlib
import json
import os
import time
from pathlib import Path

HOME_ROOT = Path("/hpc2hdd/home")
AIMSLAB = HOME_ROOT / "aimslab"

SEED_DIRS = [
    AIMSLAB / "jinghw/scripts/gpu_tra",
    AIMSLAB / "jinghw/scripts/gpu_tra/evomind_mle22",
    AIMSLAB / "jinghw/scripts/gpu_tra/ev_calibration_results",
    AIMSLAB / "playground_series_s6e6",
]
SEED_FILES = ["gpu_train_v3.py"]
KEYWORDS = ("evomind", "siim", "mlebench", "mle-bench", "kaggle", "playground",
            "dec2021", "cure", "titanic", "bike", "digit", "dog-breed", "covid",
            "ps3e", "ps4e", "ps5e", "s6e")
RESULT_PREFIX = ("metrics", "summary", "result", "score", "oof", "submission",
                 "cv", "log")
SKIP_DIRS = {"mlebench_official_data", "datasets", "competition_data",
             "mlebench_model_cache", "cache", "catboost_info", ".deps",
             "__pycache__", ".t", "node_modules", "site-packages", "venv",
             ".venv", ".git", "huggingface", "torch", "xdg"}
SENSITIVE = ("private", "answer", "label", "solution.csv", "token", "credential",
             "secret", "dpapi", "known_hosts", "id_rsa", "id_ed25519")
METRIC_KEYS = ("roc_auc", "roc-auc", "auc", "pr_auc", "f1", "accuracy", "acc",
               "rmse", "rmsle", "logloss", "log_loss", "gll", "sq", "precision",
               "recall", "map", "score", "mean_auc", "test_auc", "valid_auc")
FOLD_KEYS = ("fold_roc_auc", "folds", "fold_scores", "fold_auc")
MAX_FILES_PER_AREA = 2000
MAX_METRIC_FILES = 300
MAX_READ_BYTES = 1_048_576
WALK_DEPTH = 6
ACCOUNT_WALK_CAP = 60000

started = time.time()
out = {"schema": "evomind.hpc_training_census.v1",
       "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "scope": {"home_root": str(HOME_ROOT), "aimslab": str(AIMSLAB),
                 "seed_dirs": [str(p) for p in SEED_DIRS],
                 "keywords": list(KEYWORDS)}}
skipped = []
notes = []


def sensitive(path):
    low = str(path).lower()
    return any(word in low for word in SENSITIVE)


def is_result(name):
    low = name.lower()
    if not (low.endswith(".json") or low.endswith(".csv") or low.endswith(".txt")):
        return False
    return any(low.startswith(prefix) or ("_" + prefix) in low or
               (prefix + ".") in low for prefix in RESULT_PREFIX)


def read_metrics(path):
    """Extract scalar metrics from a small result file. Read-only."""
    try:
        if path.stat().st_size > MAX_READ_BYTES or sensitive(path):
            return None
        blob = path.read_bytes()
    except OSError:
        return None
    found = {}
    text = blob.decode("utf-8", "replace")
    if path.suffix.lower() == ".json":
        try:
            data = json.loads(text)
        except Exception:
            return None
        stack = [data]
        depth = 0
        while stack and depth < 40:
            node = stack.pop()
            depth += 1
            if isinstance(node, dict):
                for key, value in node.items():
                    low = str(key).lower()
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        if any(k == low or k in low for k in METRIC_KEYS):
                            found.setdefault(low, value)
                    elif isinstance(value, list) and any(k in low for k in FOLD_KEYS):
                        nums = [v for v in value if isinstance(v, (int, float))]
                        if nums:
                            found.setdefault(low + ":folds", nums[:10])
                    elif isinstance(value, (dict, list)):
                        stack.append(value)
            elif isinstance(node, list):
                stack.extend(node[:50])
    else:
        lines = text.splitlines()[:40]
        for line in lines:
            low = line.lower()
            for key in METRIC_KEYS:
                if key in low and any(ch.isdigit() for ch in line):
                    found.setdefault(key, line.strip()[:120])
                    break
    return found or None


def scan_area(path, owner, area_id):
    entry = {"area_id": area_id, "path": str(path), "owner": owner,
             "is_ours": "jinghw" in str(path) or any(
                 word in str(path).lower() for word in ("evomind", "siim", "mlebench")),
             "files": 0, "bytes": 0, "oldest": None, "newest": None,
             "result_files": [], "metrics": [], "truncated": False,
             "skipped_dirs": []}
    if not path.exists():
        entry["status"] = "missing"
        return entry
    if not os.access(str(path), os.R_OK):
        skipped.append({"path": str(path), "reason": "permission_denied"})
        entry["status"] = "permission_denied"
        return entry
    metric_budget = MAX_METRIC_FILES
    base_depth = len(path.parts)
    for dirpath, dirnames, filenames in os.walk(path):
        depth = len(Path(dirpath).parts) - base_depth
        if depth >= WALK_DEPTH:
            dirnames[:] = []
        kept = []
        for name in sorted(dirnames):
            if name in SKIP_DIRS or sensitive(name):
                entry["skipped_dirs"].append(name)
            else:
                kept.append(name)
        dirnames[:] = kept
        for name in sorted(filenames):
            full = Path(dirpath) / name
            try:
                stat = full.stat()
            except OSError:
                skipped.append({"path": str(full), "reason": "stat_failed"})
                continue
            entry["files"] += 1
            entry["bytes"] += stat.st_size
            stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stat.st_mtime))
            entry["oldest"] = min(entry["oldest"] or stamp, stamp)
            entry["newest"] = max(entry["newest"] or stamp, stamp)
            if entry["files"] > MAX_FILES_PER_AREA:
                entry["truncated"] = True
                break
            if is_result(name):
                record = {"path": str(full), "bytes": stat.st_size, "mtime": stamp}
                if sensitive(full):
                    record["content_read"] = False
                else:
                    try:
                        record["sha16"] = hashlib.sha256(full.read_bytes()).hexdigest()[:16]
                    except OSError:
                        record["sha16"] = "unreadable"
                    if metric_budget > 0 and full.suffix.lower() in (".json", ".csv", ".txt"):
                        metrics = read_metrics(full)
                        if metrics:
                            metric_budget -= 1
                            entry["metrics"].append({"path": str(full), "values": metrics})
                entry["result_files"].append(record)
        if entry["truncated"]:
            break
    entry["status"] = "scanned"
    return entry


# 1) account-level metadata under /hpc2hdd/home (metadata only, no descent)
accounts = []
try:
    for account in sorted(p for p in HOME_ROOT.iterdir() if p.is_dir()):
        row = {"account": account.name, "files": 0, "bytes": 0, "newest": None,
               "readable": os.access(str(account), os.R_OK), "truncated": False}
        if account.name == "aimslab":
            accounts.append(row)
            continue
        if not row["readable"]:
            accounts.append(row)
            continue
        seen = 0
        for dirpath, dirnames, filenames in os.walk(account):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS][:20]
            for name in filenames:
                try:
                    stat = (Path(dirpath) / name).stat()
                except OSError:
                    continue
                row["files"] += 1
                row["bytes"] += stat.st_size
                stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stat.st_mtime))
                row["newest"] = max(row["newest"] or stamp, stamp)
                seen += 1
            if seen >= ACCOUNT_WALK_CAP:
                row["truncated"] = True
                break
        accounts.append(row)
except OSError as exc:
    notes.append("home_root_listing_failed:" + type(exc).__name__)
out["accounts"] = accounts

# 2) deep scan of the seeds and everything discovered under aimslab
areas = []
scanned_paths = set()
for index, seed in enumerate(SEED_DIRS, start=1):
    areas.append(scan_area(seed, "aimslab", "seed-%d" % index))
    scanned_paths.add(str(seed))

for name in SEED_FILES:
    target = AIMSLAB / name
    if target.is_file():
        areas.append(scan_area(target, "aimslab", "home-file-" + name))

discovered = []
try:
    for entry in sorted(AIMSLAB.iterdir()):
        if not entry.is_dir() or entry.name in SKIP_DIRS:
            continue
        low = entry.name.lower()
        if any(word in low for word in KEYWORDS):
            discovered.append(entry)
except OSError as exc:
    notes.append("aimslab_listing_failed:" + type(exc).__name__)

for entry in discovered[:20]:
    if any(str(entry).startswith(path) for path in scanned_paths):
        continue
    areas.append(scan_area(entry, "aimslab", "discovered-" + entry.name))
    scanned_paths.add(str(entry))

# nested discovery one level deeper (jinghw/<x>, playground-like roots)
try:
    for parent in sorted(AIMSLAB.iterdir()):
        if not parent.is_dir() or parent.name in SKIP_DIRS:
            continue
        for child in sorted(parent.iterdir()):
            if not child.is_dir() or child.name in SKIP_DIRS:
                continue
            if any(str(child).startswith(path) for path in scanned_paths):
                continue
            low = child.name.lower()
            if any(word in low for word in KEYWORDS):
                areas.append(scan_area(child, "aimslab", "nested-" + parent.name + "-" + child.name))
                scanned_paths.add(str(child))
except OSError as exc:
    notes.append("nested_listing_failed:" + type(exc).__name__)

out["areas"] = areas
out["skipped"] = skipped[:200]
out["notes"] = notes
out["totals"] = {
    "areas": len(areas),
    "files": sum(a.get("files", 0) for a in areas),
    "bytes": sum(a.get("bytes", 0) for a in areas),
    "result_files": sum(len(a.get("result_files", [])) for a in areas),
    "metrics": sum(len(a.get("metrics", [])) for a in areas),
    "truncated_areas": [a["area_id"] for a in areas if a.get("truncated")],
}
out["duration_s"] = round(time.time() - started, 1)
print(json.dumps(out, ensure_ascii=True, default=str))
'''


def observer_identity(binding: dict, tenant: str, owner: str, namespace: str):
    import http.client

    token = (BASE / "data/workspace/runtime/runtime.token").read_text(encoding="utf-8").strip()

    def api(method: str, path: str, body=None, timeout: int = 300):
        conn = http.client.HTTPConnection("127.0.0.1", 8765, timeout=timeout)
        try:
            conn.request(method, path,
                         json.dumps(body).encode() if body is not None else None,
                         {"Authorization": "Bearer " + token,
                          "Content-Type": "application/json"})
            response = conn.getresponse()
            raw = response.read(4 * 1024 * 1024)
            if response.status not in {200, 201}:
                raise ValueError("operator_api_failed:" + str(response.status))
            return json.loads(raw) if raw else {}
        finally:
            conn.close()

    session = "session_hpc_census_" + namespace
    api("POST", "/v1/sessions", {
        "session_id": session,
        "objective": "Read-only HPC training-artifact census; no training, no writes.",
        "permission_level": "observe",
        "workspace_root": str(BASE / "data/acceptance/siim-calibration-20260908/operator"),
        "metadata": {"managed_hpc_identity": binding, "tenant_id": tenant,
                     "owner_principal_id": owner,
                     "run_allowed_tool_names": ["hpc_verify"]}})
    verified = api("POST", "/v1/sessions/" + session + "/tools",
                   {"tool_name": "hpc_verify", "arguments": {},
                    "idempotency_key": "census-identity-" + namespace})
    evidence = (verified.get("result") or {}).get("content") or {}
    from xsci.terminal_tools import hpc_identity_evidence_complete
    if not (verified.get("result") or {}).get("ok") or not hpc_identity_evidence_complete(
            evidence, expected_profile=binding["credential_profile"],
            expected_job_id=binding["job_id"]):
        raise ValueError("census_operator_identity_failed")
    return session, evidence


def main() -> int:
    namespace = sys.argv[1] if len(sys.argv) > 1 else "census-" + str(time.time_ns())[-8:]
    receipt_path = OUTPUT / ("hpc-census-" + namespace + ".json")
    if receipt_path.exists():
        raise ValueError("operation_exists")
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8-sig"))
    binding = policy["managed_hpc_identity"]
    sys.path[:0] = [str(BASE / "bundle/runtime"), RELEASE_SRC]
    session, evidence = observer_identity(binding, policy["tenant_id"],
                                          policy["owner_principal_id"], namespace)
    import shlex
    from evomind_runtime.tools import _load_bound_hpc_config
    from research_agent_workstation.server.core.gpu_credentials import connect_ssh
    client = connect_ssh(_load_bound_hpc_config(binding["credential_profile"],
                                                binding["job_id"], session), timeout=30)
    started = time.time()
    try:
        _, stdout, stderr = client.exec_command("python3 -c " + shlex.quote(WORKER),
                                                timeout=1500)
        raw = stdout.read(32 * 1024 * 1024)
        code = stdout.channel.recv_exit_status()
        if code:
            raise ValueError("census_worker_failed:" +
                             stderr.read(4096).decode("utf-8", "replace")[:300])
        payload = json.loads(raw.decode("utf-8", "replace")) if raw else {}
    finally:
        client.close()
    receipt = {"schema": "evomind.hpc_training_census_operator.v1", "namespace": namespace,
               "at": datetime.now(timezone.utc).isoformat(), "session_id": session,
               "job_id": binding["job_id"], "identity_samples": evidence.get("samples_passed"),
               "duration_s": round(time.time() - started, 1),
               "remote_writes": 0, "training_started": False,
               "signals_sent": 0, "other_processes_modified": False,
               "census": payload}
    receipt_path.write_text(json.dumps(receipt, indent=1, ensure_ascii=False), encoding="utf-8")
    totals = (payload or {}).get("totals") or {}
    print(json.dumps({"status": "ok", "receipt": str(receipt_path),
                      "areas": totals.get("areas"), "files": totals.get("files"),
                      "result_files": totals.get("result_files"),
                      "metrics": totals.get("metrics")}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as _exc:  # noqa: BLE001
        import traceback
        _ns = sys.argv[1] if len(sys.argv) > 1 else "unknown"
        _path = OUTPUT / ("hpc-census-" + _ns + ".error.json")
        try:
            OUTPUT.mkdir(parents=True, exist_ok=True)
            _path.write_text(json.dumps({
                "schema": "evomind.hpc_training_census_operator.error.v1",
                "namespace": _ns, "error": type(_exc).__name__ + ": " + str(_exc)[:400],
                "traceback": traceback.format_exc()[-1200:],
                "at": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False, indent=2),
                encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
        raise SystemExit(1)
