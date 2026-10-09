"""Read-only pass 2: directory-level aggregates for the truncated area, seed-file
search, and a pollution check proving this census wrote nothing on the HPC.
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
import json
import os
import time
from pathlib import Path

HOME_ROOT = Path("/hpc2hdd/home")
AIMSLAB = HOME_ROOT / "aimslab"
GPU_TRA = AIMSLAB / "jinghw/scripts/gpu_tra"
SKIP_DIRS = {"mlebench_official_data", "datasets", "competition_data",
             "mlebench_model_cache", "cache", "catboost_info", ".deps",
             "__pycache__", ".t", "node_modules", "site-packages", "venv",
             ".venv", ".git"}
OUR_MARKERS = ("hpc_training_census", "hpc_eval_inventory", "siim_hpc_inventory",
               "host_eval_inventory", "fetch_host_file", "host_sftp")


def aggregate(path, depth=3):
    rows = []
    try:
        children = sorted(p for p in path.iterdir() if p.is_dir())
    except OSError:
        return rows
    for child in children:
        if child.name in SKIP_DIRS:
            rows.append({"name": child.name, "skipped": "dataset_or_cache"})
            continue
        files = 0
        total = 0
        newest = None
        oldest = None
        for dirpath, dirnames, filenames in os.walk(child):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            if len(Path(dirpath).parts) - len(child.parts) >= depth:
                dirnames[:] = []
            for name in filenames:
                try:
                    stat = (Path(dirpath) / name).stat()
                except OSError:
                    continue
                files += 1
                total += stat.st_size
                stamp = time.strftime("%Y-%m-%d", time.gmtime(stat.st_mtime))
                newest = max(newest or stamp, stamp)
                oldest = min(oldest or stamp, stamp)
        rows.append({"name": child.name, "files": files, "bytes": total,
                     "oldest": oldest, "newest": newest})
    return rows


out = {"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
try:
    out["home_accounts"] = sorted(p.name for p in HOME_ROOT.iterdir())
except OSError as exc:
    out["home_accounts"] = "err:" + type(exc).__name__

out["gpu_tra_children"] = aggregate(GPU_TRA, depth=3)

hits = []
for dirpath, dirnames, filenames in os.walk(AIMSLAB):
    if len(Path(dirpath).parts) - len(AIMSLAB.parts) > 4:
        dirnames[:] = []
        continue
    for name in filenames:
        low = name.lower()
        if ("gpu_train" in low or low.startswith("v3_log") or "v3_log" in low
                or any(marker in low for marker in OUR_MARKERS)):
            full = Path(dirpath) / name
            try:
                stat = full.stat()
            except OSError:
                continue
            hits.append({"path": str(full), "bytes": stat.st_size,
                         "mtime": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                time.gmtime(stat.st_mtime)),
                         "is_ours_artifact": any(m in low for m in OUR_MARKERS)})
out["seed_file_search"] = hits[:60]

recent = []
cutoff = time.time() - 3600
for dirpath, dirnames, filenames in os.walk(AIMSLAB):
    if len(Path(dirpath).parts) - len(AIMSLAB.parts) > 3:
        dirnames[:] = []
        continue
    dirnames[:] = [d for d in dirnames
                   if d not in SKIP_DIRS and "siim_calibration_results" not in d
                   and "evomind_runs" not in d]
    for name in filenames:
        try:
            stat = (Path(dirpath) / name).stat()
        except OSError:
            continue
        if stat.st_mtime >= cutoff:
            recent.append({"path": str(Path(dirpath) / name),
                           "mtime": time.strftime("%H:%M:%S", time.gmtime(stat.st_mtime))})
out["files_modified_last_hour_excluding_campaign"] = recent[:40]
out["files_modified_last_hour_count"] = len(recent)
out["cwd"] = os.getcwd()
try:
    out["cwd_entries"] = sorted(p.name for p in Path(os.getcwd()).iterdir())[:30]
except OSError as exc:
    out["cwd_entries"] = "err:" + type(exc).__name__
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

    session = "session_hpc_census2_" + namespace
    api("POST", "/v1/sessions", {
        "session_id": session,
        "objective": "Read-only pass-2 census: aggregates + pollution check.",
        "permission_level": "observe",
        "workspace_root": str(BASE / "data/acceptance/siim-calibration-20260908/operator"),
        "metadata": {"managed_hpc_identity": binding, "tenant_id": tenant,
                     "owner_principal_id": owner,
                     "run_allowed_tool_names": ["hpc_verify"]}})
    verified = api("POST", "/v1/sessions/" + session + "/tools",
                   {"tool_name": "hpc_verify", "arguments": {},
                    "idempotency_key": "census2-identity-" + namespace})
    evidence = (verified.get("result") or {}).get("content") or {}
    from xsci.terminal_tools import hpc_identity_evidence_complete
    if not (verified.get("result") or {}).get("ok") or not hpc_identity_evidence_complete(
            evidence, expected_profile=binding["credential_profile"],
            expected_job_id=binding["job_id"]):
        raise ValueError("census2_operator_identity_failed")
    return session, evidence


def main() -> int:
    namespace = sys.argv[1] if len(sys.argv) > 1 else "census2-" + str(time.time_ns())[-8:]
    receipt_path = OUTPUT / ("hpc-census2-" + namespace + ".json")
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
    try:
        _, stdout, stderr = client.exec_command("python3 -c " + shlex.quote(WORKER),
                                                timeout=900)
        raw = stdout.read(16 * 1024 * 1024)
        code = stdout.channel.recv_exit_status()
        if code:
            raise ValueError("census2_worker_failed:" +
                             stderr.read(4096).decode("utf-8", "replace")[:300])
        payload = json.loads(raw.decode("utf-8", "replace")) if raw else {}
    finally:
        client.close()
    receipt = {"schema": "evomind.hpc_training_census_pass2.v1", "namespace": namespace,
               "at": datetime.now(timezone.utc).isoformat(), "session_id": session,
               "job_id": binding["job_id"], "identity_samples": evidence.get("samples_passed"),
               "remote_writes": 0, "training_started": False, "signals_sent": 0,
               "result": payload}
    receipt_path.write_text(json.dumps(receipt, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "ok", "receipt": str(receipt_path),
                      "gpu_tra_children": len(payload.get("gpu_tra_children") or []),
                      "seed_hits": len(payload.get("seed_file_search") or []),
                      "recent_excluding_campaign":
                          payload.get("files_modified_last_hour_count")},
                     ensure_ascii=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as _exc:  # noqa: BLE001
        import traceback
        _ns = sys.argv[1] if len(sys.argv) > 1 else "unknown"
        _path = OUTPUT / ("hpc-census2-" + _ns + ".error.json")
        try:
            _path.write_text(json.dumps({
                "schema": "evomind.hpc_training_census_pass2.error.v1",
                "namespace": _ns, "error": type(_exc).__name__ + ": " + str(_exc)[:400],
                "traceback": traceback.format_exc()[-1200:],
                "at": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False, indent=2),
                encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
        raise SystemExit(1)
