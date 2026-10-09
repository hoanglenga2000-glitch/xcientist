"""Read-only operator inventory of the SIIM artifacts inside the HPC container.

Runs on the application host under the service account (start_siim_hpc_inventory.ps1)
so the named DPAPI HPC profile can be used. It lists directories and hashes files
only; it never trains, never writes to the HPC and never reads private answers.
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

WORKER = r'''
import hashlib, json
from pathlib import Path

ROOT = Path("/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra")
out = {"root": str(ROOT)}


def listing(path, limit=60):
    try:
        return sorted(p.name for p in Path(path).iterdir())[:limit]
    except OSError:
        return None


out["top"] = listing(ROOT)
res = ROOT / "siim_calibration_results"
out["siim_results_top"] = listing(res)
base = res / "siim_mlebench_calibration_20260908"
results = {}
if base.is_dir():
    for case in sorted(p.name for p in base.iterdir() if p.is_dir()):
        rows = []
        for attempt in sorted(p.name for p in (base / case).iterdir() if p.is_dir()):
            entry = {"attempt": attempt}
            for name in ("metrics.json", "failure.json", "submission.csv",
                         "executor-exit-receipt.json"):
                target = base / case / attempt / name
                if target.is_file():
                    blob = target.read_bytes()
                    entry[name] = {"bytes": len(blob),
                                   "sha16": hashlib.sha256(blob).hexdigest()[:16]}
                    if name == "metrics.json":
                        try:
                            data = json.loads(blob.decode("utf-8", "replace"))
                            entry["oof"] = data.get("oof_roc_auc")
                            entry["folds"] = data.get("fold_roc_auc")
                        except Exception:
                            pass
            rows.append(entry)
        results[case] = rows
out["results"] = results
out["official_data"] = listing(ROOT / "mlebench_official_data")
out["other_assets"] = [{"name": name, "entries": listing(ROOT / name, 25)}
                       for name in ("mlebench_model_cache", "mlebench", "datasets",
                                    "siim_private_score", "logs")
                       if (ROOT / name).is_dir()]
print(json.dumps(out, ensure_ascii=True, default=str))
'''


def sha(path: Path) -> str:
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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

    session = "session_siim_inventory_" + namespace
    api("POST", "/v1/sessions", {
        "session_id": session,
        "objective": "Read-only SIIM HPC inventory operator; no training, no writes.",
        "permission_level": "observe",
        "workspace_root": str(BASE / "data/acceptance/siim-calibration-20260908/operator"),
        "metadata": {"managed_hpc_identity": binding, "tenant_id": tenant,
                     "owner_principal_id": owner,
                     "run_allowed_tool_names": ["hpc_verify"]}})
    verified = api("POST", "/v1/sessions/" + session + "/tools",
                   {"tool_name": "hpc_verify", "arguments": {},
                    "idempotency_key": "inventory-identity-" + namespace})
    evidence = (verified.get("result") or {}).get("content") or {}
    from xsci.terminal_tools import hpc_identity_evidence_complete
    if not (verified.get("result") or {}).get("ok") or not hpc_identity_evidence_complete(
            evidence, expected_profile=binding["credential_profile"],
            expected_job_id=binding["job_id"]):
        raise ValueError("inventory_operator_identity_failed")
    return session, evidence


def main() -> int:
    namespace = sys.argv[1] if len(sys.argv) > 1 else "inv-" + str(time.time_ns())[-8:]
    receipt_path = OUTPUT / ("hpc-inventory-" + namespace + ".json")
    if receipt_path.exists():
        raise ValueError("operation_exists")
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8-sig"))
    binding = policy["managed_hpc_identity"]
    import shlex
    sys.path[:0] = [str(BASE / "bundle/runtime"),
                    "C:/EvoMind/releases/664a636ddd419a66f73cc10820c0a16429784866/src"]
    session, evidence = observer_identity(binding, policy["tenant_id"],
                                          policy["owner_principal_id"], namespace)
    from evomind_runtime.tools import _load_bound_hpc_config
    from research_agent_workstation.server.core.gpu_credentials import connect_ssh
    client = connect_ssh(_load_bound_hpc_config(binding["credential_profile"],
                                                binding["job_id"], session), timeout=30)
    try:
        _, stdout, stderr = client.exec_command("python3 -c " + shlex.quote(WORKER), timeout=600)
        raw = stdout.read(8 * 1024 * 1024)
        code = stdout.channel.recv_exit_status()
        payload = json.loads(raw) if raw else {"status": "empty"}
        if code:
            raise ValueError("inventory_worker_failed:" +
                             stderr.read(4096).decode("utf-8", "replace")[:200])
    finally:
        client.close()
    receipt = {"schema": "evomind.siim_hpc_inventory_operator.v1", "namespace": namespace,
               "at": datetime.now(timezone.utc).isoformat(), "session_id": session,
               "job_id": binding["job_id"], "identity_samples": evidence.get("samples_passed"),
               "result": payload, "remote_writes": 0, "training_started": False}
    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "ok", "receipt": str(receipt_path),
                      "cases": len(payload.get("results") or {})}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as _exc:  # noqa: BLE001
        import traceback
        _ns = sys.argv[1] if len(sys.argv) > 1 else "unknown"
        _path = OUTPUT / ("hpc-inventory-" + _ns + ".error.json")
        try:
            OUTPUT.mkdir(parents=True, exist_ok=True)
            _path.write_text(json.dumps({
                "schema": "evomind.siim_hpc_inventory_operator.error.v1",
                "namespace": _ns, "error": type(_exc).__name__ + ": " + str(_exc)[:400],
                "traceback": traceback.format_exc()[-1200:],
                "at": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False, indent=2),
                encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
        raise SystemExit(1)
