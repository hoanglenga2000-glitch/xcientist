from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


FILES = (
    "scripts/build_super_agent_v1_baseline.py",
    "scripts/build_super_agent_v1_release_manifest.py",
    "scripts/run_super_agent_sftp_canary.py",
    "src/evomind_runtime/__init__.py",
    "src/evomind_runtime/agent_kernel_v2.py",
    "src/evomind_runtime/assistant_runs.py",
    "src/evomind_runtime/capabilities.py",
    "src/evomind_runtime/connectors.py",
    "src/evomind_runtime/competition_data.py",
    "src/evomind_runtime/credential_leases.py",
    "src/evomind_runtime/directory_broker.py",
    "src/evomind_runtime/ecosystem.py",
    "src/evomind_runtime/evidence_memory.py",
    "src/evomind_runtime/http_server.py",
    "src/evomind_runtime/hpc_runtime_overlay.py",
    "src/evomind_runtime/models.py",
    "src/evomind_runtime/policy.py",
    "src/evomind_runtime/recovery.py",
    "src/evomind_runtime/remote_connectors.py",
    "src/evomind_runtime/runtime.py",
    "src/evomind_runtime/store.py",
    "src/evomind_runtime/super_agent_runtime.py",
    "src/evomind_runtime/super_agent_store.py",
    "src/evomind_runtime/super_agent_tools.py",
    "src/evomind_runtime/tool_package_runtime.py",
    "src/evomind_runtime/tools.py",
    "tests/test_agent_kernel_v2.py",
    "tests/test_credential_leases_v2.py",
    "tests/test_evomind_runtime_http_server.py",
    "tests/test_remote_connectors_v1.py",
    "tests/test_super_agent_capabilities.py",
    "tests/test_super_agent_ecosystem.py",
    "tests/test_super_agent_runtime_integration.py",
    "tests/test_super_agent_runtime_v1.py",
    "tests/test_super_agent_store.py",
    "tests/test_super_agent_tools_v1.py",
    "tests/test_tool_package_runtime_v1.py",
    "web/research-agent-workstation/src/app/api/super-agent/status/route.ts",
    "web/research-agent-workstation/src/components/workstation/SuperAgentV1Panel.tsx",
    "web/research-agent-workstation/src/components/workstation/screens/RuntimeScreen.tsx",
    "web/research-agent-workstation/src/lib/super-agent-v1.test.ts",
    "web/research-agent-workstation/src/lib/super-agent-v1.ts",
)


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve(strict=True)
    rows = []
    for relative in FILES:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(relative)
        data = path.read_bytes()
        rows.append({"path": relative, "bytes": len(data), "sha256": sha(data)})
    canonical = "\n".join(
        f"{row['path']}|{row['bytes']}|{row['sha256']}" for row in rows
    ).encode("utf-8")
    payload = {
        "schema": "evomind.super_agent_v1_release_manifest.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "file_count": len(rows),
        "source_tree_sha256": sha(canonical),
        "files": rows,
    }
    output = Path(args.output).resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "file_count": len(rows),
        "source_tree_sha256": payload["source_tree_sha256"],
        "manifest_sha256": sha(output.read_bytes()),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
