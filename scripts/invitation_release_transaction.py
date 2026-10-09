"""Guarded code/config cutover. Ownership migration is a separate operation."""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import urllib.request
import uuid


ROOT = Path("C:/ProgramData/EvoMind")
ROLES = ("tenant_enrollment_queue", "llm_gateway", "python_runtime", "web")
MANUAL_LAUNCHER_SHA256 = "55ef6200195f104c989fcb054dfb68f82c1a9c888be04bd80839c7cd2bd62a67"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def code_tree(path: Path) -> dict:
    rows = []
    for file in sorted(path.glob("*.py"), key=lambda value: value.name):
        if file.is_symlink():
            raise ValueError("runtime_source_alias")
        rows.append(f"{file.name}|{file.stat().st_size}|{sha(file)}")
    return {"files": len(rows), "sha256": hashlib.sha256("\n".join(rows).encode()).hexdigest()}


def health() -> dict:
    with urllib.request.urlopen("http://127.0.0.1:8088/api/healthz", timeout=8) as response:
        body = json.load(response)
        return {"http_status": response.status, "status": body.get("status"), "build_id": body.get("build_id")}


def quiescent_history(root: Path, *, recovering_ids=()) -> dict:
    """Fingerprint reviewed history without settling, cancelling, or replaying it."""
    launcher = root / 'bundle/runtime/run_python_runtime.py'
    if sha(launcher) != MANUAL_LAUNCHER_SHA256:
        raise RuntimeError('manual_startup_contract_changed')
    config = json.loads((root / 'config/node-config.json').read_text(encoding='utf-8-sig'))
    if config.get('hpc', {}).get('state') != 'blocked':
        raise RuntimeError('hpc_startup_not_blocked')
    allowed_recovery = sorted(set(recovering_ids))
    connection = sqlite3.connect((root / 'data/workspace/runtime/runtime.sqlite3').as_uri() + '?mode=ro', uri=True, timeout=5)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute('PRAGMA query_only=ON')
        connection.execute('BEGIN')
        runs = connection.execute("SELECT * FROM assistant_runs WHERE status NOT IN ('completed','failed','blocked','cancelled') ORDER BY id").fetchall()
        for run in runs:
            session = connection.execute('SELECT * FROM sessions WHERE id=?', (run['session_id'],)).fetchone()
            if session is None or session['status'] != run['status']:
                raise RuntimeError('history_session_state_mismatch')
            if run['status'] == 'recovering':
                if run['id'] not in allowed_recovery:
                    raise RuntimeError('unreviewed_recovering_run')
            elif run['status'] == 'paused':
                if json.loads(session['metadata_json'] or '{}').get('user_pause_requested') is not True:
                    raise RuntimeError('pause_not_user_requested')
            else:
                raise RuntimeError('active_run_prevents_cutover')
        if sorted(run['id'] for run in runs if run['status'] == 'recovering') != allowed_recovery:
            raise RuntimeError('reviewed_recovery_set_changed')
        if connection.execute("SELECT count(*) FROM tool_calls WHERE status NOT IN ('completed','failed','cancelled','rejected','denied')").fetchone()[0]:
            raise RuntimeError('active_tool_prevents_cutover')
        if connection.execute("SELECT count(*) FROM approvals WHERE status NOT IN ('approved','expired','rejected','cancelled')").fetchone()[0]:
            raise RuntimeError('pending_approval_prevents_cutover')
        if connection.execute("SELECT count(*) FROM sessions WHERE status IN ('queued','planning','running','verifying','pausing','waiting_approval')").fetchone()[0]:
            raise RuntimeError('active_session_prevents_cutover')
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        fingerprints = {}
        for table in ('assistant_runs','sessions','tool_calls','approvals','events','turns','checkpoints','artifacts','benchmark_runs'):
            if table not in tables:
                continue
            digest = hashlib.sha256()
            count = 0
            columns = [row[1] for row in connection.execute(f'PRAGMA table_info({table})')]
            order = ','.join('"' + name + '"' for name in columns)
            for row in connection.execute(f'SELECT * FROM {table} ORDER BY {order}'):
                encoded = json.dumps(list(row), ensure_ascii=True, separators=(',', ':'), default=lambda value: value.hex()).encode()
                digest.update(len(encoded).to_bytes(8, 'big') + encoded)
                count += 1
            fingerprints[table] = {'rows': count, 'sha256': digest.hexdigest()}
        return {'schema': 'evomind.quiescent_history.v1', 'launcher_sha256': sha(launcher),
                'recovering_ids': allowed_recovery, 'preserved_runs': len(runs), 'tables': fingerprints}
    finally:
        connection.close()


def assert_idle(root: Path, *, preservation=None) -> None:
    if preservation is not None:
        observed = quiescent_history(root, recovering_ids=preservation.get('recovering_ids', []))
        if observed != preservation:
            raise RuntimeError('preserved_history_changed')
        return
    connection = sqlite3.connect((root / "data/workspace/runtime/runtime.sqlite3").as_uri() + "?mode=ro", uri=True, timeout=5)
    try:
        count = connection.execute("SELECT count(*) FROM assistant_runs WHERE status IN ('queued','planning','running','verifying','recovering','paused','waiting_approval')").fetchone()[0]
        if count:
            raise RuntimeError("nonterminal_runs_prevent_cutover")
    finally:
        connection.close()


def service(action: str, logs: Path) -> None:
    if action not in {"Stop", "Start"}:
        raise ValueError("service_action_rejected")
    command = ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "C:/SecureInput/Invoke-ServiceAccountAction.ps1", "-Action", action, "-TimeoutMinutes", "12"]
    print(json.dumps({"phase": "managed_service_" + action.lower(), "status": "running"}), flush=True)
    result = subprocess.run(command, capture_output=True, timeout=900)
    # Only aggregate log identities are published; service credentials remain private.
    (logs / (action.lower() + "-diagnostic.sha256")).write_text(hashlib.sha256(result.stdout + result.stderr).hexdigest())
    if result.returncode:
        raise RuntimeError("managed_service_" + action.lower() + "_failed")
    try:
        response = json.loads(result.stdout.decode("utf-8-sig").strip())
    except (ValueError, UnicodeError):
        raise RuntimeError("managed_service_response_invalid") from None
    if response.get("status") != "completed" or response.get("action") != action:
        raise RuntimeError("managed_service_action_unconfirmed")


def bundle_manifest(root: Path, baseline: dict, *, allow_runtime_change: bool) -> dict:
    """Verify all unchanged bundle files; reseal only the replaced runtime."""
    bundle = (root / 'bundle').resolve()
    if baseline.get('schema') != 'evomind.windows_bundle_integrity.v1' or Path(baseline.get('bundle_root', '')).resolve() != bundle:
        raise RuntimeError('bundle_seal_identity_invalid')
    previous = {}
    for row in baseline['files']:
        relative = row['path']
        target = contained(bundle / relative, bundle)
        if target.relative_to(bundle).as_posix() != relative or relative in previous:
            raise RuntimeError('bundle_seal_path_invalid')
        previous[relative] = {'path': relative, 'size': row['size'], 'sha256': row['sha256']}
    current = {}
    for directory, folders, files in os.walk(bundle, followlinks=False):
        folders[:] = [name for name in folders if name not in {'__pycache__', 'dist'}]
        for name in [*folders, *files]:
            item = Path(directory) / name
            if item.is_symlink() or getattr(item.stat(), 'st_file_attributes', 0) & 0x400:
                raise RuntimeError('bundle_alias_rejected')
        for name in files:
            item = Path(directory) / name
            relative = item.relative_to(bundle).as_posix()
            current[relative] = {'path': relative, 'size': item.stat().st_size, 'sha256': sha(item)}
    for relative in set(previous) | set(current):
        if allow_runtime_change and relative.startswith('runtime/evomind_runtime/'):
            continue
        if previous.get(relative) != current.get(relative):
            raise RuntimeError('unrelated_bundle_file_changed')
    if not current:
        raise RuntimeError('bundle_seal_empty')
    return {**baseline, 'sealed_at_utc': datetime.now(timezone.utc).isoformat(),
            'file_count': len(current), 'files': [current[name] for name in sorted(current)]}


def seal(root: Path, baseline: dict) -> None:
    # The immutable base release and toolchain were not changed by this
    # transaction. Rewalking their virtualenv in takeown can exceed the stop
    # window. Keep their existing seals; verify every unrelated bundle file.
    manifest = bundle_manifest(root, baseline, allow_runtime_change=True)
    config = json.loads((root / 'config/node-config.json').read_text(encoding='utf-8-sig'))
    account = str(config['dedicated_user'])
    runtime = contained(root / 'bundle/runtime/evomind_runtime', root / 'bundle')
    print(json.dumps({'phase': 'scoped_bundle_seal', 'status': 'running', 'file_count': manifest['file_count']}), flush=True)
    commands = [
        ['icacls.exe', str(runtime), '/inheritance:r', '/grant:r', account + ':RX', '*S-1-5-18:F', '*S-1-5-32-544:F', '/T', '/C', '/Q'],
        ['icacls.exe', str(runtime), '/grant:r', account + ':(OI)(CI)RX', '*S-1-5-18:(OI)(CI)F', '*S-1-5-32-544:(OI)(CI)F', '/Q'],
    ]
    for command in commands:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
    target = root / 'state/bundle-integrity.json'
    temporary = target.with_name('bundle-integrity.invitation-' + uuid.uuid4().hex + '.json')
    write_json(temporary, manifest)
    subprocess.run(['icacls.exe', str(temporary), '/inheritance:r', '/grant:r', account + ':R', '*S-1-5-18:F', '*S-1-5-32-544:F', '/Q'],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
    os.replace(temporary, target)
    print(json.dumps({'phase': 'scoped_bundle_seal', 'status': 'passed'}), flush=True)


def wait_health(build_id: str) -> dict:
    deadline = time.monotonic() + 120
    consecutive = 0
    observations = []
    while time.monotonic() < deadline:
        try:
            observed = health()
            observations.append(observed)
            if observed == {"http_status": 200, "status": "ready", "build_id": build_id}:
                consecutive += 1
                if consecutive == 3:
                    return {"status": "ready", "samples": observations[-3:]}
            else:
                consecutive = 0
        except Exception:
            consecutive = 0
        time.sleep(1)
    raise RuntimeError("candidate_health_not_confirmed")


def contained(path: Path, root: Path) -> Path:
    absolute = path.absolute()
    absolute.relative_to(root.absolute())
    if absolute.resolve(strict=False) != absolute:
        raise ValueError("transaction_path_alias_rejected")
    return absolute


def database_schema_sha256(database: Path) -> str:
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True, timeout=5)
    try:
        rows = connection.execute("SELECT type,name,tbl_name,COALESCE(sql,'') FROM sqlite_master WHERE type IN ('table','index','view','trigger') AND name NOT LIKE 'sqlite_%' ORDER BY type,name,tbl_name")
        digest = hashlib.sha256()
        for kind, name, table, sql in rows:
            digest.update(("\0".join((kind, name, table, re.sub(r"\s+", " ", sql.strip()))) + "\n").encode("utf-8"))
        return digest.hexdigest()
    finally:
        connection.close()


def production_preflight(root: Path, candidate: Path, build_id: str, source_identity: str, schema_identity: str, receipt: Path, *, schema_source_identity: str) -> dict:
    candidate = contained(candidate, root / "web-overlays")
    config_path = root / "config/node-config.json"
    config_hash = sha(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    if candidate == Path(config["web_runtime_root"]).absolute():
        raise ValueError("preflight_requires_inactive_candidate")
    if not all(re.fullmatch(r"[a-f0-9]{64}", value) for value in (source_identity, schema_identity, schema_source_identity)):
        raise ValueError("preflight_expected_identity_invalid")
    script = r"""
$ErrorActionPreference='Stop'
try {
  . (Join-Path $env:EVOMIND_PREFLIGHT_ROOT 'bundle/scripts/lib/Runtime.ps1')
  $cfg=Get-Content -LiteralPath (Join-Path $env:EVOMIND_PREFLIGHT_ROOT 'config/node-config.json') -Raw | ConvertFrom-Json
  $cfg | Add-Member -NotePropertyName web_runtime_root -NotePropertyValue $env:EVOMIND_PREFLIGHT_CANDIDATE -Force
  $verified=Resolve-WebRuntimeIdentity -Config $cfg
  $runtime=Get-Content -LiteralPath $verified.runtime_manifest_path -Raw | ConvertFrom-Json
  $source=Get-Content -LiteralPath (Join-Path $verified.root 'release-source-manifest.json') -Raw | ConvertFrom-Json
  $schema=Join-Path $verified.root 'node_modules/.prisma/client/schema.prisma'
  if ($verified.verified -ne $true -or $verified.mode -ne 'operational_overlay' -or
      $verified.overlay_id -cne $env:EVOMIND_PREFLIGHT_BUILD -or $runtime.build_id -cne $env:EVOMIND_PREFLIGHT_BUILD -or
      $source.build_id -cne $env:EVOMIND_PREFLIGHT_BUILD -or
      $verified.source_identity_sha256 -cne $env:EVOMIND_PREFLIGHT_SOURCE -or
      $source.source_tree_sha256 -cne $env:EVOMIND_PREFLIGHT_SOURCE -or
      $source.production_schema_sha256 -cne $env:EVOMIND_PREFLIGHT_SCHEMA_SOURCE -or
      $source.database_schema_sha256 -cne $env:EVOMIND_PREFLIGHT_SCHEMA -or
      $runtime.database_schema_sha256 -cne $env:EVOMIND_PREFLIGHT_SCHEMA -or
      (Get-Sha256File $schema) -cne $env:EVOMIND_PREFLIGHT_SCHEMA_SOURCE) { throw 'PREFLIGHT_IDENTITY_REJECTED' }
  [ordered]@{status='passed';build_id=[string]$runtime.build_id;source_identity_sha256=[string]$verified.source_identity_sha256;database_schema_sha256=[string]$runtime.database_schema_sha256;prisma_schema_source_sha256=[string]$source.production_schema_sha256;verified=$true} | ConvertTo-Json -Compress
} catch {
  $code=if([string]$_.Exception.Message -match '^([A-Z][A-Z0-9_]+)(?::|$)'){$Matches[1]}else{'PREFLIGHT_DIAGNOSTIC_WITHHELD'}
  [ordered]@{status='failed';error_code=$code} | ConvertTo-Json -Compress
  exit 1
}
"""
    environment = dict(os.environ)
    environment.update(EVOMIND_PREFLIGHT_ROOT=str(root), EVOMIND_PREFLIGHT_CANDIDATE=str(candidate),
                       EVOMIND_PREFLIGHT_BUILD=build_id, EVOMIND_PREFLIGHT_SOURCE=source_identity,
                       EVOMIND_PREFLIGHT_SCHEMA=schema_identity, EVOMIND_PREFLIGHT_SCHEMA_SOURCE=schema_source_identity)
    print(json.dumps({"phase": "production_startup_preflight", "status": "running", "build_id": build_id}), flush=True)
    result = {"schema": "evomind.production_startup_preflight.v1", "status": "failed", "build_id": build_id,
              "startup_contract_sha256": sha(root / "bundle/scripts/lib/Runtime.ps1"),
              "common_contract_sha256": sha(root / "bundle/scripts/lib/Common.ps1")}
    try:
        if database_schema_sha256(Path(config["data_root"]) / "prisma/workstation.db") != schema_identity:
            raise RuntimeError("production_database_schema_changed")
        process = subprocess.run(["powershell.exe", "-NoProfile", "-Command", script], env=environment, capture_output=True, timeout=240)
        answer = json.loads(process.stdout.decode("utf-8-sig").strip())
        if (process.returncode or answer.get("status") != "passed" or answer.get("verified") is not True
                or answer.get("build_id") != build_id or answer.get("source_identity_sha256") != source_identity
                or answer.get("database_schema_sha256") != schema_identity
                or answer.get("prisma_schema_source_sha256") != schema_source_identity):
            code = str(answer.get("error_code", "PREFLIGHT_RESPONSE_REJECTED"))
            result["error_code"] = code if re.fullmatch(r"[A-Z0-9_]{1,120}", code) else "PREFLIGHT_RESPONSE_REJECTED"
            raise RuntimeError("production_startup_preflight_rejected")
        if sha(config_path) != config_hash:
            raise RuntimeError("production_configuration_changed_during_preflight")
        result.update(status="passed", source_identity_sha256=source_identity, database_schema_sha256=schema_identity,
                      prisma_schema_source_sha256=schema_source_identity, production_config_unchanged=True)
        return result
    finally:
        write_json(receipt, result)
        print(json.dumps({"phase": "production_startup_preflight", **result}), flush=True)


def switch_files(root: Path, staged_runtime: Path, staged_web: Path, staged_acl: Path | None, backup: Path, build_id: str, *, stop, start, reseal, verify, preflight, prepare=None) -> dict:
    """Pure transaction core, also exercised against a disposable local fixture."""
    root = root.absolute()
    target_runtime = contained(root / "bundle/runtime/evomind_runtime", root)
    target_web = contained(root / "web-overlays" / build_id, root)
    target_acl = contained(root / "data/workspace/runtime/principal_access.sqlite3", root)
    config_path = contained(root / "config/node-config.json", root)
    backup = contained(backup, root)
    if target_web.exists() or backup.exists():
        raise ValueError("transaction_target_exists")
    # A rejected candidate must never enter the service-stop or rollback path.
    preflight()
    backup.mkdir(parents=True)
    old_config = config_path.read_bytes()
    config = json.loads(old_config.decode("utf-8-sig"))
    old_build = Path(config["web_runtime_root"]).name
    (backup / "node-config.json").write_bytes(old_config)
    had_acl = target_acl.is_file()
    if had_acl:
        shutil.copyfile(target_acl, backup / "principal_access.sqlite3")
    stopped = False
    runtime_moved = False
    web_moved = False
    acl_replaced = False
    result = {"status": "pending", "backup_root": str(backup), "old_build_id": old_build, "new_build_id": build_id}
    try:
        stop()
        stopped = True
        if prepare:
            prepare(backup)
        # All rename targets are resolved and checked above. No recursive delete.
        target_runtime.rename(backup / "evomind_runtime")
        runtime_moved = True
        staged_runtime.rename(target_runtime)
        staged_web.rename(target_web)
        web_moved = True
        # None is the code-only path. The legacy explicit core argument remains
        # for separately authorized migrations, never supplied by this CLI.
        if staged_acl is not None:
            temporary_acl = target_acl.with_name("principal_access.sqlite3.candidate")
            shutil.copyfile(staged_acl, temporary_acl)
            os.replace(temporary_acl, target_acl)
            acl_replaced = True
        config["web_runtime_root"] = str(target_web)
        write_json(config_path, config)
        reseal()
        # Start may launch processes and then fail its receipt checks. Treat
        # the node as potentially running before invoking it, so rollback
        # always confirms managed Stop before renaming live runtime files.
        stopped = False
        start()
        accepted = verify(build_id)
        result.update(status="canary_activated", verification=accepted, research_database_restored=False)
        return result
    except Exception as error:
        result.update(status="failed", error_class=type(error).__name__, error_code=str(error) if str(error).replace("_", "").isalnum() else "diagnostic_withheld")
        try:
            if not stopped:
                stop()
                stopped = True
            if runtime_moved:
                if target_runtime.exists():
                    target_runtime.rename(backup / "failed_runtime")
                (backup / "evomind_runtime").rename(target_runtime)
            if web_moved and target_web.exists():
                target_web.rename(backup / "failed_web")
            temporary_config = config_path.with_name("node-config.json.restore")
            temporary_config.write_bytes(old_config)
            os.replace(temporary_config, config_path)
            # ACL expansion is safe to preserve if new writes occurred. The old
            # runtime ignores it; never restore the research DB over new work.
            result["acl_expansion_preserved"] = acl_replaced
            reseal()
            stopped = False
            start()
            result["rollback"] = verify(old_build)
            result["rollback_status"] = "restored_and_verified"
        except Exception as rollback_error:
            result["rollback_status"] = "failed"
            result["rollback_error_class"] = type(rollback_error).__name__
        return result
    finally:
        write_json(backup / "transaction-result.json", result)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--expected-plan-sha256")
    parser.add_argument('--history-acceptance', type=Path)
    args = parser.parse_args()
    stage = contained(args.stage_root, ROOT / "staging")
    accepted = json.loads((stage / "acceptance-result.json").read_text())
    if (accepted.get("status") != "passed" or not accepted.get("checks")
            or not all(row.get("passed") is True for row in accepted["checks"])
            or accepted.get("harness_sha256") != sha(stage / "verify_invitation_server_candidate.py")):
        raise RuntimeError("server_isolation_acceptance_required")
    manifest = json.loads((stage / "web/runtime-build-manifest.json").read_text())
    project_acceptance = None
    tensor_acceptance = None
    if (stage / "runtime/evomind_runtime/managed_tensor_train.py").is_file():
        tensor_acceptance = json.loads((stage / "tensor-acceptance.json").read_text())
        required_tensor = {"independent_tensor_reload", "tampered_model_rejected", "self_assertion_rejected", "foreign_protocol_rejected", "uncertain_gpu_budget_blocks_replacement", "managed_training_permission_contract"}
        tensor_passed = {row.get("name") for row in tensor_acceptance.get("checks", []) if row.get("passed") is True}
        if (tensor_acceptance.get("status") != "passed" or not required_tensor.issubset(tensor_passed)
                or tensor_acceptance.get("web_sha256") != sha(stage / "web.zip") or tensor_acceptance.get("runtime_sha256") != sha(stage / "runtime.zip")):
            raise RuntimeError("tensor_contract_acceptance_required")
    if (stage / "runtime/evomind_runtime/user_projects.py").is_file():
        project_report = stage / "project-acceptance.json"
        project_acceptance = json.loads(project_report.read_text())
        required = {"project_unauthenticated_denied", "project_csrf_required", "project_create", "project_replay", "project_payload_conflict", "project_foreign_read_denied", "project_foreign_list_empty", "project_owner_spoof_denied", "project_foreign_run_denied", "project_run_binding", "project_run_replay", "project_run_fingerprint", "project_survives_restart", "project_run_survives_restart", "project_single_invocation", "production_unchanged"}
        passed = {row.get("name") for row in project_acceptance.get("checks", []) if row.get("passed") is True}
        if (project_acceptance.get("status") != "passed" or not required.issubset(passed)
                or project_acceptance.get("web_sha256") != sha(stage / "web.zip")
                or project_acceptance.get("runtime_sha256") != sha(stage / "runtime.zip")):
            raise RuntimeError("project_api_acceptance_required")
    source_manifest = stage / "web/release-source-manifest.json"
    source = json.loads(source_manifest.read_text())
    build_id = accepted["build_id"]
    if build_id != manifest.get("build_id") or build_id != source.get("build_id"):
        raise RuntimeError("cutover_candidate_identity_mismatch")
    verifier = load_module(stage / "verify_invitation_server_candidate.py", "server_candidate_verifier")
    database = ROOT / "data/workspace/runtime/runtime.sqlite3"
    baseline_bundle = json.loads((ROOT / 'state/bundle-integrity.json').read_text(encoding='utf-8-sig'))
    bundle_manifest(ROOT, baseline_bundle, allow_runtime_change=False)
    preservation = None
    if args.history_acceptance:
        receipt_path = contained(args.history_acceptance, stage)
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get('status') != 'passed' or receipt.get('startup_replayed') is not False
                or receipt.get('runtime_sha256') != sha(stage / 'runtime.zip')
                or receipt.get('candidate_code') != code_tree(stage / 'runtime/evomind_runtime')
                or receipt.get('harness_sha256') != sha(stage / 'verify_invitation_history_preservation.py')
                or receipt.get('transaction_sha256') != sha(Path(__file__))):
            raise RuntimeError('history_preservation_acceptance_required')
        preservation = receipt['preservation']
    assert_idle(ROOT, preservation=preservation)
    candidate_web = contained(ROOT / "web-overlays" / ("preflight-invitation-" + uuid.uuid4().hex), ROOT / "web-overlays")
    verifier.extract(stage / "web.zip", sha(stage / "web.zip"), candidate_web, "operational-overlay-manifest.json")
    preflight_receipt = stage / ("production-preflight-" + uuid.uuid4().hex + ".json")
    preflight_result = production_preflight(ROOT, candidate_web, build_id, source["source_tree_sha256"],
                                            source["database_schema_sha256"], preflight_receipt,
                                            schema_source_identity=source["production_schema_sha256"])
    proposal = {
        "schema": "evomind.invitation_cutover_plan.v1", "build_id": build_id,
        "web_sha256": sha(stage / "web.zip"), "runtime_sha256": sha(stage / "runtime.zip"),
        "source_manifest_sha256": sha(source_manifest), "acceptance_sha256": sha(stage / "acceptance-result.json"),
        "harness_sha256": sha(stage / "verify_invitation_server_candidate.py"),
        "transaction_script_sha256": sha(Path(__file__)),
        "config_sha256": sha(ROOT / "config/node-config.json"), "old_runtime": code_tree(ROOT / "bundle/runtime/evomind_runtime"),
        "old_health": health(), "ownership": "unchanged_no_migration",
        "production_startup_preflight": preflight_result,
        "project_acceptance_sha256": sha(stage / "project-acceptance.json") if project_acceptance is not None else None,
        "tensor_acceptance_sha256": sha(stage / "tensor-acceptance.json") if tensor_acceptance is not None else None,
        "training_control_sha256": sha(ROOT / "config/research-control/policy.json") if (ROOT / "config/research-control/policy.json").is_file() else None,
        'history_acceptance_sha256': sha(args.history_acceptance) if args.history_acceptance else None,
        'preserved_history': preservation,
        'bundle_integrity_sha256': sha(ROOT / 'state/bundle-integrity.json'),
        'unchanged_release_seals': {name: sha(ROOT / 'state' / name) for name in ('release-acl-seal.json', 'runtime-artifact-integrity.json')},
    }
    if not args.apply:
        if args.plan.exists():
            raise ValueError("cutover_plan_already_exists")
        write_json(args.plan, proposal)
        print(json.dumps({"status": "planned", "plan_sha256": sha(args.plan), "build_id": build_id, "old_build_id": proposal["old_health"]["build_id"], "ownership": proposal["ownership"]}))
        return 0
    if not args.expected_plan_sha256 or sha(args.plan) != args.expected_plan_sha256 or json.loads(args.plan.read_text()) != proposal:
        raise RuntimeError("cutover_plan_or_live_state_changed")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateMutexW.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    kernel.ReleaseMutex.argtypes = [ctypes.c_void_p]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    mutex = kernel.CreateMutexW(None, False, "Global\\EvoMind-Byoa-V12-Deployment")
    if not mutex or kernel.WaitForSingleObject(mutex, 0) != 0:
        raise RuntimeError("deployment_lock_unavailable")
    try:
        assert_idle(ROOT, preservation=preservation)
        if (sha(ROOT / "config/node-config.json") != proposal["config_sha256"]
                or code_tree(ROOT / "bundle/runtime/evomind_runtime") != proposal["old_runtime"]):
            raise RuntimeError("cutover_live_state_changed_under_lock")
        candidate_runtime = stage / ("activation-runtime-" + uuid.uuid4().hex)
        verifier.extract(stage / "runtime.zip", proposal["runtime_sha256"], candidate_runtime, "runtime-hotfix-manifest.json")
        config = json.loads((ROOT / "config/node-config.json").read_text(encoding="utf-8-sig"))
        service_user = str(config["dedicated_user"])
        for folder in (candidate_runtime, candidate_web):
            subprocess.run(["icacls.exe", str(folder), "/grant", service_user + ":(OI)(CI)RX", "/T", "/Q"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        backup = ROOT / "backups" / ("invitation-cutover-" + uuid.uuid4().hex)

        def prepare(directory):
            assert_idle(ROOT, preservation=preservation)
            if sha(ROOT / 'state/bundle-integrity.json') != proposal['bundle_integrity_sha256']:
                raise RuntimeError('bundle_seal_changed_under_lock')
            for name, digest in proposal['unchanged_release_seals'].items():
                if sha(ROOT / 'state' / name) != digest:
                    raise RuntimeError('immutable_release_seal_changed')
            shutil.copyfile(ROOT / 'state/bundle-integrity.json', directory / 'bundle-integrity.json')
            subprocess.run(["icacls.exe", str(directory), "/inheritance:r", "/grant:r", "*S-1-5-18:(OI)(CI)F", "*S-1-5-32-544:(OI)(CI)F"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for name in ("runtime.sqlite3", "principal_access.sqlite3", "run_requests.sqlite3", "projects.sqlite3", "gpu_budget.sqlite3", "user_tasks.sqlite3", "model_profiles.sqlite3"):
                path = database.parent / name
                if path.exists():
                    original = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
                    copy = sqlite3.connect(directory / name)
                    try:
                        original.backup(copy)
                    finally:
                        copy.close(); original.close()

        def verified(expected):
            value = wait_health(expected)
            assert_idle(ROOT, preservation=preservation)
            return value

        result = switch_files(ROOT, candidate_runtime / "evomind_runtime", candidate_web, None, backup, build_id,
                              stop=lambda: service("Stop", stage), start=lambda: service("Start", stage),
                              reseal=lambda: seal(ROOT, baseline_bundle), verify=verified, prepare=prepare,
                              preflight=lambda: production_preflight(ROOT, candidate_web, build_id, source["source_tree_sha256"],
                                                                   source["database_schema_sha256"], stage / "activation-preflight.json",
                                                                   schema_source_identity=source["production_schema_sha256"]))
        print(json.dumps(result), flush=True)
        return 0 if result["status"] == "canary_activated" else 1
    finally:
        kernel.ReleaseMutex(mutex)
        kernel.CloseHandle(mutex)


if __name__ == "__main__":
    raise SystemExit(main())
