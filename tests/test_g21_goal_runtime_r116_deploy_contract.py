from __future__ import annotations

import hashlib
import http.client
import json
import shutil
import subprocess
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from evomind_runtime.competition_goal import FIXED_RUN_ID, load_goal_spec
from evomind_runtime.http_server import ensure_token, make_handler
from evomind_runtime.models import utc_now
from evomind_runtime.runtime import AgentRuntime


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts" / "Deploy-G21GoalRuntimeR116.ps1"
REMOTE = ROOT / "scripts" / "Deploy-G21GoalRuntimeR116Remote.ps1"
APPROVAL_BUILDER = ROOT / "scripts" / "new_g21_goal_runtime_deployment_approval.py"
VERIFIER = ROOT / "scripts" / "verify_g21_goal_runtime_r116_candidate.py"
BOARD = ROOT / "configs" / "g21_five_competition_goal_board_bootstrap.json"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def _request(port: int, token: str, method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    raw = json.dumps(body).encode() if body is not None else None
    headers = {"Authorization": f"Bearer {token}"}
    if raw is not None:
        headers.update({"Content-Type": "application/json", "Content-Length": str(len(raw))})
    try:
        connection.request(method, path, body=raw, headers=headers)
        response = connection.getresponse()
        return response.status, json.loads(response.read().decode())
    finally:
        connection.close()


def test_r116_powershell_scripts_parse() -> None:
    shells = [value for name in ("pwsh", "powershell") if (value := shutil.which(name))]
    if not shells:
        pytest.skip("PowerShell parser unavailable")
    for shell in shells:
        for path in (WRAPPER, REMOTE):
            command = (
                "$tokens=$null;$errors=$null;"
                f"[Management.Automation.Language.Parser]::ParseFile('{path}',[ref]$tokens,[ref]$errors)|Out-Null;"
                "if(@($errors).Count){$errors|%{$_.Message};exit 1}"
            )
            completed = subprocess.run(
                [shell, "-NoProfile", "-NonInteractive", "-Command", command],
                capture_output=True,
                text=True,
                check=False,
            )
            assert completed.returncode == 0, completed.stderr + completed.stdout


def test_service_mutations_use_only_the_managed_wrapper() -> None:
    wrapper = _text(WRAPPER)
    remote = _text(REMOTE)
    combined = wrapper + "\n" + remote
    for forbidden in (
        "Start-ScheduledTask",
        "Stop-ScheduledTask",
        "Register-ScheduledTask",
        "Unregister-ScheduledTask",
        "Stop-Process",
        "taskkill",
    ):
        assert forbidden not in combined
    assert "$action = 'C:\\SecureInput\\Invoke-ServiceAccountAction.ps1'" in remote
    assert "Invoke-ManagedAction Stop" in remote
    assert "Invoke-ManagedAction Start" in remote
    assert "managed_service_actions=@('Stop','Start','Stop','Start')" in remote


def test_nonterminal_run_fails_before_service_bundle_or_database_mutation() -> None:
    remote = _text(REMOTE)
    gate = remote.index("$runBefore=Read-RunGate")
    stop = remote.index("$stop=Invoke-ManagedAction Stop")
    database_backup = remote.index("$databaseBackup=Backup-Database")
    bundle_candidate = remote.index("$bundleCandidate=Join-Path $bundleRuntime")
    target_swap = remote.index("Move-Item -LiteralPath $target -Destination")
    assert gate < stop < database_backup < bundle_candidate < target_swap
    read_candidate = remote[remote.index("function Read-Candidate"):remote.index("function Backup-Database")]
    assert "$candidate=Join-Path $stage 'evomind_runtime'" in read_candidate
    assert "$bundleRuntime" not in read_candidate
    run_gate = remote[remote.index("function Read-RunGate"):remote.index("function Read-HpcState")]
    assert "$snapshot.terminal -isnot [bool]" in run_gate
    assert "-not [bool]$snapshot.terminal" in run_gate
    assert "R116_TARGET_RUN_NOT_TERMINAL_QUIESCENT" in run_gate
    assert "@('blocked','failed','cancelled','completed')" in run_gate
    assert "$active -ne 0 -or $pending -ne 0" in run_gate


def test_runtime_and_database_backup_rollback_are_closed() -> None:
    remote = _text(REMOTE)
    assert "foreach($suffix in @('','-wal','-shm'))" in remote
    assert "Copy-Item -LiteralPath $source -Destination" in remote
    assert "Restore-Database $backup $databaseBackup" in remote
    assert "Move-Item -LiteralPath (Join-Path $backup 'original\\evomind_runtime') -Destination $target" in remote
    assert "R116_DATABASE_ROLLBACK_HASH_REJECTED" in remote
    assert "R116_ROLLBACK_RUNTIME_REJECTED" in remote
    assert "rollback='restored_and_verified'" in remote
    assert "Wait-Exact 'http://127.0.0.1:8088/api/healthz'" in remote
    assert "Wait-Exact 'https://evomind.zhjjq.tech/api/healthz'" in remote


def test_fault_candidate_introduced_then_target_backup_move_fails_preserves_r115() -> None:
    remote = _text(REMOTE)
    introduced = remote.index("$bundleCandidateIntroduced=$true")
    target_move = remote.index("Move-Item -LiteralPath $target -Destination")
    backed_up = remote.index("$targetBackedUp=$true")
    assert introduced < target_move < backed_up
    catch = remote[remote.index("}catch{"):]
    assert "if($targetBackedUp)" in catch
    assert "if($targetBackedUp-and(Test-Path -LiteralPath $target)){Remove-Item" in catch
    assert "if($backup-and$mutated)" not in catch
    # A failed target move leaves targetBackedUp=false, so rollback only
    # restarts the managed service and finally removes bundle staging.
    assert "elseif($serviceStopped)" in catch
    assert "if($bundleCandidateIntroduced-and$bundleCandidate" in catch


def test_fault_target_backed_up_then_candidate_swap_fails_restores_without_delete() -> None:
    remote = _text(REMOTE)
    target_move = remote.index("Move-Item -LiteralPath $target -Destination")
    backed_up = remote.index("$targetBackedUp=$true")
    candidate_move = remote.index("Move-Item -LiteralPath $candidate.path -Destination $target")
    swapped = remote.index("$targetSwapped=$true")
    assert target_move < backed_up < candidate_move < swapped
    assert "if($targetBackedUp-and(Test-Path -LiteralPath $target)){Remove-Item" in remote
    assert "R116_RUNTIME_BACKUP_MISSING" in remote
    assert "Move-Item -LiteralPath (Join-Path $backup 'original\\evomind_runtime') -Destination $target" in remote


def test_fault_goal_written_then_restart_fails_restores_runtime_database_and_health() -> None:
    remote = _text(REMOTE)
    swapped = remote.index("$targetSwapped=$true")
    goal = remote.index("$goalBootstrap=Invoke-GoalBootstrap $approval")
    restart = remote.index("$restartStop=Invoke-ManagedAction Stop")
    assert swapped < goal < restart
    catch = remote[remote.index("}catch{"):]
    assert "Restore-Database $backup $databaseBackup" in catch
    assert "R116_ROLLBACK_RUNTIME_REJECTED" in catch
    assert "R116_ROLLBACK_WEB_TREE_CHANGED" in catch
    assert "Wait-Exact 'http://127.0.0.1:8088/api/healthz'" in catch
    assert "Wait-Exact 'https://evomind.zhjjq.tech/api/healthz'" in catch
    assert "rollback='restored_and_verified'" in catch


def test_backup_root_is_unique_and_never_force_merged() -> None:
    remote = _text(REMOTE)
    assert "+'-'+$TransactionId" in remote
    assert "R116_BACKUP_PATH_EXISTS" in remote
    assert "R116_DATABASE_BACKUP_PATH_EXISTS" in remote
    backup_section = remote[remote.index("$backup=Join-Path"):remote.index("$databaseBackup=Backup-Database")]
    assert "New-Item -ItemType Directory" in backup_section
    assert "-Force" not in backup_section


def test_preflight_and_postflight_cover_roles_ports_r115_run_and_local_hpc_hashes() -> None:
    remote = _text(REMOTE)
    for role in ("tenant_enrollment_queue", "llm_gateway", "python_runtime", "web"):
        assert role in remote
    for port in ("65068", "8765", "8088", "7890"):
        assert port in remote
    assert "EvoMindSvc" in remote
    assert "R116_ROLE_COUNT_REJECTED" in remote
    assert "R116_ROLE_PID_DUPLICATE" in remote
    assert "R116_LISTENER_COUNT_REJECTED" in remote
    assert "R116_LISTENER_PID_DUPLICATE" in remote
    assert "R116_ROLE_LISTENER_PID_MISMATCH" in remote
    assert "R116_ROLE_PID_NOT_FRESH" in remote
    assert "R116_LISTENER_PID_NOT_FRESH" in remote
    assert "Assert-FreshManaged $managedBefore $managedAfter" in remote
    assert "fresh_role_pids=$true" in remote
    assert "fresh_listener_pids=$true" in remote
    assert "R116_CURRENT_RUNTIME_NOT_R115" in remote
    assert "$webBefore=Tree $webRoot" in remote
    assert "R116_WEB_TREE_CHANGED" in remote
    assert "web_tree_unchanged=$true" in remote
    assert "b92cc96dae5ac1c1ed3641999eec56eda782b4959ac8709a0c0584ac1438955a" in remote
    assert "binding_sha256=Sha $bindingPath" in remote
    assert "byoa_tree=Tree (Join-Path $root 'byoa')" in remote
    assert "profile_tree=Tree $profileRoot" in remote
    assert "Assert-HpcSame $beforeHpc $afterHpc" in remote
    assert "hpc_accessed=$false" in remote
    assert "gpu_touched=$false" in remote


def test_goal_bootstrap_uses_http_and_proves_restart_reuse() -> None:
    remote = _text(REMOTE)
    section = remote[remote.index("function Read-RawSessionEvents"):remote.index("if((Test-Path -LiteralPath $resultPath)")]
    for path in (
        "/v1/goals",
        "/v1/goals/goal_g21_five_competition",
        "/v1/runs/$fixedRun/goal",
        "/v1/sessions/$fixedRun/events?after=$cursor",
    ):
        assert path in section
    assert "-Method Post" in section
    assert "-Method Put" in section
    assert "R116_GOAL_POST_AND_RECOVERY_GET_FAILED" in section
    assert "R116_GOAL_CREATED_EVENT_COUNT_REJECTED" in section
    assert "R116_GOAL_CREATED_EVENT_BINDING_REJECTED" in section
    assert "R116_GOAL_EVENT_PAGE_GAP" in section
    assert "R116_GOAL_EVENT_CURSOR_STALLED" in section
    assert "R116_GOAL_EVENT_PAGINATION_LIMIT" in section
    assert "/v1/runs/$fixedRun/events?after=0" not in remote
    assert "Invoke-GoalBootstrap $approval ([int64]$runBefore.last_event_seq)" in remote
    assert "/v1/sessions/$fixedRun" in section
    assert "goal_human_baseline_sha256" in section
    assert "R116_SESSION_GOAL_METADATA_REJECTED" in section
    assert "$goalAfter=Invoke-GoalBootstrap $approval" in remote
    assert "R116_GOAL_RESTART_REUSE_REJECTED" in remote
    assert "Restore-Database" in remote
    assert "sqlite3.connect" not in remote
    assert "UPDATE goal_records" not in remote


def test_raw_goal_event_window_ignores_more_than_500_historical_events_and_survives_restart(tmp_path: Path) -> None:
    runtime = AgentRuntime(tmp_path)
    runtime.create_session(session_id=FIXED_RUN_ID, workspace_root=str(tmp_path), objective="R116 event window fixture")
    now = utc_now()
    runtime.store.create_assistant_run(
        {
            "id": FIXED_RUN_ID, "session_id": FIXED_RUN_ID, "conversation_id": "r116-event-window",
            "prompt": "fixture", "task_root": str(tmp_path), "status": "cancelled", "plan": {},
            "attachment_ids": [], "retry_count": 0, "error_class": "", "error_message": "",
            "model_provider": "", "model": "", "created_at": now, "updated_at": now, "completed_at": now,
        }
    )
    for index in range(1205):
        runtime.store.append_event(FIXED_RUN_ID, "history.fixture", {"index": index})
    anchor = runtime.assistant.snapshot(FIXED_RUN_ID)["last_event_seq"]
    assert anchor >= 1205
    # Inject more than one raw API page after the deployment anchor. This
    # models unrelated concurrent events and proves the cursor loop, while the
    # 1,205 earlier events prove that pre-anchor history is excluded.
    for index in range(1105):
        runtime.store.append_event(FIXED_RUN_ID, "window.fixture", {"index": index})
    token = ensure_token(runtime.runtime_root)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(runtime, token))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = int(server.server_address[1])
    board = json.loads(BOARD.read_text(encoding="utf-8"))
    payload = {"run_id": FIXED_RUN_ID, "spec": load_goal_spec(str(ROOT / "configs" / "g21_five_competition_goal.json")), "board": board, "status": "blocked"}
    try:
        post_status, created = _request(port, token, "POST", "/v1/goals", payload)
        assert post_status == 201 and created["created"] is True
        upper = runtime.assistant.snapshot(FIXED_RUN_ID)["last_event_seq"]
        cursor = anchor
        window: list[dict] = []
        pages = 0
        while cursor < upper:
            raw_status, raw = _request(port, token, "GET", f"/v1/sessions/{FIXED_RUN_ID}/events?after={cursor}")
            assert raw_status == 200 and raw["events"]
            rows = [item for item in raw["events"] if item["seq"] <= upper]
            assert rows and all(item["seq"] > cursor for item in rows)
            window.extend(rows)
            cursor = rows[-1]["seq"]
            pages += 1
        assert pages >= 2 and cursor == upper
        created_rows = [item for item in window if item["event_type"] == "goal.created"]
        assert len(created_rows) == 1
        assert created_rows[0]["seq"] > anchor
        assert created_rows[0]["payload"]["goal_id"] == "goal_g21_five_competition"
        replay_status, replay = _request(port, token, "POST", "/v1/goals", payload)
        assert replay_status == 200 and replay["created"] is False
        put_status, _ = _request(port, token, "PUT", "/v1/goals/goal_g21_five_competition", {"board": board, "status": "blocked"})
        assert put_status == 200
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=5); runtime.close()

    restarted = AgentRuntime(tmp_path)
    restarted_token = ensure_token(restarted.runtime_root)
    restarted_server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(restarted, restarted_token))
    restarted_thread = threading.Thread(target=restarted_server.serve_forever, daemon=True)
    restarted_thread.start()
    try:
        upper = restarted.assistant.snapshot(FIXED_RUN_ID)["last_event_seq"]
        cursor = anchor
        window: list[dict] = []
        while cursor < upper:
            status, raw = _request(int(restarted_server.server_address[1]), restarted_token, "GET", f"/v1/sessions/{FIXED_RUN_ID}/events?after={cursor}")
            assert status == 200 and raw["events"]
            rows = [item for item in raw["events"] if item["seq"] <= upper]
            window.extend(rows)
            cursor = rows[-1]["seq"]
        assert [item["event_type"] for item in window].count("goal.created") == 1
        assert [item["event_type"] for item in window].count("goal.updated") == 0
    finally:
        restarted_server.shutdown(); restarted_server.server_close(); restarted_thread.join(timeout=5); restarted.close()


def test_success_receipt_reports_production_writes_truthfully() -> None:
    wrapper = _text(WRAPPER)
    remote = _text(REMOTE)
    assert "production_writes_performed=$true" in remote
    assert "production_write_category_count=5" in remote
    for category in (
        "runtime_backup",
        "database_backup",
        "runtime_atomic_swap",
        "release_reseal",
        "goal_api_persistence",
    ):
        assert category in remote
    assert "hpc_remote_writes=0" in remote
    assert "direct_signals_sent=0" in remote
    assert "production_writes_performed=[bool]" in remote
    assert "-not [bool]$result.production_writes_performed" in wrapper
    assert "[int]$result.hpc_remote_writes -ne 0" in wrapper
    assert "[int]$result.direct_signals_sent -ne 0" in wrapper


def test_deployer_accepts_only_the_separate_hash_bound_approval() -> None:
    wrapper = _text(WRAPPER)
    remote = _text(REMOTE)
    assert "[Parameter(Mandatory)][string]$ApprovalPath" in wrapper
    assert "evomind.g21_goal_runtime_deployment_approval.v1" in wrapper
    assert "evomind.g21_goal_runtime_deployment_approval.v1" in remote
    for binding in (
        "r115-progress-parser-r114-base",
        "r116-g21-goal-r115-base",
        "run_7b1efb878afb40f396db431e91f093a5",
        "overlay-controlled-secret-binding-r112-standalone-f38c40f2dbdc",
        "goal_spec_file_sha256",
        "human_baseline_file_sha256",
        "goal_board_file_sha256",
        "goal_board_sha256",
        "test_receipts",
        "remote_runner_sha256",
    ):
        assert binding in wrapper
        assert binding in remote or binding in {"r115-progress-parser-r114-base", "r116-g21-goal-r115-base"}
    assert "R116_APPROVAL_EXPIRED_OR_TIME_REJECTED" in wrapper
    assert "R116_APPROVAL_EXPIRED_OR_TIME_REJECTED" in remote
    assert "if ($ValidateOnly)" in wrapper


def test_formal_verifier_is_required_before_approval_generation() -> None:
    verifier = _text(VERIFIER)
    approval = _text(APPROVAL_BUILDER)
    for required in (
        "evomind.g21_goal_runtime_test_receipt.v1",
        "dual_extract_rounds",
        "python_files_compiled",
        "goal_http_smoke_rounds",
        "baseline_unchanged_files",
        "unsafe_entries",
        "duplicate_entries",
        "symlinks",
        "secret_scan",
        "git_diff_check",
        "hpc_accessed",
        "gpu_touched",
        "remote_writes",
        "targeted_goal_builder_junit",
        "runtime_regression_junit",
        "bootstrap_artifact_sha256",
    ):
        assert required in verifier
        assert required in approval
    assert "INDEPENDENT_TEST_RECEIPT_REQUIRED" in approval
    assert "CANDIDATE_PRODUCTION_TREE_HASH_REJECTED" in approval
    assert "R116_RUNTIME_TREE_HASH_REJECTED" in verifier
    assert "R115_MANIFEST_HASH_REJECTED" in verifier
    assert "R116_PATCH_DIFF_EVIDENCE_REJECTED" in verifier
    assert "R116_MANIFEST_ENTRY_DUPLICATE_OR_INVALID" in verifier
    assert 'manifest.get("file_count") != 32' in verifier
    assert 'metadata["goal_human_baseline_sha256"]' in verifier
    assert "bootstrap[\"goal_spec\"]" in verifier
    assert 'parser.add_argument("--goal-spec"' not in verifier
    assert 'parser.add_argument("--goal-board"' not in verifier
    assert "verify_no_plaintext_secrets.py" in verifier
    assert '["git", "diff", "--check"]' in verifier
    assert "run_repository_checks" not in verifier
    rejected = "37929a1c9baf682e4f946450f834f10fe3b7a78323084ae96948bd0e92f8de8d"
    rejection_receipt = "8900c32d41a47561715dc215ec2e560e338c2d164d6fa410554b7a2b83b5fb83"
    assert rejected in verifier and rejected in approval
    assert rejected in _text(WRAPPER) and rejected in _text(REMOTE)
    assert rejection_receipt in approval and rejection_receipt in _text(WRAPPER) and rejection_receipt in _text(REMOTE)
    assert "REJECTED_CANDIDATE_ZIP_SHA256" in verifier


def test_bootstrap_board_is_fixed_blocked_and_hash_bound() -> None:
    board = json.loads(BOARD.read_text(encoding="utf-8"))
    assert board["schema"] == "evomind.goal-board.v2"
    assert board["run_id"] == "run_7b1efb878afb40f396db431e91f093a5"
    assert board["allocation"] == "G21"
    assert board["goal_record_status"] == "blocked"
    assert board["completion_count"] == 0
    assert board["weather_actions"] == 0
    assert {item["competition"] for item in board["competitions"]} == {
        "cure_bench", "e2lmc", "mindgames", "open_polymer", "ariel_2025"
    }
    assert all(item["goal_status"] == "WAITING_EXACT_GATE" for item in board["competitions"])
    assert hashlib.sha256(BOARD.read_bytes()).hexdigest() == "7d5dcb469fb8debfd7f615957294ce46d24af2834532b7c15946737b51bde999"
    canonical = json.dumps(board, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(canonical).hexdigest() == "85288de01557f7801804c4a5e0c86d3dbb07a4d93acba1ec9c3277835f9a4cc6"
