#!/usr/bin/env python3
"""Activate a prepared SIIM bundle upgrade only at an idle same-Run boundary."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for entry in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from scripts import manage_siim_job89508_campaign as campaign  # noqa: E402
from scripts.watch_siim_job89508_gate import (  # noqa: E402
    exclusive_process_lock,
    is_transient_error,
)


class SameRunBundleUpgradeError(RuntimeError):
    pass


class UpgradeApi(Protocol):
    def status(self, run_id: str) -> Mapping[str, Any]: ...

    def local_supervisor_active(self, run_id: str) -> bool: ...

    def source_bundle_sha256(self) -> str: ...

    def ablation_report(self, run_id: str) -> Mapping[str, Any]: ...

    def prepare(self, run_id: str) -> Mapping[str, Any]: ...

    def gate(
        self,
        run_id: str,
        *,
        interval_seconds: int,
        output_path: Path,
    ) -> Mapping[str, Any]: ...

    def launch(
        self,
        run_id: str,
        *,
        gate_path: Path,
        launch_path: Path,
    ) -> Mapping[str, Any]: ...

    def start_local_supervisor(self, run_id: str) -> int: ...


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_value(value: object, *, label: str) -> str:
    normalized = str(value or "").strip().lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise SameRunBundleUpgradeError(f"{label} is not SHA-256")
    return normalized


def read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SameRunBundleUpgradeError(f"{label} is unreadable") from exc
    if not isinstance(payload, dict):
        raise SameRunBundleUpgradeError(f"{label} is not a JSON object")
    return payload


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(payload), ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        process_query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(
            process_query_limited_information,
            False,
            pid,
        )
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def process_command_line(pid: int) -> str | None:
    """Read a live process identity without sending signals or opening write handles."""

    if pid <= 0 or not process_exists(pid):
        return None
    if os.name == "nt":
        script = (
            "$ErrorActionPreference='Stop';"
            "$p=Get-CimInstance Win32_Process -Filter "
            "\"ProcessId=$env:EVOMIND_QUERY_PID\";"
            "if($null -eq $p){exit 3};[Console]::Out.Write($p.CommandLine)"
        )
        environment = os.environ.copy()
        environment["EVOMIND_QUERY_PID"] = str(pid)
        completed = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=environment,
            timeout=15,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None
    try:
        raw = (Path("/proc") / str(pid) / "cmdline").read_bytes()
    except OSError:
        return None
    return raw.replace(b"\0", b" ").decode("utf-8", "replace").strip() or None


def validate_upgrade_plan(path: Path, run_id: str) -> dict[str, Any]:
    plan = read_json(path, label="same-Run bundle upgrade plan")
    expected = {
        "schema": "evomind.siim.same_run_bundle_upgrade_plan.v1",
        "run_id": run_id,
        "job_id": campaign.HPC_JOB_ID,
        "credential_profile": campaign.CREDENTIAL_PROFILE,
        "signals_sent": 0,
        "other_processes_modified": False,
    }
    for key, value in expected.items():
        if plan.get(key) != value:
            raise SameRunBundleUpgradeError(f"bundle upgrade plan changed: {key}")
    old_bundle = sha256_value(plan.get("old_bundle_sha256"), label="old bundle")
    new_bundle = sha256_value(plan.get("new_bundle_sha256"), label="new bundle")
    if old_bundle == new_bundle:
        raise SameRunBundleUpgradeError("bundle upgrade plan does not change the bundle")
    preconditions = plan.get("activation_preconditions")
    if not isinstance(preconditions, dict):
        raise SameRunBundleUpgradeError("bundle upgrade preconditions are missing")
    if preconditions.get("remote_supervisor_process_exists") is not False:
        raise SameRunBundleUpgradeError("bundle upgrade must wait for remote process exit")
    if set(preconditions.get("allowed_remote_states") or ()) != {"failed", "needs_continuation"}:
        raise SameRunBundleUpgradeError("bundle upgrade boundary states changed")
    if preconditions.get("same_run_required") is not True:
        raise SameRunBundleUpgradeError("bundle upgrade lost the same-Run requirement")
    return plan


def validate_ablation_report(report: Mapping[str, Any], plan: Mapping[str, Any]) -> None:
    preconditions = plan["activation_preconditions"]
    if report.get("passed") is not True:
        raise SameRunBundleUpgradeError("ablation report has not passed")
    if int(report.get("fold_count") or 0) != int(preconditions["ablation_fold_count"]):
        raise SameRunBundleUpgradeError("ablation fold count differs from the upgrade plan")
    seeds = [int(value) for value in report.get("evaluation_seeds") or ()]
    expected_seeds = [int(value) for value in preconditions["ablation_evaluation_seeds"]]
    if seeds != expected_seeds or int(report.get("evaluation_seed_count") or 0) != len(expected_seeds):
        raise SameRunBundleUpgradeError("ablation evaluation seeds differ from the upgrade plan")
    if report.get("run_directory_action") != "resumed":
        raise SameRunBundleUpgradeError("ablation report was not produced by the bound resume run")


def boundary_decision(
    status_payload: Mapping[str, Any],
    *,
    local_supervisor_active: bool,
    plan: Mapping[str, Any],
) -> str:
    state = status_payload.get("state")
    remote = state if isinstance(state, dict) else {}
    remote_status = str(remote.get("status") or "missing")
    remote_process_exists = status_payload.get("process_exists") is True
    if remote_process_exists:
        return "waiting_remote_supervisor_boundary"
    allowed = set(plan["activation_preconditions"]["allowed_remote_states"])
    if remote_status not in allowed:
        raise SameRunBundleUpgradeError(
            f"idle remote campaign is not at an allowed upgrade boundary: {remote_status}"
        )
    if local_supervisor_active:
        return "waiting_local_supervisor_exit"
    return "activate"


class ProductionApi:
    def status(self, run_id: str) -> Mapping[str, Any]:
        return campaign.status(run_id)

    def local_supervisor_active(self, run_id: str) -> bool:
        state_path = campaign.local_paths(run_id)["root"] / "end_to_end_supervisor.json"
        if not state_path.is_file():
            return False
        state = read_json(state_path, label="local end-to-end supervisor state")
        pid = int(state.get("supervisor_pid") or 0)
        if not process_exists(pid):
            return False
        command_line = process_command_line(pid)
        if command_line is None:
            raise SameRunBundleUpgradeError(
                "live local supervisor PID identity is unreadable"
            )
        expected_script = str(
            PROJECT_ROOT / "scripts" / "supervise_siim_job89508_end_to_end.py"
        )
        return bool(
            expected_script.lower() in command_line.lower()
            and "--run-id" in command_line
            and run_id in command_line
        )

    def source_bundle_sha256(self) -> str:
        return str(campaign.bundle_manifest(campaign.source_records())["bundle_sha256"])

    def ablation_report(self, run_id: str) -> Mapping[str, Any]:
        local_plan = read_json(campaign.local_paths(run_id)["plan"], label="campaign plan")
        remote = campaign.remote_paths(run_id, str(local_plan["bundle_sha256"]))
        report_path = campaign.ensure_remote(
            f"{remote['campaign']}/ablation/runs/{run_id}_ablation/siim_preprocessing_ablation.json"
        )
        client, _config = campaign.connect()
        try:
            with client.open_sftp() as sftp:
                try:
                    with sftp.open(report_path, "r") as handle:
                        payload = json.loads(handle.read().decode("utf-8"))
                except OSError as exc:
                    raise SameRunBundleUpgradeError("remote ablation report is missing") from exc
        finally:
            client.close()
        if not isinstance(payload, dict):
            raise SameRunBundleUpgradeError("remote ablation report is invalid")
        return payload

    def prepare(self, run_id: str) -> Mapping[str, Any]:
        return campaign.prepare(run_id)

    def gate(
        self,
        run_id: str,
        *,
        interval_seconds: int,
        output_path: Path,
    ) -> Mapping[str, Any]:
        return campaign.gate(
            run_id,
            interval_seconds=interval_seconds,
            output_path=output_path,
            publish_default=False,
        )

    def launch(
        self,
        run_id: str,
        *,
        gate_path: Path,
        launch_path: Path,
    ) -> Mapping[str, Any]:
        return campaign.launch(
            run_id,
            max_gate_age=600,
            gate_path=gate_path,
            launch_record_path=launch_path,
        )

    def start_local_supervisor(self, run_id: str) -> int:
        if self.local_supervisor_active(run_id):
            raise SameRunBundleUpgradeError("old local end-to-end supervisor is still active")
        root = campaign.local_paths(run_id)["root"]
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        stdout_path = root / f"end_to_end_supervisor_bundle_upgrade_{stamp}.stdout.log"
        stderr_path = root / f"end_to_end_supervisor_bundle_upgrade_{stamp}.stderr.log"
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "supervise_siim_job89508_end_to_end.py"),
            "--run-id",
            run_id,
            "--poll-seconds",
            "60",
            "--sample-interval-seconds",
            "15",
            "--max-wait-seconds",
            "0",
            "--max-transient-failures",
            "8",
        ]
        environment = os.environ.copy()
        environment["EVOMIND_SIIM_HPC_JOB_ID"] = str(campaign.HPC_JOB_ID)
        environment["EVOMIND_HPC_CREDENTIAL_PROFILE"] = campaign.CREDENTIAL_PROFILE
        environment["EVOMIND_SIIM_RUN_ID"] = run_id
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        with stdout_path.open("ab", buffering=0) as stdout, stderr_path.open("ab", buffering=0) as stderr:
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                creationflags=creationflags,
                start_new_session=os.name != "nt",
            )
        time.sleep(2)
        if process.poll() is not None:
            raise SameRunBundleUpgradeError("replacement local supervisor exited during startup")
        return int(process.pid)


def next_sequence(root: Path) -> int:
    return len(list(root.glob("bundle_upgrade_launch_*.json"))) + 1


def run_iteration(
    run_id: str,
    plan: Mapping[str, Any],
    *,
    sample_interval_seconds: int,
    api: UpgradeApi,
) -> dict[str, Any]:
    status_payload = api.status(run_id)
    decision = boundary_decision(
        status_payload,
        local_supervisor_active=api.local_supervisor_active(run_id),
        plan=plan,
    )
    base = {
        "schema": "evomind.siim.same_run_bundle_upgrade_event.v1",
        "captured_at": utc_now(),
        "run_id": run_id,
        "status": decision,
        "signals_sent": 0,
        "other_processes_modified": False,
    }
    if decision != "activate":
        return base

    actual_source_bundle = sha256_value(api.source_bundle_sha256(), label="current source bundle")
    expected_bundle = sha256_value(plan["new_bundle_sha256"], label="planned new bundle")
    if actual_source_bundle != expected_bundle:
        raise SameRunBundleUpgradeError("current source bundle differs from the upgrade plan")
    report = api.ablation_report(run_id)
    validate_ablation_report(report, plan)

    prepared = api.prepare(run_id)
    if sha256_value(prepared.get("bundle_sha256"), label="prepared bundle") != expected_bundle:
        raise SameRunBundleUpgradeError("prepared deployment used a different bundle")
    root = campaign.local_paths(run_id)["root"]
    sequence = next_sequence(root)
    gate_path = root / f"bundle_upgrade_gpu_gate_{sequence:03d}.json"
    launch_path = root / f"bundle_upgrade_launch_{sequence:03d}.json"
    gate = api.gate(
        run_id,
        interval_seconds=sample_interval_seconds,
        output_path=gate_path,
    )
    if gate.get("passed") is not True:
        return {
            **base,
            "status": "activation_gate_hold",
            "bundle_sha256": expected_bundle,
            "gate_path": str(gate_path),
            "hold_reasons": list(gate.get("hold_reasons") or ()),
        }
    launch = api.launch(run_id, gate_path=gate_path, launch_path=launch_path)
    if launch.get("action") != "new_supervisor_started":
        raise SameRunBundleUpgradeError("bundle upgrade did not start a new remote supervisor")
    if sha256_value(launch.get("bundle_sha256"), label="launch bundle") != expected_bundle:
        raise SameRunBundleUpgradeError("bundle upgrade launch used a different bundle")
    local_pid = api.start_local_supervisor(run_id)
    return {
        **base,
        "status": "launched",
        "bundle_sha256": expected_bundle,
        "remote_supervisor_pid": int(launch.get("supervisor_pid") or 0),
        "local_supervisor_pid": local_pid,
        "gate_path": str(gate_path),
        "launch_path": str(launch_path),
        "ablation_fold_count": int(report["fold_count"]),
        "ablation_evaluation_seeds": list(report["evaluation_seeds"]),
        "ablation_run_directory_action": report["run_directory_action"],
    }


def watch(
    run_id: str,
    *,
    plan_path: Path,
    poll_seconds: int,
    sample_interval_seconds: int,
    max_wait_seconds: int,
    max_transient_failures: int,
    api: UpgradeApi | None = None,
) -> dict[str, Any]:
    if os.environ.get("EVOMIND_SIIM_HPC_JOB_ID", "").strip() != str(campaign.HPC_JOB_ID):
        raise SameRunBundleUpgradeError("HPC job binding is missing")
    if os.environ.get("EVOMIND_HPC_CREDENTIAL_PROFILE", "").strip() != campaign.CREDENTIAL_PROFILE:
        raise SameRunBundleUpgradeError("named HPC credential profile binding is missing")
    if poll_seconds < 1 or sample_interval_seconds < 0 or max_wait_seconds < 0:
        raise ValueError("bundle upgrade watcher timing is invalid")
    if max_transient_failures < 1:
        raise ValueError("bundle upgrade watcher retry limit is invalid")

    plan = validate_upgrade_plan(plan_path, run_id)
    api = api or ProductionApi()
    root = campaign.local_paths(run_id)["root"]
    state_path = root / "bundle_upgrade_watch.json"
    events_path = root / "bundle_upgrade_watch.jsonl"
    lock_path = root / ".bundle_upgrade_watch.lock"
    started = time.monotonic()
    iteration = 0
    transient_failures = 0
    with exclusive_process_lock(lock_path):
        while True:
            iteration += 1
            try:
                state = run_iteration(
                    run_id,
                    plan,
                    sample_interval_seconds=sample_interval_seconds,
                    api=api,
                )
                transient_failures = 0
            except Exception as exc:
                if not is_transient_error(exc):
                    raise
                transient_failures += 1
                if transient_failures > max_transient_failures:
                    raise SameRunBundleUpgradeError(
                        "bundle upgrade transport retry limit was exceeded"
                    ) from exc
                state = {
                    "schema": "evomind.siim.same_run_bundle_upgrade_event.v1",
                    "captured_at": utc_now(),
                    "run_id": run_id,
                    "status": "retryable_transport_error",
                    "error_type": type(exc).__name__,
                    "transient_failures": transient_failures,
                    "signals_sent": 0,
                    "other_processes_modified": False,
                }
            state = {
                **state,
                "iteration": iteration,
                "watcher_pid": os.getpid(),
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
            append_jsonl(events_path, state)
            write_json(state_path, state)
            print(json.dumps(state, ensure_ascii=False), flush=True)
            if state["status"] == "launched":
                return state
            if max_wait_seconds and time.monotonic() - started >= max_wait_seconds:
                exhausted = {**state, "captured_at": utc_now(), "status": "wait_budget_exhausted"}
                append_jsonl(events_path, exhausted)
                write_json(state_path, exhausted)
                return exhausted
            time.sleep(poll_seconds)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=campaign.DEFAULT_RUN_ID)
    parser.add_argument("--plan-path", type=Path)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--sample-interval-seconds", type=int, default=15)
    parser.add_argument("--max-wait-seconds", type=int, default=0)
    parser.add_argument("--max-transient-failures", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    plan_path = args.plan_path or campaign.local_paths(args.run_id)["root"] / "bundle_upgrade_plan.json"
    try:
        state = watch(
            args.run_id,
            plan_path=plan_path,
            poll_seconds=args.poll_seconds,
            sample_interval_seconds=args.sample_interval_seconds,
            max_wait_seconds=args.max_wait_seconds,
            max_transient_failures=args.max_transient_failures,
        )
        return 0 if state["status"] == "launched" else 1
    except (SameRunBundleUpgradeError, ValueError, OSError) as exc:
        print(
            json.dumps(
                {
                    "schema": "evomind.siim.same_run_bundle_upgrade_failure.v1",
                    "captured_at": utc_now(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "signals_sent": 0,
                    "other_processes_modified": False,
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
