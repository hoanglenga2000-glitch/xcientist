#!/usr/bin/env python3
"""Wait for a fresh SIIM Run-bound GPU gate and launch the campaign once."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Protocol

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for entry in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from research_agent_workstation.server.core.gpu_credentials import (  # noqa: E402
    CredentialError,
    RetryableTransportError,
)
from scripts import manage_siim_job89508_campaign as campaign  # noqa: E402


class GateWatcherError(RuntimeError):
    pass


def is_transient_error(exc: BaseException) -> bool:
    """Return true only for remote failures that are safe to retry.

    Credential failures remain fail-closed unless the credential layer has
    explicitly classified the exception as a temporary transport failure.
    """
    if isinstance(exc, CredentialError):
        return isinstance(exc, RetryableTransportError)
    transient_names = {
        "AuthenticationException",
        "ChannelException",
        "ConnectionError",
        "EOFError",
        "NoValidConnectionsError",
        "SSHException",
        "TimeoutError",
    }
    return isinstance(exc, (EOFError, TimeoutError, ConnectionError)) or type(exc).__name__ in transient_names


class CampaignApi(Protocol):
    def gate(self, run_id: str, *, interval_seconds: int = 15) -> dict[str, Any]: ...

    def publish_status(self, run_id: str) -> dict[str, Any]: ...

    def launch(self, run_id: str, *, max_gate_age: int = 600) -> dict[str, Any]: ...


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


@contextmanager
def exclusive_process_lock(path: Path) -> Iterator[None]:
    """Hold a non-blocking one-byte lock for the watcher's full lifetime."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    acquired = False
    try:
        if path.stat().st_size == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except OSError as exc:
            raise GateWatcherError(
                f"a SIIM {campaign.JOB_TAG} gate watcher is already running"
            ) from exc
        yield
    finally:
        try:
            if acquired:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _paths(run_id: str) -> dict[str, Path]:
    root = campaign.local_paths(run_id)["root"]
    return {
        "root": root,
        "state": root / "gate_watch.json",
        "events": root / "gate_watch.jsonl",
        "lock": root / ".gate_watch.lock",
    }


def gate_summary(run_id: str, gate_payload: Mapping[str, Any], iteration: int) -> dict[str, Any]:
    samples = gate_payload.get("samples")
    sample_list = samples if isinstance(samples, list) else []
    last_sample = sample_list[-1] if sample_list and isinstance(sample_list[-1], dict) else {}
    gpus = last_sample.get("gpus") if isinstance(last_sample, dict) else []
    gpu = gpus[0] if isinstance(gpus, list) and gpus and isinstance(gpus[0], dict) else {}
    passed = gate_payload.get("passed") is True
    return {
        "schema": campaign.BINDING.schema("gate_watch_event"),
        "captured_at": utc_now(),
        "run_id": run_id,
        "iteration": iteration,
        "status": "go" if passed else "hold",
        "launch_decision": "GO" if passed else "HOLD",
        "gate_created_at": gate_payload.get("created_at"),
        "samples": len(sample_list),
        "free_memory_mib": int(gpu.get("memory_free_mib") or 0),
        "utilization_percent": int(gpu.get("utilization_percent") or 0),
        "hold_reasons": sorted(str(item) for item in (gate_payload.get("hold_reasons") or [])),
        "signals_sent": 0,
        "other_processes_modified": False,
    }


def run_iteration(
    run_id: str,
    *,
    iteration: int,
    sample_interval_seconds: int,
    max_gate_age: int,
    api: CampaignApi = campaign,
) -> dict[str, Any]:
    gate_payload = api.gate(run_id, interval_seconds=sample_interval_seconds)
    api.publish_status(run_id)
    summary = gate_summary(run_id, gate_payload, iteration)
    if gate_payload.get("passed") is True:
        launch_payload = api.launch(run_id, max_gate_age=max_gate_age)
        summary.update(
            {
                "status": "launched",
                "launch_decision": "GO",
                "launch_action": launch_payload.get("action"),
                "supervisor_pid": launch_payload.get("supervisor_pid"),
            }
        )
    return summary


def watch(
    run_id: str,
    *,
    poll_seconds: int,
    sample_interval_seconds: int,
    max_gate_age: int,
    max_wait_seconds: int,
    once: bool,
    api: CampaignApi = campaign,
) -> dict[str, Any]:
    if os.environ.get("EVOMIND_SIIM_HPC_JOB_ID", "").strip() != str(campaign.HPC_JOB_ID):
        raise GateWatcherError(
            f"EVOMIND_SIIM_HPC_JOB_ID must be {campaign.HPC_JOB_ID} for {campaign.JOB_TAG}"
        )
    if (
        os.environ.get("EVOMIND_HPC_CREDENTIAL_PROFILE", "").strip()
        != campaign.CREDENTIAL_PROFILE
    ):
        raise GateWatcherError(
            f"EVOMIND_HPC_CREDENTIAL_PROFILE must be {campaign.CREDENTIAL_PROFILE}"
        )
    if poll_seconds < 1 or sample_interval_seconds < 0 or max_gate_age < 1 or max_wait_seconds < 0:
        raise ValueError("watcher timing values are invalid")

    paths = _paths(run_id)
    started = time.monotonic()
    iteration = 0
    with exclusive_process_lock(paths["lock"]):
        while True:
            iteration += 1
            try:
                state = run_iteration(
                    run_id,
                    iteration=iteration,
                    sample_interval_seconds=sample_interval_seconds,
                    max_gate_age=max_gate_age,
                    api=api,
                )
            except Exception as exc:
                if not is_transient_error(exc):
                    raise GateWatcherError(
                        f"deterministic {campaign.JOB_TAG} gate or launch contract failed"
                    ) from exc
                state = {
                    "schema": campaign.BINDING.schema("gate_watch_event"),
                    "captured_at": utc_now(),
                    "run_id": run_id,
                    "iteration": iteration,
                    "status": "retryable_error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "signals_sent": 0,
                    "other_processes_modified": False,
                }
            state["watcher_pid"] = os.getpid()
            state["elapsed_seconds"] = round(time.monotonic() - started, 3)
            append_jsonl(paths["events"], state)
            write_json(paths["state"], state)
            print(json.dumps(state, ensure_ascii=False), flush=True)

            if state["status"] == "launched" or once:
                return state
            if max_wait_seconds and time.monotonic() - started >= max_wait_seconds:
                state = {
                    **state,
                    "captured_at": utc_now(),
                    "status": "wait_budget_exhausted",
                }
                append_jsonl(paths["events"], state)
                write_json(paths["state"], state)
                return state
            time.sleep(poll_seconds)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=campaign.DEFAULT_RUN_ID)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--sample-interval-seconds", type=int, default=15)
    parser.add_argument("--max-gate-age", type=int, default=600)
    parser.add_argument("--max-wait-seconds", type=int, default=0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        state = watch(
            args.run_id,
            poll_seconds=args.poll_seconds,
            sample_interval_seconds=args.sample_interval_seconds,
            max_gate_age=args.max_gate_age,
            max_wait_seconds=args.max_wait_seconds,
            once=args.once,
        )
        return 0 if state["status"] == "launched" else 4 if state["status"] == "hold" else 1
    except (GateWatcherError, ValueError, OSError) as exc:
        print(
            json.dumps(
                {
                    "schema": campaign.BINDING.schema("gate_watch_failure"),
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
