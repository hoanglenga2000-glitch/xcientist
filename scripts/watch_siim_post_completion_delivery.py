#!/usr/bin/env python3
"""Publish the current SIIM Run immediately after its existing supervisor exits.

The watcher is local and downstream-only: it creates no Run, opens no HPC
connection, sends no signal, performs no grading, and never records video.  It
only waits for the already-running end-to-end supervisor to report completion,
then transactionally refreshes the nine-page R1-R4 delivery and builds the
evidence-bound 92-second narration manifest.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for _entry in (str(PROJECT_ROOT), str(SRC_ROOT)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from scripts import manage_siim_job89508_campaign as campaign  # noqa: E402
from scripts import refresh_siim_multiround_delivery as refresh  # noqa: E402
from scripts import supervise_siim_job89508_end_to_end as supervisor  # noqa: E402

SCHEMA = "evomind.siim.post_completion_delivery_watch.v1"


class PostCompletionWatchError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise PostCompletionWatchError(f"missing or unsafe {label}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PostCompletionWatchError(f"invalid {label}") from exc
    if not isinstance(payload, dict):
        raise PostCompletionWatchError(f"{label} is not a JSON object")
    return payload


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        import psutil

        return psutil.pid_exists(pid)
    except ImportError:
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True


def acquire_lock(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        try:
            pid = int(path.read_text(encoding="ascii").strip())
        except (OSError, UnicodeError, ValueError):
            pid = -1
        if process_exists(pid):
            raise PostCompletionWatchError(f"post-completion watcher already runs as PID {pid}")
        path.unlink(missing_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(descriptor, str(os.getpid()).encode("ascii"))
    return descriptor


def load_narration_module(project_root: Path) -> Any:
    script = (
        project_root
        / "video-production"
        / "siim-isic-melanoma-commercial-v1"
        / "scripts"
        / "build_narration_manifest.py"
    )
    scripts_dir = str(script.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("evomind_siim_narration", script)
    if spec is None or spec.loader is None:
        raise PostCompletionWatchError("narration builder could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def publish(
    project_root: Path,
    run_id: str,
    *,
    refresh_fn: Any = refresh.refresh_delivery,
) -> dict[str, Any]:
    refresh_result = refresh_fn(project_root, run_id)
    if refresh_result.get("status") != "verified":
        raise PostCompletionWatchError("multiround delivery refresh did not verify")
    video_root = project_root / "video-production" / "siim-isic-melanoma-commercial-v1"
    contract = video_root / "preproduction" / "production-contract.json"
    narration_path = video_root / "preproduction" / "narration-92s.json"
    narration = load_narration_module(project_root)
    narration_result = narration.build_manifest(project_root, contract)
    if narration_result.get("run_id") != run_id or float(narration_result.get("duration_seconds") or 0) != 92.0:
        raise PostCompletionWatchError("narration manifest contract changed")
    atomic_json(narration_path, narration_result)
    return {
        "refresh": refresh_result,
        "narration_manifest": str(narration_path),
    }


def watch(project_root: Path, run_id: str, *, poll_seconds: int) -> dict[str, Any]:
    paths = supervisor.SupervisorPaths.build(project_root, run_id)
    state_path = paths.campaign_dir / "post_completion_delivery_watch.json"
    events_path = paths.campaign_dir / "post_completion_delivery_watch.jsonl"
    lock_path = paths.campaign_dir / ".post_completion_delivery_watch.lock"
    descriptor = acquire_lock(lock_path)
    started_at = utc_now()
    try:
        while True:
            current = read_json(paths.state, "end-to-end supervisor state")
            if current.get("run_id") != run_id:
                raise PostCompletionWatchError("end-to-end supervisor belongs to another Run")
            status = str(current.get("status") or "")
            phase = str(current.get("phase") or "")
            event = {
                "schema": SCHEMA,
                "captured_at": utc_now(),
                "started_at": started_at,
                "run_id": run_id,
                "status": "waiting" if status != "completed" else "publishing",
                "supervisor_status": status,
                "supervisor_phase": phase,
                "official_submission_executed": False,
                "signals_sent": 0,
                "other_processes_modified": False,
                "watcher_pid": os.getpid(),
            }
            atomic_json(state_path, event)
            with events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
            if status == "completed":
                products = publish(project_root, run_id)
                result = {
                    **event,
                    "captured_at": utc_now(),
                    "status": "completed",
                    "products": products,
                }
                atomic_json(state_path, result)
                with events_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                return result
            if status in {"failed", "blocked", "cancelled", "aborted"}:
                raise PostCompletionWatchError(
                    f"end-to-end supervisor stopped in {status}/{phase}"
                )
            time.sleep(max(15, poll_seconds))
    finally:
        os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--run-id", default=campaign.DEFAULT_RUN_ID)
    parser.add_argument("--poll-seconds", type=int, default=60)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = watch(args.project_root.resolve(), args.run_id, poll_seconds=args.poll_seconds)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
