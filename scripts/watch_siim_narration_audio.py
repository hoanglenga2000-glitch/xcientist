#!/usr/bin/env python3
"""Build the 92-second voice track as soon as final SIIM evidence is published.

This local downstream watcher opens no HPC connection, creates no Run, performs
no grading and records no video.  It only waits for the existing delivery
watcher to finish, then invokes the already-verified voice-only narration
builder with the evidence-bound narration manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "evomind_siim_isic_a800_job90353_20260730_095826"
CAMPAIGN_ROOT = (
    PROJECT_ROOT
    / "workspace"
    / "hpc"
    / "job90353_siim_campaign"
    / RUN_ID
)
VIDEO_ROOT = PROJECT_ROOT / "video-production" / "siim-isic-melanoma-commercial-v1"
DELIVERY_WATCH = CAMPAIGN_ROOT / "post_completion_delivery_watch.json"
STATE_PATH = CAMPAIGN_ROOT / "narration_audio_watch.json"
EVENTS_PATH = CAMPAIGN_ROOT / "narration_audio_watch.jsonl"
LOCK_PATH = CAMPAIGN_ROOT / ".narration_audio_watch.lock"
NARRATION_PATH = VIDEO_ROOT / "preproduction" / "narration-92s.json"
AUDIO_ROOT = VIDEO_ROOT / "audio"
BUILDER = PROJECT_ROOT / "video-production" / "local4060-ai-scientist-v2" / "build_user_narration.py"


class NarrationAudioWatchError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise NarrationAudioWatchError(f"missing or unsafe {label}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise NarrationAudioWatchError(f"invalid {label}") from exc
    if not isinstance(payload, dict):
        raise NarrationAudioWatchError(f"{label} is not a JSON object")
    return payload


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
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
            raise NarrationAudioWatchError(f"narration watcher already runs as PID {pid}")
        path.unlink(missing_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(descriptor, str(os.getpid()).encode("ascii"))
    return descriptor


def write_event(payload: Mapping[str, Any]) -> None:
    atomic_json(STATE_PATH, payload)
    with EVENTS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True) + "\n")


def build_audio() -> dict[str, Any]:
    narration = read_json(NARRATION_PATH, "narration manifest")
    if narration.get("run_id") != RUN_ID:
        raise NarrationAudioWatchError("narration manifest belongs to another Run")
    if narration.get("status") != "ready" or float(narration.get("duration_seconds") or 0) != 92.0:
        raise NarrationAudioWatchError("narration manifest is not ready for 92 seconds")
    if narration.get("voice_count") != 1 or narration.get("background_music") is not False:
        raise NarrationAudioWatchError("narration voice contract changed")
    if not BUILDER.is_file() or BUILDER.is_symlink():
        raise NarrationAudioWatchError("narration builder is missing or unsafe")
    AUDIO_ROOT.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            sys.executable,
            str(BUILDER),
            "--input",
            str(NARRATION_PATH),
            "--output-dir",
            str(AUDIO_ROOT),
            "--total-seconds",
            "92",
            "--maximum-tempo",
            "1.20",
        ],
        cwd=str(PROJECT_ROOT),
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
    )
    required = {
        "wav": AUDIO_ROOT / "narration-user-first-48k.wav",
        "aac": AUDIO_ROOT / "narration-user-first-48k.m4a",
        "subtitles": AUDIO_ROOT / "subtitles-user-first.srt",
        "manifest": AUDIO_ROOT / "narration-build-manifest.json",
        "qa": AUDIO_ROOT / "narration-audio-qa.json",
    }
    for label, path in required.items():
        if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
            raise NarrationAudioWatchError(f"narration builder did not create {label}")
    qa = read_json(required["qa"], "narration audio QA")
    if qa.get("status") != "passed":
        raise NarrationAudioWatchError("narration audio QA is not passed")
    return {
        "outputs": {label: str(path.resolve()) for label, path in required.items()},
        "builder_stdout_tail": completed.stdout[-1000:],
    }


def watch(*, poll_seconds: int) -> dict[str, Any]:
    descriptor = acquire_lock(LOCK_PATH)
    started_at = utc_now()
    try:
        while True:
            delivery = read_json(DELIVERY_WATCH, "post-completion delivery watcher")
            if delivery.get("run_id") != RUN_ID:
                raise NarrationAudioWatchError("delivery watcher belongs to another Run")
            status = str(delivery.get("status") or "")
            event = {
                "schema": "evomind.siim.narration_audio_watch.v1",
                "captured_at": utc_now(),
                "started_at": started_at,
                "run_id": RUN_ID,
                "status": "waiting" if status != "completed" else "building",
                "delivery_watch_status": status,
                "watcher_pid": os.getpid(),
                "recording_started": False,
                "official_submission_executed": False,
                "signals_sent": 0,
                "other_processes_modified": False,
            }
            write_event(event)
            if status == "completed":
                products = build_audio()
                result = {
                    **event,
                    "captured_at": utc_now(),
                    "status": "completed",
                    "products": products,
                }
                write_event(result)
                return result
            if status in {"failed", "blocked", "cancelled", "aborted"}:
                raise NarrationAudioWatchError(f"delivery watcher stopped in {status}")
            time.sleep(max(15, poll_seconds))
    finally:
        os.close(descriptor)
        LOCK_PATH.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = watch(poll_seconds=args.poll_seconds)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
