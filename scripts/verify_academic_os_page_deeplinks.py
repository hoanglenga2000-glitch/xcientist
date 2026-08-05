from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "docs" / "evomind_pages_20260708"
CDP_VERIFIER = ROOT / "scripts" / "verify_academic_os_page_deeplinks.mjs"


def fail(message: str, evidence: dict | None = None) -> None:
    raise SystemExit(
        json.dumps(
            {"status": "failed", "message": message, "evidence": evidence or {}},
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify EvoMind page deep links with a persistent Chromium CDP session."
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8088",
        help="Running dashboard URL.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="Directory for screenshot and DOM artifacts.",
    )
    args = parser.parse_args()

    if not CDP_VERIFIER.is_file():
        fail("CDP verifier is missing", {"path": str(CDP_VERIFIER)})

    command = [
        "node",
        str(CDP_VERIFIER),
        "--url",
        args.url,
        "--out-dir",
        args.out_dir,
    ]
    env = os.environ.copy()
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=240,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        fail(
            "persistent Chromium deep-link verification timed out",
            {
                "command": command,
                "timeout_seconds": 240,
                "stdout_tail": (exc.stdout or "")[-2000:]
                if isinstance(exc.stdout, str)
                else "",
                "stderr_tail": (exc.stderr or "")[-2000:]
                if isinstance(exc.stderr, str)
                else "",
            },
        )

    if completed.stdout:
        print(completed.stdout.rstrip())
    if completed.returncode != 0:
        fail(
            "persistent Chromium deep-link verification failed",
            {
                "command": command,
                "returncode": completed.returncode,
                "stdout_tail": completed.stdout[-5000:],
                "stderr_tail": completed.stderr[-5000:],
            },
        )


if __name__ == "__main__":
    main()
