from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

EXPECTED_V4_SHA256 = "D73F3966F0B873CB83D462C76D285AB792A7117658D13DDEC143E9F2FF624550"
PUBLIC_TEXT_FILES = (
    "narration-v5.json",
    "subtitles-v5.srt",
    "subtitles-v5.ass",
)
FORBIDDEN_PATTERNS = {
    "private_gpu_model": re.compile(r"\bA800\b", re.IGNORECASE),
    "internal_run_id": re.compile(r"qwen7b_(?:qlora|refine)_\d", re.IGNORECASE),
    "internal_account": re.compile(r"\b(?:aimslab[-_/]|jinghw)\b", re.IGNORECASE),
    "private_ipv4": re.compile(
        r"\b(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})\b"
    ),
    "remote_workspace": re.compile(r"/hpc2hdd/", re.IGNORECASE),
    "windows_path": re.compile(r"\b[A-Z]:\\", re.IGNORECASE),
    "credential_label": re.compile(r"\b(?:password|passwd|secret|token)\b", re.IGNORECASE),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail-closed public wording gate for the EvoMind V5 video.")
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--require-generated", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    root = args.video_root.resolve()
    blockers: list[str] = []
    checked: list[dict[str, Any]] = []
    combined: list[str] = []

    for relative in PUBLIC_TEXT_FILES:
        source = root / relative
        if not source.is_file():
            if relative == "narration-v5.json" or args.require_generated:
                blockers.append(f"missing public text source: {relative}")
            continue
        text = read_text(source)
        combined.append(text)
        checked.append({"file": relative, "bytes": source.stat().st_size, "sha256": sha256(source)})
        for label, pattern in FORBIDDEN_PATTERNS.items():
            if pattern.search(text):
                blockers.append(f"forbidden public text ({label}) in {relative}")

    public_text = "\n".join(combined)
    if not re.search(r"\bA40\b", public_text, re.IGNORECASE):
        blockers.append("public wording does not identify the A40 recommended configuration")
    if "\u7981\u7528" not in public_text:
        blockers.append("public wording does not state that local GPU use is disabled")

    v4 = root / "draft" / "evomind-enterprise-v4-picture-edit.mp4"
    if not v4.is_file():
        blockers.append("reviewed V4 picture source is missing")
        v4_hash = None
    else:
        v4_hash = sha256(v4)
        if v4_hash != EXPECTED_V4_SHA256:
            blockers.append("V4 picture source changed after the reviewed A40-only visual baseline")

    result = {
        "schema": "evomind.video.v5_public_text_gate.v1",
        "status": "passed" if not blockers else "blocked",
        "public_gpu_wording": "A40 recommended configuration",
        "v4_picture_sha256": v4_hash,
        "checked_files": checked,
        "blockers": blockers,
    }
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if not blockers else 2


if __name__ == "__main__":
    raise SystemExit(main())
