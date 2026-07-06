#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any


BASE = Path("/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra")
COMPETITION_ID = "cdiscount-image-classification-challenge"
RAW = BASE / "mlebench_raw_data" / COMPETITION_ID
REPORTS = BASE / "reports"
LOGS = BASE / "logs"

SMALL_PROBE_FILES = [
    ("category_names.csv", 5 * 60),
    ("train_example.bson", 10 * 60),
    ("sample_submission.csv", 15 * 60),
]

EXPECTED_BYTES = {
    "category_names.csv": 416158,
    "train_example.bson": 631150,
    "sample_submission.csv": 34456858,
}

BIG_FILES = [
    ("test.bson", 8 * 60 * 60),
    ("train.bson", 30 * 60 * 60),
]


def active_big_downloads() -> list[str]:
    try:
        out = subprocess.check_output(["ps", "-eo", "pid,ppid,etime,cmd"], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return []
    active: list[str] = []
    for line in out.splitlines():
        if "kaggle competitions download" not in line:
            continue
        if COMPETITION_ID in line:
            continue
        active.append(line[:1000])
    return active


def existing_file_size(file_name: str) -> int:
    direct = RAW / file_name
    if direct.exists():
        return direct.stat().st_size
    zip_variant = RAW / f"{file_name}.zip"
    if zip_variant.exists():
        return zip_variant.stat().st_size
    return 0


def rescue_existing_generic_file(file_name: str) -> dict[str, int]:
    target = RAW / file_name
    if target.exists():
        return {}
    generic = RAW / "DownloadDataFile"
    expected = EXPECTED_BYTES.get(file_name)
    if generic.exists() and (expected is None or generic.stat().st_size == expected):
        generic.rename(target)
        return {target.name: target.stat().st_size}
    return {}


def normalize_downloaded_file(file_name: str, before: dict[str, int]) -> dict[str, int]:
    after = {candidate.name: candidate.stat().st_size for candidate in RAW.glob("*") if candidate.is_file()}
    changed = {name: size for name, size in after.items() if before.get(name) != size}
    if file_name in after or f"{file_name}.zip" in after:
        return changed

    generic = RAW / "DownloadDataFile"
    if generic.exists() and before.get(generic.name) != generic.stat().st_size:
        target = RAW / file_name
        if target.exists():
            target.unlink()
        generic.rename(target)
        changed.pop(generic.name, None)
        changed[target.name] = target.stat().st_size
        return changed

    # Some Kaggle CLI versions save single-file downloads as a zip named after the file.
    generic_zip = RAW / "DownloadDataFile.zip"
    if generic_zip.exists() and before.get(generic_zip.name) != generic_zip.stat().st_size:
        target = RAW / f"{file_name}.zip"
        if target.exists():
            target.unlink()
        generic_zip.rename(target)
        changed.pop(generic_zip.name, None)
        changed[target.name] = target.stat().st_size
    return changed


def download_file(file_name: str, timeout: int) -> dict[str, Any]:
    before = {candidate.name: candidate.stat().st_size for candidate in RAW.glob("*") if candidate.is_file()}
    started = time.time()
    rescued = rescue_existing_generic_file(file_name)
    if rescued:
        return {
            "competition_id": COMPETITION_ID,
            "file": file_name,
            "status": "rescued_existing_generic",
            "elapsed_sec": 0,
            "created_or_changed": rescued,
            "stdout_tail": "",
            "stderr_tail": "",
        }
    if existing_file_size(file_name) > 0:
        return {
            "competition_id": COMPETITION_ID,
            "file": file_name,
            "status": "skipped_existing",
            "elapsed_sec": 0,
            "created_or_changed": {},
            "stdout_tail": "",
            "stderr_tail": "",
        }

    cmd = [
        "kaggle",
        "competitions",
        "download",
        "-c",
        COMPETITION_ID,
        "-f",
        file_name,
        "-p",
        str(RAW),
        "--force",
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(BASE),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        status = "downloaded" if proc.returncode == 0 else "download_failed"
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        returncode: int | None = proc.returncode
    except subprocess.TimeoutExpired as exc:
        status = "timeout"
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        returncode = None

    combined = f"{stdout}\n{stderr}".lower()
    if "403" in combined or "forbidden" in combined or "rules" in combined or "permission" in combined:
        status = "blocked_rules_or_permission"

    changed = normalize_downloaded_file(file_name, before)
    if status == "download_failed" and existing_file_size(file_name) > 0:
        status = "downloaded"
    return {
        "competition_id": COMPETITION_ID,
        "file": file_name,
        "status": status,
        "returncode": returncode,
        "elapsed_sec": round(time.time() - started, 2),
        "created_or_changed": changed,
        "stdout_tail": stdout[-3000:],
        "stderr_tail": stderr[-3000:],
    }


def write_report(path: Path, results: list[dict[str, Any]], mode: str, blocked_reason: str | None = None) -> None:
    payload = {
        "schema": "jinghw.mlebench75.cdiscount_file_by_file.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "competition_id": COMPETITION_ID,
        "mode": mode,
        "blocked_reason": blocked_reason,
        "counts": dict(Counter(item["status"] for item in results)),
        "results": results,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["small-probe", "test-only", "train-only", "full"], default="small-probe")
    parser.add_argument("--stamp", default=time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--allow-big-while-active", action="store_true")
    args = parser.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    selected = list(SMALL_PROBE_FILES)
    if args.mode == "test-only":
        selected = [BIG_FILES[0]]
    elif args.mode == "train-only":
        selected = [BIG_FILES[1]]
    elif args.mode == "full":
        selected = list(SMALL_PROBE_FILES) + list(BIG_FILES)

    if args.mode != "small-probe" and not args.allow_big_while_active:
        active = active_big_downloads()
        if active:
            final = REPORTS / f"mlebench75_cdiscount_file_by_file_final_{args.stamp}.json"
            write_report(
                final,
                [],
                args.mode,
                blocked_reason=f"other_active_downloads={len(active)}",
            )
            print(f"BLOCKED_BY_ACTIVE_DOWNLOADS {len(active)}")
            return 2

    lock = RAW / ".cdiscount_file_by_file.lock"
    if lock.exists():
        final = REPORTS / f"mlebench75_cdiscount_file_by_file_final_{args.stamp}.json"
        write_report(final, [], args.mode, blocked_reason="lock_exists")
        print("LOCK_EXISTS")
        return 3

    lock.write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z"), encoding="utf-8")
    progress = REPORTS / f"mlebench75_cdiscount_file_by_file_progress_{args.stamp}.json"
    final = REPORTS / f"mlebench75_cdiscount_file_by_file_final_{args.stamp}.json"
    results: list[dict[str, Any]] = []
    try:
        for file_name, timeout in selected:
            result = download_file(file_name, timeout)
            results.append(result)
            write_report(progress, results, args.mode)
            if result["status"] == "blocked_rules_or_permission":
                break
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass
    write_report(final, results, args.mode)
    print(str(final))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
