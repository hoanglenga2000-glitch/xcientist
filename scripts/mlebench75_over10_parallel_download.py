#!/usr/bin/env python3
from __future__ import annotations

import concurrent.futures as cf
import importlib.util
import json
import os
import subprocess
import time
from collections import Counter
from pathlib import Path

BASE = Path("/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra")
RAW = BASE / "mlebench_raw_data"
REPORTS = BASE / "reports"
LOGS = BASE / "logs"
UNDER10_SCRIPT = BASE / "scripts" / "mlebench75_priority_under10gb_download.py"
LATEST_STATE = REPORTS / "mlebench75_latest_state_by_competition_20260701_1950.json"
MAX_WORKERS = 2
TASK_TIMEOUT_SECONDS = 12 * 3600


def assert_workspace() -> None:
    if str(BASE) != "/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra":
        raise RuntimeError(f"Refusing to run outside Jing workspace: {BASE}")
    for directory in (RAW, REPORTS, LOGS):
        directory.mkdir(parents=True, exist_ok=True)
    os.chdir(BASE)


def load_under10_module():
    spec = importlib.util.spec_from_file_location("mlebench75_under10", UNDER10_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import helper script: {UNDER10_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def du_kb(path: Path) -> int:
    try:
        return int(subprocess.check_output(["du", "-sk", str(path)], text=True, stderr=subprocess.DEVNULL).split()[0])
    except Exception:
        return 0


def classify_download_error(stdout: str, stderr: str) -> str:
    text = f"{stdout}\n{stderr}".lower()
    if "403" in text or "forbidden" in text or "must accept" in text or ("rules" in text and "accept" in text):
        return "needs_api_account_rules_or_permission"
    if "404" in text or "not found" in text:
        return "not_found_or_unavailable"
    return "download_failed"


def load_candidates(module) -> list[dict]:
    if not LATEST_STATE.exists():
        raise RuntimeError(f"Missing latest state report: {LATEST_STATE}")
    payload = json.loads(LATEST_STATE.read_text(encoding="utf-8"))
    candidates: list[dict] = []
    for item in payload.get("state", []):
        if item.get("priority_class") == "under_10gb":
            continue
        if item.get("current_status") == "downloaded_or_useful_data_present":
            continue
        competition_id = item.get("competition_id")
        if not competition_id:
            continue
        target = RAW / competition_id
        actual_kb = du_kb(target)
        estimated_gb = item.get("estimated_gb")
        is_large_estimate = isinstance(estimated_gb, (int, float)) and estimated_gb >= 10
        has_partial_data = actual_kb > 1024
        if not (is_large_estimate or has_partial_data):
            continue
        try:
            useful_now = module.has_useful_files(target)
        except Exception:
            useful_now = False
        if useful_now:
            continue
        candidates.append(
            {
                "competition_id": competition_id,
                "estimated_gb": estimated_gb,
                "priority_class": item.get("priority_class"),
                "latest_status": item.get("latest_status"),
                "current_status": item.get("current_status"),
                "actual_gb_before": round(actual_kb / 1024 / 1024, 3),
                "force": has_partial_data,
            }
        )
    candidates.sort(key=lambda item: (0 if item["actual_gb_before"] > 0 else 1, -(item["actual_gb_before"] or 0), item["competition_id"]))
    return candidates


def download_one(plan: dict) -> dict:
    module = load_under10_module()
    competition_id = plan["competition_id"]
    target = RAW / competition_id
    target.mkdir(parents=True, exist_ok=True)
    lock = target / ".over10_parallel.lock"
    started = time.time()
    if module.has_useful_files(target):
        return {
            **plan,
            "status": "skipped_existing",
            "elapsed_sec": round(time.time() - started, 2),
            "actual_gb_after": round(du_kb(target) / 1024 / 1024, 3),
            "useful_files_after": True,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
    if lock.exists() or (target / ".priority_under10gb.lock").exists() or (target / ".fastlane.lock").exists():
        return {
            **plan,
            "status": "locked_by_other_downloader",
            "elapsed_sec": round(time.time() - started, 2),
            "actual_gb_after": round(du_kb(target) / 1024 / 1024, 3),
            "useful_files_after": module.has_useful_files(target),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
    lock.write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z"), encoding="utf-8")
    stdout_tail = ""
    stderr_tail = ""
    extracted: list[str] = []
    try:
        cmd = ["kaggle", "competitions", "download", "-c", competition_id, "-p", str(target)]
        if plan.get("force"):
            cmd.append("--force")
        proc = subprocess.run(cmd, cwd=BASE, text=True, capture_output=True, timeout=TASK_TIMEOUT_SECONDS)
        stdout_tail = proc.stdout[-1200:]
        stderr_tail = proc.stderr[-1200:]
        if proc.returncode == 0:
            extracted = module.extract_zip_archives(target)
            status = "downloaded" if module.has_useful_files(target) else "downloaded_but_incomplete"
        else:
            status = classify_download_error(proc.stdout, proc.stderr)
    except subprocess.TimeoutExpired as exc:
        status = "timeout"
        stdout_tail = (exc.stdout or "")[-1200:] if isinstance(exc.stdout, str) else ""
        stderr_tail = (exc.stderr or "")[-1200:] if isinstance(exc.stderr, str) else ""
    except Exception as exc:
        status = f"exception:{type(exc).__name__}"
        stderr_tail = repr(exc)[-1200:]
    finally:
        try:
            lock.unlink()
        except OSError:
            pass
    return {
        **plan,
        "status": status,
        "elapsed_sec": round(time.time() - started, 2),
        "actual_gb_after": round(du_kb(target) / 1024 / 1024, 3),
        "useful_files_after": module.has_useful_files(target),
        "extracted_archives": extracted,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def main() -> int:
    assert_workspace()
    module = load_under10_module()
    candidates = load_candidates(module)
    plan_path = REPORTS / "mlebench75_over10_parallel_plan_20260701.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema": "jinghw.mlebench75.over10_parallel_plan.v1",
                "base": str(BASE),
                "max_workers": MAX_WORKERS,
                "task_timeout_seconds": TASK_TIMEOUT_SECONDS,
                "total_candidates": len(candidates),
                "candidates": candidates,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    progress_path = REPORTS / "mlebench75_over10_parallel_progress_20260701.json"
    results: list[dict] = []
    with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(download_one, candidate): candidate["competition_id"] for candidate in candidates}
        for future in cf.as_completed(futures):
            record = future.result()
            results.append(record)
            payload = {
                "schema": "jinghw.mlebench75.over10_parallel_download.v1",
                "base": str(BASE),
                "max_workers": MAX_WORKERS,
                "total_candidates": len(candidates),
                "completed": len(results),
                "counts": dict(Counter(item["status"] for item in results)),
                "results": results,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            progress_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(
                json.dumps(
                    {
                        "competition_id": record["competition_id"],
                        "status": record["status"],
                        "actual_gb_before": record["actual_gb_before"],
                        "actual_gb_after": record["actual_gb_after"],
                        "useful_files_after": record["useful_files_after"],
                        "elapsed_sec": record["elapsed_sec"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    final_path = REPORTS / "mlebench75_over10_parallel_final_20260701.json"
    final_path.write_text(
        json.dumps(
            {
                "schema": "jinghw.mlebench75.over10_parallel_download.v1",
                "base": str(BASE),
                "max_workers": MAX_WORKERS,
                "total_candidates": len(candidates),
                "completed": len(results),
                "counts": dict(Counter(item["status"] for item in results)),
                "results": results,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
