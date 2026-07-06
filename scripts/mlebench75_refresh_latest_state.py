#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_BASE = Path("/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra")
RAW_DIR_NAME = "mlebench_raw_data"
REPORTS_DIR_NAME = "reports"
UNDER10_SCRIPT_NAME = "mlebench75_priority_under10gb_download.py"


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        return {"_read_error": f"json_decode:{exc}", "_path": str(path)}


def du_kb(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return int(
            subprocess.check_output(["du", "-sk", str(path)], text=True, stderr=subprocess.DEVNULL).split()[0]
        )
    except Exception:
        total = 0
        for candidate in path.rglob("*"):
            if candidate.is_file():
                try:
                    total += candidate.stat().st_size
                except OSError:
                    pass
        return total // 1024


def load_split75(base: Path) -> list[str]:
    script = base / "scripts" / UNDER10_SCRIPT_NAME
    text = script.read_text(encoding="utf-8")
    start = text.index('for line in """') + len('for line in """')
    end = text.index('""".splitlines()', start)
    return [line.strip() for line in text[start:end].splitlines() if line.strip()]


def has_useful_files(path: Path) -> bool:
    if not path.exists():
        return False
    if path.name == "cdiscount-image-classification-challenge":
        files_by_name = {candidate.name.lower(): candidate for candidate in path.rglob("*") if candidate.is_file()}
        full_zip = files_by_name.get("cdiscount-image-classification-challenge.zip")
        if full_zip is not None:
            marker = full_zip.parent / f"{full_zip.name}.extracted.ok"
            try:
                if marker.exists() or zipfile.is_zipfile(full_zip):
                    return True
            except OSError:
                pass
        required = ["category_names.csv", "sample_submission.csv", "test.bson", "train.bson"]
        return all(name in files_by_name and files_by_name[name].stat().st_size > 0 for name in required)
    files: list[str] = []
    complete_archives = 0
    for candidate in path.rglob("*"):
        if not candidate.is_file():
            continue
        name = candidate.name.lower()
        files.append(name)
        if name.endswith(".zip"):
            marker = candidate.parent / f"{candidate.name}.extracted.ok"
            try:
                if marker.exists() or zipfile.is_zipfile(candidate):
                    complete_archives += 1
            except OSError:
                pass
        elif name.endswith((".7z", ".tar.gz", ".bson")) and candidate.stat().st_size > 0:
            complete_archives += 1
        if len(files) > 5000:
            break
    if not files:
        return False
    train_like = any(name.startswith("train.") or name in {"train.csv", "training.csv"} for name in files)
    test_like = any(name.startswith("test.") or name in {"test.csv", "testing.csv"} for name in files)
    sample_like = any("sample" in name and "submission" in name for name in files)
    media_like = any(
        name.endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff", ".wav", ".mp3", ".json", ".parquet", ".npy", ".nii", ".dcm"))
        for name in files
    )
    return (
        (train_like and (test_like or sample_like))
        or (sample_like and media_like)
        or complete_archives > 0
        or (media_like and len(files) > 20)
    )


def collect_report_results(reports: Path) -> dict[str, dict[str, Any]]:
    patterns = [
        "mlebench75_fastlane_*_20260701.json",
        "mlebench75_download_*_20260701.json",
        "mlebench75_under10gb_priority_*_20260701.json",
        "mlebench75_over10_parallel_*_20260701.json",
        "mlebench75_over10_extra_probe_*_20260701.json",
    ]
    latest: dict[str, dict[str, Any]] = {}
    for pattern in patterns:
        for path in sorted(reports.glob(pattern), key=lambda p: p.stat().st_mtime):
            payload = read_json(path)
            for record in payload.get("results", []) or []:
                competition_id = record.get("competition_id")
                status = record.get("status") or record.get("latest_status")
                if not competition_id or not status:
                    continue
                status_history = list(latest.get(competition_id, {}).get("_status_history", []))
                status_history.append(str(status))
                latest[competition_id] = {
                    **record,
                    "_source_report": path.name,
                    "_source_mtime": path.stat().st_mtime,
                    "_status_history": status_history,
                }
    return latest


def load_inventory(reports: Path) -> dict[str, dict[str, Any]]:
    candidates = sorted(reports.glob("mlebench75_size_inventory_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return {}
    payload = read_json(candidates[0])
    inventory: dict[str, dict[str, Any]] = {}
    for item in payload.get("items", []) or []:
        competition_id = item.get("competition_id")
        if competition_id:
            inventory[competition_id] = item
    return inventory


def active_downloads() -> dict[str, str]:
    try:
        out = subprocess.check_output(["ps", "-eo", "pid,ppid,etime,cmd"], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return {}
    active: dict[str, str] = {}
    marker = "kaggle competitions download -c "
    for line in out.splitlines():
        if marker not in line:
            continue
        tail = line.split(marker, 1)[1]
        competition_id = tail.split()[0]
        active[competition_id] = line[:1000]
    return active


def classify_status(useful: bool, active: bool, latest_status: str, file_count: int | None, status_history: list[str]) -> str:
    lowered = latest_status.lower()
    history_text = " ".join(status_history).lower()
    if useful:
        return "downloaded_or_useful_data_present"
    if active:
        return "downloading"
    if "permission" in lowered or "rules" in lowered or "403" in lowered or "forbidden" in lowered:
        return "blocked_rules_or_permission"
    if ("incomplete" in lowered or "empty" in lowered or file_count == 0) and (
        "permission" in history_text or "rules" in history_text or "403" in history_text or "forbidden" in history_text
    ):
        return "blocked_rules_or_permission"
    if "incomplete" in lowered or "empty" in lowered or file_count == 0:
        return "incomplete_download_or_empty_listing"
    if "failed" in lowered or "exception" in lowered or "timeout" in lowered:
        return "download_failed"
    return "not_attempted_or_unknown"


def build_state(base: Path) -> dict[str, Any]:
    raw = base / RAW_DIR_NAME
    reports = base / REPORTS_DIR_NAME
    split75 = load_split75(base)
    inventory = load_inventory(reports)
    report_results = collect_report_results(reports)
    active = active_downloads()

    rows: list[dict[str, Any]] = []
    for index, competition_id in enumerate(split75, 1):
        inv = inventory.get(competition_id, {})
        latest = report_results.get(competition_id, {})
        path = raw / competition_id
        actual_kb = du_kb(path)
        useful = has_useful_files(path)
        latest_status = str(latest.get("status") or latest.get("latest_status") or "")
        if not latest_status and inv.get("has_data_before"):
            latest_status = "downloaded"
        priority_class = inv.get("priority_class")
        estimated_gb = inv.get("estimated_gb")
        file_count = inv.get("file_count")
        status_history = [str(item) for item in latest.get("_status_history", []) if item]
        current_status = classify_status(useful, competition_id in active, latest_status, file_count, status_history)
        rows.append(
            {
                "index": index,
                "competition_id": competition_id,
                "current_status": current_status,
                "latest_status": latest_status or None,
                "status_history": status_history[-8:],
                "latest_source": latest.get("_source_report"),
                "priority_class": priority_class,
                "estimated_gb": estimated_gb,
                "list_status": inv.get("list_status"),
                "file_count": file_count,
                "actual_gb": round(actual_kb / 1024 / 1024, 3),
                "actual_kb": actual_kb,
                "useful_files": useful,
                "active_download": active.get(competition_id),
            }
        )

    counts = Counter(row["current_status"] for row in rows)
    under10 = [row for row in rows if row.get("priority_class") == "under_10gb"]
    over10 = [row for row in rows if row.get("priority_class") != "under_10gb"]
    return {
        "schema": "jinghw.mlebench75.latest_state.v2",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "root": str(base),
        "total_tasks": len(rows),
        "counts": dict(counts),
        "under10_total": len(under10),
        "under10_counts": dict(Counter(row["current_status"] for row in under10)),
        "over10_or_unknown_total": len(over10),
        "over10_or_unknown_counts": dict(Counter(row["current_status"] for row in over10)),
        "actual_data_total_gb": round(du_kb(raw) / 1024 / 1024, 3),
        "actual_nonempty_over_1mb": sum(1 for row in rows if row["actual_kb"] > 1024),
        "actual_over_100mb": sum(1 for row in rows if row["actual_kb"] > 100 * 1024),
        "actual_over_1gb": sum(1 for row in rows if row["actual_kb"] > 1024 * 1024),
        "active_downloads": active,
        "state": rows,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# MLE-Bench75 当前下载状态（{payload['generated_at']}）",
        "",
        f"- 总任务数：`{payload['total_tasks']}`",
        f"- 当前状态计数：`{payload['counts']}`",
        f"- `<10GB` 任务数：`{payload['under10_total']}`，状态计数：`{payload['under10_counts']}`",
        f"- `>10GB / unknown` 任务数：`{payload['over10_or_unknown_total']}`，状态计数：`{payload['over10_or_unknown_counts']}`",
        f"- 实际数据总量：`{payload['actual_data_total_gb']}` GB",
        f"- 非空数据目录（>1MB）：`{payload['actual_nonempty_over_1mb']}`",
        f"- 较实质数据目录（>100MB）：`{payload['actual_over_100mb']}`",
        f"- 大数据目录（>1GB）：`{payload['actual_over_1gb']}`",
        "",
        "## 正在下载",
    ]
    active_rows = [row for row in payload["state"] if row["current_status"] == "downloading"]
    if not active_rows:
        lines.append("- 无")
    else:
        for row in active_rows:
            lines.append(f"- `{row['competition_id']}` | actual `{row['actual_gb']}` GB | `{row['priority_class']}`")

    lines.extend(["", "## 已有可用数据"])
    for row in payload["state"]:
        if row["current_status"] == "downloaded_or_useful_data_present":
            lines.append(f"- `{row['competition_id']}` | actual `{row['actual_gb']}` GB | `{row['priority_class']}`")

    lines.extend(["", "## 仍需 Kaggle Rules/Permission"])
    blocked = [row for row in payload["state"] if row["current_status"] == "blocked_rules_or_permission"]
    if not blocked:
        lines.append("- 无")
    else:
        for row in blocked:
            cid = row["competition_id"]
            lines.append(
                f"- `{cid}` | `{row['priority_class']}` | estimated `{row['estimated_gb']}` GB | "
                f"[rules](https://www.kaggle.com/competitions/{cid}/rules) | "
                f"[data](https://www.kaggle.com/competitions/{cid}/data)"
            )

    lines.extend(["", "## 非权限失败/不完整"])
    for row in payload["state"]:
        if row["current_status"] in {"download_failed", "incomplete_download_or_empty_listing", "not_attempted_or_unknown"}:
            lines.append(
                f"- `{row['competition_id']}` | `{row['current_status']}` | latest `{row['latest_status']}` | actual `{row['actual_gb']}` GB"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh MLE-Bench75 data readiness state from remote reports and raw data.")
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--stamp", default=time.strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()

    base = args.base
    if str(base) != str(DEFAULT_BASE):
        raise RuntimeError(f"Refusing to run outside Jing workspace: {base}")
    payload = build_state(base)
    reports = base / REPORTS_DIR_NAME
    reports.mkdir(parents=True, exist_ok=True)
    json_path = reports / f"mlebench75_latest_state_by_competition_{args.stamp}.json"
    md_path = reports / f"MLEBENCH75_LATEST_STATE_{args.stamp}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "md": str(md_path), "counts": payload["counts"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
