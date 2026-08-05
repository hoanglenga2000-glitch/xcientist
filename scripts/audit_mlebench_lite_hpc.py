#!/usr/bin/env python3
"""Audit the official MLE-bench Lite split on the gated HPC workspace.

The audit is read-only with respect to datasets. It records a versioned JSON
inventory under the allowed remote workspace and mirrors the current report to
the local ``workspace`` directory. Credentials are resolved exclusively by the
shared DPAPI/environment loader and are never printed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from research_agent_workstation.server.core.gpu_credentials import (  # noqa: E402
    ALLOWED_GPU_REMOTE_ROOT,
    connect_ssh,
    load_gpu_ssh_config,
)

LITE_DATASET_GB = {
    "aerial-cactus-identification": 0.0254,
    "aptos2019-blindness-detection": 10.22,
    "denoising-dirty-documents": 0.06,
    "detecting-insults-in-social-commentary": 0.002,
    "dog-breed-identification": 0.75,
    "dogs-vs-cats-redux-kernels-edition": 0.85,
    "histopathologic-cancer-detection": 7.76,
    "jigsaw-toxic-comment-classification-challenge": 0.06,
    "leaf-classification": 0.036,
    "mlsp-2013-birds": 0.5851,
    "new-york-city-taxi-fare-prediction": 5.7,
    "nomad2018-predict-transparent-conductors": 0.00624,
    "plant-pathology-2020-fgvc7": 0.8,
    "random-acts-of-pizza": 0.003,
    "ranzcr-clip-catheter-line-classification": 13.13,
    "siim-isic-melanoma-classification": 116.16,
    "spooky-author-identification": 0.0019,
    "tabular-playground-series-dec-2021": 0.7,
    "tabular-playground-series-may-2022": 0.57,
    "text-normalization-challenge-english-language": 0.01,
    "text-normalization-challenge-russian-language": 0.01,
    "the-icml-2013-whale-challenge-right-whale-redux": 0.29314,
}


def _remote_source() -> str:
    payload = json.dumps(LITE_DATASET_GB, sort_keys=True)
    return f'''from __future__ import annotations
import json
import os
import pathlib
import shutil
import subprocess
from datetime import datetime, timezone

ROOT = pathlib.Path({ALLOWED_GPU_REMOTE_ROOT!r})
LITE = json.loads({payload!r})
RAW_ROOT = ROOT / "mlebench_raw_data"
LEGACY_PREPARED_ROOT = ROOT / "mlebench_prepared"
OFFICIAL_ROOTS = [
    ROOT / "mlebench_official_data",
    ROOT / "mlebench_lite",
    ROOT / ".cache" / "mle-bench" / "data",
]


def tree_stats(path: pathlib.Path) -> dict:
    if not path.is_dir():
        return {{"exists": False, "bytes": 0, "files": 0, "largest": []}}
    total = 0
    files = 0
    largest = []
    for base, _dirs, names in os.walk(path):
        for name in names:
            item = pathlib.Path(base) / name
            try:
                size = item.stat().st_size
            except OSError:
                continue
            total += size
            files += 1
            largest.append((size, str(item.relative_to(path))))
            if len(largest) > 32:
                largest = sorted(largest, reverse=True)[:12]
    largest = sorted(largest, reverse=True)[:12]
    return {{
        "exists": True,
        "bytes": total,
        "files": files,
        "largest": [{{"bytes": size, "path": rel}} for size, rel in largest],
    }}


def nonempty(path: pathlib.Path) -> bool:
    try:
        return path.is_dir() and any(path.iterdir())
    except OSError:
        return False


rows = []
for competition_id, expected_gb in LITE.items():
    raw = tree_stats(RAW_ROOT / competition_id)
    legacy_prepared = tree_stats(LEGACY_PREPARED_ROOT / competition_id)
    official = []
    for data_root in OFFICIAL_ROOTS:
        competition_root = data_root / competition_id
        if not competition_root.exists():
            continue
        public = competition_root / "prepared" / "public"
        private = competition_root / "prepared" / "private"
        official.append({{
            "path": str(competition_root),
            "public_nonempty": nonempty(public),
            "private_nonempty": nonempty(private),
            "raw": tree_stats(competition_root / "raw"),
            "prepared": tree_stats(competition_root / "prepared"),
        }})
    expected_bytes = int(expected_gb * 1_000_000_000)
    ratio = (raw["bytes"] / expected_bytes) if expected_bytes else 0.0
    official_prepared = any(
        item["public_nonempty"] and item["private_nonempty"] for item in official
    )
    if official_prepared:
        status = "official_prepared"
    elif raw["files"] == 0:
        status = "missing"
    elif ratio < 0.5:
        status = "partial_by_size"
    else:
        status = "raw_present_unverified"
    rows.append({{
        "competition_id": competition_id,
        "expected_gb": expected_gb,
        "expected_bytes": expected_bytes,
        "raw": raw,
        "legacy_prepared": legacy_prepared,
        "raw_size_ratio": round(ratio, 4),
        "official": official,
        "official_prepared": official_prepared,
        "status": status,
    }})

disk = shutil.disk_usage(ROOT)
repo = ROOT / "mle-bench"
try:
    repo_commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
    ).strip()
except Exception:
    repo_commit = None

summary = {{
    "official_prepared": sum(row["status"] == "official_prepared" for row in rows),
    "raw_present_unverified": sum(row["status"] == "raw_present_unverified" for row in rows),
    "partial_by_size": sum(row["status"] == "partial_by_size" for row in rows),
    "missing": sum(row["status"] == "missing" for row in rows),
    "raw_bytes": sum(row["raw"]["bytes"] for row in rows),
    "expected_bytes": sum(row["expected_bytes"] for row in rows),
}}
report = {{
    "schema": "evomind.mlebench_lite_inventory.v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "source": {{
        "repository": "https://github.com/openai/mle-bench",
        "split": "experiments/splits/low.txt",
        "competition_count": len(LITE),
        "declared_total_gb": sum(LITE.values()),
        "remote_repo_commit": repo_commit,
    }},
    "remote": {{
        "root": str(ROOT),
        "raw_root": str(RAW_ROOT),
        "official_roots": [str(path) for path in OFFICIAL_ROOTS],
        "disk_total": disk.total,
        "disk_used": disk.used,
        "disk_free": disk.free,
    }},
    "summary": summary,
    "competitions": rows,
}}

audit_root = ROOT / "_evomind_audits"
audit_root.mkdir(parents=True, exist_ok=True)
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
versioned = audit_root / f"mlebench_lite_inventory_{{stamp}}.json"
current = audit_root / "mlebench_lite_inventory_current.json"
encoded = json.dumps(report, ensure_ascii=False, indent=2)
versioned.write_text(encoded, encoding="utf-8")
tmp = current.with_suffix(".json.tmp")
tmp.write_text(encoded, encoding="utf-8")
tmp.replace(current)
print(json.dumps(report, ensure_ascii=False))
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "workspace" / "mlebench_lite_inventory_current.json",
    )
    args = parser.parse_args()

    config = load_gpu_ssh_config()
    config.jump_host = None
    client = connect_ssh(config, timeout=20)
    remote_audit_dir = f"{ALLOWED_GPU_REMOTE_ROOT}/_evomind_audits"
    remote_script = f"{remote_audit_dir}/audit_mlebench_lite.py"
    source = _remote_source().encode("utf-8")
    try:
        client.exec_command(f"mkdir -p {shlex.quote(remote_audit_dir)}", timeout=30)
        with client.open_sftp() as sftp:
            with sftp.file(remote_script, "wb") as handle:
                handle.write(source)
        command = f"python3 {shlex.quote(remote_script)}"
        _stdin, stdout, stderr = client.exec_command(command, timeout=900)
        raw_stdout = stdout.read().decode("utf-8", "replace")
        raw_stderr = stderr.read().decode("utf-8", "replace")
        exit_code = stdout.channel.recv_exit_status()
    finally:
        client.close()
    if exit_code != 0:
        print(raw_stderr or raw_stdout, file=sys.stderr)
        return exit_code

    report = json.loads(raw_stdout)
    report["local"] = {
        "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(args.output)

    summary = report["summary"]
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "summary": summary,
                "disk_free": report["remote"]["disk_free"],
                "remote_repo_commit": report["source"]["remote_repo_commit"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
