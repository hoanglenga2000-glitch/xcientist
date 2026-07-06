from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_TERMS = {
    "whoami": ["whoami"],
    "hostname": ["hostname"],
    "pwd": ["pwd"],
    "python": ["Python"],
    "nvidia_smi": ["nvidia-smi", "NVIDIA-SMI"],
    "filesystem": ["Filesystem", "df -hT"],
    "memory": ["Mem:", "free -h"],
}


def fail(message: str, evidence: dict[str, Any]) -> None:
    raise SystemExit(json.dumps({"status": "failed", "message": message, "evidence": evidence}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify pasted HKUST(GZ) HPC Web Terminal evidence before marking GPU fully ready.")
    parser.add_argument("probe_file", help="Text file containing pasted Web Terminal commands and output.")
    args = parser.parse_args()

    path = Path(args.probe_file)
    if not path.is_file():
        fail("probe file is missing", {"path": str(path)})
    text = path.read_text(encoding="utf-8", errors="replace")
    missing = [
        key
        for key, terms in REQUIRED_TERMS.items()
        if not any(term in text for term in terms)
    ]
    if missing:
        fail(
            "HPC Web Terminal evidence is incomplete; do not mark GPU fully ready",
            {"path": str(path), "missing_sections": missing},
        )

    gpu_count = text.count("NVIDIA A800") + text.count("NVIDIAA800") + text.count("A800-SXM4")
    print(json.dumps({
        "status": "passed",
        "path": str(path),
        "fully_ready_allowed": gpu_count >= 4,
        "gpu_evidence": "4 x A800 detected" if gpu_count >= 4 else "nvidia-smi present, but 4 x A800 not proven by text count",
        "required_sections": sorted(REQUIRED_TERMS),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
