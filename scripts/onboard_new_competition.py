#!/usr/bin/env python3
"""One-command onboarding for a new Kaggle competition.

Turns a competition name/URL into a validated COMPETITIONS-registry entry that
scripts/gpu_batch_trainer_v1.py can train, without touching the live trainer
until a human approves (Gate rule #3/#7). Safe by default:

  * --dry-run (default): detect config from local data, print it, write nothing.
  * --write-registry: persist the detected entry to a JSON side-file
                       (benchmark/auto_onboarded/<name>.json), NOT into the
                       trainer source. A human/tool merges it after review.
  * --download: fetch data via the Kaggle CLI (requires configured kaggle.json).
                Off by default so the tool runs offline in CI/tests.

This script deliberately does NOT: submit to Kaggle, launch GPU training, or edit
gpu_batch_trainer_v1.py in place. Those remain human-gated actions.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]

# Load the sibling detection module by path (scripts/ is not a package).
_SPEC = importlib.util.spec_from_file_location(
    "auto_detect_competition", Path(__file__).resolve().parent / "auto_detect_competition.py"
)
_adc = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _adc
_SPEC.loader.exec_module(_adc)


def parse_competition_name(name_or_url: str) -> str:
    """Accept a bare slug or a full Kaggle URL and return the slug."""
    text = name_or_url.strip().rstrip("/")
    match = re.search(r"kaggle\.com/(?:c|competitions)/([^/?#]+)", text)
    if match:
        return match.group(1)
    return text


def _find_csv(data_dir: Path, *candidates: str) -> Optional[Path]:
    for name in candidates:
        hit = data_dir / name
        if hit.exists():
            return hit
    # Case-insensitive fallback.
    lowered = {p.name.lower(): p for p in data_dir.glob("*.csv")}
    for name in candidates:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def download_competition(name: str, dest: Path) -> None:
    """Download + unzip competition files via the Kaggle CLI (opt-in)."""
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "competitions", "download", "-c", name, "-p", str(dest)],
        check=True,
    )
    for zip_path in dest.glob("*.zip"):
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest)


def onboard(
    name_or_url: str,
    *,
    data_dir: Optional[Path] = None,
    download: bool = False,
    target_override: Optional[str] = None,
    id_override: Optional[str] = None,
    bronze: Optional[float] = None,
) -> dict:
    """Run detection and return a report dict (no side effects beyond optional download)."""
    import pandas as pd

    name = parse_competition_name(name_or_url)
    data_dir = data_dir or (ROOT / "datasets" / name)

    if download:
        download_competition(name, data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}. Use --download or point --data-dir at the files."
        )

    train_csv = _find_csv(data_dir, "train.csv", f"{name}_train.csv")
    if train_csv is None:
        raise FileNotFoundError(f"No train.csv found under {data_dir}")
    sample_csv = _find_csv(data_dir, "sample_submission.csv", "sampleSubmission.csv")

    train_df = pd.read_csv(train_csv)
    sample_df = pd.read_csv(sample_csv) if sample_csv else None

    config = _adc.detect_competition_config(
        train_df,
        sample_df,
        target_override=target_override,
        id_override=id_override,
        bronze=bronze,
    )
    return {
        "competition": name,
        "data_dir": str(data_dir),
        "train_csv": str(train_csv),
        "sample_submission": str(sample_csv) if sample_csv else None,
        "n_rows": int(len(train_df)),
        "n_cols": int(train_df.shape[1]),
        "registry_entry": config.to_registry_entry(),
        "rationale": config.rationale,
        "n_classes": config.n_classes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("competition", help="Kaggle competition slug or URL")
    parser.add_argument("--data-dir", type=Path, default=None, help="local dir with train.csv/sample_submission.csv")
    parser.add_argument("--download", action="store_true", help="download data via Kaggle CLI first")
    parser.add_argument("--target", default=None, help="override detected target column")
    parser.add_argument("--id-col", default=None, help="override detected id column")
    parser.add_argument("--bronze", type=float, default=None, help="bronze-medal threshold if known")
    parser.add_argument("--write-registry", action="store_true", help="persist entry to benchmark/auto_onboarded/")
    args = parser.parse_args()

    try:
        report = onboard(
            args.competition,
            data_dir=args.data_dir,
            download=args.download,
            target_override=args.target,
            id_override=args.id_col,
            bronze=args.bronze,
        )
    except Exception as exc:  # noqa: BLE001 - surface a clean CLI error
        print(json.dumps({"status": "error", "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False, indent=2))
        return 1

    if args.write_registry:
        out_dir = ROOT / "benchmark" / "auto_onboarded"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{report['competition']}.json"
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        report["written_to"] = str(out_path)
        report["merge_note"] = "Review, then add registry_entry to gpu_batch_trainer_v1.py COMPETITIONS (human-gated)."

    report["status"] = "ok"
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
