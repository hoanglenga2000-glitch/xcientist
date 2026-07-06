#!/usr/bin/env python3
"""Offline mle-bench medal grading harness.

Given a competition id, a staged raw train file, and a submission.csv, this:
  1. runs the competition's prepare_fn(raw, public, private) directly (NO Kaggle),
     splitting raw train into public/ (train + unlabeled test) and private/ (answers)
  2. grades a submission.csv against private answers using the shipped leaderboard
     -> real gold/silver/bronze verdict + thresholds

Split is deterministic (random_state=0) so the public/test.csv the solver predicts
matches the private answers grade_csv scores against.

Usage:
  python scripts/medal_grade.py prepare  <cid> --raw-dir <dir_with_raw_files>
  python scripts/medal_grade.py grade    <cid> --submission <submission.csv>
  python scripts/medal_grade.py status   <cid>

--data-dir defaults to workspace/mle_data (local, we control it).
"""
from __future__ import annotations
import argparse, json, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external-projects" / "mle-bench"))

from mlebench.registry import registry  # noqa: E402
from mlebench.grade import grade_csv     # noqa: E402

DEFAULT_DATA = ROOT / "workspace" / "mle_data"


def _comp(cid: str, data_dir: Path):
    reg = registry.set_data_dir(data_dir)
    return reg.get_competition(cid)


def cmd_prepare(a):
    comp = _comp(a.cid, a.data_dir)
    comp.raw_dir.mkdir(parents=True, exist_ok=True)
    comp.public_dir.mkdir(parents=True, exist_ok=True)
    comp.private_dir.mkdir(parents=True, exist_ok=True)
    raw_src = Path(a.raw_dir)
    copied = []
    for f in raw_src.iterdir():
        if f.is_file():
            dst = comp.raw_dir / f.name
            if not dst.exists():
                shutil.copyfile(f, dst)
            copied.append(f.name)
        elif f.is_dir():
            dst = comp.raw_dir / f.name
            if not dst.exists():
                shutil.copytree(f, dst)
            copied.append(f.name + "/")
    print(f"[prepare] staged raw for {a.cid}: {sorted(copied)}")
    comp.prepare_fn(raw=comp.raw_dir, public=comp.public_dir, private=comp.private_dir)
    ok = comp.answers.is_file()
    print(f"[prepare] answers written: {comp.answers} exists={ok}")
    print(f"[prepare] public dir contents: {sorted(p.name for p in comp.public_dir.iterdir())}")
    return 0 if ok else 3


def cmd_status(a):
    comp = _comp(a.cid, a.data_dir)
    print(json.dumps({
        "cid": a.cid,
        "raw_dir": str(comp.raw_dir), "raw_exists": comp.raw_dir.exists(),
        "answers": str(comp.answers), "answers_exists": comp.answers.is_file(),
        "public_dir": str(comp.public_dir),
        "public_files": sorted(p.name for p in comp.public_dir.iterdir()) if comp.public_dir.exists() else [],
        "leaderboard": str(comp.leaderboard), "leaderboard_exists": Path(comp.leaderboard).is_file(),
    }, ensure_ascii=False, indent=2))
    return 0


def cmd_grade(a):
    comp = _comp(a.cid, a.data_dir)
    if not comp.answers.is_file():
        print(f"[grade] ERROR: answers missing, run prepare first: {comp.answers}", file=sys.stderr)
        return 3
    report = grade_csv(Path(a.submission), comp)
    d = report.to_dict()
    medal = "gold" if d.get("gold_medal") else "silver" if d.get("silver_medal") else \
            "bronze" if d.get("bronze_medal") else ("above_median" if d.get("above_median") else "none")
    out = {
        "cid": a.cid, "submission": a.submission,
        "score": d.get("score"), "medal": medal,
        "gold_medal": d.get("gold_medal"), "silver_medal": d.get("silver_medal"),
        "bronze_medal": d.get("bronze_medal"), "above_median": d.get("above_median"),
        "gold_threshold": d.get("gold_threshold"), "silver_threshold": d.get("silver_threshold"),
        "bronze_threshold": d.get("bronze_threshold"), "median_threshold": d.get("median_threshold"),
        "valid_submission": d.get("valid_submission"), "is_lower_better": d.get("is_lower_better"),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    outdir = ROOT / "workspace" / "medal_reports"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{a.cid}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("prepare", "grade", "status"):
        p = sub.add_parser(name)
        p.add_argument("cid")
        p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
        if name == "prepare":
            p.add_argument("--raw-dir", required=True)
        if name == "grade":
            p.add_argument("--submission", required=True)
    a = ap.parse_args()
    return {"prepare": cmd_prepare, "grade": cmd_grade, "status": cmd_status}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
