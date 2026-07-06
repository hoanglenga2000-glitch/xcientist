"""Live/read-only event watcher for xsci research runs."""
from __future__ import annotations

import time
import json
from pathlib import Path
from typing import Optional

from research_os import events as ev

from .results import evolution_dir, find_result, list_results


def _run_dir_from_selector(selector: str | None) -> Optional[Path]:
    base = evolution_dir()
    if selector:
        candidate = Path(selector)
        if candidate.is_dir():
            return candidate
        if candidate.is_file() and candidate.name == "events.jsonl":
            return candidate.parent
        exact = base / selector
        if exact.is_dir():
            return exact
        result = find_result(selector)
        return result.run_dir if result else None

    if not base.is_dir():
        return None
    candidates = [p for p in base.iterdir() if p.is_dir() and (p / "events.jsonl").exists()]
    if candidates:
        return max(candidates, key=lambda p: (p / "events.jsonl").stat().st_mtime)
    completed = list_results()
    if completed:
        return completed[0].run_dir
    return None


def _render_new(path: Path, *, seen: int, limit: int | None = None) -> int:
    events = ev.read_events(path)
    chunk = events[seen:]
    if limit is not None:
        chunk = chunk[-limit:]
    for event in chunk:
        print(ev.format_event(event), flush=True)
    return len(events)


def _render_summary(run_dir: Path) -> int:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        print(f"no events.jsonl or summary.json under {run_dir}")
        return 1
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"summary unreadable: {type(exc).__name__}")
        return 1

    print(f"summary-only run: {run_dir}")
    print(f"task       : {data.get('task', run_dir.name)}")
    print(f"metric     : {data.get('metric', 'cv_score')} ({data.get('metric_direction', 'maximize')})")
    print(f"best       : {data.get('best_exp_id')}  cv={data.get('best_cv_score')}")
    print(f"promotions : {data.get('n_promotions', 0)}/{data.get('n_iterations', 0)}")
    iterations = data.get("iterations") or []
    if iterations:
        print("\niterations:")
        for item in iterations[-12:]:
            print(
                f"  {item.get('exp_id', '?'):<8} "
                f"{item.get('mode', ''):<10} "
                f"success={item.get('success')} "
                f"cv={item.get('cv_score')} "
                f"promoted={item.get('promoted')}"
            )
    return 0


def run_watch(
    selector: str = "",
    *,
    follow: bool = False,
    lines: int = 80,
    interval: float = 1.0,
) -> int:
    """Render a run's events.jsonl once, or follow it live."""
    run_dir = _run_dir_from_selector(selector or None)
    if run_dir is None:
        print("no run events found. Start a run with `xsci run` or `xsci agent` first.")
        return 1

    events_path = run_dir / "events.jsonl"
    if not events_path.exists():
        if follow:
            print("cannot follow: this historical run has no events.jsonl stream.")
            return 1
        return _render_summary(run_dir)

    print(f"watching: {events_path}")
    seen = _render_new(events_path, seen=0, limit=max(1, lines))
    if not follow:
        return 0

    try:
        while True:
            seen = _render_new(events_path, seen=seen)
            time.sleep(max(0.2, interval))
    except KeyboardInterrupt:
        print("\nstopped.")
        return 0
