"""Render research events as a staged terminal narrative."""
from __future__ import annotations

import itertools
import os
import sys
import threading
import time
from contextlib import contextmanager
from typing import Iterator, Optional

from research_os import events as ev

STAGES = (
    "Understanding task",
    "Data audit",
    "Hypothesis",
    "Search decision",
    "Code generation",
    "Training",
    "Gate",
    "Memory",
    "Report",
)

_TOOL_STAGE = {
    "inspect_data": "Data audit",
    "recommend_strategies": "Data audit",
    "read_memory": "Memory",
    "read_search_tree": "Search decision",
    "plan_next_experiment": "Search decision",
    "run_experiment": "Code generation",
    "evaluate_promotion": "Gate",
    "record_lesson": "Memory",
    "audit_conclusion": "Gate",
    "request_audit": "Gate",
    "submit_to_kaggle": "Report",
    "finish": "Report",
}

_EVENT_STAGE = {
    ev.SELECT: "Search decision",
    ev.PROPOSE: "Hypothesis",
    ev.EXEC_BEGIN: "Training",
    ev.SCORE: "Training",
    ev.PROMOTE: "Gate",
    ev.REPAIR: "Training",
    ev.LESSON: "Memory",
}

_GLYPHS_UNICODE = {"arrow": "▸", "bullet": "●", "sep": " → ", "dot": " · ", "ell": "..."}
_GLYPHS_ASCII = {"arrow": ">", "bullet": "*", "sep": " -> ", "dot": " - ", "ell": "..."}


def _pick_glyphs(stream) -> dict:
    enc = getattr(stream, "encoding", None) or "utf-8"
    try:
        "".join(_GLYPHS_UNICODE.values()).encode(enc)
        return _GLYPHS_UNICODE
    except (UnicodeEncodeError, LookupError):
        return _GLYPHS_ASCII


def _stage_for(event: dict, *, seen_first_msg: bool) -> Optional[str]:
    t = event.get("type")
    if t in _EVENT_STAGE:
        return _EVENT_STAGE[t]
    if t in (ev.TOOL_CALL, ev.TOOL_RESULT):
        return _TOOL_STAGE.get(event.get("tool", ""))
    if t == ev.AGENT_MSG:
        return STAGES[0] if not seen_first_msg else None
    return None


class StageRenderer:
    def __init__(self, *, color: Optional[bool] = None, stream=None) -> None:
        self._out = stream if stream is not None else sys.stdout
        if color is None:
            color = bool(getattr(self._out, "isatty", lambda: False)()) and not os.environ.get("NO_COLOR")
        self._color = color
        self._g = _pick_glyphs(self._out)
        self._current: Optional[str] = None
        self._seen_first_msg = False
        self._iteration = 0

    def _c(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self._color else text

    def _print(self, text: str = "") -> None:
        try:
            print(text, file=self._out, flush=True)
        except UnicodeEncodeError:
            enc = getattr(self._out, "encoding", None) or "utf-8"
            safe = text.encode(enc, "replace").decode(enc, "replace")
            try:
                print(safe, file=self._out, flush=True)
            except Exception:  # noqa: BLE001
                pass
        except Exception:  # noqa: BLE001
            pass

    def __call__(self, event: dict) -> None:
        try:
            self._handle(event)
        except Exception:  # noqa: BLE001
            pass

    def _handle(self, event: dict) -> None:
        t = event.get("type")
        if t == ev.RUN_BEGIN:
            self._on_run_begin(event)
            return
        if t == ev.RUN_END:
            self._on_run_end(event)
            return
        if t == ev.ITER_BEGIN:
            self._iteration = int(event.get("iteration", self._iteration + 1) or 0)
            return
        if t in (ev.ITER_END, ev.COMPACTION):
            return

        detail = _strip_tag(ev.format_event(event))
        if t == ev.AGENT_MSG:
            if not self._seen_first_msg:
                self._seen_first_msg = True
                self._enter("Understanding task", detail)
            elif detail:
                self._continue(detail)
            return

        stage = _stage_for(event, seen_first_msg=self._seen_first_msg)
        if stage is None:
            if detail:
                self._continue(detail)
            return
        if stage == self._current:
            self._continue(detail)
        else:
            self._enter(stage, detail)

    def _enter(self, stage: str, detail: str) -> None:
        self._current = stage
        try:
            idx = STAGES.index(stage) + 1
        except ValueError:
            idx = 0
        tag = self._c("96", f"{self._g['arrow']} {stage}")
        counter = self._c("90", f"[{idx}/{len(STAGES)}]")
        body = f"  {self._c('90', detail)}" if detail else ""
        self._print(f"{tag} {counter}{body}")

    def _continue(self, detail: str) -> None:
        if detail:
            self._print("    " + self._c("90", detail))

    def _on_run_begin(self, event: dict) -> None:
        task = event.get("task", "?")
        metric = event.get("metric", "?")
        direction = event.get("metric_direction", "")
        head = self._c("97;1", f"{self._g['bullet']} Research run{self._g['dot']}{task}")
        sub = self._c("90", f"metric={metric}({direction}){self._g['dot']}stages: " + self._g["sep"].join(STAGES))
        self._print()
        self._print(head)
        self._print("  " + sub)
        self._print("  " + self._c("90", "live events also stream to events.jsonl and the 8088 dashboard; Ctrl+C aborts"))
        self._print()

    def _on_run_end(self, event: dict) -> None:
        best = event.get("best_exp_id") or "-"
        cv = event.get("best_cv_score")
        cv_txt = f"{cv:.4f}" if isinstance(cv, (int, float)) else "-"
        promos = event.get("n_promotions", "?")
        iters = event.get("n_iterations", "?")
        self._enter("Report", "")
        dot = self._g["dot"]
        self._print("    " + self._c("90", f"best={best}{dot}cv={cv_txt}{dot}promotions={promos}/{iters}"))
        self._print()


def _strip_tag(line: str) -> str:
    line = (line or "").strip()
    if line.startswith("[") and "]" in line:
        return line[line.index("]") + 1:].strip()
    if line.startswith("- "):
        return line[2:].strip()
    return line


_SPINNER_UNICODE = ("·", "•", "●", "•")
_SPINNER_ASCII = ("-", "\\", "|", "/")


@contextmanager
def thinking(label: str = "thinking", *, stream=None) -> Iterator[None]:
    out = stream if stream is not None else sys.stdout
    is_tty = bool(getattr(out, "isatty", lambda: False)())
    if not is_tty or os.environ.get("NO_COLOR"):
        yield
        return

    glyphs = _pick_glyphs(out)
    frames = _SPINNER_UNICODE if glyphs is _GLYPHS_UNICODE else _SPINNER_ASCII
    stop = threading.Event()

    def _spin() -> None:
        for frame in itertools.cycle(frames):
            if stop.is_set():
                break
            try:
                out.write(f"\r\033[90m{frame} {label}...\033[0m")
                out.flush()
            except Exception:  # noqa: BLE001
                break
            time.sleep(0.12)

    worker = threading.Thread(target=_spin, daemon=True)
    worker.start()
    try:
        yield
    finally:
        stop.set()
        worker.join(timeout=0.3)
        try:
            out.write("\r\033[2K")
            out.flush()
        except Exception:  # noqa: BLE001
            pass
