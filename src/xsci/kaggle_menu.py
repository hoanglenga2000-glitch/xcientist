"""Arrow-key selection menu for the setup wizard (Claude-Code style).

Real TTY -> a highlighted up/down list you drive with the arrow keys; piped /
NO_COLOR / tests -> a numbered text fallback driven by an injectable reader, so
the wizard works everywhere and never hangs a non-interactive run.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Callable, Optional, Sequence


@dataclass
class Choice:
    """One selectable row: a bright label and an optional dim hint."""

    label: str
    hint: str = ""


_UNI = {"arrow": "❯", "dot": "·"}   # ❯  ·
_ASC = {"arrow": ">", "dot": "-"}


def _glyphs(stream) -> dict:
    """Richest glyph set the stream's encoding can actually emit (gbk-safe)."""
    enc = getattr(stream, "encoding", None) or "utf-8"
    try:
        "".join(_UNI.values()).encode(enc)
        return _UNI
    except (UnicodeEncodeError, LookupError):
        return _ASC


def _enable_windows_vt() -> bool:
    """Best-effort enable of ANSI/VT processing on a Windows console.

    Without this, the cursor-movement escapes the arrow menu relies on would
    print as literal garbage, so a failure here disables the interactive path.
    """
    if os.name != "nt":
        return True
    try:
        import ctypes

        kernel = ctypes.windll.kernel32
        handle = kernel.GetStdHandle(-11)              # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if not kernel.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        if not (mode.value & 0x0004):                  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
            kernel.SetConsoleMode(handle, mode.value | 0x0004)
        return True
    except Exception:  # noqa: BLE001 - any failure -> fall back to numbered mode
        return False


def _interactive(stream) -> bool:
    """True only when a live keyboard menu is safe: a real TTY, colour allowed,
    not under pytest, and (on Windows) VT successfully enabled."""
    if os.environ.get("XSCI_NO_MENU") or os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return False
    try:
        if not (sys.stdin.isatty() and stream.isatty()):
            return False
    except Exception:  # noqa: BLE001
        return False
    return _enable_windows_vt()


def _w(stream, text: str) -> None:
    try:
        stream.write(text)
        stream.flush()
    except UnicodeEncodeError:
        # Legacy console code page (e.g. gbk): re-emit with unrepresentable chars
        # replaced so the line still shows (degraded) instead of vanishing.
        enc = getattr(stream, "encoding", None) or "utf-8"
        try:
            stream.write(text.encode(enc, "replace").decode(enc, "replace"))
            stream.flush()
        except Exception:  # noqa: BLE001
            pass
    except Exception:  # noqa: BLE001 - a render write must never crash setup
        pass


def _read_key() -> str:
    """Block for one keystroke, return a token: up/down/left/right/enter/esc, a
    single char (e.g. a digit), or '' for an unhandled key. Ctrl+C raises."""
    if os.name == "nt":
        import msvcrt

        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):                       # arrow / function-key prefix
            return {"H": "up", "P": "down", "K": "left", "M": "right"}.get(msvcrt.getwch(), "")
        if ch == "\x03":
            raise KeyboardInterrupt
        if ch in ("\r", "\n"):
            return "enter"
        if ch == "\x1b":
            return "esc"
        return ch
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x03":
            raise KeyboardInterrupt
        if ch in ("\r", "\n"):
            return "enter"
        if ch == "\x1b":
            return {"[A": "up", "[B": "down", "[C": "right", "[D": "left"}.get(sys.stdin.read(2), "esc")
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _row(choice: Choice, selected: bool, g: dict, color: bool) -> str:
    hint = f"  {g['dot']} {choice.hint}" if choice.hint else ""
    if not color:
        return f"{g['arrow'] if selected else ' '} {choice.label}{hint}"
    if selected:
        body = f"\033[96;1m{g['arrow']} {choice.label}\033[0m"
    else:
        body = f"  {choice.label}"
    if choice.hint:
        body += f"\033[90m  {g['dot']} {choice.hint}\033[0m"
    return body


def _select_interactive(title, choices, default, allow_skip, stream, g, color) -> int:
    idx = max(0, min(default, len(choices) - 1))
    n = len(choices)
    if title:
        _w(stream, title + "\n")
    hint_txt = f"Up/Down move {g['dot']} Enter select" + (f" {g['dot']} Esc skip" if allow_skip else "")
    footer = f"\033[90m  {hint_txt}\033[0m" if color else f"  {hint_txt}"

    def paint(first: bool = False) -> None:
        if not first:
            _w(stream, f"\033[{n + 1}A")                 # cursor back up to the first row
        for i, c in enumerate(choices):
            _w(stream, "\033[2K" + _row(c, i == idx, g, color) + "\n")
        _w(stream, "\033[2K" + footer + "\n")

    paint(first=True)
    while True:
        key = _read_key()
        if key == "up":
            idx = (idx - 1) % n
        elif key == "down":
            idx = (idx + 1) % n
        elif key == "enter":
            return idx
        elif key == "esc" and allow_skip:
            return -1
        elif key and key.isdigit() and 1 <= int(key) <= n:
            idx = int(key) - 1
        else:
            continue
        paint()


def _default_reader(prompt: str, default: str = "") -> str:
    try:
        raw = input(f"  {prompt}> ").strip()
    except EOFError:
        return default
    return raw or default


def _select_numbered(title, choices, default, allow_skip, reader, stream) -> int:
    if title:
        _w(stream, title.strip() + "\n")
    for i, c in enumerate(choices, 1):
        hint = f"   {c.hint}" if c.hint else ""
        _w(stream, f"    {i}) {c.label}{hint}\n")
    if allow_skip:
        _w(stream, "    s) skip for now\n")
    raw = (reader(f"Choose [{default + 1}]", str(default + 1)) or "").strip().lower()
    if allow_skip and raw in ("s", "skip"):
        return -1
    if raw.isdigit() and 1 <= int(raw) <= len(choices):
        return int(raw) - 1
    return default


def select(title: str, choices: Sequence[Choice], *, default: int = 0,
           allow_skip: bool = False, reader: Optional[Callable[[str, str], str]] = None,
           stream=None) -> int:
    """Return the chosen index (0-based), or -1 if skipped.

    A real TTY gets the arrow-key menu; otherwise a numbered prompt driven by
    ``reader`` (defaults to ``input()``). ``reader`` is used ONLY in the fallback,
    so pipes/tests stay deterministic while real terminals get keyboard control."""
    stream = stream if stream is not None else sys.stdout
    if not choices:
        return -1
    if _interactive(stream):
        try:
            return _select_interactive(title, list(choices), default, allow_skip,
                                       stream, _glyphs(stream), True)
        except KeyboardInterrupt:
            raise
        except Exception:  # noqa: BLE001 - any console quirk -> numbered fallback
            pass
    return _select_numbered(title, list(choices), default, allow_skip,
                            reader or _default_reader, stream)
