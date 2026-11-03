"""t238 C7c-5 HARD RULE: every HPC-side path EvoMind builds lives under /hpc2hdd/home/aimslab/jinghw.

``confine_remote_path(root, *parts)`` is the one constructor. It fails closed (RemotePathError)
on: ``..``/``.``/empty components (``//``, trailing ``/``), absolute re-roots inside parts, other
home directories, ``~``, environment-variable expansion, NUL/newline/control characters, shell
metacharacters and whitespace, ``%`` (double-encoding), Unicode look-alikes of ``.``, ``/`` or
``\\``, and prefix-sibling tricks (``/hpc2hdd/home/aimslab/jinghw_evil``). Matching is
case-sensitive. When a remote resolve step is used, ``verify_resolved`` checks the real path
still has the prefix (symlink escape). Shell text must be built with ``shell_command`` (every
argument confined or quoted with ``shlex.quote``); creating symlinks is refused.
"""
from __future__ import annotations

import posixpath
import re
import shlex
import unicodedata

HPC_HOME_ROOT = "/hpc2hdd/home/aimslab/jinghw"
# Carried on the trainer's command line (``python3 -X <marker>=<job id>``, ignored by Python) so the
# dead-man guard and orphan cleanup match the real process, which runs as ``bundle_*/work/train_gpu.py``.
TRAINER_MARKER = "evomind_gpu_trainer"

_SHELL_META = set(" \t;&|<>()$`\\\"'*?[]{}!#~%")
# Separators that are not ASCII but read (or normalise) as '.', '/' or '\'.
_LOOKALIKES = {"․", "‥", "…", "。", "．", "｡", "﹒", "∕", "⁄",
               "⧸", "⧹", "／", "＼", "∖", "﹨", "̸", "⳺"}
_REFUSED_PROGRAMS = {"ln", "symlink", "link"}


class RemotePathError(ValueError):
    """Fail-closed confinement violation. ``code`` is stable; the text never echoes the path."""

    code = "remote_path_rejected"

    def __init__(self, reason: str) -> None:
        super().__init__("remote path escapes the dedicated EvoMind root")
        self.reason = reason


def _check_component(component: str) -> None:
    if component in {"", ".", ".."}:
        raise RemotePathError("dot_or_empty_component")
    if component.startswith("-"):
        raise RemotePathError("option_like_component")
    for char in component:
        if char in _SHELL_META:
            raise RemotePathError("shell_metacharacter")
        if char in _LOOKALIKES:
            raise RemotePathError("unicode_lookalike")
        if ord(char) < 0x20 or ord(char) == 0x7F:
            raise RemotePathError("control_character")
        if ord(char) > 0x7F:
            category = unicodedata.category(char)
            if category[0] in {"C", "Z", "M"}:
                raise RemotePathError("invisible_or_combining_character")
            folded = unicodedata.normalize("NFKC", char)
            if any(item in folded for item in "./\\") or any(item in _SHELL_META for item in folded):
                raise RemotePathError("unicode_lookalike")


def _components(value: str) -> list[str]:
    if not isinstance(value, str) or "\x00" in value:
        raise RemotePathError("not_text")
    parts = value.split("/")
    for part in parts:
        _check_component(part)
    return parts


def _check_root(root: str) -> str:
    if not isinstance(root, str) or not root.startswith("/") or root.endswith("/"):
        raise RemotePathError("root_invalid")
    _components(root[1:])
    if root != HPC_HOME_ROOT and not root.startswith(HPC_HOME_ROOT + "/"):
        raise RemotePathError("outside_hpc_home")
    return root


def confine_remote_path(root: str, *parts: str) -> str:
    """Join ``parts`` under ``root`` (itself under HPC_HOME_ROOT); never escapes, never normalises silently."""

    root = _check_root(root)
    joined = [root]
    for part in parts:
        if not isinstance(part, str) or part.startswith("/"):
            raise RemotePathError("absolute_reroot")
        joined.extend(_components(part))
    path = "/".join(joined)
    if posixpath.normpath(path) != path:
        raise RemotePathError("not_normalised")
    return path


def confine_absolute(path: str, root: str = HPC_HOME_ROOT) -> str:
    """Validate an already-built absolute path against ``root`` with the same rules."""

    root = _check_root(root)
    if not isinstance(path, str) or not path.startswith("/"):
        raise RemotePathError("not_absolute")
    if path == root:
        return path
    if not path.startswith(root + "/"):
        raise RemotePathError("outside_root")
    return confine_remote_path(root, path[len(root) + 1:])


_OUTPUT_NAME = re.compile(r"[\w .,()+=@\[\]-]+")
_WINDOWS_DEVICE = re.compile(r"(?:con|prn|aux|nul|com[0-9]|lpt[0-9])(?:\..*)?", re.IGNORECASE)


def output_file_name(name: str) -> str:
    """One file name written by a remote job, for the SFTP download only (never put in a shell command).

    Wider than ``confine_remote_path``: spaces, CJK and parentheses are normal output names. It stays
    a single safe component on both ends (the host is Windows): no separators, traversal, control
    characters, shell metacharacters or device names. Anything else raises and is skipped by the caller.
    """

    if not isinstance(name, str) or name in {".", ".."} or not _OUTPUT_NAME.fullmatch(name):
        raise RemotePathError("output_name_unsafe")
    if name[0] in " -" or name[-1] in " ." or _WINDOWS_DEVICE.fullmatch(name):
        raise RemotePathError("output_name_unsafe")
    return name


def verify_resolved(resolved: str, root: str = HPC_HOME_ROOT) -> str:
    """After a remote realpath/normalize, the real path must still be under ``root`` (symlink escape)."""

    try:
        return confine_absolute(str(resolved), root)
    except RemotePathError:
        raise RemotePathError("symlink_escape") from None


def resolve_and_verify(sftp, path: str, root: str = HPC_HOME_ROOT) -> str:
    """Resolve on the remote (SFTP REALPATH follows symlinks) and fail closed on escape or error."""

    confined = confine_absolute(path, root)
    try:
        resolved = sftp.normalize(confined)
    except Exception:
        raise RemotePathError("resolve_failed") from None
    return verify_resolved(resolved, root)


def shell_command(program: str, *args: str, paths: tuple[int, ...] = (), root: str = HPC_HOME_ROOT) -> str:
    """Quote every argument; arguments at ``paths`` indexes must be confined absolute paths."""

    if program in _REFUSED_PROGRAMS or posixpath.basename(program) in _REFUSED_PROGRAMS:
        raise RemotePathError("symlink_creation_refused")
    values = list(args)
    for index in paths:
        values[index] = confine_absolute(values[index], root)
    return " ".join(shlex.quote(item) for item in (program, *values))
