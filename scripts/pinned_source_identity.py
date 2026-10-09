"""Portable resolution for hash-bound source and artifact records.

Frozen plans may retain the absolute path observed when evidence was created,
but release verification must bind the recorded ``relative_path`` to the
current checkout.  Absolute paths remain a legacy fallback only when they are
already confined to the current project root.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any, Mapping


class PinnedPathError(ValueError):
    """Raised when a pinned record cannot resolve inside the project root."""


def _relative_parts(value: str) -> tuple[str, ...]:
    normalized = value.strip().replace("\\", "/")
    relative = PurePosixPath(normalized)
    if (
        not normalized
        or relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
        or (relative.parts and relative.parts[0].endswith(":"))
    ):
        raise PinnedPathError("pinned relative_path is invalid")
    return relative.parts


def bind_legacy_record_to_project_anchor(
    record: Mapping[str, Any],
    project_anchor: str,
) -> dict[str, Any]:
    """Add a portable identity to a legacy absolute-path record.

    ``project_anchor`` is a trusted repository directory such as
    ``workspace/hpc/job123``.  The suffix at or below that anchor is retained;
    the machine-specific prefix is discarded and later confinement is still
    enforced by :func:`resolve_pinned_project_path`.
    """

    if not isinstance(record, Mapping):
        raise PinnedPathError("legacy pinned record must be an object")
    anchor = "/".join(_relative_parts(project_anchor))
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise PinnedPathError("legacy pinned record is missing path identity")
    normalized = raw_path.strip().replace("\\", "/")
    marker = f"/{anchor}/"
    marker_index = normalized.casefold().rfind(marker.casefold())
    if marker_index < 0:
        raise PinnedPathError("legacy pinned path does not contain the project anchor")
    relative_path = normalized[marker_index + 1 :]
    _relative_parts(relative_path)
    return {**record, "relative_path": relative_path}


def resolve_pinned_project_path(
    record: Mapping[str, Any],
    project_root: Path,
    *,
    label: str = "pinned path",
) -> Path:
    """Resolve one record inside ``project_root`` without machine-path drift.

    A valid relative identity is authoritative across checkouts.  Records that
    predate ``relative_path`` may still use ``path``, but an absolute legacy
    path is accepted only when it is already inside the current project root.
    Symlink and ``..`` escapes are rejected after canonical resolution.
    """

    if not isinstance(record, Mapping):
        raise PinnedPathError(f"{label} record must be an object")
    root = Path(project_root).resolve()
    relative_value = record.get("relative_path")
    if isinstance(relative_value, str) and relative_value.strip():
        candidate = root.joinpath(*_relative_parts(relative_value)).resolve()
    else:
        raw_path = record.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise PinnedPathError(f"{label} is missing path identity")
        candidate = Path(raw_path.strip()).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        candidate = candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise PinnedPathError(f"{label} escaped the project root") from exc
    return candidate
