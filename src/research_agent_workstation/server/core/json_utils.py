from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any


def _coerce_numpy(value: Any) -> Any | None:
    """Best-effort conversion of numpy scalars/arrays to native Python types.

    Returns the converted value, or ``None`` when ``value`` is not a numpy
    object. numpy is imported lazily so this module has no hard dependency on
    it; if numpy is absent the function is a no-op.
    """
    module = type(value).__module__
    if not module or not module.startswith("numpy"):
        return None
    # numpy scalar (np.bool_, np.int64, np.float32, ...)
    item = getattr(value, "item", None)
    if callable(item) and getattr(value, "ndim", None) == 0:
        return item()
    # numpy array / matrix
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    return None


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Enum):
        return to_jsonable(value.value)
    if isinstance(value, Decimal):
        return float(value)
    # bool is a subclass of int; handle before the numpy/int checks below.
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    coerced = _coerce_numpy(value)
    if coerced is not None:
        return to_jsonable(coerced)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def append_jsonl(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_jsonable(payload), ensure_ascii=False) + "\n")
    return path
