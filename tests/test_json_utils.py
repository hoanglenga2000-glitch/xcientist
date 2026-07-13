"""Regression tests for the central JSON serializer.

Guards the numpy-serialization fix: gate/evidence/result dataclasses flow
through write_json, so a numpy value reaching the serializer must never crash.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path

import pytest

from research_agent_workstation.server.core.json_utils import (
    append_jsonl,
    to_jsonable,
    write_json,
)

np = pytest.importorskip("numpy")


def _roundtrip(value):
    return json.loads(json.dumps(to_jsonable(value)))


def test_native_scalars_passthrough():
    assert to_jsonable(True) is True
    assert to_jsonable(1) == 1
    assert to_jsonable(1.5) == 1.5
    assert to_jsonable("x") == "x"
    assert to_jsonable(None) is None


def test_numpy_bool_serializes_to_python_bool():
    out = to_jsonable(np.bool_(True))
    assert out is True
    assert isinstance(out, bool)


@pytest.mark.parametrize(
    "value,expected",
    [
        (np.int64(42), 42),
        (np.int32(7), 7),
        (np.float64(2.5), 2.5),
        (np.float32(0.5), 0.5),
    ],
)
def test_numpy_scalars(value, expected):
    out = to_jsonable(value)
    assert out == expected
    json.dumps(out)


def test_numpy_arrays_become_lists():
    assert _roundtrip(np.array([1, 2, 3])) == [1, 2, 3]
    assert _roundtrip(np.array([[1, 2], [3, 4]])) == [[1, 2], [3, 4]]


def test_nested_structures_with_numpy():
    payload = {"a": np.int64(5), "b": [np.float64(1.5), np.bool_(False)], "c": (np.int32(1),)}
    out = _roundtrip(payload)
    assert out == {"a": 5, "b": [1.5, False], "c": [1]}


def test_dataclass_with_numpy_fields():
    @dataclass
    class Result:
        score: object
        flags: object

    out = _roundtrip(Result(score=np.float64(0.95), flags=np.array([True, False])))
    assert out == {"score": 0.95, "flags": [True, False]}


def test_enum_decimal_bytes_datetime():
    class Color(Enum):
        RED = "red"

    assert to_jsonable(Color.RED) == "red"
    assert to_jsonable(Decimal("2.5")) == 2.5
    assert to_jsonable(b"hi") == "hi"
    assert to_jsonable(datetime(2026, 7, 1, 12, 0)) == "2026-07-01T12:00:00"
    assert to_jsonable(date(2026, 7, 1)) == "2026-07-01"


def test_set_and_frozenset_become_lists():
    assert sorted(to_jsonable({1, 2, 3})) == [1, 2, 3]
    assert sorted(to_jsonable(frozenset({4, 5}))) == [4, 5]


def test_dataclass_type_is_not_treated_as_instance():
    @dataclass
    class Empty:
        value: int = 0

    # Passing the class itself must not raise (is_dataclass(type) is True).
    assert to_jsonable(Empty) is Empty


def test_write_json_with_numpy(tmp_path: Path):
    target = tmp_path / "nested" / "out.json"
    payload = {"medal_rate": np.float64(0.33), "passed": np.bool_(True), "ranks": np.array([1, 2])}
    write_json(target, payload)
    loaded = json.loads(target.read_text(encoding="utf-8"))
    assert loaded == {"medal_rate": 0.33, "passed": True, "ranks": [1, 2]}


def test_append_jsonl_with_numpy(tmp_path: Path):
    target = tmp_path / "audit.jsonl"
    append_jsonl(target, {"step": np.int64(1)})
    append_jsonl(target, {"step": np.int64(2)})
    lines = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    assert lines == [{"step": 1}, {"step": 2}]
