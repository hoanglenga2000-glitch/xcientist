from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any


MAX_FILES = 60_000
MAX_TEXT_BYTES = 256 * 1024
MAX_DOCS = 80
TEXT_SUFFIXES = {".md", ".rst", ".txt", ".yaml", ".yml", ".toml", ".ini"}
SECRET_LABEL = re.compile(
    r"(?i)(password|passwd|pwd|credential|token|secret|api[ _-]*key|authorization|bearer|"
    r"密码|口令|凭据|账号|登录)"
)
SECRET_VALUE = re.compile(
    r"(?i)(-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|"
    r"AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{30,}|"
    r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,})"
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def safe_relative(path: Path, root: Path) -> str:
    relative = path.relative_to(root).as_posix()
    if not relative or relative.startswith("/") or ".." in Path(relative).parts:
        raise ValueError("unsafe relative path")
    return relative


def redact_text(value: str) -> tuple[str, int]:
    rows: list[str] = []
    redacted = 0
    for line in value.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if SECRET_LABEL.search(line) or SECRET_VALUE.search(line):
            rows.append("[REDACTED SENSITIVE LINE]")
            redacted += 1
        else:
            rows.append(line)
    result = "\n".join(rows)
    if SECRET_VALUE.search(result):
        raise ValueError("sensitive value remained after redaction")
    return result, redacted


def bounded_text(path: Path) -> tuple[str, int, bool]:
    raw = path.read_bytes()[: MAX_TEXT_BYTES + 1]
    truncated = len(raw) > MAX_TEXT_BYTES
    text = raw[:MAX_TEXT_BYTES].decode("utf-8", "replace")
    cleaned, redacted = redact_text(text)
    return cleaned, redacted, truncated


def json_shape(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    if len(raw) > 4 * 1024 * 1024:
        return {"parsed": False, "reason": "size_cap"}
    value = json.loads(raw.decode("utf-8"))
    if isinstance(value, dict):
        return {"parsed": True, "type": "object", "keys": sorted(map(str, value))[:200]}
    if isinstance(value, list):
        first = value[0] if value else None
        return {
            "parsed": True,
            "type": "array",
            "length": len(value),
            "first_item_type": type(first).__name__ if first is not None else None,
            "first_item_keys": sorted(map(str, first))[:200] if isinstance(first, dict) else [],
        }
    return {"parsed": True, "type": type(value).__name__}


def jsonl_shape(path: Path) -> dict[str, Any]:
    keys: Counter[str] = Counter()
    rows = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if rows >= 20:
                break
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                keys.update(map(str, value))
            rows += 1
    return {"sampled_rows": rows, "keys": sorted(keys)[:200]}


def csv_shape(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        sampled = sum(1 for _, _row in zip(range(5), reader))
    return {"columns": [str(value)[:200] for value in header[:500]], "sampled_rows": sampled}


def notebook_shape(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    if len(raw) > 8 * 1024 * 1024:
        return {"parsed": False, "reason": "size_cap"}
    notebook = json.loads(raw.decode("utf-8"))
    cells = notebook.get("cells") if isinstance(notebook, dict) else []
    extracted: list[dict[str, Any]] = []
    redactions = 0
    for cell in cells or []:
        if len(extracted) >= 80 or not isinstance(cell, dict):
            break
        source = cell.get("source") or []
        text = "".join(source) if isinstance(source, list) else str(source)
        if not text.strip():
            continue
        cleaned, count = redact_text(text[:12_000])
        redactions += count
        extracted.append({"cell_type": str(cell.get("cell_type") or ""), "source": cleaned})
    return {
        "parsed": True,
        "cell_count": len(cells or []),
        "extracted_cells": extracted,
        "redacted_lines": redactions,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    data_root = Path(args.data_dir).resolve(strict=True)
    output_root = Path(args.out_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if not data_root.is_dir():
        raise ValueError("data directory is missing")

    extension_counts: Counter[str] = Counter()
    extension_bytes: Counter[str] = Counter()
    files: list[dict[str, Any]] = []
    documents: list[dict[str, Any]] = []
    total_bytes = 0
    redacted_lines = 0

    for current_root, directories, names in os.walk(data_root, followlinks=False):
        directories[:] = sorted(
            name
            for name in directories
            if name != ".evomind" and not (Path(current_root) / name).is_symlink()
        )
        for name in sorted(names):
            path = Path(current_root) / name
            if path.is_symlink() or not path.is_file():
                continue
            if len(files) >= MAX_FILES:
                raise ValueError("file count exceeds discovery cap")
            relative = safe_relative(path, data_root)
            size = path.stat().st_size
            suffix = path.suffix.lower() or "[none]"
            extension_counts[suffix] += 1
            extension_bytes[suffix] += size
            total_bytes += size
            files.append({"path": relative, "bytes": size, "suffix": suffix})

            lower_name = name.casefold()
            likely_doc = (
                suffix in TEXT_SUFFIXES
                or suffix in {".ipynb", ".json", ".jsonl", ".csv", ".tsv"}
                or any(marker in lower_name for marker in ("readme", "metric", "evaluation", "baseline", "schema"))
            )
            if not likely_doc or len(documents) >= MAX_DOCS:
                continue
            item: dict[str, Any] = {"path": relative, "bytes": size, "suffix": suffix}
            try:
                if suffix == ".ipynb":
                    item["notebook"] = notebook_shape(path)
                elif suffix == ".json":
                    item["shape"] = json_shape(path)
                    if size <= MAX_TEXT_BYTES:
                        text, count, truncated = bounded_text(path)
                        item.update(text=text, text_truncated=truncated)
                        redacted_lines += count
                elif suffix == ".jsonl":
                    item["shape"] = jsonl_shape(path)
                elif suffix in {".csv", ".tsv"}:
                    item["shape"] = csv_shape(path)
                else:
                    text, count, truncated = bounded_text(path)
                    item.update(text=text, text_truncated=truncated)
                    redacted_lines += count
            except (OSError, UnicodeError, json.JSONDecodeError, csv.Error, ValueError) as exc:
                item["read_error_class"] = type(exc).__name__
            documents.append(item)

    largest = sorted(files, key=lambda item: (-int(item["bytes"]), str(item["path"])))[:100]
    payload = {
        "schema": "evomind.competition_task_discovery.v1",
        "read_only": True,
        "training_performed": False,
        "gpu_compute_requested": False,
        "regular_files": len(files),
        "total_bytes": total_bytes,
        "extension_counts": dict(sorted(extension_counts.items())),
        "extension_bytes": dict(sorted(extension_bytes.items())),
        "largest_files": largest,
        "documents": documents,
        "redacted_sensitive_lines": redacted_lines,
        "remote_absolute_paths_emitted": False,
        "data_body_exported": False,
    }
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    target = output_root / "task-discovery.json"
    temporary = output_root / ".task-discovery.json.tmp"
    temporary.write_bytes(encoded)
    os.replace(temporary, target)
    receipt = {
        "schema": "evomind.competition_task_discovery_receipt.v1",
        "artifact": "task-discovery.json",
        "bytes": len(encoded),
        "sha256": sha256_bytes(encoded),
        "regular_files": len(files),
        "total_bytes": total_bytes,
        "read_only": True,
        "training_performed": False,
    }
    (output_root / "task-discovery-receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
