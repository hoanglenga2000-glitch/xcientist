"""Extract privacy-safe interaction signals from operator-owned chat exports.

The output intentionally contains aggregate behaviour counts and source hashes,
not prompts, answers, credentials, hidden reasoning, or private-grader feedback.
It is an offline curation aid for the versioned assistant behaviour board.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

SCHEMA = "evomind.assistant_behavior_distillation.v1"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _iter_messages(value: Any) -> Iterable[tuple[str, str]]:
    if isinstance(value, dict):
        role = str(value.get("role") or "").strip().lower()
        content = value.get("content")
        if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
            yield role, content.strip()
            return
        for key, item in value.items():
            if str(key).lower() in {"reasoning", "thinking", "hidden_reasoning", "private_grader"}:
                continue
            yield from _iter_messages(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_messages(item)


def _read_export(path: Path) -> tuple[list[tuple[str, str]], str]:
    raw = path.read_bytes()
    messages: list[tuple[str, str]] = []
    try:
        payload = json.loads(raw.decode("utf-8-sig"))
        messages.extend(_iter_messages(payload))
    except (UnicodeDecodeError, json.JSONDecodeError):
        for line in raw.decode("utf-8", errors="replace").splitlines():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            messages.extend(_iter_messages(payload))
    return messages, _sha256_bytes(raw)


def _signals(answer: str) -> set[str]:
    folded = answer.casefold()
    first = folded[:180]
    found: set[str] = set()
    if any(term in first for term in ("结论", "结果是", "当前", "已完成", "go", "no-go", "done")):
        found.add("conclusion_first")
    if any(term in folded for term in ("证据", "依据", "sha-256", "run id", "已验证", "verified")):
        found.add("evidence_grounded")
    if any(term in folded for term in ("下一步", "接下来", "next step", "立即执行", "验证方式")):
        found.add("actionable_next_step")
    if any(term in folded for term in ("直接复制", "你可以直接", "copy this", "直接发")):
        found.add("copyable_followup")
    if any(term in folded for term in ("未训练", "不训练", "未提交", "不提交", "no training", "no submission")):
        found.add("constraint_carryover")
    if any(term in folded for term in ("原因", "根因", "修复", "重试", "排障", "diagnos", "verify")):
        found.add("failure_recovery")
    if any(term in folded for term in ("下载", "文件", "artifact", "bytes", "sha256")):
        found.add("artifact_handoff")
    if len(answer) <= 2600:
        found.add("progressive_disclosure")
    return found


def distill(paths: Iterable[str | Path]) -> dict[str, Any]:
    sources: list[dict[str, Any]] = []
    pairs = 0
    signals: Counter[str] = Counter()
    answer_hashes: set[str] = set()
    for value in sorted({str(Path(item).resolve()) for item in paths}):
        path = Path(value)
        messages, source_hash = _read_export(path)
        sources.append({
            "name": path.name,
            "sha256": source_hash,
            "message_count": len(messages),
        })
        previous_role = ""
        for role, content in messages:
            if role == "assistant" and previous_role == "user":
                digest = _sha256_bytes(content.encode("utf-8"))
                if digest not in answer_hashes:
                    answer_hashes.add(digest)
                    pairs += 1
                    signals.update(_signals(content))
            previous_role = role
    return {
        "schema": SCHEMA,
        "source_policy": {
            "operator_owned_exports_only": True,
            "raw_content_retained": False,
            "hidden_reasoning_retained": False,
            "private_grader_feedback_used": False,
        },
        "sources": sources,
        "unique_user_assistant_pairs": pairs,
        "behavior_signal_counts": dict(sorted(signals.items())),
        "candidate_signal_order": [
            item
            for item, _ in sorted(signals.items(), key=lambda pair: (-pair[1], pair[0]))
        ],
    }


def write_report(path: str | Path, report: dict[str, Any]) -> Path:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary_name, target)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, help="Operator-owned JSON/JSONL export")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = distill(args.input)
    target = write_report(args.output, report)
    print(json.dumps({"status": "passed", "output": str(target), "pairs": report["unique_user_assistant_pairs"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
