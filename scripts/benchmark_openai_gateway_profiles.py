#!/usr/bin/env python3
"""Benchmark EvoMind's OpenAI-compatible model profiles without logging secrets."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def endpoint(base_url: str) -> str:
    base = base_url.rstrip("/")
    return base + "/chat/completions" if base.endswith("/v1") else base + "/v1/chat/completions"


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "minimum_ms": None, "median_ms": None, "maximum_ms": None}
    return {
        "count": len(values),
        "minimum_ms": round(min(values), 1),
        "median_ms": round(statistics.median(values), 1),
        "maximum_ms": round(max(values), 1),
    }


def profile_order(round_index: int, names: list[str]) -> list[str]:
    offset = round_index % len(names)
    return names[offset:] + names[:offset]


def request_once(
    *,
    url: str,
    key: str,
    model: str,
    profile: dict[str, str],
    stream: bool,
    timeout: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": "Return exactly PERF_OK"}],
        "temperature": 0,
        "max_tokens": 24,
        "stream": stream,
        **profile,
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json",
            "User-Agent": "evomind-gateway-profile-benchmark/1.0",
        },
        method="POST",
    )
    started = time.perf_counter()
    first_ms: float | None = None
    content: list[str] = []
    served_model = ""
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            if stream:
                for raw_line in response:
                    line = raw_line.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        body = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    served_model = str(body.get("model") or served_model)
                    for choice in body.get("choices") or []:
                        delta = choice.get("delta") or {}
                        chunk = str(delta.get("content") or "")
                        reasoning = str(
                            delta.get("reasoning_content") or delta.get("thinking") or ""
                        )
                        if (chunk or reasoning) and first_ms is None:
                            first_ms = round((time.perf_counter() - started) * 1000, 1)
                        if chunk:
                            content.append(chunk)
            else:
                body = json.loads(response.read().decode("utf-8"))
                served_model = str(body.get("model") or "")
                content.append(
                    str(
                        ((body.get("choices") or [{}])[0].get("message") or {}).get(
                            "content"
                        )
                        or ""
                    )
                )
                first_ms = round((time.perf_counter() - started) * 1000, 1)
        total_ms = round((time.perf_counter() - started) * 1000, 1)
        return {
            "ok": status == 200 and "PERF_OK" in "".join(content),
            "status": status,
            "served_model": served_model,
            "first_ms": first_ms,
            "total_ms": total_ms,
            "error_type": None,
        }
    except Exception as exc:  # sanitized: never persist key-bearing details
        return {
            "ok": False,
            "status": None,
            "served_model": "",
            "first_ms": None,
            "total_ms": round((time.perf_counter() - started) * 1000, 1),
            "error_type": type(exc).__name__,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:65068/v1"),
    )
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5.6-sol"))
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("workspace/llm/gpt56_performance_profiles_current.json"),
    )
    args = parser.parse_args()
    if args.samples < 2 or args.samples > 10:
        raise ValueError("samples must be within 2..10")
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not configured")

    profiles = {
        "baseline": {},
        "interactive": {"reasoning_effort": "low", "service_tier": "priority"},
        "research": {"reasoning_effort": "high", "service_tier": "priority"},
    }
    records: dict[str, list[dict[str, Any]]] = {name: [] for name in profiles}
    names = list(profiles)
    for round_index in range(args.samples):
        for name in profile_order(round_index, names):
            for stream in (False, True):
                observation = request_once(
                    url=endpoint(args.base_url),
                    key=key,
                    model=args.model,
                    profile=profiles[name],
                    stream=stream,
                    timeout=args.timeout,
                )
                observation.update({"round": round_index + 1, "stream": stream})
                records[name].append(observation)

    summary: dict[str, Any] = {}
    for name, observations in records.items():
        nonstream = [item for item in observations if not item["stream"]]
        stream = [item for item in observations if item["stream"]]
        summary[name] = {
            "profile": profiles[name],
            "success_count": sum(bool(item["ok"]) for item in observations),
            "request_count": len(observations),
            "nonstream_total": summarize(
                [float(item["total_ms"]) for item in nonstream if item["ok"]]
            ),
            "stream_first_delta": summarize(
                [float(item["first_ms"]) for item in stream if item["ok"] and item["first_ms"]]
            ),
            "stream_total": summarize(
                [float(item["total_ms"]) for item in stream if item["ok"]]
            ),
        }

    baseline_median = float(summary["baseline"]["stream_first_delta"]["median_ms"] or 0)
    interactive_median = float(
        summary["interactive"]["stream_first_delta"]["median_ms"] or 0
    )
    result = {
        "schema": "evomind.openai_gateway_profile_benchmark.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provider": "openai",
        "requested_model": args.model,
        "samples_per_mode": args.samples,
        "summary": summary,
        "observations": records,
        "interactive_stream_median_delta_percent": (
            round((interactive_median / baseline_median - 1) * 100, 1)
            if baseline_median and interactive_median
            else None
        ),
        "selected_profiles": {
            "interactive": profiles["interactive"],
            "research": profiles["research"],
        },
        "ok": bool(
            all(
                item["success_count"] == item["request_count"]
                for item in summary.values()
            )
            and all(
                observation["served_model"] == args.model
                for observations in records.values()
                for observation in observations
            )
        ),
        "secret_policy": "API key remained environment/DPAPI-only and is absent from evidence.",
    }
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({**result, "observations": "persisted", "output": str(args.output)}, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
