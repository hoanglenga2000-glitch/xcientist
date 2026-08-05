#!/usr/bin/env python3
"""Verify a configured OpenAI-compatible gateway without printing its secret."""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def endpoint(base_url: str, suffix: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return base + suffix
    return base + "/v1" + suffix


def request_json(url: str, key: str, payload: dict[str, Any] | None, timeout: int) -> tuple[int, dict[str, Any], float]:
    headers = {
        "Authorization": f"Bearer {key}",
        "User-Agent": "evomind-openai-gateway-smoke/1.0",
    }
    data = None
    method = "GET"
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
        method = "POST"
    started = time.perf_counter()
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
        return response.status, body, round((time.perf_counter() - started) * 1000, 1)


def verify_stream(
    base_url: str,
    key: str,
    model: str,
    timeout: int,
    request_profile: dict[str, str],
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Return exactly STREAM_OK"}],
        "temperature": 0,
        "max_tokens": 32,
        "stream": True,
        "stream_options": {"include_usage": True},
        **request_profile,
    }
    req = urllib.request.Request(
        endpoint(base_url, "/chat/completions"),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "User-Agent": "evomind-openai-gateway-smoke/1.0",
        },
        method="POST",
    )
    started = time.perf_counter()
    first_delta_ms: float | None = None
    text: list[str] = []
    events = 0
    usage: dict[str, Any] = {}
    with urllib.request.urlopen(req, timeout=timeout) as response:
        status = response.status
        for raw_line in response:
            line = raw_line.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            raw_data = line[5:].strip()
            if raw_data == "[DONE]":
                break
            try:
                body = json.loads(raw_data)
            except json.JSONDecodeError:
                continue
            events += 1
            usage = body.get("usage") or usage
            for choice in body.get("choices") or []:
                delta = choice.get("delta") or {}
                content = str(delta.get("content") or "")
                reasoning = str(delta.get("reasoning_content") or delta.get("thinking") or "")
                if (content or reasoning) and first_delta_ms is None:
                    first_delta_ms = round((time.perf_counter() - started) * 1000, 1)
                if content:
                    text.append(content)
    content = "".join(text)
    return {
        "ok": status == 200 and "STREAM_OK" in content,
        "status": status,
        "first_delta_ms": first_delta_ms,
        "total_ms": round((time.perf_counter() - started) * 1000, 1),
        "event_count": events,
        "content": content[:120],
        "usage": usage,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:65068/v1"))
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5.6-sol"))
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high", "xhigh", "max", "ultra"),
        default=os.environ.get("OPENAI_REASONING_EFFORT") or None,
    )
    parser.add_argument(
        "--service-tier",
        choices=("auto", "default", "flex", "priority"),
        default=os.environ.get("OPENAI_SERVICE_TIER") or None,
    )
    args = parser.parse_args()
    key = os.environ.get("OPENAI_API_KEY", "")
    request_profile = {
        **(
            {"reasoning_effort": args.reasoning_effort}
            if args.reasoning_effort
            else {}
        ),
        **({"service_tier": args.service_tier} if args.service_tier else {}),
    }
    report: dict[str, Any] = {
        "schema": "evomind.openai_gateway_smoke.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provider": "openai",
        "base_url": args.base_url,
        "requested_model": args.model,
        "request_profile": request_profile,
        "secret_policy": "API key loaded from environment or DPAPI and never written to this report.",
    }
    if not key:
        report.update({"status": "not_configured", "ok": False, "error_type": "MissingApiKey"})
    else:
        try:
            status, models_body, latency = request_json(endpoint(args.base_url, "/models"), key, None, args.timeout)
            model_ids = [str(item.get("id")) for item in models_body.get("data", []) if isinstance(item, dict)]
            report["models"] = {
                "ok": status == 200 and args.model in model_ids,
                "status": status,
                "latency_ms": latency,
                "model_count": len(model_ids),
                "requested_model_present": args.model in model_ids,
            }

            status, body, latency = request_json(
                endpoint(args.base_url, "/chat/completions"),
                key,
                {
                    "model": args.model,
                    "messages": [
                        {"role": "system", "content": "Return only the requested token."},
                        {"role": "user", "content": "Return exactly GATEWAY_OK"},
                    ],
                    "temperature": 0,
                    "max_tokens": 32,
                    "stream": False,
                    **request_profile,
                },
                args.timeout,
            )
            message = ((body.get("choices") or [{}])[0].get("message") or {})
            content = str(message.get("content") or "")
            report["nonstream"] = {
                "ok": status == 200 and "GATEWAY_OK" in content,
                "status": status,
                "latency_ms": latency,
                "served_model": body.get("model"),
                "content": content[:120],
                "usage": body.get("usage") or {},
            }
            report["stream"] = verify_stream(
                args.base_url,
                key,
                args.model,
                args.timeout,
                request_profile,
            )

            status, body, latency = request_json(
                endpoint(args.base_url, "/chat/completions"),
                key,
                {
                    "model": args.model,
                    "messages": [{"role": "user", "content": "Call report_probe with status ok. Do not answer in text."}],
                    "temperature": 0,
                    "max_tokens": 64,
                    "tools": [{
                        "type": "function",
                        "function": {
                            "name": "report_probe",
                            "description": "Report probe state",
                            "parameters": {
                                "type": "object",
                                "properties": {"status": {"type": "string"}},
                                "required": ["status"],
                            },
                        },
                    }],
                    "tool_choice": {"type": "function", "function": {"name": "report_probe"}},
                    "stream": False,
                    **request_profile,
                },
                args.timeout,
            )
            message = ((body.get("choices") or [{}])[0].get("message") or {})
            calls = message.get("tool_calls") or []
            first_name = ((calls[0].get("function") or {}).get("name") if calls else None)
            report["tool_call"] = {
                "ok": status == 200 and first_name == "report_probe",
                "status": status,
                "latency_ms": latency,
                "tool_call_count": len(calls),
                "first_tool": first_name,
            }
            report["performance"] = {
                "advisory_ok": bool(
                    float(report["nonstream"]["latency_ms"]) <= 3000
                    and float(report["stream"]["first_delta_ms"] or 999999) <= 3000
                    and float(report["tool_call"]["latency_ms"]) <= 3000
                ),
                "nonstream_budget_ms": 3000,
                "stream_first_delta_budget_ms": 3000,
                "tool_call_budget_ms": 3000,
            }
            checks = [
                report["models"],
                report["nonstream"],
                report["stream"],
                report["tool_call"],
            ]
            report["ok"] = all(bool(check.get("ok")) for check in checks)
            report["status"] = "passed" if report["ok"] else "failed"
        except urllib.error.HTTPError as exc:
            report.update({"status": "failed", "ok": False, "error_type": "HTTPError", "http_status": exc.code})
        except Exception as exc:  # sanitized: never include URL/body/key-bearing exception text
            report.update({"status": "failed", "ok": False, "error_type": type(exc).__name__})

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temp = args.output.with_suffix(args.output.suffix + ".tmp")
        temp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(args.output)
        report["output"] = str(args.output.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
