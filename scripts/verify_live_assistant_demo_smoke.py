"""Run a real authenticated EvoMind assistant demo smoke test.

The test is intentionally isolated from the user's live 8088 browser session:

* starts a temporary dashboard/runtime pair on non-default ports;
* reads only that temporary port's one-time bootstrap URL;
* exchanges the fragment token for an HttpOnly session cookie through the same
  /api/session/bootstrap route used by the browser;
* posts novice Chinese prompts to /api/assistant/stream and requires real
  answer_completed SSE events with real tool calls, including an HPC connection
  self-check prompt that mirrors the live demo failure mode;
* stops only the verified temporary EvoMind processes on exit.

The report never writes the bootstrap token, session cookie, CSRF token, or full
answer text.
"""

from __future__ import annotations

import argparse
import hashlib
import http.cookiejar
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = ROOT / "web" / "research-agent-workstation" / ".runtime-logs"
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from xsci.assistant_quality_evaluation import governance_snapshot  # noqa: E402

DEFAULT_PROMPT = (
    "我是第一次用这个系统的小白。请你用中文帮我查看上次 SIIM 实验目前能说明什么、"
    "有哪些真实证据、下一步该怎么改进；不要启动训练，不要提交 Kaggle。"
)
DEFAULT_FOLLOWUP_PROMPT = (
    "刚才这些我听懂了一点。那如果我现在要现场演示给别人看，"
    "你基于上一轮内容给我三个最稳的下一步动作就行，别展开太长，仍然不要训练、不要提交。"
)
DEFAULT_LITERATURE_PROMPT = (
    "你刚才说的分组验证和下一轮计划，有哪些真实论文支持？"
    "请列两篇最相关的已核验文献，写标题、年份、DOI，并各用一句小白话说明。"
    "没有已核验来源就直接说不知道，不要凭记忆编 DOI。"
)
DEFAULT_CONNECTION_PROMPT = (
    "我是小白，帮我检查现在能不能连接服务器，作业号90673。"
    "请你像 Codex 一样先自己检查网关和 profile，简单告诉我结论；"
    "不要训练，不要提交 Kaggle，也不要调用 private grader。"
)
TARGET_RUN = "evomind_siim_isic_a800_job90353_20260730_095826"
REQUIRED_PROVIDER = "openai"
REQUIRED_MODEL = "gpt-5.6-sol"
REVIEWED_LITERATURE_DOIS = (
    "10.1111/jdv.20479",
    "10.1016/j.media.2021.102305",
)


@dataclass
class ProcResult:
    ok: bool
    returncode: int
    stdout_tail: str
    stderr_tail: str
    seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "returncode": self.returncode,
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
            "seconds": round(self.seconds, 3),
        }


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def run_manager(command: list[str], *, port: int, runtime_port: int, timeout: float) -> ProcResult:
    env = os.environ.copy()
    env["EVOMIND_RUNTIME_PORT"] = str(runtime_port)
    start = time.monotonic()
    proc = subprocess.run(
        [sys.executable, "scripts/manage_workstation_dashboard.py", *command, "--port", str(port), "--timeout", str(timeout)],
        cwd=ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=max(timeout + 30, 60),
        check=False,
    )
    return ProcResult(
        ok=proc.returncode == 0,
        returncode=int(proc.returncode),
        stdout_tail=proc.stdout[-4000:],
        stderr_tail=proc.stderr[-4000:],
        seconds=time.monotonic() - start,
    )


def bootstrap_path(port: int) -> Path:
    suffix = "" if port == 8088 else f".{port}"
    return RUNTIME_DIR / f"dashboard{suffix}.bootstrap.once"


def read_bootstrap_token(port: int) -> tuple[str, str]:
    path = bootstrap_path(port)
    raw = path.read_text(encoding="utf-8").strip()
    parsed = urllib.parse.urlparse(raw)
    fragment = urllib.parse.parse_qs(parsed.fragment)
    token = (fragment.get("bootstrap") or [""])[0]
    if not token:
        raise RuntimeError("temporary bootstrap token was not present")
    return raw.split("#", 1)[0], token


def request_json(
    opener: urllib.request.OpenerDirector,
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 15,
) -> tuple[int, dict[str, Any]]:
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=body, method=method)
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    if body is not None and not req.has_header("Content-Type"):
        req.add_header("Content-Type", "application/json")
    with opener.open(req, timeout=timeout) as response:
        data = response.read().decode("utf-8", errors="replace")
        parsed = json.loads(data) if data else {}
        return int(response.status), parsed


def parse_sse_line_buffer(events: list[dict[str, Any]], event_name: str | None, data_lines: list[str]) -> tuple[str | None, list[str]]:
    if not data_lines:
        return None, []
    raw_data = "\n".join(data_lines)
    try:
        data = json.loads(raw_data)
    except json.JSONDecodeError:
        data = {"raw": raw_data}
    if isinstance(data, dict):
        item = {"event": event_name or data.get("type") or "message", "data": data}
    else:
        item = {"event": event_name or "message", "data": {"value": data}}
    events.append(item)
    return None, []


def post_assistant_stream(
    opener: urllib.request.OpenerDirector,
    base_url: str,
    csrf: str,
    prompt: str,
    selected_task: str,
    timeout: float,
    *,
    session_id: str,
    history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    payload = {
        "prompt": prompt,
        "session_id": session_id,
        "selected_task": selected_task,
        "history": history or [],
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(f"{base_url}/api/assistant/stream", data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "text/event-stream")
    req.add_header("Origin", base_url)
    req.add_header("x-evomind-csrf", csrf)
    events: list[dict[str, Any]] = []
    event_name: str | None = None
    data_lines: list[str] = []
    started = time.monotonic()
    with opener.open(req, timeout=timeout) as response:
        status = int(response.status)
        while True:
            if time.monotonic() - started > timeout:
                raise TimeoutError("assistant stream timed out")
            raw_line = response.readline()
            if not raw_line:
                event_name, data_lines = parse_sse_line_buffer(events, event_name, data_lines)
                break
            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
            if line == "":
                event_name, data_lines = parse_sse_line_buffer(events, event_name, data_lines)
                if any(item.get("event") == "answer_completed" for item in events):
                    break
                continue
            if line.startswith("event:"):
                event_name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data_lines.append(line.split(":", 1)[1].lstrip())
    completed = [item for item in events if item.get("event") == "answer_completed"]
    errors = [item for item in events if item.get("event") == "error"]
    final = completed[-1]["data"] if completed else {}
    usage_events = [item for item in events if item.get("event") == "usage"]
    usage = usage_events[-1]["data"] if usage_events else {}
    answer = str(final.get("answer") or "")
    tool_names = list(final.get("tool_names") or [])
    native_tool_calls = int(final.get("native_tool_calls") or 0)
    orchestrated_tool_calls = int(final.get("orchestrated_tool_calls") or usage.get("orchestrated_tool_calls") or 0)
    tool_calls_total = int(
        final.get("tool_calls_total")
        or usage.get("tool_calls_total")
        or native_tool_calls
    )
    return {
        "http_status": status,
        "event_count": len(events),
        "event_types": [str(item.get("event")) for item in events],
        "completed": bool(completed),
        "errors": errors,
        "answer_sha256": sha256_text(answer) if answer else None,
        "answer_characters": len(answer),
        "answer_preview": answer[:500],
        "native_tool_calls": native_tool_calls,
        "orchestrated_tool_calls": orchestrated_tool_calls,
        "tool_calls_total": tool_calls_total,
        "tool_names": tool_names,
        "repair_rounds": int(final.get("repair_rounds") or usage.get("repair_rounds") or 0),
        "response_audit_passed": bool(
            ((final.get("response_audit") or {}).get("after") or {}).get("passed")
        ),
        "provider": final.get("provider") or usage.get("provider"),
        "model": final.get("model") or usage.get("model"),
        "input_tokens": int(usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
        "required_terms": {
            "target_run": TARGET_RUN in answer,
            "no_kaggle_boundary": bool(re.search(r"Kaggle|提交|官方|奖牌|私有 grader|grader", answer, re.I)),
            "next_step": bool(re.search(r"下一步|建议|计划|改进", answer)),
            "evidence": bool(re.search(r"证据|报告|哈希|artifact|验证|复核", answer, re.I)),
        },
        "_answer_private": answer,
        "seconds": round(time.monotonic() - started, 3),
    }


def public_assistant_result(result: dict[str, Any]) -> dict[str, Any]:
    public = dict(result)
    public.pop("_answer_private", None)
    return public


def first_turn_passed(assistant: dict[str, Any]) -> tuple[bool, list[str]]:
    answer = str(assistant.get("_answer_private") or "")
    required_terms = assistant.get("required_terms") or {}
    checks = {
        "assistant_http_200": assistant.get("http_status") == 200,
        "answer_completed": assistant.get("completed") is True,
        "no_errors": not assistant.get("errors"),
        "real_tool_call": int(assistant.get("tool_calls_total") or 0) >= 1,
        "tool_names": bool(assistant.get("tool_names")),
        "substantive_answer": int(assistant.get("answer_characters") or 0) >= 500,
        "core_roc_auc": "0.9225357247684676" in answer,
        "core_pr_auc": "0.23970933285011597" in answer,
        "core_brier": "0.29524735217259324" in answer,
        **{f"term:{key}": bool(value) for key, value in required_terms.items()},
    }
    return all(checks.values()), [name for name, ok in checks.items() if not ok]


def followup_turn_passed(assistant: dict[str, Any]) -> tuple[bool, list[str]]:
    answer = str(assistant.get("_answer_private") or "")
    checks = {
        "followup_http_200": assistant.get("http_status") == 200,
        "followup_answer_completed": assistant.get("completed") is True,
        "followup_no_errors": not assistant.get("errors"),
        "followup_substantive_answer": int(assistant.get("answer_characters") or 0) >= 300,
        "followup_prior_context": bool(re.search(r"SIIM|黑色素瘤|上次|刚才|上一轮|这个实验", answer, re.I)),
        "followup_three_actions": bool(re.search(r"1[\\.、)]|①|第一|三个|3 个|三步|三件", answer)),
        "followup_no_training_boundary": bool(re.search(r"不训练|不要训练|不启动训练|不提交|不要提交|Kaggle", answer, re.I)),
        "followup_actionable": bool(re.search(r"动作|步骤|先|接着|最后|演示|复核|证据|报告", answer)),
    }
    return all(checks.values()), [name for name, ok in checks.items() if not ok]


def literature_turn_passed(assistant: dict[str, Any]) -> tuple[bool, list[str]]:
    answer = str(assistant.get("_answer_private") or "")
    checks = {
        "literature_http_200": assistant.get("http_status") == 200,
        "literature_answer_completed": assistant.get("completed") is True,
        "literature_no_errors": not assistant.get("errors"),
        "literature_substantive_answer": int(assistant.get("answer_characters") or 0) >= 400,
        "literature_real_tool_call": int(assistant.get("tool_calls_total") or 0) >= 1,
        "literature_verified_context": "verified_context" in list(assistant.get("tool_names") or []),
        "literature_reviewed_doi_1": REVIEWED_LITERATURE_DOIS[0].lower() in answer.lower(),
        "literature_reviewed_doi_2": REVIEWED_LITERATURE_DOIS[1].lower() in answer.lower(),
        "literature_title_or_year": bool(re.search(r"标题|2021|2024|Effect of patient|Analysis of the ISIC", answer, re.I)),
        "literature_novice_explanation": bool(re.search(r"小白|简单|也就是|大白话|意思是|通俗", answer)),
        "literature_source_distinction": bool(re.search(r"文献|论文", answer) and re.search(r"当前|本次|Run|实验", answer, re.I)),
    }
    assistant["literature_checks"] = checks
    return all(checks.values()), [name for name, ok in checks.items() if not ok]


def connection_turn_passed(assistant: dict[str, Any]) -> tuple[bool, list[str]]:
    answer = str(assistant.get("_answer_private") or "")
    live_verified = bool(re.search(
        r"job_container_verified\s*[:=]\s*true|"
        r"目标?容器.{0,20}(?:身份|Host|GPU|root|只读|采样).{0,30}(?:通过|成功|匹配)|"
        r"(?:5|五)\s*次.{0,16}只读.{0,16}(?:采样|检查).{0,20}(?:通过|成功)",
        answer,
        re.I,
    ))
    live_blocked = bool(re.search(
        r"job_container_verified\s*[:=]\s*false|"
        r"live_status\s*[:=]\s*(?:job_container_)?(?:channel_closed|identity_failed|blocked)|"
        r"(?:目标)?容器.{0,30}(?:未通过|不可用|关闭|阻断|失败)|"
        r"(?:没有|尚未|还没有).{0,20}(?:通过|完成).{0,20}(?:容器|实时连接)",
        answer,
        re.I,
    ))
    checks = {
        "connection_http_200": assistant.get("http_status") == 200,
        "connection_answer_completed": assistant.get("completed") is True,
        "connection_no_errors": not assistant.get("errors"),
        "connection_substantive_answer": int(assistant.get("answer_characters") or 0) >= 180,
        "connection_required_model_provider": assistant.get("provider") == REQUIRED_PROVIDER and assistant.get("model") == REQUIRED_MODEL,
        "connection_hpc_tool": "hpc_connection_status" in list(assistant.get("tool_names") or []),
        "connection_job90673": "90673" in answer or "job90673" in answer.lower(),
        "connection_truthful_live_status": live_verified or live_blocked,
        "connection_no_training": bool(re.search(r"未启动训练|不训练|没有启动训练|training_started\s*[:=]\s*false", answer, re.I)),
        "connection_no_kaggle_submit": bool(re.search(
            r"未提交\s*Kaggle|不提交\s*Kaggle|没有提交\s*Kaggle|"
            r"未执行\s*Kaggle\s*提交|没有执行\s*Kaggle\s*提交|"
            r"kaggle_submissions\s*[:=]\s*0",
            answer,
            re.I,
        )),
        "connection_no_private_grader": bool(re.search(r"没有调用\s*private grader|未调用\s*private grader|不调用\s*private grader|private grader|grader_calls\s*[:=]\s*0", answer, re.I)),
        "connection_actionable": bool(re.search(r"下一步|建议|结论|证据|profile|网关|gateway|SOCKS|SSH", answer, re.I)),
    }
    assistant["connection_checks"] = checks
    return all(checks.values()), [name for name, ok in checks.items() if not ok]


def ux_budget_result(
    first: dict[str, Any],
    followup: dict[str, Any],
    literature: dict[str, Any],
    connection: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "all_turns_required_provider": all(
            str(item.get("provider") or "") == REQUIRED_PROVIDER
            for item in (first, followup, literature, connection)
        ),
        "all_turns_required_model": all(
            str(item.get("model") or "") == REQUIRED_MODEL
            for item in (first, followup, literature, connection)
        ),
        # The release quality gate already uses a 90-second p95 ceiling.  Keep
        # the live smoke aligned with that user-visible contract; provider cold
        # starts make a 75-second single-sample gate noisy without improving the
        # answer-quality guarantee.
        "first_turn_seconds_le_90": float(first.get("seconds") or 9999) <= 90.0,
        "first_turn_chars_1200_3200": 1200 <= int(first.get("answer_characters") or 0) <= 3200,
        "first_turn_input_tokens_1_12000": 1 <= int(first.get("input_tokens") or 0) <= 12000,
        # output_tokens is cumulative across the native tool-request turn and
        # the final synthesis turn.  The 1200-3200 character gate above remains
        # the direct user-facing verbosity limit.
        "first_turn_cumulative_output_tokens_1_2000": 1 <= int(first.get("output_tokens") or 0) <= 2000,
        "followup_seconds_le_45": float(followup.get("seconds") or 9999) <= 45.0,
        "followup_chars_300_1600": 300 <= int(followup.get("answer_characters") or 0) <= 1600,
        "followup_input_tokens_1_17000": 1 <= int(followup.get("input_tokens") or 0) <= 17000,
        "followup_output_tokens_1_1200": 1 <= int(followup.get("output_tokens") or 0) <= 1200,
        "literature_seconds_le_75": float(literature.get("seconds") or 9999) <= 75.0,
        "literature_chars_400_1600": 400 <= int(literature.get("answer_characters") or 0) <= 1600,
        "literature_input_tokens_1_17000": 1 <= int(literature.get("input_tokens") or 0) <= 17000,
        "literature_output_tokens_1_1200": 1 <= int(literature.get("output_tokens") or 0) <= 1200,
        "connection_seconds_le_90": float(connection.get("seconds") or 9999) <= 90.0,
        "connection_chars_180_1400": 180 <= int(connection.get("answer_characters") or 0) <= 1400,
        "connection_input_tokens_1_14000": 1 <= int(connection.get("input_tokens") or 0) <= 14000,
        "connection_output_tokens_1_1200": 1 <= int(connection.get("output_tokens") or 0) <= 1200,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "token_accounting": (
            "output_tokens are cumulative provider usage across tool selection and final synthesis; "
            "answer_characters measure the user-visible response"
        ),
        "turns": {
            "first": {
                "seconds": first.get("seconds"),
                "characters": first.get("answer_characters"),
                "input_tokens": first.get("input_tokens"),
                "output_tokens": first.get("output_tokens"),
            },
            "followup": {
                "seconds": followup.get("seconds"),
                "characters": followup.get("answer_characters"),
                "input_tokens": followup.get("input_tokens"),
                "output_tokens": followup.get("output_tokens"),
            },
            "literature": {
                "seconds": literature.get("seconds"),
                "characters": literature.get("answer_characters"),
                "input_tokens": literature.get("input_tokens"),
                "output_tokens": literature.get("output_tokens"),
            },
            "connection": {
                "seconds": connection.get("seconds"),
                "characters": connection.get("answer_characters"),
                "input_tokens": connection.get("input_tokens"),
                "output_tokens": connection.get("output_tokens"),
            },
        },
    }


def governance_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Compare governance state before/after the live browser-equivalent smoke."""

    hashes_before = before.get("tracked_hashes") or {}
    hashes_after = after.get("tracked_hashes") or {}
    changed_hashes = sorted(
        key for key in set(hashes_before) | set(hashes_after)
        if hashes_before.get(key) != hashes_after.get(key)
    )
    run_dirs_before = list(before.get("siim_run_directories") or [])
    run_dirs_after = list(after.get("siim_run_directories") or [])
    run_dirs_changed = run_dirs_before != run_dirs_after
    invariants = {
        "run_directories_unchanged": not run_dirs_changed,
        "tracked_hashes_unchanged": not changed_hashes,
        "grader_execution_count_exactly_one": after.get("grader_execution_count") == 1,
        "grader_outcome_failed_closed": after.get("grader_outcome") == "failed_closed",
        "grader_score_null": after.get("grader_score") is None,
        "official_submission_not_executed": after.get("official_submission_executed") is False,
    }
    return {
        "passed": all(invariants.values()),
        "invariants": invariants,
        "changed_hashes": changed_hashes,
        "run_dirs_changed": run_dirs_changed,
        "before": before,
        "after": after,
    }


def port_open(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/healthz", timeout=3) as response:
            return response.status == 200
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=19088)
    parser.add_argument("--runtime-port", type=int, default=19089)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--assistant-timeout", type=float, default=180.0)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--followup-prompt", default=DEFAULT_FOLLOWUP_PROMPT)
    parser.add_argument("--literature-prompt", default=DEFAULT_LITERATURE_PROMPT)
    parser.add_argument("--connection-prompt", default=DEFAULT_CONNECTION_PROMPT)
    parser.add_argument("--selected-task", default="siim-isic-melanoma-classification")
    parser.add_argument("--output", type=Path, default=ROOT / "workspace/evaluation/live_assistant_demo_smoke_current.json")
    args = parser.parse_args()

    if args.port in {8088, 8765, 65068, 17897} or args.runtime_port in {8088, 8765, 65068, 17897}:
        raise SystemExit("temporary smoke ports must not be protected production ports")
    if args.port == args.runtime_port:
        raise SystemExit("dashboard and runtime ports must differ")

    report: dict[str, Any] = {
        "schema": "evomind.live_assistant_demo_smoke.v1",
        "generated_at": now_iso(),
        "status": "failed",
        "port": args.port,
        "runtime_port": args.runtime_port,
        "prompt_sha256": sha256_text(args.prompt),
        "followup_prompt_sha256": sha256_text(args.followup_prompt),
        "literature_prompt_sha256": sha256_text(args.literature_prompt),
        "connection_prompt_sha256": sha256_text(args.connection_prompt),
        "selected_task": args.selected_task,
        "token_values_recorded": False,
        "session_values_recorded": False,
        "claim_scope": "isolated local browser-equivalent smoke, not official Kaggle or MLE-Bench proof",
    }

    stopped: ProcResult | None = None
    try:
        if port_open(args.port) or port_open(args.runtime_port):
            raise RuntimeError("temporary smoke ports are already occupied")

        governance_before = governance_snapshot(ROOT)
        report["governance_before"] = governance_before

        start_result = run_manager(["start", "--no-auto-build"], port=args.port, runtime_port=args.runtime_port, timeout=args.timeout)
        report["start"] = start_result.to_dict()
        if not start_result.ok:
            raise RuntimeError("temporary dashboard failed to start")

        base_url, token = read_bootstrap_token(args.port)
        if base_url != f"http://127.0.0.1:{args.port}/?page=assistant":
            raise RuntimeError(f"unexpected bootstrap base URL: {base_url}")
        report["bootstrap"] = {
            "file": str(bootstrap_path(args.port)),
            "token_sha256": sha256_text(token),
            "url_base": base_url,
        }

        cookie_jar = http.cookiejar.CookieJar()
        opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))
        origin = f"http://127.0.0.1:{args.port}"
        status, auth = request_json(
            opener,
            f"{origin}/api/session/bootstrap",
            method="POST",
            payload={"token": token},
            headers={"Origin": origin, "Content-Type": "application/json"},
            timeout=15,
        )
        csrf = str(auth.get("csrf_token") or "")
        report["auth"] = {
            "http_status": status,
            "ok": auth.get("ok") is True,
            "csrf_present": bool(csrf),
            "cookie_count": len(cookie_jar),
        }
        if status != 200 or auth.get("ok") is not True or not csrf or len(cookie_jar) < 1:
            raise RuntimeError("bootstrap auth failed")

        status, session = request_json(opener, f"{origin}/api/session/status", timeout=15)
        report["session_status"] = {
            "http_status": status,
            "ok": session.get("ok") is True,
            "authenticated": session.get("authenticated") is True,
            "csrf_present": bool(session.get("csrf_token")),
        }
        if status != 200 or session.get("authenticated") is not True:
            raise RuntimeError("session status failed")

        smoke_session_id = f"live_demo_smoke_{int(time.time())}"
        assistant = post_assistant_stream(
            opener,
            origin,
            csrf,
            args.prompt,
            args.selected_task,
            timeout=args.assistant_timeout,
            session_id=smoke_session_id,
        )
        report["assistant"] = public_assistant_result(assistant)
        first_ok, first_failures = first_turn_passed(assistant)

        followup_history = [
            {"role": "user", "content": args.prompt},
            {"role": "assistant", "content": str(assistant.get("_answer_private") or "")[:8000]},
        ]
        followup = post_assistant_stream(
            opener,
            origin,
            csrf,
            args.followup_prompt,
            args.selected_task,
            timeout=args.assistant_timeout,
            session_id=smoke_session_id,
            history=followup_history,
        )
        report["assistant_followup"] = public_assistant_result(followup)
        followup_ok, followup_failures = followup_turn_passed(followup)

        literature_history = [
            *followup_history,
            {"role": "user", "content": args.followup_prompt},
            {"role": "assistant", "content": str(followup.get("_answer_private") or "")[:8000]},
        ]
        literature = post_assistant_stream(
            opener,
            origin,
            csrf,
            args.literature_prompt,
            args.selected_task,
            timeout=args.assistant_timeout,
            session_id=smoke_session_id,
            history=literature_history,
        )
        literature_ok, literature_failures = literature_turn_passed(literature)
        report["assistant_literature"] = public_assistant_result(literature)

        connection_history = [
            {"role": "user", "content": args.prompt},
            {"role": "assistant", "content": str(assistant.get("_answer_private") or "")[:4000]},
            {"role": "user", "content": args.followup_prompt},
            {"role": "assistant", "content": str(followup.get("_answer_private") or "")[:4000]},
        ]
        connection = post_assistant_stream(
            opener,
            origin,
            csrf,
            args.connection_prompt,
            args.selected_task,
            timeout=args.assistant_timeout,
            session_id=smoke_session_id,
            history=connection_history,
        )
        connection_ok, connection_failures = connection_turn_passed(connection)
        report["assistant_connection"] = public_assistant_result(connection)

        ux_budget = ux_budget_result(assistant, followup, literature, connection)
        report["ux_budget"] = ux_budget

        governance_after = governance_snapshot(ROOT)
        governance = governance_delta(governance_before, governance_after)
        report["governance"] = governance

        passed = (
            first_ok
            and followup_ok
            and literature_ok
            and connection_ok
            and ux_budget.get("passed") is True
            and governance.get("passed") is True
        )
        report["conversation"] = {
            "turns": 4,
            "same_session_id": True,
            "history_sent_to_followup": True,
            "history_sent_to_literature": True,
            "history_sent_to_connection": True,
            "history_user_chars": len(args.prompt),
            "history_assistant_chars": len(str(assistant.get("_answer_private") or "")[:8000]),
            "literature_history_messages": len(literature_history),
            "connection_history_messages": len(connection_history),
            "total_native_tool_calls": (
                int(assistant.get("native_tool_calls") or 0)
                + int(followup.get("native_tool_calls") or 0)
                + int(literature.get("native_tool_calls") or 0)
                + int(connection.get("native_tool_calls") or 0)
            ),
            "total_orchestrated_tool_calls": (
                int(assistant.get("orchestrated_tool_calls") or 0)
                + int(followup.get("orchestrated_tool_calls") or 0)
                + int(literature.get("orchestrated_tool_calls") or 0)
                + int(connection.get("orchestrated_tool_calls") or 0)
            ),
            "total_real_tool_calls": (
                int(assistant.get("tool_calls_total") or 0)
                + int(followup.get("tool_calls_total") or 0)
                + int(literature.get("tool_calls_total") or 0)
                + int(connection.get("tool_calls_total") or 0)
            ),
            "first_turn_passed": first_ok,
            "followup_turn_passed": followup_ok,
            "literature_turn_passed": literature_ok,
            "connection_turn_passed": connection_ok,
        }
        report["status"] = "passed" if passed else "failed"
        report["failed_checks"] = [] if passed else first_failures + followup_failures + literature_failures + connection_failures + [
            f"ux_budget:{name}"
            for name, ok in (ux_budget.get("checks") or {}).items()
            if not ok
        ] + (
            [] if governance.get("passed") is True else [
                f"governance:{name}"
                for name, ok in (governance.get("invariants") or {}).items()
                if not ok
            ]
        )
    except Exception as exc:  # noqa: BLE001 - report exact smoke failure
        report.setdefault("failed_checks", []).append("exception")
        report["exception"] = f"{type(exc).__name__}: {exc}"
    finally:
        stopped = run_manager(["stop"], port=args.port, runtime_port=args.runtime_port, timeout=30)
        report["stop"] = stopped.to_dict()
        report["temporary_ports_released"] = not port_open(args.port) and not port_open(args.runtime_port)
        if report.get("status") == "passed" and not report["temporary_ports_released"]:
            report["status"] = "failed"
            report.setdefault("failed_checks", []).append("temporary_ports_released")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
