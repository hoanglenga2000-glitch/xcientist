"""NDJSON bridge between the EvoMind web assistant and the Python runtime."""
from __future__ import annotations

import contextlib
import io
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .assistant_context import AssistantContextPacket, build_assistant_context
from .config import active_root, inject_engine_env, load_config
from .kaggle_conversation import ConversationAgent
from .kaggle_intent import CAPABILITY, CHAT, EXECUTION, GREETING, PLANNING, TOOL_QUERY, classify
from .kaggle_session import SessionState
from .terminal_agent import TerminalAgent
from .user_request import UserRequest

SIIM_TASK_ID = "siim-isic-melanoma-classification"


def _clean_history(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    cleaned: list[dict[str, str]] = []
    for item in value[-20:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "")
        content = str(item.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            cleaned.append({"role": role, "content": content[:8000]})
    return cleaned


def _bind_requested_task(session: SessionState, requested_task: str, root: Path) -> None:
    """Bind the URL-selected task without retaining metadata from an older task.

    ``SessionState.from_root`` restores the last terminal task before the web
    request can apply its explicit ``selected_task``.  Replacing only the slug
    leaves ``task_brief`` and the recent-run digest pointing at the old task,
    which can make the LLM combine SIIM evidence with unrelated tabular schema.
    Refresh the derived fields after the explicit selection; an unregistered
    but governed Run id intentionally keeps an empty brief instead of stale data.
    """

    task = str(requested_task or "").strip()[:160]
    if not task:
        return
    session.selected_task = task
    session.task_brief = ""
    try:
        session.refresh_task_brief(root)
    except (AttributeError, OSError, ValueError):
        session.task_brief = ""
    try:
        session.refresh_recent_run(root)
    except (AttributeError, OSError, ValueError):
        session.recent_run_id = ""
        session.recent_events_path = ""
        session.recent_best_cv = None


def _should_start_siim_execution(intent: Any) -> bool:
    """Only an affirmative execution request may enter the governed SIIM run.

    Planning, explanation, literature, and "do not train" requests must remain
    LLM-first read-only turns.  They may discuss the existing Run but must not be
    collapsed into the legacy Multi-Agent Run status template.
    """

    request = getattr(intent, "request", None)
    return bool(
        getattr(intent, "kind", "") == EXECUTION
        and request is not None
        and request.task_type == "image_classification"
        and request.dataset == SIIM_TASK_ID
        and request.requests_execution
        and "no_training" not in request.negative_constraints
    )


class EventWriter:
    def __init__(self, session_id: str, runtime=None) -> None:
        self.session_id = session_id
        self.seq = 0
        self.runtime = runtime

    def emit(self, event_type: str, **payload: Any) -> None:
        self.seq += 1
        body = {
            "type": event_type,
            "seq": self.seq,
            "session_id": self.session_id,
            **payload,
        }
        line = json.dumps(body, ensure_ascii=False, separators=(",", ":")) + "\n"
        try:
            sys.stdout.buffer.write(line.encode("utf-8"))
            sys.stdout.buffer.flush()
        except AttributeError:  # pragma: no cover - StringIO/capsys compatibility
            sys.stdout.write(line)
            sys.stdout.flush()
        if self.runtime is not None:
            self.runtime.store.append_event(self.session_id, f"web.{event_type}", body)
            if event_type == "answer_completed":
                self.runtime.store.add_turn(self.session_id, "assistant", str(payload.get("answer") or ""))


def _direct_chat(
    writer: EventWriter,
    prompt: str,
    session: SessionState,
    history: list[dict[str, str]],
    context: AssistantContextPacket | None = None,
) -> int:
    writer.emit("route", route="agent", label="真实 LLM Agent", detail="模型先理解问题，再自主选择只读工具和证据")
    writer.emit("thinking_status", status="running", label="LLM 正在分析问题", detail="按需读取当前 Run 与工具证据")
    agent = ConversationAgent()

    def on_tool_event(phase: str, tool: str, ok: bool) -> None:
        if phase == "started":
            writer.emit("tool_started", tool=tool, label=f"Agent 正在调用 {tool}")
        else:
            writer.emit(
                "tool_completed",
                tool=tool,
                status="completed" if ok else "failed",
                label=f"{tool} 已返回可审计结果" if ok else f"{tool} 调用失败",
            )

    answer = agent.agent_reply(
        prompt,
        session,
        history=history,
        context=context,
        on_tool_event=on_tool_event,
    ).strip()
    evidence = dict(agent._last_llm_execution)
    if not answer:
        from .kaggle_conversation import (
            _execute_terminal_tool,
            _forced_tool_hints,
            _precheck_fallback_answer,
        )

        forced_results: list[str] = []
        tool_names = list(evidence.get("tool_names") or [])
        for tool_name in _forced_tool_hints(prompt):
            if tool_name in tool_names:
                continue
            on_tool_event("started", tool_name, True)
            try:
                tool_result = _execute_terminal_tool(tool_name, session)
                tool_ok = "status=FAILED" not in tool_result
            except Exception as exc:
                tool_result = json.dumps(
                    {
                        "ok": False,
                        "tool": tool_name,
                        "message": f"{type(exc).__name__}: {exc}",
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                tool_ok = False
            on_tool_event("completed", tool_name, tool_ok)
            forced_results.append(tool_result)
            tool_names.append(tool_name)
            evidence["orchestrated_tool_calls"] = int(
                evidence.get("orchestrated_tool_calls") or 0
            ) + 1
            evidence["tool_calls_total"] = int(
                evidence.get("tool_calls_total") or 0
            ) + 1
        evidence["tool_names"] = tool_names
        if forced_results:
            answer = _precheck_fallback_answer(
                prompt,
                "\n\n".join(forced_results),
                reason=str(evidence.get("status") or "model_transport_unavailable"),
            ).strip()
            if answer:
                evidence["status"] = "completed_with_precheck_fallback"
    provider = str(evidence.get("provider") or "")
    model = str(evidence.get("model") or "")
    if provider or model:
        writer.emit(
            "model",
            provider=provider,
            model=model,
            label=f"{model or 'model'} via {provider or 'provider'}",
        )
    if not answer:
        status = str(evidence.get("status") or "unknown")
        answer = f"真实模型网关本轮未返回答案（状态：{status}）。问题已保留，请重试一次。"
    writer.emit("answer_delta", delta=answer)
    writer.emit(
        "usage",
        provider=provider,
        model=model,
        input_tokens=int(evidence.get("input_tokens") or 0),
        output_tokens=int(evidence.get("output_tokens") or 0),
        native_tool_calls=int(evidence.get("native_tool_calls") or 0),
        orchestrated_tool_calls=int(evidence.get("orchestrated_tool_calls") or 0),
        tool_calls_total=int(
            evidence.get("tool_calls_total")
            or evidence.get("native_tool_calls")
            or 0
        ),
        repair_rounds=int(evidence.get("repair_rounds") or 0),
    )
    writer.emit(
        "thinking_status",
        status=(
            "completed"
            if evidence.get("status")
            in {
                "completed",
                "completed_after_wrap",
                "completed_with_precheck_fallback",
            }
            else "blocked"
        ),
        label="Agent 回答已完成" if answer else "模型网关未返回答案",
        detail=(
            "真实工具调用 "
            f"{int(evidence.get('tool_calls_total') or evidence.get('native_tool_calls') or 0)} 次"
        ),
    )
    writer.emit(
        "answer_completed",
        answer=answer,
        provider=provider,
        model=model,
        route="agent",
        native_tool_calls=int(evidence.get("native_tool_calls") or 0),
        orchestrated_tool_calls=int(evidence.get("orchestrated_tool_calls") or 0),
        tool_calls_total=int(
            evidence.get("tool_calls_total")
            or evidence.get("native_tool_calls")
            or 0
        ),
        tool_names=list(evidence.get("tool_names") or []),
        response_audit=dict(evidence.get("response_audit") or {}),
        repair_rounds=int(evidence.get("repair_rounds") or 0),
        llm_status=str(evidence.get("status") or "unknown"),
        stop_reason=str(evidence.get("stop_reason") or ""),
    )
    return 0


def _workflow_turn(
    writer: EventWriter,
    prompt: str,
    session: SessionState,
    root: Path,
    intent_kind: str,
) -> int:
    if intent_kind == EXECUTION:
        route = "execution"
        label = "受控执行"
        activity = "正在检查执行门禁与资源边界"
    elif intent_kind == PLANNING:
        route = "research"
        label = "科研规划"
        activity = "Supervisor 正在规划任务与证据需求"
    else:
        route = "tool"
        label = "只读查询"
        activity = "正在读取对应状态与证据"

    writer.emit("route", route=route, label=label, detail="使用 EvoMind 受控工作流")
    writer.emit("thinking_status", status="running", label=activity, detail="详细轨迹保存在审计账本")
    writer.emit("tool_started", tool=label, label=activity)
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured):
            agent = TerminalAgent(colour=False)
            if intent_kind == PLANNING:
                result = agent.handle_scientist_turn(prompt, session, root, max_tools=4)
            else:
                result = agent.handle(prompt, session, root)
        writer.emit(
            "tool_completed",
            tool=label,
            status="blocked" if result.blocked else "completed",
            label="工作流已返回结果",
        )
        answer = str(result.summary or "").strip()
        if not answer:
            answer = captured.getvalue().strip()
        if not answer:
            answer = (
                "受控工作流已完成，但本轮没有生成新的文本或证据。"
                if result.blocked
                else "只读检查已完成，当前没有新增可显示内容。"
            )
        writer.emit("answer_delta", delta=answer)
        writer.emit(
            "thinking_status",
            status="blocked" if result.blocked else "completed",
            label="需要继续处理" if result.blocked else "处理完成",
            detail="可在高级控制中查看完整审计轨迹",
        )
        writer.emit(
            "answer_completed",
            answer=answer,
            route=route,
            blocked=result.blocked,
        )
        return result.rc
    except Exception as exc:  # noqa: BLE001
        writer.emit("error", code=type(exc).__name__, message="本轮处理失败，状态已保留，可直接重试。")
        return 1


def _current_siim_run(root: Path) -> dict[str, Any] | None:
    pointer_path = root / "workspace" / "current_run.json"
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(pointer, dict) or pointer.get("task_id") != SIIM_TASK_ID:
        return None
    run_id = str(pointer.get("run_id") or "")
    if not run_id or not all(char.isalnum() or char in "_-" for char in run_id):
        return None
    run_dir = (root / "workspace" / "evomind_runs" / run_id).resolve()
    try:
        run_dir.relative_to(root.resolve())
        run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(run, dict) or run.get("run_id") != run_id:
        return None
    return {"run_id": run_id, "run_dir": run_dir, "run": run}


def _siim_research_turn(
    writer: EventWriter,
    prompt: str,
    root: Path,
    request: UserRequest,
) -> int:
    """Create or idempotently attach the assistant to one governed SIIM Run."""
    writer.emit("route", route="image_classification", label="医学影像科研工作流", detail="同一 Run 完成训练、复核与交付")
    writer.emit("thinking_status", status="running", label="正在确认数据、A800、验证和交付要求", detail="公开榜单提交保持关闭")
    current = _current_siim_run(root)
    reused = current is not None
    if current is None:
        run_id = (
            "evomind_siim_isic_hpc_"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "_"
            + uuid.uuid4().hex[:6]
        )
        writer.emit("tool_started", tool="Multi-Agent Run", label="正在建立九节点研究任务")
        from research_os.agent.siim_hpc_workflow import run_siim_hpc_research

        run = run_siim_hpc_research(root, request, run_id=run_id)
        status = str(run.status)
        current = {"run_id": run_id, "run_dir": root / "workspace" / "evomind_runs" / run_id, "run": run.to_dict()}
        writer.emit("tool_completed", tool="Multi-Agent Run", status="completed", label="同一研究 Run 已建立")
    else:
        run_id = str(current["run_id"])
        status = str(current["run"].get("status") or "unknown")
        writer.emit("tool_started", tool="Multi-Agent Run", label="正在核对现有同任务运行")
        writer.emit("tool_completed", tool="Multi-Agent Run", status="completed", label="已关联同一研究 Run")

    tasks = current["run"].get("tasks") if isinstance(current["run"], dict) else {}
    task_records = tasks if isinstance(tasks, dict) else {}
    completed = sum(
        1
        for item in task_records.values()
        if isinstance(item, dict) and item.get("status") == "completed"
    )
    total = len(task_records) or 9
    if status == "completed":
        state_text = "研究、独立复核和四项交付已经完成"
    elif status == "needs_continuation":
        state_text = "当前正在等待安全资源或下一段真实证据，之后会沿同一 Run 继续"
    else:
        state_text = "研究任务已经进入受控执行"
    answer = (
        f"已理解这项医学影像研究需求，并{'关联' if reused else '建立'}同一运行 {run_id}。"
        f"系统会检查 33,129 个文件、患者与重复图像泄漏，比较五种预处理方案，"
        f"再完成 A800 训练、独立复核、声明审计和四个下载文件；{state_text}。"
        f"当前进度 {completed}/{total}，未提交公开榜单。"
    )
    writer.emit(
        "research_run",
        label="同一研究 Run 已关联" if reused else "同一研究 Run 已建立",
        run_id=run_id,
        run_status=status,
        completed_tasks=completed,
        total_tasks=total,
        snapshot_url=f"/api/multi-agent/runs/{run_id}",
        reused=reused,
    )
    writer.emit("answer_delta", delta=answer)
    writer.emit("thinking_status", status="completed", label="需求与研究运行已绑定", detail="可在高级控制中查看实时账本")
    writer.emit(
        "answer_completed",
        answer=answer,
        route="image_classification",
        run_id=run_id,
        run_status=status,
        reused=reused,
    )
    return 0


def _read_payload_from_stdin() -> dict[str, Any]:
    """Read the web bridge payload as UTF-8 bytes.

    On Windows the Python stdio text encoding can follow the parent console code
    page instead of the Node/web bridge's UTF-8 bytes.  Using ``sys.stdin.read``
    can therefore introduce surrogate escapes for Chinese prompts, and those
    surrogates later crash SQLite when a session is created.  The bridge contract
    is JSON over UTF-8 bytes, with a conservative local-codepage fallback for
    manual CLI smoke tests.
    """

    try:
        raw = sys.stdin.buffer.read()
    except Exception:
        raw = b""
    if not raw:
        return {}
    text = ""
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if not text:
        text = raw.decode("utf-8", errors="replace")
    try:
        payload = json.loads(text or "{}")
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        return {}


def main() -> int:
    payload = _read_payload_from_stdin()
    prompt = str(payload.get("prompt") or "").strip()[:20000]
    session_id = str(payload.get("session_id") or f"chat_{uuid.uuid4().hex[:12]}")[:80]
    if not prompt:
        writer = EventWriter(session_id)
        writer.emit("error", code="invalid_prompt", message="请输入问题后再发送。")
        return 2

    root = active_root()
    from evomind_runtime import AgentRuntime
    runtime = AgentRuntime(root)
    if runtime.store.get_session(session_id) is None:
        runtime.create_session(objective=prompt, title="Web assistant", session_id=session_id)
    runtime.store.add_turn(session_id, "user", prompt)
    writer = EventWriter(session_id, runtime)
    cfg = load_config(root)
    inject_engine_env(cfg)
    session = SessionState.from_root(root, cfg=cfg)
    requested_task = str(payload.get("selected_task") or "").strip()
    _bind_requested_task(session, requested_task, root)
    history = _clean_history(payload.get("history"))
    intent = classify(prompt)
    context = build_assistant_context(root)

    writer.emit("session", workspace="active", ready=session.llm_ready)
    writer.emit("context", **context.public_status())
    if _should_start_siim_execution(intent):
        return _siim_research_turn(writer, prompt, root, intent.request)
    if intent.kind == EXECUTION:
        return _workflow_turn(writer, prompt, session, root, intent.kind)

    # Every non-execution natural-language turn is LLM-first.  The model can
    # call verified read-only tools itself; keyword adapters no longer answer
    # status/report/tool questions with fixed templates before the model sees them.
    return _direct_chat(writer, prompt, session, history, context=context)


if __name__ == "__main__":
    raise SystemExit(main())


