from __future__ import annotations

import hashlib
import html
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

DEC2021_TASK_ID = "tabular-playground-series-dec-2021"
RUN_SCHEMA = "evomind.assistant.training_run.v1"
PUBLIC_RUN_ROOT = Path("workspace") / "assistant_runs"
DEFAULT_TIMEOUT_SECONDS = 240


class AssistantTrainingError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"evomind_dec2021_demo_{stamp}_{uuid.uuid4().hex[:6]}"


def run_state_path(root: Path, run_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]{8,180}", run_id):
        raise ValueError("invalid run id")
    base = (root / PUBLIC_RUN_ROOT).resolve()
    target = (base / run_id / "run.json").resolve()
    target.relative_to(base)
    return target


def load_public_run(root: Path, run_id: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(run_state_path(root, run_id).read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) and payload.get("run_id") == run_id else None


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _safe_log_tail(value: Any) -> list[str]:
    lines = str(value or "").replace("\r", "").splitlines()
    safe: list[str] = []
    sensitive = re.compile(r"(?i)(password|token|cookie|authorization|ssh-rsa|ssh-ed25519|gpu-[0-9a-f-]{20,}|[0-9a-f]{32,})")
    for line in lines[-80:]:
        cleaned = line.strip()
        if cleaned and not sensitive.search(cleaned):
            safe.append(cleaned[:300])
    return safe[-12:]


def _progress_from_log(lines: list[str]) -> dict[str, Any]:
    iteration = None
    accuracy = None
    for line in lines:
        match = re.search(r"(?:^|\s)(\d+):.*?(?:test|validation|accuracy|Accuracy)[^0-9]*([0-9]+(?:\.[0-9]+)?)", line)
        if match:
            iteration = int(match.group(1))
            accuracy = float(match.group(2))
            if accuracy > 1.0:
                accuracy = None
    return {
        "iteration": iteration,
        "iterations_total": 380,
        "validation_accuracy": accuracy,
        "log_lines": lines,
    }


def _remote_candidate_complete(status: dict[str, Any]) -> bool:
    """Return true when a stopped candidate-only worker committed all outputs."""

    manifest = status.get("manifest") or {}
    checkpoint = status.get("checkpoint") or {}
    summary = status.get("summary") or {}
    completed = checkpoint.get("completed") or {}
    requested = checkpoint.get("requested") or []
    remaining = checkpoint.get("remaining")
    return bool(
        status.get("process") == "stopped"
        and manifest.get("status") == "candidate_complete"
        and summary.get("status") == "candidate_complete"
        and isinstance(remaining, list)
        and not remaining
        and requested
        and all(task in completed for task in requested)
    )


def _find_metric(value: Any) -> float | None:
    preferred = ("cv_score", "validation_accuracy", "accuracy", "best_score", "score")
    if isinstance(value, dict):
        for key in preferred:
            candidate = value.get(key)
            if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
                return float(candidate)
        for candidate in value.values():
            found = _find_metric(candidate)
            if found is not None:
                return found
    elif isinstance(value, list):
        for candidate in value:
            found = _find_metric(candidate)
            if found is not None:
                return found
    return None


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _build_report(root: Path, state: dict[str, Any]) -> tuple[Path, Path]:
    run_dir = run_state_path(root, state["run_id"]).parent
    report_json = run_dir / "report.json"
    report_html = run_dir / "report.html"
    coverage_path = root / "video-production" / "evomind-lite22-training-20260808" / "reports" / "mlebench_lite22_training_coverage.json"
    coverage = _read_json(coverage_path) or {}
    historical_lite22 = {
        "label": "historical_offline_evidence",
        "trained": int(coverage.get("training_attempts_completed") or 0),
        "total": int(coverage.get("lite_total_competitions") or 0),
        "private_graded": int(coverage.get("private_grades_completed") or 0),
        "offline_medal_equivalent": int((coverage.get("medal_threshold_equivalence") or {}).get("any_medal_count") or 0),
        "remaining_gate": str((coverage.get("remaining_grade_blocker") or {}).get("competition_id") or ""),
        "source_sha256": hashlib.sha256(coverage_path.read_bytes()).hexdigest() if coverage_path.is_file() else "",
    }
    public_report = {
        "schema": "evomind.assistant.training_report.v1",
        "created_at": _utc_now(),
        "run_id": state["run_id"],
        "task": state["task"],
        "gpu": state.get("gpu", "NVIDIA A800-SXM4-80GB"),
        "status": state["status"],
        "elapsed_seconds": state.get("elapsed_seconds"),
        "validation_accuracy": state.get("validation_accuracy"),
        "validation_gate": state.get("validation_gate", "offline_threshold_comparison"),
        "kaggle_submission": "disabled_human_gate",
        "artifacts": state.get("artifacts", []),
        "historical_lite22": historical_lite22,
        "evidence_boundary": "This run reports fresh local CV evidence only; it is not an official Kaggle score or rank.",
    }
    _atomic_json(report_json, public_report)
    rows = "".join(
        "<tr><td><a href='{href}'>{name}</a></td><td>{bytes}</td><td><code>{sha}</code></td></tr>".format(
            href=html.escape(str(item.get("href") or "#"), quote=True),
            name=html.escape(str(item.get("name") or "artifact")),
            bytes=int(item.get("bytes") or 0),
            sha=html.escape(str(item.get("sha256") or "")),
        )
        for item in public_report["artifacts"]
    )
    metric = public_report["validation_accuracy"]
    metric_text = f"{metric:.6f}" if isinstance(metric, (int, float)) else "见 result.json / summary.json"
    report_html.write_text(
        "<!doctype html><meta charset='utf-8'><title>EvoMind 真实训练报告</title>"
        "<style>body{font-family:system-ui;margin:40px;max-width:1100px;color:#15202b}"
        "h1{color:#3457d5}.card{padding:18px;border:1px solid #dbe3ef;border-radius:12px;margin:16px 0}"
        "table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #e5eaf2;padding:10px;text-align:left}"
        "code{font-size:12px;word-break:break-all}.ok{color:#087f5b;font-weight:700}</style>"
        f"<h1>EvoMind Dec-2021 真实 GPU 训练报告</h1><div class='card'><b>Run ID</b> {html.escape(state['run_id'])}<br>"
        f"<b>任务</b> {DEC2021_TASK_ID}<br><b>GPU</b> NVIDIA A800-SXM4-80GB<br>"
        f"<b>验证 Accuracy</b> <span class='ok'>{metric_text}</span><br>"
        "<b>Kaggle 提交</b> 已关闭（Human Gate）</div>"
        "<p>本页所有数字和文件均来自同一新 Run。验证值属于新鲜离线 CV 证据，不代表 Kaggle 官方分数、排名或奖牌。</p>"
        "<h2>可核验产物</h2><table><thead><tr><th>文件</th><th>字节</th><th>SHA-256</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
        "<div class='card'><h2>历史 Lite 22 离线覆盖（与本 Run 分开）</h2>"
        f"<p><b>{historical_lite22['trained']}/{historical_lite22['total']}</b> 已训练；"
        f"<b>{historical_lite22['private_graded']}/{historical_lite22['total']}</b> 完成 private grader；"
        f"<b>{historical_lite22['offline_medal_equivalent']}/{historical_lite22['total']}</b> 达到离线奖牌等效门槛。</p>"
        f"<p>剩余门禁：<code>{html.escape(historical_lite22['remaining_gate'])}</code>。这些是历史离线证据，不属于本次新 Run，也不是 Kaggle 官方成绩、排名或奖牌。</p>"
        f"<p>历史证据 SHA-256：<code>{historical_lite22['source_sha256']}</code></p></div>",
        encoding="utf-8",
    )
    return report_json, report_html


def _artifact_entries(root: Path, collection: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in collection.get("files") or []:
        local = Path(str(item.get("local") or ""))
        try:
            relative = _relative(root, local)
        except (OSError, ValueError):
            continue
        result.append({
            "name": local.name,
            "path": relative,
            "bytes": int(item.get("bytes") or local.stat().st_size),
            "sha256": str(item.get("sha256") or hashlib.sha256(local.read_bytes()).hexdigest()),
            "href": f"/api/artifacts?path={relative}",
        })
    return result


def execute_dec2021_training(
    root: Path,
    emit: Callable[[str], None] | Callable[..., None],
    *,
    ops: Any | None = None,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    run_id: str | None = None,
) -> dict[str, Any]:
    if ops is None:
        from scripts import mlebench_remote_ops as ops  # type: ignore[no-redef]

    run_id = run_id or _run_id()
    state_path = run_state_path(root, run_id)
    if state_path.exists():
        existing = load_public_run(root, run_id)
        if existing:
            return existing
        raise AssistantTrainingError("run id already exists with invalid state")

    started = monotonic()
    state: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "run_id": run_id,
        "task": DEC2021_TASK_ID,
        "gpu": "NVIDIA A800-SXM4-80GB",
        "status": "preflight",
        "stage": "gpu_gate",
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "elapsed_seconds": 0.0,
        "kaggle_submission": "disabled_human_gate",
        "artifacts": [],
        "log_lines": [],
    }
    _atomic_json(state_path, state)

    def save(**changes: Any) -> None:
        state.update(changes, updated_at=_utc_now(), elapsed_seconds=round(monotonic() - started, 2))
        _atomic_json(state_path, state)

    try:
        gate_path = Path(ops.DEFAULT_GATE_REPORT)
        gate = _read_json(gate_path)
        gate_is_fresh = bool(gate and gate.get("passed") and time.time() - gate_path.stat().st_mtime <= 600)
        if not gate_is_fresh:
            emit("training_progress", run_id=run_id, stage="gpu_gate", status="running", label="正在刷新 A800 5/5 GPU 只读门禁")
            gate = ops.sample_gpu_idle_gate(samples_required=5)
            _atomic_json(gate_path, gate)
        if not gate or not gate.get("passed"):
            raise AssistantTrainingError("GPU gate did not pass")
        gpu_name = str((((gate.get("samples") or [{}])[-1].get("gpus") or [{}])[0].get("name") or state["gpu"]))
        save(status="starting", stage="launch", gpu=gpu_name, gpu_gate_samples=len(gate.get("samples") or []))
        emit("training_run_started", run_id=run_id, task=DEC2021_TASK_ID, gpu=gpu_name, stage="launch", status="running", label="真实 GPU Run 已创建")

        start_payload = ops.start_run(
            Path(ops.DEFAULT_BUNDLE), gate_path, run_id=run_id, waves=["Wave1"],
            competitions=[DEC2021_TASK_ID], seed=42, optimization_plan_name=None,
            resume=False, allow_concurrent_with_cpu_light=False, max_gate_age=600,
            runner_performance_overrides=None, runner_contract_args=None,
        )
        save(status="running", stage="training", remote_started=True, human_gate_preserved=bool(start_payload.get("human_gate_preserved", True)))

        last_signature = ""
        final_status: dict[str, Any] | None = None
        while monotonic() - started <= timeout_seconds:
            status = ops.read_remote_status(run_id)
            lines = _safe_log_tail(status.get("log_tail"))
            progress = _progress_from_log(lines)
            signature = json.dumps(progress, sort_keys=True, ensure_ascii=False)
            if signature != last_signature:
                last_signature = signature
                save(stage="training", log_lines=lines, **{key: value for key, value in progress.items() if key != "log_lines"})
                emit("training_progress", run_id=run_id, task=DEC2021_TASK_ID, gpu=gpu_name, stage="training", status="running", label="CatBoost 正在训练", elapsed_seconds=state["elapsed_seconds"], **progress)
            if status.get("process") != "running":
                final_status = status
                break
            sleep(2)
        if final_status is None:
            raise AssistantTrainingError("training timed out; run state is preserved for recovery")
        if final_status.get("hold_reasons") and not _remote_candidate_complete(final_status):
            raise AssistantTrainingError("training finished with a blocked runtime state")

        save(status="collecting", stage="artifacts")
        collection = ops.collect_run(run_id, include_checkpoints=True)
        artifacts = _artifact_entries(root, collection)
        if not artifacts:
            raise AssistantTrainingError("training completed without collected artifacts")
        local_root = Path(str(collection["local_root"]))
        result_payload = _read_json(local_root / "result.json") or {}
        summary_payload = _read_json(local_root / "summary.json") or {}
        metric = _find_metric(result_payload) or _find_metric(summary_payload)
        names = {item["name"] for item in artifacts}
        required = {"result.json", "summary.json", "submission.csv", "submission_validation.json", "manifest.json"}
        missing = sorted(required - names)
        if missing:
            raise AssistantTrainingError("missing required artifacts: " + ", ".join(missing))
        save(
            status="completed", stage="completed", validation_accuracy=metric,
            validation_gate="offline_threshold_comparison", artifacts=artifacts,
            signals_sent=int(final_status.get("signals_sent") or 0),
            other_processes_modified=bool(final_status.get("other_processes_modified", False)),
        )
        report_json, report_html = _build_report(root, state)
        report_entries = []
        for report in (report_json, report_html):
            relative = _relative(root, report)
            report_entries.append({"name": report.name, "path": relative, "bytes": report.stat().st_size, "sha256": hashlib.sha256(report.read_bytes()).hexdigest(), "href": f"/api/artifacts?path={relative}"})
        save(artifacts=[*artifacts, *report_entries])
        emit("training_artifacts", run_id=run_id, task=DEC2021_TASK_ID, stage="artifacts", status="completed", label="模型、submission、manifest 和报告已收集", artifacts=state["artifacts"])
        emit("training_completed", **state, label="真实 GPU 训练已完成")
        return state
    except Exception as exc:
        save(status="failed", stage="failed", error=str(exc)[:500])
        emit("training_failed", run_id=run_id, task=DEC2021_TASK_ID, status="failed", stage="failed", label="训练未完成，Run 状态已保留", message=str(exc)[:500])
        raise


def completion_markdown(state: dict[str, Any]) -> str:
    metric = state.get("validation_accuracy")
    metric_text = f"{float(metric):.6f}" if isinstance(metric, (int, float)) else "见运行产物"
    artifacts = state.get("artifacts") or []
    rows = [f"| [{item['name']}]({item['href']}) | {item['bytes']} | `{item['sha256']}` |" for item in artifacts]
    return "\n".join([
        "## 真实 GPU 训练完成",
        "",
        f"- Run ID：`{state['run_id']}`",
        f"- 任务：`{state['task']}`",
        f"- GPU：{state.get('gpu', 'NVIDIA A800-SXM4-80GB')}",
        f"- 新鲜离线 CV Accuracy：**{metric_text}**",
        f"- 耗时：{state.get('elapsed_seconds', 0)} 秒",
        "- Kaggle 官方提交：已关闭（Human Gate）",
        "",
        "| 产物 | 字节 | SHA-256 |",
        "|---|---:|---|",
        *rows,
        "",
        "本次显示的是同一新 Run 的真实离线 CV 和文件证据，不代表 Kaggle 官方分数、排名或奖牌。",
    ])
