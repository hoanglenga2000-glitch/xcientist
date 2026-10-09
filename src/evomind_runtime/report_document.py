"""Evidence-bound report data shared by web, Markdown, Word and PDF renderers."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

SCHEMA = "evomind.report_document.v1"
METRICS = {
    "accuracy": ("Accuracy", "ratio", "higher"), "roc_auc": ("ROC AUC", "score", "higher"),
    "pr_auc": ("PR AUC", "score", "higher"), "f1": ("F1", "score", "higher"),
    "precision": ("Precision", "ratio", "higher"), "recall": ("Recall", "ratio", "higher"),
    "balanced_accuracy": ("Balanced accuracy", "ratio", "higher"), "r2": ("R squared", "score", "higher"),
    "loss": ("Loss", "loss", "lower"), "rmse": ("RMSE", "target units", "lower"),
    "mae": ("MAE", "target units", "lower"), "mse": ("MSE", "squared target units", "lower"),
    "gll": ("GLL", "score", "higher"), "sq": ("SQ", "score", "higher"),
    "scored_questions": ("Scored questions", "questions", "none"),
    "elapsed_seconds": ("Elapsed time", "seconds", "lower"),
    "initial_loss": ("Initial loss", "loss", "lower"), "final_loss": ("Final loss", "loss", "lower"),
    "fit_steps": ("Optimizer updates", "steps", "none"),
}
SECRET_LINE = re.compile(r"(?i)password|passwd|\bpwd\b|token|authorization|bearer|cookie|secret|api[_ -]?key|登录|账号|密码|口令|凭据")
MAX_SOURCE_BYTES = 8 * 1024 * 1024


def canonical(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def safe_text(value, limit=12000):
    lines = str(value or "").splitlines()
    safe = ["[敏感内容已隐藏]" if SECRET_LINE.search(line) else re.sub(r"(https?://[^\s?]+)\?\S+", r"\1?[redacted]", line) for line in lines]
    return "\n".join(safe)[:limit]


def number(value):
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def _metric_rows(value, source, prefix=""):
    if not isinstance(value, dict):
        return []
    rows = []
    metric_name = value.get("metric")
    if isinstance(metric_name, str) and metric_name in METRICS and number(value.get("value")) is not None:
        label, unit, direction = METRICS[metric_name]
        rows.append({"name": prefix + metric_name, "label": label, "value": number(value["value"]), "unit": unit,
                     "direction": direction, "source_artifact_id": source["id"], "source_sha256": source["sha256"]})
    for key, item in value.items():
        if key in METRICS and number(item) is not None:
            label, unit, direction = METRICS[key]
            rows.append({"name": prefix + key, "label": label, "value": number(item), "unit": unit,
                         "direction": direction, "source_artifact_id": source["id"], "source_sha256": source["sha256"]})
        elif key in {"metrics", "evaluation", "validation", "train", "test", "before", "after"} and isinstance(item, dict):
            rows.extend(_metric_rows(item, source, prefix + key + "."))
    return rows


def build_document(runtime, run_id, args):
    session = runtime.get_session(run_id)
    run = runtime.store.get_assistant_run(run_id)
    if run is None:
        raise ValueError("report_requires_assistant_run")
    metadata = session.get("metadata") or {}
    requested = set(args.get("artifact_ids") or [])
    all_sources = runtime.store.list_deliverables(run_id)
    known = {row["id"] for row in all_sources}
    if not requested.issubset(known):
        raise ValueError("report_artifact_not_owned_by_run")
    selected = [row for row in all_sources if (not requested or row["id"] in requested)
                and row.get("source_tool_call") != "report_generate"]
    if len(selected) > 256:
        raise ValueError("report_source_count_limit")
    sources, metrics, figures, missing = [], [], [], []
    read_bytes = 0
    for item in selected:
        source = {key: item.get(key) for key in ("id", "name", "sha256", "bytes", "media_type")}
        source["verification"] = "registered"
        path = Path(item["path"])
        path.resolve().relative_to(Path(session["workspace_root"]).resolve())
        if path.is_symlink() or not path.is_file() or path.stat().st_size != item["bytes"]:
            raise ValueError("report_source_integrity_failed")
        if item["bytes"] <= MAX_SOURCE_BYTES and read_bytes + item["bytes"] <= 32 * 1024 * 1024:
            if digest(path) != item["sha256"]:
                raise ValueError("report_source_integrity_failed")
            source["verification"] = "hash_verified"
            read_bytes += item["bytes"]
            name = str(item["name"]).lower()
            if not re.search(r"private|phase[_-]?2|hidden|test_labels", name) and path.suffix == ".json":
                try:
                    value = json.loads(path.read_text(encoding="utf-8"))
                except (ValueError, UnicodeError):
                    value = {}
                metrics.extend(_metric_rows(value, source))
                # Explicit source specs are representations of recorded data,
                # never independent evidence that a model was trained.
                if isinstance(value, dict) and value.get("schema") == "evomind.figure_source.v1":
                    for figure in value.get("figures", [])[:12]:
                        figures.append({**figure, "source_artifact_id": item["id"], "source_sha256": item["sha256"],
                                        "evidence_scope": "source_file_data_not_independent_research_validation"})
            elif path.suffix == ".jsonl" and re.search(r"telemetry|train(?:ing)?[-_](?:metrics|log)", name):
                rows = []
                for line in path.read_text(encoding="utf-8").splitlines()[:20000]:
                    try: rows.append(json.loads(line))
                    except ValueError: continue
                axis = next((key for key in ("step", "steps", "epoch") if any(isinstance(row, dict) and number(row.get(key)) is not None for row in rows)), "step")
                axis_label = "Epoch" if axis == "epoch" else "Optimizer step"
                for key, label in (("loss", "Training loss"), ("eval_loss", "Validation observations"),
                                   ("learning_rate", "Learning rate"), ("cuda_max_allocated_mb", "CUDA memory (MB)")):
                    points = [(number(row.get(axis)), number(row.get(key))) for row in rows if isinstance(row, dict)]
                    points = [(x, y) for x, y in points if x is not None]
                    if any(y is not None for _x, y in points):
                        figures.append({"id": key+"-"+item["id"][-8:], "kind": "line", "title": label,
                            "x_label": axis_label, "y_label": label, "x": [x for x, _y in points], "y": [y for _x, y in points],
                            "points_only": key == "eval_loss", "caption": "Recorded observations; missing values are not interpolated.",
                            "source_artifact_id": item["id"], "source_sha256": item["sha256"],
                            "evidence_scope": "recorded_telemetry"})
        sources.append(source)
    prompt = str(run.get("prompt") or session.get("objective") or "")
    inferred = "training" if re.search(r"训练|微调|\btrain|fine.?tun", prompt, re.I) else "inference" if re.search(r"推理|预测|inference|predict", prompt, re.I) else "diagnostic" if re.search(r"连接|健康|检查|diagnostic|health|connect", prompt, re.I) else "analysis"
    kind = args.get("report_kind") or inferred
    if kind not in {"diagnostic", "training", "inference", "analysis"}:
        raise ValueError("report_kind_invalid")
    language = args.get("language") or "zh-CN"
    if language not in {"zh-CN", "en-US"}:
        raise ValueError("report_language_invalid")
    calls = [call for call in runtime.store.list_tool_calls(run_id, limit=2000)
             if call["tool_name"] not in {"report_generate", "report_status"}]
    checks = [{"tool": call["tool_name"], "status": call["status"], "ok": (call.get("result") or {}).get("ok"),
               "call_id": call["id"], "completed_at": call.get("completed_at", "")} for call in calls]
    independent = False
    for event in runtime.store.list_events(run_id):
        payload = event.get("payload") or {}
        if event.get("event_type") == "model_verification_completed" and payload.get("verified") is True:
            independent = any(item["id"] == payload.get("artifact_id") and item["sha256"] == payload.get("receipt_sha256")
                              and item["verification"] == "hash_verified" for item in sources)
            if independent: break
    if kind in {"training", "inference"} and not metrics:
        missing.append("没有已绑定来源的指标 / No source-bound metrics")
    if kind == "training" and not independent:
        missing.append("模型独立重载尚未核验 / Independent model reload is not verified")
    if kind == "analysis" and not sources:
        missing.append("没有已发布的分析来源，当前仅为任务记录；先用 artifact_publish 发布所需产物并取得 artifact_id，再用 artifact_ids 创建新报告，不能靠恢复旧报告补入来源。 / No published analysis sources; this is a task record only. Publish the intended artifacts with artifact_publish, then create a new report with the returned artifact_ids. Resuming this frozen report cannot add sources.")
    if not checks and not sources:
        missing.append("尚无执行或产物证据 / No execution or artifact evidence")
    summary = safe_text(args.get("summary") or "")
    status = str(run.get("status") or session.get("status") or "unknown")
    if len(figures) > 24:
        raise ValueError("report_figure_count_limit")
    document = {"schema": SCHEMA, "language": language, "kind": kind,
        "identity": {"run_id": run_id, "project_id": str(metadata.get("project_id") or ""),
                     "task_id": str(metadata.get("selected_task") or "") if kind != "diagnostic" else ""},
        "title": safe_text(args.get("title") or prompt or "EvoMind Report", 200),
        "model": metadata.get("model_execution_contract") or {"provider": run.get("model_provider", ""), "model": run.get("model", "")},
        "execution_status": status,
        "evidence_status": "independently_verified" if independent else "hash_verified" if sources and all(s["verification"] == "hash_verified" for s in sources) else "unverified",
        "report_status": "partial" if missing else "ready", "author_summary": summary,
        "metrics": metrics, "figures": figures, "checks": checks, "sources": sources, "missing": missing,
        "limitations": ["文件哈希通过不等于科研结论通过 / File integrity is not scientific validity.",
                        "本报告不声明官方比赛成绩或论文优势 / No official competition score or paper superiority is claimed."],
        "formats": sorted(set(args.get("formats") or ["markdown", "html", "docx", "pdf"]))}
    if not document["formats"] or not set(document["formats"]).issubset({"markdown", "html", "docx", "pdf"}):
        raise ValueError("report_formats_invalid")
    canonical(document)
    return document
