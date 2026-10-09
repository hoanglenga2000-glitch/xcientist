from __future__ import annotations

import hashlib
import contextvars
import json
import mimetypes
import os
import re
import shutil
import subprocess
import threading
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

from .models import PermissionLevel, SessionStatus, new_id, utc_now
from .policy import argument_fingerprint

if TYPE_CHECKING:
    from .runtime import AgentRuntime


CHUNK_SIZE = 16 * 1024 * 1024
DEFAULT_MAX_UPLOAD_BYTES = 10 * 1024 * 1024 * 1024
PUBLIC_EVENT_TYPES = {
    "run_progress",
    "research_stage",
    "knowledge_retrieved",
    "research_outcome",
    "run_created",
    "plan_updated",
    "step_started",
    "tool_started",
    "tool_completed",
    "data_profiled",
    "compute_route_selected",
    "verification_completed",
    "retry_scheduled",
    "approval_required",
    "artifact_published",
    "attachments_inherited",
    "model_observed",
    "assistant_text",
    "run_completed",
    "run_blocked",
    "run_failed",
    "run_pause_requested",
    "run_paused",
    "goal_created",
    "goal_updated",
    "managed_hpc_identity.rebound",
}
TERMINAL_RUN_STATES = {"completed", "blocked", "failed", "cancelled"}
_SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]{1,180}$")
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_MANAGED_HPC_TENANT = re.compile(r"^tenant_[a-f0-9]{24}$")
_MANAGED_HPC_OWNER = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_MANAGED_HPC_PROFILE_INSTANCE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_MANAGED_HPC_BINDING = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$")
_MANAGED_HPC_REBIND_KEY = re.compile(r"^[A-Za-z0-9_.:-]{8,180}$")
_MANAGED_HPC_IDENTITY_FIELDS = (
    "tenant_id",
    "owner_principal_id",
    "job_id",
    "credential_profile",
    "allocation_generation",
    "profile_instance_id",
    "allocation_binding_id",
)
_ARTIFACT_FILE = re.compile(
    r"(?i)\b([A-Za-z0-9][A-Za-z0-9._()\-]{0,178}\.(?:csv|json|jsonl|md|pdf|zip|joblib|pkl|pt|pth|onnx|xlsx|docx|pptx|log|py|html|png|jpg|jpeg|svg|parquet|txt))\b"
)
_FINAL_ARTIFACT_MARKERS = (
    "最终产物至少包括",
    "最终交付文件",
    "交付文件至少包括",
    "final artifacts include",
    "final deliverables include",
)
_FINAL_ARTIFACT_FALLBACK_MARKERS = ("发布", "交付", "publish", "deliver")
_BLOCKER_ARTIFACT_NAME = re.compile(r"(?i)(?:^|[-_.])(blocker|failure|failed|error|superseded)(?:[-_.]|$)")
_ARTIFACT_SNAPSHOT_DIR = ".a"

# Hotfix D10: retry transient file-write failures. A 12-13 MB artifact written moments
# earlier can still be held by a virus scanner or the search indexer, which surfaced as
# `[WinError 5]` *after* a run had already delivered its bundle and turned a successful
# window into `blocked`. Retry only access/sharing failures; everything else propagates.
_TRANSIENT_FILE_WINERRORS = {5, 32, 33}


def _retry_transient_file_op(operation, *, attempts: int = 6, base_delay: float = 0.4):
    """Run one filesystem mutation, retrying transient access or sharing failures."""
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except OSError as exc:
            winerror = getattr(exc, "winerror", None)
            if isinstance(exc, PermissionError) or winerror in _TRANSIENT_FILE_WINERRORS:
                last_error = exc
                if attempt < attempts:
                    time.sleep(base_delay * attempt)
                    continue
            raise
    if last_error is not None:
        raise last_error
    raise OSError("transient_file_op_failed")
_SETTLED_OBSERVATION_TOOLS = {"competition_data_status", "competition_data_prepare"}
_CONTINUATION_TOOL_SCOPE_MARKERS = ("仅允许使用", "只允许使用", "only allow", "only use")
_MODEL_STRATEGY_ALTERNATIVE = re.compile(
    r"(?i)(?:strategy(?:[-_.a-z0-9]*\.(?:json|ya?ml|py|md|txt)|\s+artifact)|策略(?:产物|文件)?)"
    r".{0,32}(?:或|or)\s*"
    r"(?:model\s+artifact|模型(?:产物|文件|权重)?)"
    r"|(?:model\s+artifact|模型(?:产物|文件|权重)?)"
    r".{0,32}(?:或|or)\s*"
    r"(?:strategy(?:[-_.a-z0-9]*\.(?:json|ya?ml|py|md|txt)|\s+artifact)|策略(?:产物|文件)?)"
)
_OPTIONAL_MODEL_LINE_MARKERS = (
    "不得",
    "禁止",
    "不能",
    "不生成",
    "未生成",
    "无需",
    "无模型",
    "只有",
    "若",
    "if ",
    "only if",
    "without",
    "must not",
    "do not",
    "no model",
)
_CONTINUATION_SUPPORT_TOOLS = {
    "file_list",
    "file_read",
    "file_write",
    "artifact_list",
    "artifact_preview",
    "artifact_publish",
}


def _validated_managed_hpc_identity(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(_MANAGED_HPC_IDENTITY_FIELDS):
        raise ValueError("managed_hpc_identity_rebind_identity_invalid")
    tenant_id = str(value.get("tenant_id") or "")
    owner = str(value.get("owner_principal_id") or "")
    profile = str(value.get("credential_profile") or "")
    instance = str(value.get("profile_instance_id") or "")
    binding = str(value.get("allocation_binding_id") or "")
    job_id = value.get("job_id")
    generation = value.get("allocation_generation")
    if (
        not _MANAGED_HPC_TENANT.fullmatch(tenant_id)
        or not _MANAGED_HPC_OWNER.fullmatch(owner)
        or isinstance(job_id, bool)
        or not isinstance(job_id, int)
        or job_id <= 0
        or isinstance(generation, bool)
        or not isinstance(generation, int)
        or generation <= 0
        or profile != f"{tenant_id}_job{job_id}_g{generation}"
        or not _MANAGED_HPC_PROFILE_INSTANCE.fullmatch(instance)
        or not _MANAGED_HPC_BINDING.fullmatch(binding)
    ):
        raise ValueError("managed_hpc_identity_rebind_identity_invalid")
    return {field: value[field] for field in _MANAGED_HPC_IDENTITY_FIELDS}


def _managed_hpc_identity_fingerprint(value: Any) -> str:
    identity = _validated_managed_hpc_identity(value)
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    ).hexdigest()


def _artifact_requirements_from_prompt(prompt: str) -> dict[str, list[str]]:
    """Extract only an explicit final-deliverables section, never input names."""
    objective = str(prompt or "")
    # A supplied program is reference material, not an extension of the final
    # deliverable contract. Its input paths and string literals must not become
    # mandatory published artifacts (including for unfenced source appendices).
    source_appendix = re.search(
        r"(?i)(?:完整源码(?:如下)?|完整源代码(?:如下)?|full source code(?: follows)?|complete source code(?: follows)?)[：:]",
        objective,
    )
    if source_appendix:
        objective = objective[:source_appendix.start()]
    lowered = objective.casefold()
    marker_positions = [lowered.find(marker.casefold()) for marker in _FINAL_ARTIFACT_MARKERS]
    marker_positions = [position for position in marker_positions if position >= 0]
    if not marker_positions:
        marker_positions = [lowered.rfind(marker.casefold()) for marker in _FINAL_ARTIFACT_FALLBACK_MARKERS]
        marker_positions = [position for position in marker_positions if position >= 0]
    if not marker_positions:
        return {"names": [], "groups": []}
    section = objective[max(marker_positions):]
    names = list(dict.fromkeys(match.group(1) for match in _ARTIFACT_FILE.finditer(section)))
    section_lower = section.casefold()
    groups: list[str] = []
    model_strategy_alternative = bool(_MODEL_STRATEGY_ALTERNATIVE.search(section)) or bool(
        re.search(
            r"(?i)(?:frozen[-_.]?model\.(?:joblib|pkl|pt|pth|onnx)|model(?:\s+artifact)?).{0,32}(?:或|\bor\b).{0,32}(?:strategy(?:[-_.a-z0-9]*\.(?:json|ya?ml|py|md|txt))|策略(?:产物|文件)?)",
            section,
        )
    )
    if model_strategy_alternative:
        groups.append("model_or_strategy")
        names = [
            name
            for name in names
            if not (
                ("strategy" in Path(name).stem.casefold() or Path(name).stem.casefold() in {"frozen-model", "model"})
                and Path(name).suffix.casefold() in {".json", ".yaml", ".yml", ".py", ".md", ".txt", ".joblib", ".pkl", ".pt", ".pth", ".onnx"}
            )
        ]
    model_lines = [
        line
        for line in section.splitlines()
        if ("模型" in line or "model artifact" in line.casefold())
        and not _MODEL_STRATEGY_ALTERNATIVE.search(line)
        and not any(marker in line.casefold() for marker in _OPTIONAL_MODEL_LINE_MARKERS)
    ]
    if model_lines:
        groups.append("model")
    if "加载环境" in section or "environment" in section_lower:
        groups.append("environment")
    if "experience board" in section_lower or "experience card" in section_lower:
        groups.append("experience_board")
    if "search_graph" in section_lower or "search graph" in section_lower:
        groups.append("search_graph")
    if "retrospective_memory" in section_lower or "retrospective memory" in section_lower:
        groups.append("retrospective_memory")
    if "validation_contract" in section_lower or "validation contract" in section_lower:
        groups.append("validation_contract")
    if "claim_audit" in section_lower or "claim audit" in section_lower:
        groups.append("claim_audit")
    if "manifest" in section_lower or "清单" in section:
        groups.append("manifest")
    if "完整 zip" in section_lower or "complete zip" in section_lower or "complete-results.zip" in section_lower:
        groups.append("complete_bundle")
    if "markdown/pdf 报告" in section_lower or "markdown and pdf report" in section_lower:
        groups.extend(("markdown_report", "pdf_report"))
    return {"names": names, "groups": list(dict.fromkeys(groups))}


def _explicit_continuation_tool_scope(
    instruction: str,
    available_tool_names: set[str],
) -> list[str] | None:
    for line in str(instruction or "").splitlines():
        lowered = line.casefold()
        if not any(marker in lowered for marker in _CONTINUATION_TOOL_SCOPE_MARKERS):
            continue
        requested = [
            token
            for token in re.findall(r"[A-Za-z][A-Za-z0-9_]*", line)
            if token in available_tool_names
        ]
        return list(dict.fromkeys(requested))
    return None


def _automatic_retry_enabled() -> bool:
    return os.environ.get("EVOMIND_ASSISTANT_AUTO_RETRY", "").strip().casefold() in {"1", "true", "yes", "on"}


def _failure_policy(exc: Exception) -> tuple[str, bool]:
    text = f"{type(exc).__name__}: {exc}".casefold()
    # Transport retries are already bounded at the model-request layer.
    # Replaying the whole Run here could repeat previously executed tools.
    if "llmerror/" in text or "model_contract" in text:
        return "recoverable", False
    # A controlled credential is intentionally one-shot. Consumer failures
    # cannot be repaired by replaying the same model turn: the envelope has
    # already been tombstoned and replay would only produce a missing-secret
    # gate. Stop here so the user can enter a fresh credential through the
    # browser-memory form.
    if any(
        term in text
        for term in (
            "controlled_secret_required",
            "consumer_process_exit",
            "consumer_host_exit",
            "consumer_invalid_base64",
            "stage_envelope_",
        )
    ):
        return "controlled_secret_required", False
    if any(term in text for term in ("approval", "waiting_approval")):
        return "approval_required", False
    if any(term in text for term in ("attachment not found", "input required", "missing input")):
        return "input_required", False
    if any(term in text for term in ("no space left", "disk full", "enospc")):
        return "recoverable", False
    if isinstance(exc, ValueError) or any(term in text for term in ("integrity", "sha256 mismatch", "escapes", "unsafe path", "invalid artifact")):
        return "terminal", False
    # Once the model has returned a final answer, unresolved tool rows or
    # missing deliverables require a changed precondition or an explicit
    # continuation. Re-running the same external call automatically can repeat
    # downloads and cannot repair a genuinely absent model artifact.
    if "required result artifacts unavailable;" in text:
        return "recoverable", False
    if isinstance(exc, (TimeoutError, ConnectionError)) or any(
        term in text for term in ("timeout", "timed out", "429", "rate limit", "500", "502", "503", "504", "network", "connection", "gateway", "outage", "unavailable", "stream")
    ):
        return "transient", _automatic_retry_enabled()
    return "recoverable", _automatic_retry_enabled()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_name(value: str) -> str:
    name = Path(str(value or "attachment.bin").replace("\x00", "")).name.strip()
    name = re.sub(r"[^A-Za-z0-9._()\-\u3400-\u9fff]+", "_", name).strip(" ._")
    return (name or "attachment.bin")[:180]


def _snapshot_artifact_file(
    source: Path,
    outputs: Path,
    artifact_id: str,
    name: str,
) -> Path:
    """Copy a published file to an immutable, run-local snapshot."""
    resolved_source = source.resolve(strict=True)
    resolved_outputs = outputs.resolve(strict=True)
    resolved_source.relative_to(resolved_outputs)
    snapshot_root = (resolved_outputs / _ARTIFACT_SNAPSHOT_DIR).resolve(strict=False)
    snapshot_root.relative_to(resolved_outputs)
    snapshot_root.mkdir(parents=True, exist_ok=True)
    suffix = Path(_safe_name(name or resolved_source.name)).suffix[:20]
    target = snapshot_root / f"{artifact_id.rsplit('_', 1)[-1][:16]}{suffix}"
    temporary = target.with_name(target.name + ".tmp")
    try:
        _retry_transient_file_op(lambda: shutil.copyfile(resolved_source, temporary))
        _retry_transient_file_op(lambda: os.replace(temporary, target))
        return target.resolve(strict=True)
    except Exception:
        temporary.unlink(missing_ok=True)
        target.unlink(missing_ok=True)
        raise


def _media_type(path: Path, declared: str = "") -> str:
    """Derive the served type from bytes; names and browser declarations are hints only."""
    supplied = str(declared or "").split(";", 1)[0].strip().lower()
    if not path.is_file():
        if re.fullmatch(r"[a-z0-9.+-]+/[a-z0-9.+-]+", supplied):
            return supplied
        return mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    raw = path.read_bytes()[:8192] if path.is_file() else b""
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if raw.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    if raw.startswith(b"%PDF-"):
        return "application/pdf"
    if raw.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(path) as archive:
                names = {item.filename for item in archive.infolist()[:5000]}
            if "[Content_Types].xml" in names and any(name.startswith("word/") for name in names):
                return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            if "[Content_Types].xml" in names and any(name.startswith("xl/") for name in names):
                return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            if "[Content_Types].xml" in names and any(name.startswith("ppt/") for name in names):
                return "application/vnd.openxmlformats-officedocument.presentationml.presentation"
            return "application/zip"
        except (OSError, zipfile.BadZipFile):
            return "application/octet-stream"
    stripped = raw.lstrip().lower()
    if stripped.startswith(b"<svg") or (stripped.startswith(b"<?xml") and b"<svg" in stripped[:2048]):
        return "image/svg+xml"
    if stripped.startswith((b"<!doctype html", b"<html")):
        return "text/html"
    if b"\x00" in raw:
        return "application/octet-stream"
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError:
        return "application/octet-stream"
    suffix = path.suffix.lower()
    text_types = {
        ".csv": "text/csv",
        ".tsv": "text/tab-separated-values",
        ".json": "application/json",
        ".jsonl": "application/x-ndjson",
        ".md": "text/markdown",
        ".markdown": "text/markdown",
        ".yaml": "application/yaml",
        ".yml": "application/yaml",
        ".toml": "application/toml",
    }
    detected = text_types.get(suffix, "text/plain")
    if supplied.startswith("text/") and detected == "text/plain" and re.fullmatch(r"[a-z0-9.+-]+/[a-z0-9.+-]+", supplied):
        return supplied
    return detected


def _preview_kind(path: Path, media_type: str) -> str:
    suffix = path.suffix.lower()
    if media_type.startswith("image/") and media_type != "image/svg+xml":
        return "image"
    if media_type == "application/pdf":
        return "pdf"
    if media_type == "text/html":
        return "html"
    if media_type == "text/markdown":
        return "markdown"
    if media_type in {"text/csv", "text/tab-separated-values"}:
        return "table"
    if media_type in {"application/json", "application/x-ndjson", "application/yaml", "application/toml"}:
        return "structured_text"
    if media_type.startswith("text/"):
        return "text"
    if media_type in {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    }:
        return "office"
    if media_type == "application/zip":
        return "archive"
    if media_type == "image/svg+xml":
        return "svg"
    return "binary"


def public_artifact(row: dict[str, Any]) -> dict[str, Any]:
    artifact_id = str(row.get("id") or "")
    value = {
        **row,
        "preview_url": f"/api/assistant/artifacts/{artifact_id}",
        "download_url": f"/api/assistant/artifacts/{artifact_id}?download=1",
    }
    if str(row.get("preview_kind") or "") == "office":
        derived = _verified_office_preview(Path(str(row.get("path") or "")), str(row.get("sha256") or ""))
        if derived:
            value.update(
                {
                    "source_preview_kind": "office",
                    "preview_kind": "pdf",
                    "preview_url": f"/api/assistant/artifacts/{artifact_id}?derived=pdf",
                    "preview_png_url": f"/api/assistant/artifacts/{artifact_id}?derived=png" if derived.get("png_path") else "",
                    "preview_sha256": derived["pdf_sha256"],
                }
            )
    return value


def _office_preview_dir(path: Path) -> Path:
    return path.parent / ".evomind_previews"


def _verified_office_preview(path: Path, source_sha256: str) -> dict[str, Any] | None:
    if not source_sha256:
        return None
    preview_dir = _office_preview_dir(path)
    preview_key = source_sha256[:32]
    metadata_path = preview_dir / f"{preview_key}.json"
    legacy_metadata_path = preview_dir / f"{source_sha256}.json"
    if not metadata_path.is_file() and legacy_metadata_path.is_file():
        metadata_path = legacy_metadata_path
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("source_sha256") != source_sha256:
            return None
        pdf_path = (preview_dir / str(metadata["pdf_name"])).resolve(strict=True)
        pdf_path.relative_to(preview_dir.resolve(strict=True))
        if not pdf_path.read_bytes()[:5] == b"%PDF-" or _sha256_file(pdf_path) != metadata.get("pdf_sha256"):
            return None
        result = {**metadata, "pdf_path": str(pdf_path)}
        png_name = str(metadata.get("png_name") or "")
        if png_name:
            png_path = (preview_dir / png_name).resolve(strict=True)
            png_path.relative_to(preview_dir.resolve(strict=True))
            if png_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and _sha256_file(png_path) == metadata.get("png_sha256"):
                result["png_path"] = str(png_path)
        return result
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return None


def _derive_office_preview(path: Path, source_sha256: str) -> dict[str, Any] | None:
    existing = _verified_office_preview(path, source_sha256)
    if existing:
        return existing
    candidates = [
        os.environ.get("EVOMIND_SOFFICE_PATH", ""),
        shutil.which("soffice") or "",
        str(Path.home() / ".codex-office" / "bin" / "soffice.exe"),
    ]
    soffice = next((value for value in candidates if value and Path(value).is_file()), "")
    if not soffice:
        return None
    preview_dir = _office_preview_dir(path)
    preview_dir.mkdir(parents=True, exist_ok=True)
    temporary = preview_dir / f"tmp_{source_sha256[:16]}_{os.getpid()}"
    temporary.mkdir(parents=True, exist_ok=True)
    try:
        completed = subprocess.run(
            [soffice, "--headless", "--convert-to", "pdf", "--outdir", str(temporary), str(path)],
            capture_output=True,
            timeout=120,
            check=False,
        )
        converted = temporary / f"{path.stem}.pdf"
        if completed.returncode != 0 or not converted.is_file() or converted.read_bytes()[:5] != b"%PDF-":
            return None
        preview_key = source_sha256[:32]
        pdf_name = f"{preview_key}.pdf"
        pdf_path = preview_dir / pdf_name
        os.replace(converted, pdf_path)
        pdf_sha = _sha256_file(pdf_path)
        metadata: dict[str, Any] = {
            "schema": "evomind.office_preview.v1",
            "source_sha256": source_sha256,
            "pdf_name": pdf_name,
            "pdf_sha256": pdf_sha,
            "pdf_bytes": pdf_path.stat().st_size,
            "generated_at": utc_now(),
        }
        try:
            import fitz
            document = fitz.open(pdf_path)
            if document.page_count:
                png_name = f"{preview_key}.png"
                png_path = preview_dir / png_name
                document.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False).save(png_path)
                metadata.update({"png_name": png_name, "png_sha256": _sha256_file(png_path), "png_bytes": png_path.stat().st_size})
            document.close()
        except Exception:
            pass
        metadata_path = preview_dir / f"{preview_key}.json"
        metadata_tmp = metadata_path.with_suffix(".json.tmp")
        metadata_tmp.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(metadata_tmp, metadata_path)
        return _verified_office_preview(path, source_sha256)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


class AssistantRunService:
    """Durable assistant runs, uploads and verified user-facing deliverables."""

    def __init__(self, runtime: "AgentRuntime") -> None:
        self.runtime = runtime
        self.store = runtime.store
        self.root = runtime.runtime_root
        self.upload_root = self.root / "uploads"
        self.task_root = self.root / "assistant_tasks"
        self.upload_root.mkdir(parents=True, exist_ok=True)
        self.task_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._threads: dict[str, threading.Thread] = {}
        self._resume_waiters: dict[str, threading.Thread] = {}
        self._approval_workers: dict[str, threading.Thread] = {}
        self._stopping = threading.Event()

    def shutdown(self, timeout: float = 2.0) -> bool:
        """Drain local workers without cancelling or signalling remote jobs.

        False means stores must remain open and the caller must retry shutdown
        after the in-flight work settles. No replacement worker is admitted.
        """
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._lock:
            self._stopping.set()
            threads = list(dict.fromkeys([*self._threads.values(), *self._resume_waiters.values(), *self._approval_workers.values()]))
        for thread in threads:
            if thread is not threading.current_thread() and thread.is_alive():
                thread.join(timeout=max(0.0, deadline - time.monotonic()))
        return not any(thread.is_alive() for thread in threads)

    @property
    def max_upload_bytes(self) -> int:
        try:
            return max(CHUNK_SIZE, int(os.environ.get("EVOMIND_MAX_UPLOAD_BYTES", DEFAULT_MAX_UPLOAD_BYTES)))
        except ValueError:
            return DEFAULT_MAX_UPLOAD_BYTES

    def create_upload(
        self,
        *,
        name: str,
        total_bytes: int,
        media_type: str = "",
        sha256: str = "",
    ) -> dict[str, Any]:
        total = int(total_bytes)
        if total < 0 or total > self.max_upload_bytes:
            raise ValueError("upload size is outside the configured limit")
        expected = str(sha256 or "").lower()
        if expected and not _SHA256.fullmatch(expected):
            raise ValueError("expected sha256 is invalid")
        upload_id = new_id("upload")
        safe_name = _safe_name(name)
        created = utc_now()
        row = self.store.put_upload(
            {
                "id": upload_id,
                "original_name": str(name or safe_name)[:500],
                "safe_name": safe_name,
                "media_type": _media_type(Path(safe_name), media_type),
                "total_bytes": total,
                "chunk_size": CHUNK_SIZE,
                "expected_sha256": expected,
                "status": "uploading",
                "chunk_count": (total + CHUNK_SIZE - 1) // CHUNK_SIZE,
                "final_path": "",
                "final_sha256": "",
                "created_at": created,
                "updated_at": created,
            }
        )
        (self.upload_root / upload_id / "chunks").mkdir(parents=True, exist_ok=True)
        return {**row, "max_upload_bytes": self.max_upload_bytes}

    def put_chunk(self, upload_id: str, chunk_index: int, data: bytes, sha256: str) -> dict[str, Any]:
        with self._lock:
            return self._put_chunk_locked(upload_id, chunk_index, data, sha256)

    def _put_chunk_locked(self, upload_id: str, chunk_index: int, data: bytes, sha256: str) -> dict[str, Any]:
        if not _SAFE_ID.fullmatch(upload_id):
            raise ValueError("upload id is invalid")
        upload = self.store.get_upload(upload_id)
        if not upload:
            raise KeyError(upload_id)
        if upload["status"] not in {"uploading", "complete"}:
            raise ValueError("upload is not writable")
        index = int(chunk_index)
        count = int(upload["chunk_count"])
        if index < 0 or index >= count:
            raise ValueError("chunk index is outside the upload contract")
        expected_bytes = min(CHUNK_SIZE, int(upload["total_bytes"]) - index * CHUNK_SIZE)
        if len(data) != expected_bytes:
            raise ValueError("chunk byte length does not match the upload contract")
        digest = hashlib.sha256(data).hexdigest()
        supplied = str(sha256 or "").lower()
        if not _SHA256.fullmatch(supplied) or supplied != digest:
            raise ValueError("chunk sha256 mismatch")
        existing = self.store.get_upload_chunk(upload_id, index)
        if existing:
            if existing["sha256"] != digest or int(existing["bytes"]) != len(data):
                raise ValueError("chunk idempotency conflict")
            return {**existing, "replayed": True}
        target = self.upload_root / upload_id / "chunks" / f"{index:08d}.part"
        temporary = target.with_suffix(".part.tmp")
        _retry_transient_file_op(lambda: temporary.write_bytes(data))
        _retry_transient_file_op(lambda: os.replace(temporary, target))
        payload = {
            "upload_id": upload_id,
            "chunk_index": index,
            "bytes": len(data),
            "sha256": digest,
            "path": str(target),
            "created_at": utc_now(),
        }
        self.store.put_upload_chunk(payload)
        return {**payload, "replayed": False}

    def complete_upload(self, upload_id: str) -> dict[str, Any]:
        with self._lock:
            return self._complete_upload_locked(upload_id)

    def _complete_upload_locked(self, upload_id: str) -> dict[str, Any]:
        if not _SAFE_ID.fullmatch(upload_id):
            raise ValueError("upload id is invalid")
        upload = self.store.get_upload(upload_id)
        if not upload:
            raise KeyError(upload_id)
        existing_attachments = [row for row in self.store.list_attachments() if row.get("upload_id") == upload_id]
        if upload["status"] == "complete" and existing_attachments:
            return {"upload": upload, "attachment": existing_attachments[0], "replayed": True}
        chunks = self.store.list_upload_chunks(upload_id)
        expected_count = int(upload["chunk_count"])
        if len(chunks) != expected_count or [int(item["chunk_index"]) for item in chunks] != list(range(expected_count)):
            raise ValueError("upload chunks are incomplete")
        final_dir = self.upload_root / upload_id / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        final_path = final_dir / str(upload["safe_name"])
        temporary = final_path.with_suffix(final_path.suffix + ".tmp")
        digest = hashlib.sha256()
        written = 0
        with temporary.open("wb") as output:
            for item in chunks:
                part = Path(str(item["path"]))
                with part.open("rb") as source:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        output.write(block)
                        digest.update(block)
                        written += len(block)
            output.flush()
            os.fsync(output.fileno())
        actual_sha = digest.hexdigest()
        if written != int(upload["total_bytes"]):
            temporary.unlink(missing_ok=True)
            raise ValueError("final upload byte length mismatch")
        if upload["expected_sha256"] and upload["expected_sha256"] != actual_sha:
            temporary.unlink(missing_ok=True)
            raise ValueError("final upload sha256 mismatch")
        os.replace(temporary, final_path)
        detected_media_type = _media_type(final_path, str(upload["media_type"]))
        upload = self.store.update_upload(
            upload_id,
            status="complete",
            final_path=str(final_path),
            final_sha256=actual_sha,
            media_type=detected_media_type,
        )
        attachment = {
            "id": new_id("attachment"),
            "upload_id": upload_id,
            "run_id": "",
            "name": str(upload["safe_name"]),
            "path": str(final_path),
            "media_type": str(upload["media_type"]),
            "bytes": written,
            "sha256": actual_sha,
            "created_at": utc_now(),
        }
        self.store.put_attachment(attachment)
        return {"upload": upload, "attachment": attachment, "replayed": False}

    def _latest_conversation_attachment_ids(
        self,
        conversation_id: str,
        *,
        exclude_run_id: str = "",
    ) -> tuple[list[str], str]:
        """Return the newest complete attachment set from this conversation only."""
        for previous in self.store.list_assistant_runs(conversation_id, 200):
            previous_run_id = str(previous.get("id") or "")
            if not previous_run_id or previous_run_id == exclude_run_id:
                continue
            candidate_ids = [str(value) for value in (previous.get("attachment_ids") or []) if str(value)][:50]
            if not candidate_ids:
                continue
            valid = True
            for attachment_id in candidate_ids:
                source = self.store.get_attachment(attachment_id)
                if not source:
                    valid = False
                    break
                try:
                    source_path = Path(str(source["path"])).resolve(strict=True)
                except (OSError, RuntimeError):
                    valid = False
                    break
                if (
                    not source_path.is_file()
                    or source_path.is_symlink()
                    or source_path.stat().st_size != int(source["bytes"])
                ):
                    valid = False
                    break
            if valid:
                return candidate_ids, previous_run_id
        return [], ""

    def _bind_attachment_ids(self, run_id: str, root: Path, attachment_ids: list[str]) -> list[dict[str, Any]]:
        bound: list[dict[str, Any]] = []
        for attachment_id in list(dict.fromkeys(attachment_ids))[:50]:
            source = self.store.get_attachment(str(attachment_id))
            if not source:
                raise ValueError(f"attachment not found: {attachment_id}")
            source_path = Path(str(source["path"])).resolve(strict=True)
            target = root / "inputs" / _safe_name(str(source["name"]))
            if target.exists():
                target = target.with_name(f"{target.stem}_{len(bound) + 1}{target.suffix}")
            # Inputs are versioned copies. A hard link lets one Run overwrite
            # the library file (and another Run's input) through the same inode.
            _retry_transient_file_op(lambda: shutil.copy2(source_path, target))
            if _sha256_file(target) != source["sha256"]:
                target.unlink(missing_ok=True)
                raise ValueError("attachment copy verification failed")
            clone = {
                **source, "id": new_id("attachment"), "run_id": run_id,
                "path": str(target), "created_at": utc_now(),
            }
            self.store.put_attachment(clone)
            bound.append(clone)
        return bound

    def repair_run_attachments(self, run_id: str) -> dict[str, Any]:
        """Restore a legacy empty Run from the latest valid attachment set in its conversation."""
        run = self.store.get_assistant_run(run_id)
        if not run:
            raise KeyError(run_id)
        if self.store.list_attachments(run_id):
            return self.snapshot(run_id)
        if self._effective_active_tool_calls(run_id):
            raise ValueError("cannot repair attachments while tools are active")
        attachment_ids, source_run_id = self._latest_conversation_attachment_ids(
            str(run["conversation_id"]),
            exclude_run_id=run_id,
        )
        if not attachment_ids:
            raise ValueError("no prior conversation attachments available")
        root = Path(str(run["task_root"])).resolve(strict=True)
        expected_root = (self.task_root / run_id).resolve(strict=True)
        if root != expected_root:
            raise ValueError("run task root is invalid")
        bound = self._bind_attachment_ids(run_id, root, attachment_ids)
        plan = dict(run.get("plan") or {})
        plan["require_artifact"] = True
        self.store.update_assistant_run(run_id, attachment_ids=[item["id"] for item in bound], plan=plan)
        session = self.store.get_session(str(run["session_id"])) or {}
        metadata = dict(session.get("metadata") or {})
        metadata["attachment_context"] = {"mode": "repaired", "source_run_id": source_run_id}
        self.store.update_session(str(run["session_id"]), metadata_json=metadata)
        self.store.append_event(
            run_id,
            "attachments_inherited",
            {
                "run_id": run_id,
                "status": "completed",
                "attachment_count": len(bound),
                "source_run_id": source_run_id,
                "mode": "repaired",
            },
        )
        return self.snapshot(run_id)

    def create_run(
        self,
        *,
        prompt: str,
        conversation_id: str = "",
        attachment_ids: list[str] | None = None,
        managed_hpc_identity: dict[str, Any] | None = None,
        secret_refs: list[str] | None = None,
        selected_task: str = "",
        start: bool = True,
        _reserved_run_id: str = "",
        project_id: str = "",
        user_task_id: str = "",
        model_profile_id: str = "",
        model_profile_version: int = 0,
    ) -> dict[str, Any]:
        objective = str(prompt or "").strip()
        profile_binding = None
        if model_profile_id:
            from .model_profiles import ModelProfiles
            from .tenant_access import current_principal
            profile_binding = ModelProfiles(self.runtime.runtime_root).binding(current_principal.get(), model_profile_id, model_profile_version)
        if not objective or len(objective) > 20_000:
            raise ValueError("prompt is required and must be at most 20000 characters")
        conversation = str(conversation_id or "conversation_default").strip()
        if not _SAFE_ID.fullmatch(conversation):
            raise ValueError("conversation id is invalid")
        task = str(selected_task or "").strip()
        if not task:
            match = re.match(r"^\[Selected task: ([A-Za-z0-9_.-]{1,160})\]\s*", objective)
            task = match.group(1) if match else ""
        if task and not _SAFE_ID.fullmatch(task):
            raise ValueError("selected task id is invalid")
        if project_id:
            from .tenant_access import current_principal
            from .user_projects import ProjectStore
            ProjectStore(self.runtime.runtime_root).get_owned(current_principal.get(), project_id)
        run_id = _reserved_run_id or new_id("run")
        if not _SAFE_ID.fullmatch(run_id) or self.store.get_session(run_id):
            raise ValueError("reserved Run identity is invalid or already in use")
        root = self.task_root / run_id
        for name in ("inputs", "work", "outputs", "logs", "evidence"):
            (root / name).mkdir(parents=True, exist_ok=True)
        resolved_attachment_ids = list(dict.fromkeys(str(value) for value in (attachment_ids or []) if str(value)))[:50]
        attachment_context = {"mode": "explicit" if resolved_attachment_ids else "none", "source_run_id": ""}
        bound = self._bind_attachment_ids(run_id, root, resolved_attachment_ids)
        artifact_requirements = _artifact_requirements_from_prompt(objective)
        require_artifact = bool(bound) or bool(artifact_requirements["names"] or artifact_requirements["groups"]) or bool(re.search(
            r"(?i)(生成|创建|写|修改|分析|报告|文件|产物|表格|代码|导出|压缩|清洗|可视化|训练|模型|交付|generate|create|write|edit|analy[sz]e|report|file|artifact|code|export|train|model|deliver)",
            objective,
        ))
        plan = {
            "schema": "evomind.assistant.plan.v1",
            "require_artifact": require_artifact,
            "expected_artifacts": artifact_requirements["names"],
            "expected_artifact_groups": artifact_requirements["groups"],
            "steps": [
                {"id": "understand", "label": "理解目标与输入", "status": "pending"},
                {"id": "execute", "label": "调用工具执行任务", "status": "pending"},
                {"id": "verify", "label": "验证结果与文件", "status": "pending"},
                {"id": "deliver", "label": "发布可预览下载产物", "status": "pending"},
            ],
        }
        created = utc_now()
        claimed_secret_refs: list[str] = []
        if secret_refs:
            from .run_secrets import RunSecretStore

            claimed_secret_refs = RunSecretStore(self.root).claim_refs(
                run_id,
                secret_refs,
                managed_hpc_identity=(
                    dict(managed_hpc_identity)
                    if isinstance(managed_hpc_identity, dict)
                    else {}
                ),
            )
        self.runtime.create_session(
            session_id=run_id,
            objective=objective,
            title=objective[:80],
            permission_level=PermissionLevel.WORKSPACE_WRITE.value,
            workspace_root=str(root),
            metadata={
                "managed_hpc_identity": dict(managed_hpc_identity) if isinstance(managed_hpc_identity, dict) else None,
                "run_secret_refs": claimed_secret_refs,
                "selected_task": task,
                "project_id": project_id,
                "user_task_id": user_task_id,
                "model_profile": profile_binding,
                "attachment_context": attachment_context,
            },
        )
        row = self.store.create_assistant_run(
            {
                "id": run_id,
                "session_id": run_id,
                "conversation_id": conversation,
                "prompt": objective,
                "task_root": str(root),
                "status": SessionStatus.QUEUED.value,
                "plan": plan,
                "attachment_ids": [item["id"] for item in bound],
                "retry_count": 0,
                "error_class": "",
                "error_message": "",
                "model_provider": "",
                "model": "",
                "created_at": created,
                "updated_at": created,
                "completed_at": "",
            }
        )
        self.store.append_event(run_id, "run_created", {"run_id": run_id, "status": "queued", "attachment_count": len(bound), "attachment_context": attachment_context, "controlled_secret_count": len(claimed_secret_refs)})
        if attachment_context["mode"] == "inherited":
            self.store.append_event(
                run_id,
                "attachments_inherited",
                {
                    "run_id": run_id,
                    "status": "completed",
                    "attachment_count": len(bound),
                    "source_run_id": attachment_context["source_run_id"],
                    "mode": "inherited",
                },
            )
        self.store.append_event(run_id, "plan_updated", {"run_id": run_id, "plan": plan})
        if start:
            self.start(run_id)
        return self.snapshot(run_id)

    def _set_step(self, run_id: str, step_id: str, status: str, detail: str = "") -> dict[str, Any]:
        run = self.store.get_assistant_run(run_id)
        if not run:
            raise KeyError(run_id)
        plan = dict(run.get("plan") or {})
        steps = []
        for raw in plan.get("steps") or []:
            item = dict(raw)
            if item.get("id") == step_id:
                item["status"] = status
                if detail:
                    item["detail"] = detail[:500]
            steps.append(item)
        plan["steps"] = steps
        self.store.update_assistant_run(run_id, plan=plan)
        self.store.append_event(run_id, "plan_updated", {"run_id": run_id, "plan": plan})
        return plan

    def _conversation_context(self, run: dict[str, Any]) -> str:
        from .tenant_access import AccessStore
        access = AccessStore(self.root)
        owner = access.owner_for("session", str(run["session_id"]))
        rows = self.store.list_assistant_runs(str(run["conversation_id"]), 8)
        history: list[dict[str, str]] = []
        for row in reversed(rows):
            if row["id"] == run["id"]:
                continue
            if access.owner_for("session", str(row["session_id"])) != owner:
                continue
            turns = self.store.list_turns(str(row["session_id"]))
            answer = next((str(item.get("content") or "") for item in reversed(turns) if item.get("role") == "assistant"), "")
            history.append({"user": str(row["prompt"])[:2000], "assistant": answer[:3000]})
        return json.dumps(history[-6:], ensure_ascii=False)

    def _execution_prompt(self, run: dict[str, Any], *, resume: bool) -> str:
        root = Path(str(run["task_root"]))
        attachments = self.store.list_attachments(str(run["id"]))
        session = self.runtime.get_session(str(run["id"])) or {}
        metadata = session.get("metadata") if isinstance(session.get("metadata"), dict) else {}
        selected_task = str(metadata.get("selected_task") or "")
        missing_artifacts = self._missing_expected_artifacts(run) if resume else []
        plan = run.get("plan") if isinstance(run.get("plan"), dict) else {}
        continuation_instructions = [
            str(value).strip()
            for value in (plan.get("continuation_instructions") or [])
            if str(value).strip()
        ][-8:]
        enforced_tool_scope = [
            str(value)
            for value in (metadata.get("run_allowed_tool_names") or [])
            if str(value)
        ]
        attachment_lines = [f"- inputs/{Path(str(item['path'])).name} ({item['bytes']} bytes, sha256={item['sha256']})" for item in attachments]
        output_lines: list[str] = []
        outputs = root / "outputs"
        if outputs.is_dir():
            for item in sorted(outputs.iterdir(), key=lambda candidate: candidate.name.casefold())[:100]:
                if item.is_file() and not item.is_symlink():
                    output_lines.append(f"- outputs/{item.name} ({item.stat().st_size} bytes)")

        continuation_rows: list[dict[str, Any]] = []
        controlled_secret_availability: dict[str, int] = {}
        if resume:
            from .run_secrets import RunSecretStore

            controlled_secret_availability = RunSecretStore(self.root).available_purposes_for_run(
                str(run["id"])
            )
            approvals = {
                str(item.get("tool_call_id") or ""): item
                for item in self.store.list_approvals(limit=2000)
                if str(item.get("session_id") or "") == str(run["id"])
            }
            settled = []
            for item in self.store.list_tool_calls(str(run["id"]), limit=2000):
                status = str(item.get("status") or "")
                approval = approvals.get(str(item.get("id") or "")) or {}
                approval_status = str(approval.get("status") or "")
                if status not in {"running", "waiting_approval"} or (
                    status == "waiting_approval" and approval_status in {"expired", "rejected"}
                ):
                    settled.append(item)
            for item in sorted(settled, key=self._tool_transition_key)[-16:]:
                result = item.get("result") if isinstance(item.get("result"), dict) else {}
                arguments = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
                path_value = str(arguments.get("path") or "")
                path_receipt = ""
                if path_value:
                    try:
                        path_receipt = Path(path_value).resolve(strict=False).relative_to(root.resolve(strict=False)).as_posix()
                    except ValueError:
                        path_receipt = f"outside-current-run:{Path(path_value).name}"
                approval = approvals.get(str(item.get("id") or "")) or {}
                raw_error = str(result.get("error") or "")
                continuation_rows.append(
                    {
                        "tool_call_id": str(item.get("id") or ""),
                        "tool": str(item.get("tool_name") or ""),
                        "status": str(item.get("status") or ""),
                        "approval_status": str(approval.get("status") or "not_required"),
                        "path": path_receipt,
                        "summary": str(result.get("summary") or "")[:240],
                        "error_class": raw_error.split(":", 1)[0][:120] if raw_error else "",
                    }
                )
        return "\n".join(
            [
                "You are executing one durable EvoMind assistant run.",
                "Complete the user's objective, use tools when needed, verify every claimed result, and keep working until the task is genuinely complete.",
                f"Task root: {root}",
                "For every tool path argument, use workspace-relative paths such as inputs/name, work/name, or outputs/name.",
                "Never copy the absolute Task root or any Run ID into a tool argument; the runtime binds relative paths to this exact Run.",
                "Read inputs only from inputs/. Put working files in work/. Put final user-facing files in outputs/.",
                "For every final file call artifact_publish. Do not treat a file write as delivery until it is verified and published.",
                "Workspace-local file writes, commands and tests are allowed. Stop at exact approval for external or irreversible effects.",
                "Hotfix D8: training-run tuning: this host has no GPU and no torch/GPU stack. Never execute training, fine-tuning, inference or heavy data scripts on the local interpreter; run them on the managed HPC allocation through hpc_execute_solution (or managed_tensor_train), which enforces the 5/5 identity gate. A local 'ModuleNotFoundError: No module named torch' is expected and is NOT a valid smoke test - go straight to the managed HPC execution.",
                "Hotfix D8: training-run tuning: keep the per-turn tool surface small. Do not call capability_discover unless you genuinely need capabilities you do not already have: it expands the visible tool list and makes every later turn much slower. Reuse settled receipts instead of repeating the same status reads in one run.",
                "Hotfix D9: long-run tuning: training, inference and preprocessing scripts must use a parallel, cache-aware data path. Decode/resize must run through worker processes or threads (for example torch DataLoader with num_workers >= 8, or a thread/process pool) and must reuse any cached, pre-scaled artifact that already exists under the managed data root. A single-threaded Python loop over hundreds of thousands of images is not acceptable: report the decode throughput (images/second) and the worker count in the run report.",
                "When an action needs approval, call the exact tool with its final arguments so the runtime can create the approval card; do not ask for approval only in prose.",
                "If a tool fails, inspect its error and logs, correct the narrow cause, then retry with the same objective.",
                "For competition or benchmark data, use competition_data_prepare and competition_data_status. Do not invent an ad-hoc downloader, guess a prior Run path, or treat exit code 0 as data readiness.",
                "For the six-source workflow, the fixed IDs are cure_bench, e2lmc, mindgames, ariel_2025, weather4cast, and open_polymer. Call competition_data_status for all six first, then competition_data_prepare only for items that are not ready. Do not call kaggle_list as a prerequisite: the official URLs and adapter slugs are already pinned, and a live listing failure must not block these six adapters. Continue independent sources after one source reaches a human, credential, network, or storage gate. Derive the summary mechanically: FULL_DATA_READY only for 6/6 ready, PARTIAL for 1-5/6 ready, and BLOCKED for 0/6 ready or a failed status audit.",
                "Sources whose access is local_prepared_mlebench (for example histopathologic_cancer, the Kaggle histopathologic-cancer-detection archive) are managed competition sources too: the official archive is already prepared on the shared GPU data root, so call competition_data_status, then competition_data_prepare to publish the persistent pointer (it symlinks $ROOT/prepared and records counts, label rows and a listing hash), and train from the prepared public/ layout. Never route a local_prepared source to the managed Kaggle connector and never report it as unavailable; a missing prepared layout is SOURCE_UNAVAILABLE, not an authorization problem.",
                "Competition credentials must come only from a tenant-bound controlled secret reference or the already-configured server-managed Kaggle connector. Never request or place passwords, tokens, cookies, or access codes in the prompt or tool arguments. The managed Kaggle connector authorizes only the requested read/download action; it never authorizes Join, rule acceptance, account creation, or submission. Weather4cast SFTP still requires its tenant-bound one-time controlled secret.",
                "For a new, replaced, reclaimed, or restored HPC allocation, the secure enrollment surface is Workbench > Data / Kaggle > AIMSLAB 60-second quick replacement. Direct the user to that browser-memory encrypted form and stop at its credential-entry gate. Do not claim the enrollment surface is unavailable merely because no enrollment tool schema is exposed to the model, and never request HPC or proxy credentials in the prompt.",
                "When a selected research task is present, call verified_context before judging whether prior evidence exists; an empty task inputs/ directory is not proof that historical run evidence is absent.",
                f"Resume mode: {str(resume).lower()}",
                "User continuation instructions: " + (
                    json.dumps(continuation_instructions, ensure_ascii=False)
                    if continuation_instructions
                    else "none"
                ),
                "Approval/tool continuation receipt: " + (json.dumps(continuation_rows, ensure_ascii=False) if continuation_rows else "none"),
                "Controlled secret availability (counts only; values and references are never exposed): "
                + (
                    json.dumps(controlled_secret_availability, ensure_ascii=False, sort_keys=True)
                    if controlled_secret_availability
                    else "none"
                ),
                "When the required controlled secret count is positive, call the requested competition tool once; the tool consumes the secret internally. Do not ask the user to provide or repeat it in the prompt.",
                "Treat settled tool receipts as durable evidence. Do not repeat settled external reads, downloads, or submissions unless the user contract explicitly permits a retry and a relevant precondition changed.",
                "A rejected or expired approval is settled negative evidence. Never recreate the same or an equivalent approval unless the latest continuation instruction explicitly asks to retry that exact action.",
                "Continuation precedence: when continuation instructions exist, the latest instruction is the active objective and overrides conflicting earlier objectives, assistant plans, and pending approval requests.",
                "Enforced available tool scope: " + (", ".join(enforced_tool_scope) if enforced_tool_scope else "default prompt-routed tools"),
                "Missing required final artifacts: " + (", ".join(missing_artifacts) if missing_artifacts else "none"),
                "Existing outputs:",
                *(output_lines or ["- none"]),
                f"Selected research task: {selected_task or '(none)'}",
                "Attachments:",
                *(attachment_lines or ["- none"]),
                f"Recent conversation: {self._conversation_context(run)}",
                (
                    "Original user objective (historical context only; overridden wherever it conflicts with the active continuation):"
                    if continuation_instructions
                    else "User objective:"
                ),
                str(run["prompt"]),
                *( ["Active continuation objective:", continuation_instructions[-1]] if continuation_instructions else [] ),
            ]
        )

    def start(self, run_id: str, *, resume: bool = False) -> bool:
        if not _SAFE_ID.fullmatch(run_id):
            raise ValueError("run id is invalid")
        with self._lock:
            if self._stopping.is_set():
                return False
            current = self._threads.get(run_id)
            if current and current.is_alive():
                return False
            run = self.store.get_assistant_run(run_id)
            if not run:
                raise KeyError(run_id)
            if run["status"] in {"completed", "cancelled"} or self.runtime.user_pause_requested(run_id):
                return False
            thread = threading.Thread(
                target=self._run_thread,
                args=(run_id, resume),
                name=f"evomind-assistant-{run_id[-12:]}",
                daemon=True,
            )
            self._threads[run_id] = thread
            thread.start()
            return True

    def _start_or_defer(self, run_id: str, *, resume: bool) -> bool:
        """Resume once the worker that emitted an approval card has unwound."""
        if self.runtime.user_pause_requested(run_id):
            return False
        with self._lock:
            if self._stopping.is_set():
                return False
            current = self._threads.get(run_id)
            if current is None or not current.is_alive():
                immediate = True
            else:
                immediate = False
                waiter = self._resume_waiters.get(run_id)
                if waiter is not None and waiter.is_alive():
                    return False

                def wait_then_start() -> None:
                    try:
                        current.join()
                        if self._stopping.is_set():
                            return
                        run = self.store.get_assistant_run(run_id)
                        if run and run["status"] not in {"completed", "cancelled"}:
                            self.start(run_id, resume=resume)
                    finally:
                        # Keep the waiter registered until its last store read.
                        with self._lock:
                            self._resume_waiters.pop(run_id, None)

                waiter = threading.Thread(
                    target=wait_then_start,
                    name=f"evomind-assistant-resume-{run_id[-12:]}",
                    daemon=True,
                )
                self._resume_waiters[run_id] = waiter
                waiter.start()
        if immediate:
            return self.start(run_id, resume=resume)
        return False

    def _run_thread(self, run_id: str, resume: bool) -> None:
        try:
            self._execute(run_id, resume=resume)
        finally:
            with self._lock:
                self._threads.pop(run_id, None)
            self.runtime.settle_user_pause(run_id)

    def _execute(self, run_id: str, *, resume: bool) -> None:
        run = self.store.get_assistant_run(run_id)
        if not run:
            return
        attempts = int(run.get("retry_count") or 0)
        while attempts < 3:
            if self._stopping.is_set():
                return
            try:
                current_run = self.store.get_assistant_run(run_id)
                if not current_run or current_run["status"] == "cancelled":
                    return
                if self.runtime.user_pause_requested(run_id):
                    self.runtime.settle_user_pause(run_id)
                    return
                self.store.update_assistant_run(run_id, status=SessionStatus.PLANNING.value, error_class="", error_message="")
                self.runtime.store.update_session(run_id, status=SessionStatus.PLANNING.value)
                self.store.append_event(run_id, "step_started", {"run_id": run_id, "step_id": "understand", "label": "理解目标与输入"})
                self._set_step(run_id, "understand", "completed")
                self.store.update_assistant_run(run_id, status=SessionStatus.RUNNING.value)
                self.runtime.store.update_session(run_id, status=SessionStatus.RUNNING.value)
                self.store.append_event(run_id, "step_started", {"run_id": run_id, "step_id": "execute", "label": "调用工具执行任务"})
                result = self.runtime.message(run_id, self._execution_prompt(run, resume=resume or attempts > 0), max_steps=48)
                if self.runtime.user_pause_requested(run_id):
                    self.runtime.settle_user_pause(run_id)
                    return
                current_run = self.store.get_assistant_run(run_id)
                if not current_run or current_run["status"] == "cancelled":
                    return
                model = result.get("model_execution") if isinstance(result.get("model_execution"), dict) else {}
                provider = str(model.get("provider") or result.get("model", {}).get("provider") or "")
                model_name = str(model.get("model") or result.get("model", {}).get("model") or "")
                self.store.update_assistant_run(run_id, model_provider=provider, model=model_name)
                self.store.append_event(
                    run_id,
                    "model_observed",
                    {"run_id": run_id, "provider": provider, "model": model_name, "tool_calls": model.get("native_tool_calls", 0)},
                )
                status = str(result.get("status") or "failed")
                if status == "waiting_approval":
                    self._set_step(run_id, "execute", "waiting_approval")
                    self.store.update_assistant_run(run_id, status=SessionStatus.WAITING_APPROVAL.value)
                    approval = result.get("approval") if isinstance(result.get("approval"), dict) else {}
                    self.store.append_event(run_id, "approval_required", {"run_id": run_id, "approval": approval})
                    return
                if status != "completed":
                    raise RuntimeError(str(result.get("text") or f"runtime returned {status}"))
                self._set_step(run_id, "execute", "completed")
                self.store.update_assistant_run(run_id, status=SessionStatus.VERIFYING.value)
                self.runtime.store.update_session(run_id, status=SessionStatus.VERIFYING.value)
                self.store.append_event(run_id, "step_started", {"run_id": run_id, "step_id": "verify", "label": "验证结果与文件"})
                answer = str(result.get("text") or "").strip()
                if provider and model_name:
                    observed_label = f"{provider} / {model_name}"
                    answer = f"**本轮实际模型**：`{observed_label}`（来自受管 model_observed 事件）\n\n{answer}".strip()
                artifacts = self._finalize_outputs(run_id, answer)
                self._set_step(run_id, "verify", "completed", f"verified {len(artifacts)} artifact(s)")
                self.store.append_event(
                    run_id,
                    "verification_completed",
                    {"run_id": run_id, "artifact_count": len(artifacts), "artifacts": [public_artifact(item) for item in artifacts]},
                )
                self.store.append_event(run_id, "step_started", {"run_id": run_id, "step_id": "deliver", "label": "发布可预览下载产物"})
                self._set_step(run_id, "deliver", "completed")
                self.store.update_assistant_run(
                    run_id,
                    status="completed",
                    completed_at=utc_now(),
                    error_class="",
                    error_message="",
                )
                self.runtime.store.update_session(run_id, status=SessionStatus.COMPLETED.value)
                self.store.append_event(
                    run_id,
                    "run_completed",
                    {"run_id": run_id, "status": "completed", "answer": answer, "artifacts": [public_artifact(item) for item in artifacts]},
                )
                return
            except Exception as exc:
                if self.runtime.user_pause_requested(run_id):
                    self.store.append_event(run_id, "run_paused", {"error_type": type(exc).__name__, "budget_reset": False})
                    self.runtime.settle_user_pause(run_id)
                    return
                attempts += 1
                failure_class, auto_retry = _failure_policy(exc)
                self.store.update_assistant_run(
                    run_id,
                    status=SessionStatus.RECOVERING.value if auto_retry and attempts < 3 else SessionStatus.FAILED.value if failure_class == "terminal" else SessionStatus.BLOCKED.value,
                    retry_count=attempts,
                    error_class=failure_class,
                    error_message=str(exc)[:1000],
                )
                if auto_retry and attempts < 3:
                    delay = 2 ** (attempts - 1)
                    self.store.append_event(
                        run_id,
                        "retry_scheduled",
                        {"run_id": run_id, "attempt": attempts, "delay_seconds": delay, "error_class": failure_class, "error_type": type(exc).__name__},
                    )
                    time.sleep(delay)
                    resume = True
                    run = self.store.get_assistant_run(run_id) or run
                    continue
                terminal = failure_class == "terminal"
                self.runtime.store.update_session(run_id, status=SessionStatus.FAILED.value if terminal else SessionStatus.BLOCKED.value)
                self.store.append_event(
                    run_id,
                    "run_failed" if terminal else "run_blocked",
                    {
                        "run_id": run_id,
                        "status": "failed" if terminal else "blocked",
                        "error_class": failure_class,
                        "error_type": type(exc).__name__,
                        "message": str(exc)[:1000],
                        "retry_count": attempts,
                        "recoverable": not terminal,
                    },
                )
                return

    @staticmethod
    def _tool_transition_key(item: dict[str, Any]) -> tuple[str, str, str]:
        settled_at = str(item.get("completed_at") or item.get("started_at") or item.get("created_at") or "")
        return settled_at, str(item.get("created_at") or ""), str(item.get("id") or "")

    def _effective_active_tool_calls(
        self,
        run_id: str,
        calls: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        history = calls if calls is not None else self.store.list_tool_calls(run_id, limit=2000)
        latest_assistant_turn_at = max(
            (
                str(item.get("created_at") or "")
                for item in self.store.list_turns(run_id)
                if str(item.get("role") or "") == "assistant"
            ),
            default="",
        )
        completed_by_name: dict[str, list[dict[str, Any]]] = {}
        completed_downloads: list[dict[str, Any]] = []
        for item in history:
            if str(item.get("status") or "") != "completed":
                continue
            name = str(item.get("tool_name") or "")
            completed_by_name.setdefault(name, []).append(item)
            if name == "kaggle_download":
                completed_downloads.append(item)

        active: list[dict[str, Any]] = []
        for item in history:
            status = str(item.get("status") or "")
            if status not in {"running", "waiting_approval"}:
                continue
            if status == "waiting_approval":
                approval = self.store.get_approval(str(item.get("approval_id") or ""))
                if approval and str(approval.get("status") or "") in {"expired", "rejected"}:
                    continue
                active.append(item)
                continue
            name = str(item.get("tool_name") or "")
            started_at = str(item.get("started_at") or item.get("created_at") or "")
            # The runtime records an assistant turn only after model execution
            # has returned. A still-running tool row from before that turn is
            # therefore an orphan left by an interrupted persistence update,
            # not live work. A tool started by a later recovery attempt remains
            # active because its timestamp is newer than the persisted turn.
            if latest_assistant_turn_at and latest_assistant_turn_at > started_at:
                continue
            successors = completed_by_name.get(name, [])
            if name == "kaggle_list":
                successors = [*successors, *completed_downloads]
            # A durable running row cannot represent live work after a later
            # successful call has settled.  This occurs when the process that
            # owned the older request exited before persisting its terminal
            # state.  Ignore only success-backed supersession; an unrelated
            # failure never hides a possibly live call.
            if any(str(candidate.get("completed_at") or "") > started_at for candidate in successors):
                continue
            active.append(item)
        return active

    def _latest_tool_blockers(
        self,
        run_id: str,
        *,
        final_answer_ready: bool = False,
    ) -> list[dict[str, Any]]:
        history = self.store.list_tool_calls(run_id, limit=2000)
        latest_assistant_turn_at = max(
            (
                str(item.get("created_at") or "")
                for item in self.store.list_turns(run_id)
                if str(item.get("role") or "") == "assistant"
            ),
            default="",
        )
        run = self.store.get_assistant_run(run_id)
        required_artifacts_satisfied = bool(run) and not self._missing_expected_artifacts(run)
        verification_tools = {
            "artifact_list",
            "artifact_preview",
            "artifact_publish",
            "file_list",
            "file_read",
            # A later completed HPC execution is stronger evidence than an
            # earlier optional local shell preflight.  Keep the shell failure
            # in the durable tool history, but do not relabel a SHA-bound,
            # artifact-producing remote execution as unresolved merely because
            # that preflight failed or its terminal state was persisted late.
            "hpc_execute_solution",
        }
        finalization_orphans: set[str] = set()
        if final_answer_ready and required_artifacts_satisfied and latest_assistant_turn_at:
            completed_artifact_lists = [
                item
                for item in history
                if str(item.get("tool_name") or "") == "artifact_list"
                and str(item.get("status") or "") == "completed"
                and str(item.get("completed_at") or "")
            ]
            for item in history:
                if str(item.get("status") or "") != "running":
                    continue
                started_at = str(item.get("started_at") or item.get("created_at") or "")
                call_id = str(item.get("id") or "")
                # Runtime.message is synchronous: once it has returned a final
                # assistant answer, a running row that predates a completed
                # artifact inventory cannot still represent live work.  The
                # artifact and final-turn checks keep this exception scoped to
                # finalization; normal snapshots and genuinely newer work stay
                # fail-closed.  Use >= because timestamps have millisecond
                # precision and adjacent persistence writes can be equal.
                if call_id and any(
                    started_at <= str(candidate.get("completed_at") or "") <= latest_assistant_turn_at
                    for candidate in completed_artifact_lists
                ):
                    finalization_orphans.add(call_id)
        by_name: dict[str, list[dict[str, Any]]] = {}
        for item in history:
            name = str(item.get("tool_name") or "")
            if name:
                by_name.setdefault(name, []).append(item)
        effective_active_ids = {
            str(item.get("id") or "")
            for item in self._effective_active_tool_calls(run_id, history)
            if str(item.get("id") or "") not in finalization_orphans
        }
        latest: dict[str, dict[str, Any]] = {}
        blockers: list[dict[str, Any]] = []
        for name, calls in by_name.items():
            active = [
                item for item in calls
                if str(item.get("id") or "") in effective_active_ids
            ]
            if active:
                authoritative = max(active, key=self._tool_transition_key)
                latest[name] = authoritative
                blockers.append(authoritative)
                continue
            terminal = [item for item in calls if str(item.get("status") or "") not in {"running", "waiting_approval"}]
            if not terminal:
                continue
            authoritative = max(terminal, key=self._tool_transition_key)
            latest[name] = authoritative
            if str(authoritative.get("status") or "") == "failed":
                result = authoritative.get("result") if isinstance(authoritative.get("result"), dict) else {}
                # A status/read tool can legitimately return a structured
                # negative observation (for example DATA_UNAVAILABLE or a
                # permission/identity gate). The call is settled even though
                # its observed outcome is negative. Keep the error in the
                # durable receipt and final report, but never relabel it as an
                # unresolved/live tool call or repeat it just to clear Harness.
                if (
                    name in _SETTLED_OBSERVATION_TOOLS
                    and bool(str(result.get("error") or "").strip() or str(result.get("summary") or "").strip())
                ):
                    continue
                settled_at = str(authoritative.get("completed_at") or authoritative.get("started_at") or "")
                recovered_after_failure = any(
                    str(candidate.get("status") or "") == "completed"
                    and str(candidate.get("tool_name") or "") in verification_tools
                    and str(candidate.get("completed_at") or "") > settled_at
                    for candidate in history
                )
                # A failed optional/diagnostic call is no longer a completion
                # blocker once a later verification tool succeeds, every
                # required artifact exists, and the model persists a final
                # answer after both. Missing deliverables still fail closed.
                if (
                    required_artifacts_satisfied
                    and recovered_after_failure
                    and latest_assistant_turn_at
                    and latest_assistant_turn_at > settled_at
                ):
                    continue
                blockers.append(authoritative)
        completed_download = str((latest.get("kaggle_download") or {}).get("status") or "") == "completed"
        if completed_download:
            blockers = [item for item in blockers if str(item.get("tool_name") or "") != "kaggle_list"]
        return blockers

    def _substantive_deliverables(self, run_id: str) -> list[dict[str, Any]]:
        return [
            item for item in self.store.list_deliverables(run_id)
            if str(item.get("source_tool_call") or "") != "verified_answer_fallback"
        ]

    @staticmethod
    def _artifact_group_satisfied(group: str, deliverables: list[dict[str, Any]]) -> bool:
        names = [str(item.get("name") or "").casefold() for item in deliverables]
        if group == "model":
            return any(
                Path(name).suffix in {".joblib", ".pkl", ".pt", ".pth", ".onnx"}
                or ("model" in Path(name).stem and Path(name).suffix in {".bin", ".cbm", ".json"})
                for name in names
            )
        if group == "model_or_strategy":
            return AssistantRunService._artifact_group_satisfied("model", deliverables) or any(
                any(token in Path(name).stem.replace("-", "_") for token in ("strategy", "policy"))
                and Path(name).suffix in {".json", ".yaml", ".yml", ".py", ".bin"}
                for name in names
            )
        if group == "environment":
            return any(
                name in {"environment.json", "requirements.txt", "requirements.lock", "training-manifest.json"}
                or name.startswith("requirements.")
                or (
                    "environment" in Path(name).stem.replace("-", "_")
                    and Path(name).suffix in {".json", ".txt", ".lock", ".yaml", ".yml"}
                )
                for name in names
            )
        if group == "experience_board":
            return any("experience" in name and Path(name).suffix in {".json", ".jsonl", ".md"} for name in names)
        if group == "search_graph":
            return any("search_graph" in name.replace("-", "_") and Path(name).suffix == ".json" for name in names)
        if group == "retrospective_memory":
            return any("retrospective_memory" in name.replace("-", "_") and Path(name).suffix == ".json" for name in names)
        if group == "validation_contract":
            return any("validation_contract" in name.replace("-", "_") and Path(name).suffix == ".json" for name in names)
        if group == "claim_audit":
            return any("claim_audit" in name.replace("-", "_") and Path(name).suffix == ".json" for name in names)
        if group == "manifest":
            return any("manifest" in name and Path(name).suffix == ".json" for name in names)
        if group == "complete_bundle":
            return any(Path(name).suffix == ".zip" for name in names)
        if group == "markdown_report":
            return any("report" in name and Path(name).suffix == ".md" for name in names)
        if group == "pdf_report":
            return any("report" in name and Path(name).suffix == ".pdf" for name in names)
        return False

    def _artifact_requirements(self, run: dict[str, Any]) -> dict[str, list[str]]:
        plan = run.get("plan") if isinstance(run.get("plan"), dict) else {}
        names = [str(item) for item in list(plan.get("expected_artifacts") or []) if isinstance(item, str)]
        groups = [str(item) for item in list(plan.get("expected_artifact_groups") or []) if isinstance(item, str)]
        if names or groups:
            return {"names": names, "groups": groups}
        return _artifact_requirements_from_prompt(str(run.get("prompt") or ""))

    def _missing_expected_artifacts(
        self,
        run: dict[str, Any],
        deliverables: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        requirements = self._artifact_requirements(run)
        current = deliverables if deliverables is not None else self.store.list_deliverables(str(run["id"]))
        available = {str(item.get("name") or "").casefold() for item in current}
        missing = [name for name in requirements["names"] if name.casefold() not in available]
        named_requirement_rows = [{"name": name} for name in requirements["names"]]
        missing.extend(
            f"artifact-group:{group}"
            for group in requirements["groups"]
            if not self._artifact_group_satisfied(group, current)
            and not self._artifact_group_satisfied(group, named_requirement_rows)
        )
        return list(dict.fromkeys(missing))

    @staticmethod
    def _project_final_deliverables(
        run: dict[str, Any],
        deliverables: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        latest_by_name: dict[str, dict[str, Any]] = {}
        for item in deliverables:
            name = str(item.get("name") or "").casefold()
            previous = latest_by_name.get(name)
            if previous is None or str(item.get("verified_at") or "") >= str(previous.get("verified_at") or ""):
                latest_by_name[name] = item
        projected = list(latest_by_name.values())
        if str(run.get("status") or "") == "completed" and any(
            not _BLOCKER_ARTIFACT_NAME.search(str(item.get("name") or "")) for item in projected
        ):
            projected = [
                item for item in projected
                if not _BLOCKER_ARTIFACT_NAME.search(str(item.get("name") or ""))
            ]
        return sorted(projected, key=lambda item: str(item.get("verified_at") or ""))

    def _recoverable_false_completion(self, run_id: str) -> bool:
        run = self.store.get_assistant_run(run_id)
        return bool(
            run
            and bool((run.get("plan") or {}).get("require_artifact"))
            and (
                self._latest_tool_blockers(run_id)
                or self._missing_expected_artifacts(run)
            )
        )

    def _remove_fallback_deliverables(self, run_id: str) -> None:
        run = self.store.get_assistant_run(run_id)
        if not run:
            return
        outputs = (Path(str(run["task_root"])).resolve(strict=True) / "outputs").resolve(strict=True)
        for item in self.store.remove_deliverables_by_source(run_id, "verified_answer_fallback"):
            path = Path(str(item.get("path") or "")).resolve(strict=False)
            try:
                path.relative_to(outputs)
            except ValueError:
                continue
            if path.is_file() and not path.is_symlink() and _sha256_file(path) == str(item.get("sha256") or ""):
                path.unlink()

    def _finalize_outputs(self, run_id: str, answer: str) -> list[dict[str, Any]]:
        run = self.store.get_assistant_run(run_id)
        if not run:
            raise KeyError(run_id)
        root = Path(str(run["task_root"])).resolve(strict=True)
        outputs = (root / "outputs").resolve(strict=True)
        current = self.store.list_deliverables(run_id)
        published_paths = {str(Path(str(item["path"])).resolve(strict=False)) for item in current}
        for candidate in sorted(outputs.rglob("*")):
            if (
                ".evomind_previews" in candidate.parts
                or _ARTIFACT_SNAPSHOT_DIR in candidate.parts
                or not candidate.is_file()
                or candidate.is_symlink()
            ):
                continue
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(outputs)
            if str(resolved) in published_paths:
                continue
            self.publish_path(run_id, resolved, source_tool_call="auto_finalize")
        current = self.store.list_deliverables(run_id)
        current = self._prune_legacy_superseded_deliverables(run_id, current)
        if bool((run.get("plan") or {}).get("require_artifact")):
            blockers = self._latest_tool_blockers(run_id, final_answer_ready=bool(answer.strip()))
            if blockers:
                names = ", ".join(sorted({str(item.get("tool_name") or "unknown") for item in blockers}))
                raise RuntimeError(f"required result artifacts unavailable; unresolved tool calls: {names}")
            missing = self._missing_expected_artifacts(run, current)
            if missing:
                raise RuntimeError("required result artifacts unavailable; missing final artifacts: " + ", ".join(missing))
        if not current and bool((run.get("plan") or {}).get("require_artifact")):
            result_path = outputs / "result.md"
            result_path.write_text(
                "# EvoMind Task Result\n\n" + (answer or "Task execution completed; inspect the run evidence for details.") + "\n",
                encoding="utf-8",
            )
            self.publish_path(run_id, result_path, source_tool_call="verified_answer_fallback")
            current = self.store.list_deliverables(run_id)
        for item in current:
            path = Path(str(item["path"]))
            if not path.is_file() or path.is_symlink() or path.stat().st_size != int(item["bytes"]) or _sha256_file(path) != item["sha256"]:
                raise ValueError(f"artifact verification failed: {item['id']}")
        return current

    def _prune_legacy_superseded_deliverables(
        self,
        run_id: str,
        deliverables: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Drop invalid pre-snapshot rows when a later row owns the same live path.

        Older runtimes published mutable output paths directly. A retry could
        overwrite ``training.log`` and leave earlier database rows pointing at
        bytes that no longer existed. Keep the newest row whose hash still
        matches the live file and remove only its invalid same-path siblings.
        """
        by_path: dict[str, list[dict[str, Any]]] = {}
        for item in deliverables:
            key = str(Path(str(item.get("path") or "")).resolve(strict=False))
            by_path.setdefault(key, []).append(item)
        remove_ids: list[str] = []
        for path_text, rows in by_path.items():
            if len(rows) < 2 or _ARTIFACT_SNAPSHOT_DIR in Path(path_text).parts:
                continue
            path = Path(path_text)
            if not path.is_file() or path.is_symlink():
                continue
            size = path.stat().st_size
            digest = _sha256_file(path)
            valid = []
            for item in rows:
                declared_bytes = item.get("bytes")
                if declared_bytes is None:
                    continue
                if int(declared_bytes) == size and str(item.get("sha256") or "") == digest:
                    valid.append(item)
            if not valid:
                continue
            winner = max(valid, key=lambda item: (str(item.get("verified_at") or ""), str(item.get("id") or "")))
            remove_ids.extend(str(item["id"]) for item in rows if item["id"] != winner["id"])
        if remove_ids:
            self.store.remove_deliverables_by_ids(run_id, remove_ids)
            removed = set(remove_ids)
            deliverables = [item for item in deliverables if str(item.get("id") or "") not in removed]
        return deliverables

    def publish_path(self, run_id: str, path: Path, *, name: str = "", source_tool_call: str = "") -> dict[str, Any]:
        run = self.store.get_assistant_run(run_id)
        if not run:
            raise KeyError(run_id)
        root = Path(str(run["task_root"])).resolve(strict=True)
        resolved = path.resolve(strict=True)
        outputs = (root / "outputs").resolve(strict=True)
        resolved.relative_to(outputs)
        if not resolved.is_file() or resolved.is_symlink():
            raise ValueError("deliverable must be a regular file")
        artifact_id = new_id("artifact")
        display_name = _safe_name(name or resolved.name)
        snapshot = _snapshot_artifact_file(
            resolved,
            outputs,
            artifact_id,
            display_name,
        )
        size = snapshot.stat().st_size
        digest = _sha256_file(snapshot)
        media = _media_type(snapshot)
        preview_kind = _preview_kind(snapshot, media)
        if preview_kind == "office":
            _derive_office_preview(snapshot, digest)
        payload = {
            "id": artifact_id,
            "run_id": run_id,
            "session_id": str(run["session_id"]),
            "name": display_name,
            "path": str(snapshot),
            "media_type": media,
            "bytes": size,
            "sha256": digest,
            "source_tool_call": str(source_tool_call or "")[:180],
            "preview_kind": preview_kind,
            "verified_at": utc_now(),
        }
        row = self.store.publish_deliverable(payload)
        if str(row.get("id") or "") != artifact_id:
            snapshot.unlink(missing_ok=True)
        self.store.append_event(run_id, "artifact_published", {"run_id": run_id, "artifact": public_artifact(row)})
        return row

    def artifact(self, artifact_id: str) -> tuple[dict[str, Any], Path]:
        if not _SAFE_ID.fullmatch(artifact_id):
            raise ValueError("artifact id is invalid")
        row = self.store.get_deliverable(artifact_id)
        if not row:
            raise KeyError(artifact_id)
        path = Path(str(row["path"])).resolve(strict=True)
        if not path.is_file() or path.is_symlink() or path.stat().st_size != int(row["bytes"]) or _sha256_file(path) != row["sha256"]:
            raise ValueError("artifact integrity verification failed")
        return public_artifact(row), path

    def artifact_file(self, artifact_id: str, derived: str = "") -> tuple[dict[str, Any], Path]:
        row, path = self.artifact(artifact_id)
        kind = str(derived or "").lower()
        if kind not in {"pdf", "png"}:
            return row, path
        preview = _verified_office_preview(path, str(row["sha256"]))
        key = f"{kind}_path"
        if not preview or not preview.get(key):
            raise KeyError(f"{artifact_id}:{kind}")
        derived_path = Path(str(preview[key])).resolve(strict=True)
        derived_sha = _sha256_file(derived_path)
        return {
            **row,
            "name": f"{Path(str(row['name'])).stem}.preview.{kind}",
            "media_type": "application/pdf" if kind == "pdf" else "image/png",
            "bytes": derived_path.stat().st_size,
            "sha256": derived_sha,
            "derived_from": artifact_id,
        }, derived_path

    def preview(self, artifact_id: str) -> dict[str, Any]:
        row, path = self.artifact(artifact_id)
        kind = str(row["preview_kind"])
        preview: Any = None
        if kind in {"text", "markdown", "structured_text", "table"}:
            preview = path.read_text(encoding="utf-8", errors="replace")[:131_072]
        elif kind == "archive":
            with zipfile.ZipFile(path) as bundle:
                preview = [
                    {"name": item.filename, "bytes": item.file_size}
                    for item in bundle.infolist()[:500]
                    if not item.is_dir() and not Path(item.filename).is_absolute() and ".." not in Path(item.filename).parts
                ]
        return {"artifact": row, "preview": preview}

    def snapshot(self, run_id: str) -> dict[str, Any]:
        run = self.store.get_assistant_run(run_id)
        if not run:
            raise KeyError(run_id)
        session = self.store.get_session(str(run["session_id"])) or {}
        session_metadata = session.get("metadata") if isinstance(session.get("metadata"), dict) else {}
        selected_task = str(session_metadata.get("selected_task") or "")
        active_tools = [
            item for item in self._effective_active_tool_calls(str(run["session_id"]))
            if str(item.get("status") or "") == "running"
        ]
        effective_run = dict(run)
        reconciliation_required = str(run.get("error_class") or "") in {
            "execution_reconciliation_required", "approval_execution_unconfirmed",
        }
        if reconciliation_required and str(run.get('status') or '') != 'cancelled':
            effective_run.update(status="blocked", completed_at="")
        if active_tools and not reconciliation_required and str(run.get("status") or "") not in {"completed", "cancelled"}:
            effective_run.update(status="running", error_class="", error_message="", completed_at="")
        if session_metadata.get("user_pause_requested") and not reconciliation_required and str(run.get('status') or '') not in {'completed', 'cancelled'}:
            worker = self._threads.get(run_id)
            raw_running = any(item["status"] == "running" for item in self.store.list_tool_calls(run_id, limit=2000))
            effective_run.update(status="pausing" if raw_running or (worker and worker.is_alive()) else "paused", completed_at="")
        completion_repair_required = False
        missing_expected_artifacts: list[str] = []
        unresolved_tool_names: list[str] = []
        if not active_tools and str(run.get("status") or "") == "completed":
            missing_expected_artifacts = self._missing_expected_artifacts(run)
            unresolved_tools = self._latest_tool_blockers(run_id)
            unresolved_tool_names = sorted(
                {
                    str(item.get("tool_name") or "unknown")
                    for item in unresolved_tools
                }
            )
            completion_repair_required = bool(
                bool((run.get("plan") or {}).get("require_artifact"))
                and (missing_expected_artifacts or unresolved_tool_names)
            )
            if completion_repair_required:
                details: list[str] = []
                if missing_expected_artifacts:
                    details.append("missing final artifacts: " + ", ".join(missing_expected_artifacts))
                if unresolved_tool_names:
                    details.append("unresolved tool calls: " + ", ".join(unresolved_tool_names))
                effective_run.update(
                    status="blocked",
                    error_class="incomplete_deliverables",
                    error_message="Run completion requires repair; " + "; ".join(details),
                    completed_at="",
                )
        turns = self.store.list_turns(str(run["session_id"]))
        from .assistant_history import project_message_history
        message_history = project_message_history(effective_run, turns)
        partial = self.store.latest_event(str(run['session_id']), 'assistant.stream')
        streaming_text = (partial['payload'].get('text', '') if partial and partial['payload'].get('phase') == 'streaming'
                          and effective_run['status'] in {'running', 'planning', 'verifying'} else None)
        answer = message_history["answer"]
        approvals = [item for item in self.store.list_approvals(limit=200) if item.get("session_id") == run["session_id"]]
        evidence_deliverables = self.store.list_deliverables(run_id)
        projected_deliverables = self._project_final_deliverables(effective_run, evidence_deliverables)
        projected_ids = {str(item.get("id") or "") for item in projected_deliverables}
        attachments = self.store.list_attachments(run_id)
        attachment_context = session_metadata.get("attachment_context")
        if not isinstance(attachment_context, dict):
            attachment_context = {"mode": "explicit" if attachments else "none", "source_run_id": ""}
        from .run_secrets import RunSecretStore

        controlled_secret_availability = RunSecretStore(self.root).available_purposes_for_run(run_id)
        persisted_goal = self.store.get_goal_for_run(run_id)
        goal_projection = None
        if persisted_goal is not None:
            goal_projection = {
                key: persisted_goal.get(key)
                for key in (
                    "id", "run_id", "goal_key", "schema", "allocation", "status",
                    "spec_sha256", "board_sha256", "human_baseline_sha256", "created_at", "updated_at",
                )
            }
        return {
            **effective_run,
            "selected_task": selected_task,
            "project_id": str(session_metadata.get("project_id") or ""),
            "user_task_id": str(session_metadata.get("user_task_id") or ""),
            "model_profile": session_metadata.get("model_profile"),
            "user_pause_requested": bool(session_metadata.get("user_pause_requested")),
            "model_execution_contract": session_metadata.get("model_execution_contract"),
            "answer": answer,
            "streaming_text": streaming_text,
            "answer_is_current": message_history["answer_is_current"],
            "message_history": {key: message_history[key] for key in ("schema", "messages", "active_message_id")},
            "attachments": attachments,
            "attachment_context": attachment_context,
            "artifacts": [public_artifact(item) for item in projected_deliverables],
            "artifact_evidence": [
                public_artifact(item)
                for item in evidence_deliverables
                if str(item.get("id") or "") not in projected_ids
            ],
            "artifact_evidence_count": len(evidence_deliverables),
            "approvals": approvals,
            "controlled_secret_availability": controlled_secret_availability,
            "goal": goal_projection,
            "active_tool_calls": [str(item.get("id") or "") for item in active_tools],
            "completion_repair_required": completion_repair_required,
            "missing_expected_artifacts": missing_expected_artifacts,
            "unresolved_tool_names": unresolved_tool_names,
            "last_event_seq": self.store.latest_event_seq(str(run["session_id"])),
            "terminal": effective_run["status"] in TERMINAL_RUN_STATES,
            "progress": self.runtime.progress.read(run_id, status=str(effective_run["status"])),
        }

    def list_runs(self, conversation_id: str = "", limit: int = 50) -> list[dict[str, Any]]:
        return [self.snapshot(str(item["id"])) for item in self.store.list_assistant_runs(conversation_id, limit)]

    def events(self, run_id: str, after: int = 0) -> list[dict[str, Any]]:
        if not self.store.get_assistant_run(run_id):
            raise KeyError(run_id)
        mapped: list[dict[str, Any]] = []
        aliases = {
            "execution_progress": "run_progress",
            "tool.started": "tool_started",
            "tool.completed": "tool_completed",
            "tool.failed": "tool_completed",
            "approval.requested": "approval_required",
            "model.response": "model_observed",
            "assistant.stream": "assistant_text",
            "goal.created": "goal_created",
            "goal.updated": "goal_updated",
        }
        for item in self.store.list_events(run_id, int(after), 1000):
            event_type = aliases.get(str(item["event_type"]), str(item["event_type"]))
            if event_type not in PUBLIC_EVENT_TYPES:
                continue
            payload = dict(item.get("payload") or {})
            if event_type == 'assistant_text':
                payload = {'run_id': run_id, 'text': str(payload.get('text', '')), 'phase': payload.get('phase')}
            if event_type == "run_progress":
                payload = {"run_id": run_id, "progress": payload.get("progress", payload)}
            elif event_type == "tool_started":
                payload = {"run_id": run_id, "tool": payload.get("tool_name"), "tool_call_id": payload.get("id"), "status": "running"}
            elif event_type == "tool_completed" and str(item["event_type"]).startswith("tool."):
                completed_call = self.store.get_tool_call(str(payload.get("tool_call_id") or "")) or {}
                payload = {
                    "tool": completed_call.get("tool_name"),
                    "run_id": run_id,
                    "tool_call_id": payload.get("tool_call_id"),
                    "status": "completed" if payload.get("ok") else "failed",
                    "summary": payload.get("summary"),
                    "error": payload.get("error"),
                }
            mapped.append({**item, "event_type": event_type, "payload": payload})
        return mapped

    def rebind_managed_hpc_identity(
        self,
        run_id: str,
        managed_hpc_identity: dict[str, Any] | None,
        idempotency_key: str,
    ) -> dict[str, Any]:
        """CAS the fixed terminal Run to the current server-bound allocation.

        This operation changes no Run lifecycle state and performs no network
        or HPC action.  Historical calls, approvals, events, and consumed
        secret tombstones remain immutable.
        """

        from .competition_goal import FIXED_RUN_ID
        from .run_secrets import RunSecretStore

        if run_id != FIXED_RUN_ID:
            raise ValueError("managed_hpc_identity_rebind_fixed_run_required")
        key = str(idempotency_key or "").strip()
        if not _MANAGED_HPC_REBIND_KEY.fullmatch(key):
            raise ValueError("managed_hpc_identity_rebind_idempotency_key_invalid")
        next_identity = _validated_managed_hpc_identity(managed_hpc_identity)
        with self.runtime._invoke_lock:
            run = self.store.get_assistant_run(run_id)
            if not run or str(run.get("session_id") or "") != run_id:
                raise ValueError("managed_hpc_identity_rebind_run_binding_invalid")
            session = self.store.get_session(run_id) or {}
            if (
                str(run.get("status") or "") not in TERMINAL_RUN_STATES
                or str(session.get("status") or "") not in TERMINAL_RUN_STATES
            ):
                raise ValueError("managed_hpc_identity_rebind_run_not_terminal")
            if self._effective_active_tool_calls(run_id):
                raise ValueError("managed_hpc_identity_rebind_active_tool_calls")
            pending = [
                item
                for item in self.store.list_approvals(status="pending", limit=2000)
                if str(item.get("session_id") or "") == run_id
            ]
            if pending:
                raise ValueError("managed_hpc_identity_rebind_pending_approvals")
            secret_store = RunSecretStore(self.root)
            if secret_store.available_purposes_for_run(run_id):
                raise ValueError("managed_hpc_identity_rebind_live_secret")
            metadata = session.get("metadata") if isinstance(session.get("metadata"), dict) else {}
            current_identity = _validated_managed_hpc_identity(metadata.get("managed_hpc_identity"))
            current_sha = _managed_hpc_identity_fingerprint(current_identity)
            next_sha = _managed_hpc_identity_fingerprint(next_identity)
            idempotency_sha = hashlib.sha256(key.encode("utf-8")).hexdigest()
            prior = metadata.get("managed_hpc_identity_rebind")
            if current_identity == next_identity:
                if not isinstance(prior, dict) or (
                    prior.get("schema") != "evomind.managed_hpc_identity_rebind.v1"
                    or prior.get("run_id") != run_id
                    or prior.get("to_identity_sha256") != next_sha
                    or prior.get("idempotency_key_sha256") != idempotency_sha
                    or not _SHA256.fullmatch(str(prior.get("from_identity_sha256") or ""))
                ):
                    raise ValueError("managed_hpc_identity_rebind_idempotency_mismatch")
                migration_record = dict(prior)
            else:
                if (
                    current_identity["tenant_id"] != next_identity["tenant_id"]
                    or current_identity["owner_principal_id"] != next_identity["owner_principal_id"]
                ):
                    raise ValueError("managed_hpc_identity_rebind_principal_mismatch")
                if int(next_identity["allocation_generation"]) <= int(current_identity["allocation_generation"]):
                    raise ValueError("managed_hpc_identity_rebind_generation_not_monotonic")
                migration_record = {
                    "schema": "evomind.managed_hpc_identity_rebind.v1",
                    "run_id": run_id,
                    "from_identity_sha256": current_sha,
                    "to_identity_sha256": next_sha,
                    "idempotency_key_sha256": idempotency_sha,
                }
            result = self.store.rebind_session_managed_hpc_identity(
                run_id,
                expected_identity=current_identity,
                next_identity=next_identity,
                migration_record=migration_record,
                event_payload={
                    "run_id": run_id,
                    "status": "completed",
                    "from_identity_sha256": current_sha,
                    "to_identity_sha256": next_sha,
                    "from_job_id": current_identity["job_id"],
                    "to_job_id": next_identity["job_id"],
                    "from_allocation_generation": current_identity["allocation_generation"],
                    "to_allocation_generation": next_identity["allocation_generation"],
                    "idempotency_key_sha256": idempotency_sha,
                    "run_resumed": False,
                    "hpc_accessed": False,
                },
            )
            snapshot = self.snapshot(run_id)
            snapshot["managed_hpc_identity_rebind"] = {
                "updated": bool(result.get("updated")),
                "from_identity_sha256": migration_record["from_identity_sha256"],
                "to_identity_sha256": next_sha,
                "run_resumed": False,
                "hpc_accessed": False,
            }
            return snapshot

    def action(
        self,
        run_id: str,
        action: str,
        instruction: str = "",
        secret_refs: list[str] | None = None,
        managed_hpc_identity: dict[str, Any] | None = None,
        *,
        idempotency_key: str = "",
    ) -> dict[str, Any]:
        run = self.store.get_assistant_run(run_id)
        if not run:
            raise KeyError(run_id)
        command = str(action or "").lower()
        if command == "repair_attachments":
            return self.repair_run_attachments(run_id)
        if command == "rebind_managed_hpc_identity":
            if str(instruction or "").strip() or secret_refs:
                raise ValueError("managed_hpc_identity_rebind_rejects_instruction_or_secret_refs")
            return self.rebind_managed_hpc_identity(
                run_id,
                managed_hpc_identity,
                idempotency_key,
            )
        if command == "cancel":
            self.runtime.cancel(run_id)
            self.store.update_assistant_run(run_id, status="cancelled", completed_at=utc_now())
            return self.snapshot(run_id)
        if command == "pause":
            if secret_refs or str(instruction or "").strip():
                raise ValueError("pause_action_rejects_instruction_or_secret_refs")
            self.runtime.pause(run_id)
            self.store.append_event(run_id, "run_pause_requested", {"signals_sent": 0, "new_work_blocked": True})
            return self.snapshot(run_id)
        if command == "attach_secret_refs":
            if str(instruction or "").strip():
                raise ValueError("attach_secret_refs_rejects_instruction")
            return self.attach_secret_refs(run_id, secret_refs, managed_hpc_identity)
        if command not in {"retry", "resume"}:
            raise ValueError("unsupported run action")
        continuation = str(instruction or "").strip()
        if len(continuation) > 4_000:
            raise ValueError("run continuation instruction must be at most 4000 characters")
        resume_user_pause = self.runtime.user_pause_requested(run_id)
        if resume_user_pause and any(item["status"] == "running" for item in self.store.list_tool_calls(run_id, limit=2000)):
            return self.snapshot(run_id)
        active_calls = self._effective_active_tool_calls(run_id)
        approved_waiting = [
            (call, approval)
            for call in active_calls
            if call["status"] == "waiting_approval"
            and (approval := self.store.get_approval(str(call.get("approval_id") or "")))
            and approval.get("status") == "approved"
        ]
        if approved_waiting and (continuation or secret_refs or managed_hpc_identity):
            raise ValueError("approved_action_resume_rejects_new_parameters")
        if active_calls and not resume_user_pause and len(approved_waiting) != len(active_calls):
            return self.snapshot(run_id)
        if secret_refs:
            from .run_secrets import RunSecretStore

            session = self.store.get_session(str(run["session_id"])) or {}
            metadata = dict(session.get("metadata") or {})
            claimed = RunSecretStore(self.root).claim_refs(
                run_id,
                secret_refs,
                managed_hpc_identity=dict(metadata.get("managed_hpc_identity") or {}),
            )
            existing_refs = [str(value) for value in (metadata.get("run_secret_refs") or []) if str(value)]
            metadata["run_secret_refs"] = list(dict.fromkeys([*existing_refs, *claimed]))[-4:]
            self.store.update_session(str(run["session_id"]), metadata_json=metadata)
            self.store.append_event(
                run_id,
                "run_secret_refs_added",
                {"run_id": run_id, "status": "completed", "controlled_secret_count": len(claimed)},
            )
        if run["status"] == "completed":
            if not self._recoverable_false_completion(run_id):
                return self.snapshot(run_id)
            self._remove_fallback_deliverables(run_id)
        if continuation:
            plan = dict(run.get("plan") or {})
            instructions = [
                str(value).strip()
                for value in (plan.get("continuation_instructions") or [])
                if str(value).strip()
            ]
            instructions.append(continuation)
            plan["continuation_instructions"] = instructions[-8:]
            continuation_requirements = _artifact_requirements_from_prompt(continuation)
            if continuation_requirements["names"] or continuation_requirements["groups"]:
                history = [
                    dict(value)
                    for value in (plan.get("artifact_requirement_history") or [])
                    if isinstance(value, dict)
                ]
                history.append(
                    {
                        "source": "continuation",
                        "previous_names": list(plan.get("expected_artifacts") or []),
                        "previous_groups": list(plan.get("expected_artifact_groups") or []),
                    }
                )
                plan["artifact_requirement_history"] = history[-8:]
                plan["expected_artifacts"] = continuation_requirements["names"]
                plan["expected_artifact_groups"] = continuation_requirements["groups"]
                plan["require_artifact"] = True
            available_tool_names = {
                str(item.get("name") or "")
                for item in self.runtime.tools()
                if str(item.get("name") or "")
            }
            requested_tool_scope = _explicit_continuation_tool_scope(
                continuation,
                available_tool_names,
            )
            if requested_tool_scope is not None:
                if not requested_tool_scope:
                    raise ValueError("explicit continuation tool scope did not name any available tools")
                enforced_tool_scope = sorted(
                    set(requested_tool_scope) | (_CONTINUATION_SUPPORT_TOOLS & available_tool_names)
                )
                plan["allowed_tool_names"] = enforced_tool_scope
                session = self.store.get_session(str(run["session_id"])) or {}
                metadata = dict(session.get("metadata") or {})
                metadata["run_allowed_tool_names"] = enforced_tool_scope
                self.store.update_session(str(run["session_id"]), metadata_json=metadata)
            self.store.update_assistant_run(run_id, plan=plan)
            self.store.append_event(
                run_id,
                "run_instruction_added",
                {
                    "run_id": run_id,
                    "status": "completed",
                    "instruction_chars": len(continuation),
                },
            )
        with self._lock:
            worker = self._threads.get(run_id)
            if worker is not None and worker.is_alive():
                return self.snapshot(run_id)
            from .aibuild_model_recovery import reconcile_model_failure
            if resume_user_pause:
                self.runtime.clear_user_pause(run_id)
            elif not approved_waiting:
                reconcile_model_failure(self.runtime, run_id, continuation, idempotency_key)
            self.store.update_assistant_run(
                run_id,
                status=SessionStatus.RECOVERING.value,
                retry_count=int(run.get("retry_count") or 0) if resume_user_pause else 0,
                error_class="",
                error_message="",
                completed_at="",
            )
            if approved_waiting:
                # A decision made during a user pause is durable but undispatched.
                # Explicit resume must settle that exact action before the model
                # journal can advance. The existing executor revalidates the
                # fingerprint, expiry and live call state and will not run twice.
                call, approval = approved_waiting[0]
                self._queue_approved_action(approval, call)
            else:
                self.start(run_id, resume=True)
            return self.snapshot(run_id)

    def attach_secret_refs(
        self,
        run_id: str,
        secret_refs: list[str] | None,
        managed_hpc_identity: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Attach fresh opaque references to a pending approval without executing it."""

        refs = list(dict.fromkeys(str(value) for value in (secret_refs or []) if str(value)))
        if len(refs) != 1:
            raise ValueError("attach_secret_refs_requires_one_ref")
        from .run_secrets import RunSecretStore

        with self.runtime._invoke_lock:
            run = self.store.get_assistant_run(run_id)
            if not run or str(run.get("session_id") or "") != run_id:
                raise ValueError("attach_secret_refs_run_binding_invalid")
            session = self.store.get_session(run_id) or {}
            if (
                str(run.get("status") or "") != SessionStatus.WAITING_APPROVAL.value
                or str(session.get("status") or "") != SessionStatus.WAITING_APPROVAL.value
            ):
                raise ValueError("attach_secret_refs_run_not_waiting_approval")
            metadata = dict(session.get("metadata") or {})
            current_identity = metadata.get("managed_hpc_identity")
            if not isinstance(current_identity, dict) or not isinstance(managed_hpc_identity, dict):
                raise ValueError("attach_secret_refs_identity_required")
            if current_identity != managed_hpc_identity:
                raise ValueError("attach_secret_refs_identity_mismatch")
            calls = self._effective_active_tool_calls(run_id)
            if len(calls) != 1 or str(calls[0].get("status") or "") != "waiting_approval":
                raise ValueError("attach_secret_refs_requires_one_pending_approval")
            item = calls[0]
            if str(item.get("tool_name") or "") != "competition_data_accelerate":
                raise ValueError("attach_secret_refs_tool_not_supported")
            arguments = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
            if arguments.get("competition") != "weather4cast":
                raise ValueError("attach_secret_refs_competition_mismatch")
            approval = self.store.get_approval(str(item.get("approval_id") or ""))
            if not approval or str(approval.get("status") or "") != "pending":
                raise ValueError("attach_secret_refs_approval_not_pending")
            try:
                expires_at = datetime.fromisoformat(str(approval.get("expires_at") or "").replace("Z", "+00:00"))
                if expires_at.tzinfo is None:
                    expires_at = expires_at.replace(tzinfo=timezone.utc)
            except ValueError as exc:
                raise ValueError("attach_secret_refs_approval_expiry_invalid") from exc
            if expires_at <= datetime.now(timezone.utc):
                raise ValueError("attach_secret_refs_approval_expired")
            if (
                str(approval.get("session_id") or "") != run_id
                or str(approval.get("tool_call_id") or "") != str(item.get("id") or "")
                or str(approval.get("tool_name") or "") != str(item.get("tool_name") or "")
                or approval.get("normalized_arguments") != arguments
                or str(approval.get("argument_fingerprint") or "") != argument_fingerprint("competition_data_accelerate", arguments)
            ):
                raise ValueError("attach_secret_refs_approval_binding_mismatch")
            secret_store = RunSecretStore(self.root)
            if secret_store.claimed_ref_is_live(
                run_id,
                refs[0],
                managed_hpc_identity=dict(current_identity),
                purpose="weather4cast_sftp",
            ):
                # A response can be lost after the filesystem claim succeeds.
                # Repair only the same Run's local metadata and return the
                # unchanged pending approval; never claim or execute twice.
                if metadata.get("run_secret_refs") != refs:
                    metadata["run_secret_refs"] = refs
                    self.store.update_session(run_id, metadata_json=metadata)
                return self.snapshot(run_id)
            if secret_store.available_purposes_for_run(run_id):
                raise ValueError("attach_secret_refs_live_secret_already_present")
            claimed = secret_store.claim_refs(
                run_id,
                refs,
                managed_hpc_identity=dict(current_identity),
            )
            metadata["run_secret_refs"] = claimed
            self.store.update_session(run_id, metadata_json=metadata)
            self.store.append_event(
                run_id,
                "run_secret_refs_added",
                {"run_id": run_id, "status": "completed", "controlled_secret_count": len(claimed), "attach_only": True},
            )
        return self.snapshot(run_id)

    def _queue_approved_action(self, approval: dict[str, Any], call: dict[str, Any]) -> bool:
        approval_id, run_id = str(approval["id"]), str(approval["session_id"])
        with self._lock:
            if self.runtime.user_pause_requested(run_id):
                return False
            if self._stopping.is_set():
                raise RuntimeError("approval_executor_stopping")
            existing = self._approval_workers.get(approval_id)
            if existing is not None and existing.is_alive():
                return False
            current = self.store.get_assistant_run(run_id)
            if not current or current["status"] in {"completed", "cancelled"}:
                return False
            if (self.store.get_tool_call(str(call["id"])) or {}).get("status") in {"running", "completed", "failed"}:
                return False
            self.store.update_assistant_run(run_id, status=SessionStatus.RECOVERING.value, error_class="", error_message="")
            self.store.append_event(run_id, "approval_execution_queued", {"approval_id": approval_id, "tool_call_id": call["id"], "status": "queued"})
            context = contextvars.copy_context()

            def execute() -> None:
                try:
                    if (self.store.get_assistant_run(run_id) or {}).get("status") == "cancelled":
                        return
                    self.store.update_assistant_run(run_id, status=SessionStatus.RUNNING.value)
                    outcome = self.runtime.invoke_tool(run_id, call["tool_name"], call["arguments"],
                                                       tool_call_id=call["id"], approved_fingerprint=approval["argument_fingerprint"],
                                                       idempotency_key=call.get("idempotency_key", ""))
                    state = str(outcome.get("status", "failed"))
                    current = self.store.get_assistant_run(run_id) or {}
                    if current.get("status") == "cancelled":
                        return
                    if state in {"completed", "failed"}:
                        if not self._stopping.is_set():
                            self.store.update_assistant_run(run_id, status=SessionStatus.RECOVERING.value)
                            self._start_or_defer(run_id, resume=True)
                    elif state == "waiting_approval":
                        self.store.update_assistant_run(run_id, status=SessionStatus.WAITING_APPROVAL.value)
                except Exception as exc:
                    self.store.update_assistant_run(run_id, status=SessionStatus.BLOCKED.value,
                                                    error_class="approval_execution_unconfirmed",
                                                    error_message="Approval is recorded; execution requires identity-bound reconciliation.")
                    self.store.append_event(run_id, "run_blocked", {"run_id": run_id, "status": "blocked", "recoverable": True,
                                                                    "error_class": "approval_execution_unconfirmed", "error_type": type(exc).__name__})
                finally:
                    with self._lock:
                        self._approval_workers.pop(approval_id, None)
                    self.runtime.settle_user_pause(run_id)

            worker = threading.Thread(target=lambda: context.run(execute), name=f"evomind-approval-{approval_id[-12:]}", daemon=True)
            self._approval_workers[approval_id] = worker
            try:
                worker.start()
            except RuntimeError:
                self._approval_workers.pop(approval_id, None)
                raise
            return True

    def decide_approval(self, approval_id: str, approved: bool, note: str = "", *, background: bool = False) -> dict[str, Any]:
        pending = self.store.get_approval(approval_id) or {}
        assistant_run = self.store.get_assistant_run(str(pending.get("session_id") or ""))
        background = bool(background and assistant_run)
        if background and self._stopping.is_set():
            raise RuntimeError("approval_executor_stopping")
        result = self.runtime.decide_approval(approval_id, approved, note, execute=not background)
        approval = result.get("approval") if isinstance(result, dict) else None
        if not isinstance(approval, dict):
            approval = self.store.get_approval(approval_id) or {}
        run_id = str(approval.get("session_id") or "")
        if run_id and self.runtime.user_pause_requested(run_id):
            self.runtime.settle_user_pause(run_id)
            return {**result, "status": "paused", "execution_enqueued": False}
        if run_id and self.store.get_assistant_run(run_id):
            if approval.get("status") == "approved":
                if background:
                    call = result.get("tool_call") or self.store.get_tool_call(str(approval["tool_call_id"]))
                    if not call:
                        raise KeyError(approval["tool_call_id"])
                    queued = self._queue_approved_action(approval, call)
                    result = {**result, "status": "queued" if queued else str(call.get("status") or "approved"), "execution_enqueued": queued}
                else:
                    self.store.update_assistant_run(run_id, status=SessionStatus.RECOVERING.value)
                    self._start_or_defer(run_id, resume=True)
            else:
                status = str(approval.get("status") or "rejected")
                message = (
                    "Approval expired before decision. Resume this Run to request a fresh approval."
                    if status == "expired"
                    else "Approval rejected. Resume this Run when you are ready to request a new approval."
                )
                self.store.update_assistant_run(
                    run_id,
                    status=SessionStatus.BLOCKED.value,
                    error_class=f"approval_{status}",
                    error_message=message,
                )
                self.store.append_event(
                    run_id,
                    "run_blocked",
                    {"run_id": run_id, "status": "blocked", "message": message, "recoverable": True},
                )
        return {"approval": approval, "tool_result": result, "run": self.snapshot(run_id) if run_id and assistant_run else None}

    def recover_incomplete(self) -> list[str]:
        recovered: list[str] = []
        for run in self.store.list_assistant_runs(limit=200):
            status = str(run["status"])
            if status == "blocked" and str(run.get("error_class") or "") in {
                "execution_reconciliation_required", "approval_execution_unconfirmed",
            }:
                continue
            calls = self.store.list_tool_calls(str(run["id"]), limit=2000)
            running_calls = [call for call in calls if call["status"] == "running"]
            if running_calls and status in {"queued", "planning", "running", "verifying", "recovering", "pausing", "paused", SessionStatus.WAITING_APPROVAL.value}:
                with self._lock:
                    local = [self._threads.get(str(run["id"])), *[self._approval_workers.get(str(call.get("approval_id") or "")) for call in running_calls]]
                    locally_owned = any(thread is not None and thread.is_alive() for thread in local)
                if not locally_owned:
                    self.store.update_assistant_run(str(run["id"]), status=SessionStatus.BLOCKED.value,
                                                    error_class="execution_reconciliation_required",
                                                    error_message="A previous execution has no confirmed settlement. Reconcile its identity before resuming.")
                    self.store.append_event(str(run["id"]), "run_blocked", {
                        "run_id": run["id"], "status": "blocked", "recoverable": True,
                        "error_class": "execution_reconciliation_required",
                        "tool_call_ids": [call["id"] for call in running_calls],
                    })
                continue
            if self.runtime.user_pause_requested(str(run["id"])):
                self.runtime.settle_user_pause(str(run["id"]))
                continue
            if status in {"queued", "planning", "running", "verifying", "recovering", "paused", SessionStatus.WAITING_APPROVAL.value} and not any(call["status"] == "running" for call in calls):
                approved_pending = next(((call, self.store.get_approval(str(call.get("approval_id") or ""))) for call in reversed(calls)
                                         if call["status"] == "waiting_approval" and (self.store.get_approval(str(call.get("approval_id") or "")) or {}).get("status") == "approved"), None)
                if approved_pending is not None:
                    call, approval = approved_pending
                    if self._queue_approved_action(approval, call):
                        recovered.append(str(run["id"]))
                    continue
            if status == SessionStatus.WAITING_APPROVAL.value and self.store.pending_tool_call(str(run["id"])):
                continue
            if status in {"queued", "planning", "running", "verifying", "recovering", "paused", SessionStatus.WAITING_APPROVAL.value}:
                self.store.update_assistant_run(str(run["id"]), status=SessionStatus.RECOVERING.value)
                if self.start(str(run["id"]), resume=True):
                    recovered.append(str(run["id"]))
        return recovered

    def wait(self, run_id: str, timeout: float = 30.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            snapshot = self.snapshot(run_id)
            if snapshot["terminal"] or snapshot["status"] == "waiting_approval":
                return snapshot
            time.sleep(0.05)
        return self.snapshot(run_id)
