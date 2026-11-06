from __future__ import annotations

import hashlib
import contextvars
import json
import logging
import mimetypes
import os
import re
import shutil
import struct
import subprocess
import threading
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

from .activity_labels import duration_ms, error_label, step_position, tool_label
from .execution_progress import safe_text
from .models import PermissionLevel, SessionStatus, new_id, utc_now
from .policy import argument_fingerprint
from .experience_entry import PlanDecisions
from .user_memory import UserMemoryStore

if TYPE_CHECKING:
    from .runtime import AgentRuntime

logger = logging.getLogger(__name__)


CHUNK_SIZE = 16 * 1024 * 1024
DEFAULT_MAX_UPLOAD_BYTES = 10 * 1024 * 1024 * 1024
# t238 C7a D1: per-tenant quota and global disk low-water (env 0 = unlimited quota).
DEFAULT_TENANT_QUOTA_BYTES = 5 * 1024 * 1024 * 1024
DEFAULT_TENANT_QUOTA_FILES = 500
DEFAULT_DISK_LOW_WATER_PCT = 15.0
DEFAULT_UPLOAD_ABANDON_TTL_HOURS = 24.0
UPLOAD_GC_INTERVAL_SECONDS = 60.0
# t238 C7b m6: run inputs/ copies of terminal runs older than this are removed (0 = never).
DEFAULT_RUN_INPUT_RETENTION_DAYS = 7.0
RUN_INPUT_GC_INTERVAL_SECONDS = 3600.0
# Uploads with no tenant principal (service token / local operator) share this ledger key, uncapped.
OPERATOR_USAGE_KEY = "__operator__"


class UploadLimitError(ValueError):
    """A user-actionable upload rejection with a stable code and Chinese product copy."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(code)
        self.status = status
        self.code = code
        self.message = message


def _env_number(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, "") or default)
    except ValueError:
        return default
    return value if value >= 0 else default


_NBSP = "\u00a0"  # t238 C7c-7a: number and unit never wrap apart ("5 GiB", "500 个")


def _format_bytes(value: int) -> str:
    size = float(max(0, int(value)))
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            text = f"{int(size)} {unit}" if unit == "B" else f"{size:.1f} {unit}".replace(".0 ", " ")
            return text.replace(" ", _NBSP)
        size /= 1024
    return f"{size:.1f}{_NBSP}TiB"


def _quota_message(max_bytes: int, max_files: int, usage: dict[str, Any], *, adding_bytes: int = 0) -> str:
    """t238 C7c-7a: one sentence per limit that was hit, showing that dimension at its limit."""
    used_bytes, used_files = int(usage.get("bytes") or 0), int(usage.get("files") or 0)
    files_full = max_files > 0 and used_files + 1 > max_files
    bytes_full = max_bytes > 0 and used_bytes + max(0, int(adding_bytes)) > max_bytes
    if files_full and not bytes_full:
        return f"文件数已达上限（已用 {used_files}{_NBSP}个，上限 {max_files}{_NBSP}个）。删掉不用的文件后再上传。"
    return f"存储空间已用完（已用 {_format_bytes(used_bytes)}，上限 {_format_bytes(max_bytes)}）。删掉不用的文件后再上传。"


DISK_LOW_WATER_MESSAGE = "暂时无法上传。请稍后再试，或联系管理员。"
UPLOAD_COMPLETING_MESSAGE = "文件还在处理，稍后点「刷新」。"
# t238 C7b COPY: the single-file hint names only the limit that was hit.
TRAINABILITY_BYTES_MESSAGE = "文件超过训练上限（256\u00a0MiB）。可以直接对话和分析；要训练，先缩小或预处理。"
TRAINABILITY_FILES_MESSAGE = "文件数超过训练上限（256\u00a0个）。可以直接对话和分析；要训练，先缩小或预处理。"
TRAINABILITY_LIMIT_MESSAGE = TRAINABILITY_BYTES_MESSAGE


def _training_input_limits() -> tuple[int, int]:
    """The exact bounds enforced by the training entry (hpc_runtime); never duplicated here."""
    from research_os.hpc_runtime import _MAX_SOLUTION_INPUT_BYTES, _MAX_SOLUTION_INPUT_FILES
    return int(_MAX_SOLUTION_INPUT_FILES), int(_MAX_SOLUTION_INPUT_BYTES)


_EOCD = b"PK\x05\x06"
_ZIP64_LOCATOR = b"PK\x06\x07"
_CENTRAL = b"PK\x01\x02"


def _central_directory_entries(path: Path):
    """Yield (is_dir, uncompressed_size) per central-directory entry without decoding names.

    Streams the directory (no full infolist), tolerates prefixed (SFX) archives and never
    decodes names, so a bad UTF-8 flag cannot raise. Malformed input raises ValueError.
    """
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        end = handle.tell()
        tail_start = max(0, end - (65535 + 22))
        handle.seek(tail_start)
        tail = handle.read()
        at = tail.rfind(_EOCD)
        if at < 0 or len(tail) - at < 22:
            raise ValueError("zip end of central directory not found")
        eocd = tail_start + at
        entries, cd_size = struct.unpack_from("<HI", tail, at + 10)
        (comment_len,) = struct.unpack_from("<H", tail, at + 20)
        if at + 22 + comment_len > len(tail):
            raise ValueError("zip end record is truncated")
        cd_start = eocd - cd_size
        if entries == 0xFFFF or cd_size == 0xFFFFFFFF:
            locator = at - 20
            if locator < 0 or tail[locator:locator + 4] != _ZIP64_LOCATOR:
                raise ValueError("zip64 locator missing")
            (record_offset,) = struct.unpack_from("<Q", tail, locator + 8)
            handle.seek(record_offset)
            record = handle.read(56)
            if record[:4] != b"PK\x06\x06" or len(record) < 56:
                record_pos = tail_start + locator - 56
                handle.seek(max(0, record_pos))
                record = handle.read(56)
                if record[:4] != b"PK\x06\x06":
                    raise ValueError("zip64 end record missing")
                record_offset = record_pos
            entries, cd_size = struct.unpack_from("<Q", record, 32)[0], struct.unpack_from("<Q", record, 40)[0]
            cd_start = record_offset - cd_size
        if cd_start < 0:
            raise ValueError("zip central directory offset invalid")
        handle.seek(cd_start)
        for _ in range(int(entries)):
            header = handle.read(46)
            if len(header) != 46 or header[:4] != _CENTRAL:
                raise ValueError("zip central directory entry invalid")
            size = struct.unpack_from("<I", header, 24)[0]
            name_len, extra_len, comment_len = struct.unpack_from("<HHH", header, 28)
            name = handle.read(name_len)
            extra = handle.read(extra_len)
            handle.seek(comment_len, os.SEEK_CUR)
            if size == 0xFFFFFFFF:
                offset = 0
                while offset + 4 <= len(extra):
                    tag, length = struct.unpack_from("<HH", extra, offset)
                    if tag == 0x0001 and length >= 8:
                        (size,) = struct.unpack_from("<Q", extra, offset + 4)
                        break
                    offset += 4 + length
            yield name.endswith(b"/"), int(size)


def upload_trainability(path: Path, size: int) -> dict[str, Any]:
    """Cheap advisory precheck (size, archive member count via the zip directory; no content parsing).

    Never raises: any archive problem is reported as `archive_unreadable`. The directory scan
    stops after max_files + 1 file entries.
    """
    max_files, max_bytes = _training_input_limits()
    files, total = 1, int(size)
    reasons: list[str] = []
    if path.suffix.lower() == ".zip":
        try:
            members, unpacked = 0, 0
            for is_dir, member_size in _central_directory_entries(path):
                if is_dir:
                    continue
                members += 1
                unpacked += member_size
                if members > max_files:
                    break
            files = members
            total = max(total, unpacked)
        except Exception:
            reasons.append("archive_unreadable")
    if total > max_bytes:
        reasons.append("input_bytes_exceeded")
    if files > max_files:
        reasons.append("input_files_exceeded")
    message = TRAINABILITY_BYTES_MESSAGE if "input_bytes_exceeded" in reasons else TRAINABILITY_FILES_MESSAGE if "input_files_exceeded" in reasons else ""
    return {
        "trainable": not reasons,
        "reasons": reasons,
        "limits": {"max_files": max_files, "max_bytes": max_bytes},
        "bytes": total,
        "files": files,
        "message": message,
    }


def _reflink(source: Path, target: Path) -> None:
    """Copy-on-write clone (Linux FICLONE: btrfs/XFS). Raises OSError where unsupported."""
    import fcntl  # POSIX only; ImportError is handled by the caller like an unsupported volume.
    with source.open("rb") as src, target.open("wb") as dst:
        fcntl.ioctl(dst.fileno(), 0x40049409, src.fileno())  # FICLONE
    shutil.copystat(source, target)


def _is_reparse_point(path: Path) -> bool:
    """A Windows junction / reparse point (FILE_ATTRIBUTE_REPARSE_POINT); False elsewhere."""
    try:
        return bool(getattr(path.lstat(), "st_file_attributes", 0) & 0x400)
    except OSError:
        return False


def _clone_or_copy(source: Path, target: Path) -> str:
    """Materialize an immutable library file as a Run input without sharing a writable inode.

    A hard link is not safe here: Run code (and directory_copy/shell tools) may write inputs/ in
    place, which would rewrite the stored original. A reflink gives a separate inode that shares
    blocks until written; elsewhere (NTFS, ext4, cross-device) this falls back to a full copy.
    """
    try:
        _reflink(source, target)
        return "reflink"
    except (OSError, ImportError, AttributeError):
        target.unlink(missing_ok=True)
    _retry_transient_file_op(lambda: shutil.copy2(source, target))
    return "copy"
PUBLIC_EVENT_TYPES = {
    "memory_saved",
    "plan_proposed",
    "project_files_unreadable",
    "run_progress",
    "research_stage",
    "knowledge_retrieved",
    "research_outcome",
    "run_created",
    "plan_updated",
    "step_started",
    "step_completed",
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
    "user_steered",
    "reasoning_summary",
    "goal_created",
    "goal_updated",
    "managed_hpc_identity.rebound",
}
ACTIVITY_LIMIT = 20  # t237 C6a: snapshot timeline rows (most recent)
MEMORY_PROMPT_TOP_K = 20  # t237 C3.4 / C6c: live LTM rows injected per prompt (pinned, then most related)
TERMINAL_RUN_STATES = {"completed", "blocked", "failed", "cancelled"}
_SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]{1,180}$")
# Exactly this module's per-claim tmp: `.<new_id("claim")>.tmp` (new_id = prefix + uuid4().hex), 43 bytes
# whatever the filename (t238 C7b N4). A leading dot is never a `_safe_name` output (it strips "."),
# so the tmp cannot be a committed final's name.
_CLAIM_TMP = re.compile(r"\.(claim_[a-f0-9]{32})\.tmp")
# C7a's legacy shape `<safe_name>.<claim>.tmp`; still swept so orphans from that release are removed.
_LEGACY_CLAIM_TMP = re.compile(r"(.+)\.(claim_[a-f0-9]{32})\.tmp")
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
# SPEC §4.4: unregistered files auto-published from outputs/final/ only when their type is a real deliverable.
_AUTO_FINALIZE_SUFFIXES = frozenset({
    ".md", ".txt", ".csv", ".json", ".jsonl", ".xlsx", ".docx", ".pptx", ".pdf", ".py", ".ipynb", ".zip", ".html",
    ".png", ".jpg", ".jpeg", ".svg", ".parquet", ".joblib", ".pkl", ".pt", ".pth", ".onnx", ".yaml", ".yml",
})
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


DELIVERY_MODES = ("inline", "inline+file", "file_only")
# SPEC §4.2 (C6-answer): a file is owed only on an explicit file ask. A bare 写/分析/模型/代码/报告 does not count,
# and neither does a bound attachment or library file — those are inputs, not an output intent.
_FILE_NOUN = r"(?:文件|文档|表格|报告|PDF|Word|Excel|PPT|幻灯片|CSV|JSON|markdown|md|zip|代码文件)"
_FILE_INTENT_ZH = re.compile(
    r"(?i)(?:保存|导出|下载|生成|输出|写成|整理成|做成|另存)\s*(?:为|成|到)?\s*[^，。,.!?！？\n]{0,12}?" + _FILE_NOUN
)
_FILE_INTENT_EN = re.compile(r"(?i)\b(?:save|export|download|give me|produce)\b.{0,20}?\b(?:file|pdf|docx|xlsx|pptx|csv|zip)\b")
_OUTPUT_FILENAME = re.compile(
    r"(?i)(?:输出|生成|保存|写入|导出|存为|另存|output|save|write|export)[^。！？!?\n]{0,24}?"
    r"([A-Za-z0-9_\-㐀-鿿][\w.\-]*\.(?:md|csv|xlsx|docx|pptx|pdf|py|json|zip))\b"
)
_DELIVERY_MARKER = re.compile(r"(?i)交付文件|最终产物|发布|交付|\bpublish\b|\bdeliver(?:ables?)?\b")
_QUESTION = re.compile(r"(?i)[?？]|吗|是什么|怎么|为什么|如何|\b(?:what|how|why)\b")
# 「不要生成文件」/ "don't create a file" is the opposite of a file ask.
_NO_FILE = re.compile(
    r"(?i)(?:不要|不用|无需|不需要|别)\s*(?:再)?(?:生成|创建|输出|保存|导出|写成|做成)?\s*(?:任何|一个)?" + _FILE_NOUN
    + r"|\b(?:no|without(?: a)?|don'?t (?:create|make|generate|save)(?: a| any)?)\s+files?\b"
)
_FILE_ONLY = re.compile(r"(?i)(?:只|仅)(?:给我|要|需要|输出|导出)?\s*(?:一个)?(?:文件|附件)|不用(?:解释|说明)|\b(?:just|only)\s+(?:give me\s+)?the\s+file\b")


def classify_delivery_intent(objective: str, attachments: Any = None) -> dict[str, Any]:
    """SPEC §4.2: decide whether a run owes a file. ``attachments`` are inputs and never decide it."""
    del attachments
    requirements = _artifact_requirements_from_prompt(str(objective or ""))
    text = _NO_FILE.sub(" ", str(objective or ""))
    explicit = bool(_FILE_INTENT_ZH.search(text) or _FILE_INTENT_EN.search(text) or _OUTPUT_FILENAME.search(text))
    marker = bool(requirements["names"] or requirements["groups"]) or (
        bool(_DELIVERY_MARKER.search(text)) and not _QUESTION.search(text)
    )
    require = explicit or marker
    mode = "inline"
    if require:
        mode = "file_only" if _FILE_ONLY.search(text) else "inline+file"
    # Names the user asked to save are the only outputs/ files auto-finalize may publish outside outputs/final/.
    requested = list(dict.fromkeys(match.group(1) for match in _OUTPUT_FILENAME.finditer(text))) if require else []
    return {"require_artifact": require, "delivery_mode": mode, "requested_files": requested}


def _delivery_mode(plan: dict[str, Any], deliverables: list[dict[str, Any]]) -> str:
    """SPEC §4.1: the declared mode, raised to inline+file once a file was actually delivered (§4.2 rule 4)."""
    declared = str(plan.get("delivery_mode") or "")
    if declared not in DELIVERY_MODES:
        declared = "inline"
    if deliverables and declared == "inline":
        return "inline+file"
    return declared


# Accepts the legacy internal wording too (stored answers); new answers say 「来自模型输出」 (t238 C7b).
_MODEL_PREFIX = re.compile(r"\*\*本轮实际模型\*\*：`[^`]*`（来自(?:受管 model_observed 事件|模型输出)）\s*")


def _visible_answer(answer: str) -> bool:
    return bool(_MODEL_PREFIX.sub("", str(answer or "")).strip())


def _files_summary(deliverables: list[dict[str, Any]]) -> str:
    names = [str(item.get("name") or "") for item in deliverables if str(item.get("name") or "")]
    shown = "、".join(f"`{name}`" for name in names[:3])
    more = f" 等 {len(names)} 个文件" if len(names) > 3 else ""
    return f"已生成 {shown}{more}。"


_MD_LINK = re.compile(r"(?<!!)\[([^\]\n]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_DOWNLOAD_TEXT = re.compile(r"(?i)^\s*(?:click\s+(?:here\s+)?(?:to\s+)?)?download\b|^\s*(?:点击|立即)?下载")
_DELIVERY_URL = re.compile(r"(?i)/api/assistant/artifacts/|^sandbox:|^(?:\./)?outputs/|[?&]download=1")
_DELIVERY_LABEL = re.compile(r"(?i)^\s*(?:[-*+>]\s*)*(?:download|下载)\s*[:：]?\s*$")
_ENGLISH_PUBLISH = re.compile(
    r"(?i)^[^\u4e00-\u9fff]*\b(?:verified|validated|checked)\b[^\u4e00-\u9fff]{0,40}\b(?:published|delivered|uploaded)\b[^\u4e00-\u9fff]*$"
)
# t237 C6d-pin: plain-text variants the live file answers still produced (VERIFY DL_LINK_RE / EN_PUBLISH_RE).
_DOWNLOAD_WORD = re.compile(r"(?i)\bdownload\b|下载")
_RAW_DELIVERY_URL = re.compile(
    r"(?i)(?:https?://[^\s()（）]*?)?/api/assistant/artifacts/[^\s()（）\[\]，。；]*|sandbox:/[^\s()（）\[\]，。；]*"
    r"|[^\s()（）\[\]，。；]*[?&]download=1[^\s()（）\[\]，。；]*"
)
_URL_LABEL_ONLY = re.compile(r"(?i)^(?:文件|链接|地址|下载|下载链接|下载地址|文件链接|文件地址|download|link|file|url)?$")
_LEADING_DOWNLOAD = re.compile(r"(?i)^(\s*(?:[-*+>]\s+)*)(?:[^\w\s\u4e00-\u9fff]+\s*)*download\b[*_`]*\s*[:：]?\s*")
_CJK = re.compile(r"[\u4e00-\u9fff]")
_PUBLISH_PHRASE = re.compile(
    r"(?i)(已\s*|\s+(?:(?:has|have)\s+been|was|were|is|are)\s+)?\b(?:verified|validated|checked)\s+and\s+(?:published|delivered|uploaded)\b"
)


def _publish_to_chinese(match: re.Match[str]) -> str:
    lead = match.group(1) or ""
    return " 已生成" if lead[:1].isspace() else "已生成"


def strip_delivery_boilerplate(answer: str, deliverables: list[dict[str, Any]]) -> str:
    """t237 C6d (d): the file chip is the only download affordance once a file is delivered.

    Drops Markdown links that point at the run's own files (artifact / download / outputs URLs, or a link to a
    delivered file name) or whose text mentions Download/下载, plus lines that are only English publish-verify boilerplate.
    C6d-pin: also drops raw artifact / sandbox: / ?download=1 URLs and plain-text "Download …" lines, and turns English
    verify-and-publish wording inside a Chinese line into 已生成. Links to anything else are user content and keep their
    text. No deliverables ⇒ the answer is returned unchanged.
    """
    text = str(answer or "")
    if not deliverables:
        return text
    names = {str(item.get("name") or "").casefold() for item in deliverables} - {""}
    ids = [str(item.get("id") or "") for item in deliverables if str(item.get("id") or "")]

    def delivery_link(match: re.Match[str]) -> bool:
        label, url = match.group(1), match.group(2)
        basename = url.split("?", 1)[0].rstrip("/").rsplit("/", 1)[-1].casefold()
        return bool(
            _DOWNLOAD_WORD.search(label)
            or _DELIVERY_URL.search(url)
            or any(artifact_id in url for artifact_id in ids)
            or (basename in names and not url.lower().startswith(("http://", "https://")))
        )

    kept: list[str] = []
    for line in text.split("\n"):
        links = list(_MD_LINK.finditer(line))
        if links and any(delivery_link(match) for match in links):
            rest = _MD_LINK.sub(lambda m: "" if delivery_link(m) else m.group(0), line)
            if _DELIVERY_LABEL.match(rest) or not re.sub(r"[\s\-*+>:：|·•,，.。`]", "", rest):
                continue  # the line was only the download affordance
            label_kept = _MD_LINK.sub(lambda m: (m.group(1) if not _DOWNLOAD_TEXT.search(m.group(1)) else "") if delivery_link(m) else m.group(0), line)
            line = label_kept
        bare = _RAW_DELIVERY_URL.sub("", line)
        if bare != line:
            bare = re.sub(r"[（(]\s*[)）]", "", bare)
            if _URL_LABEL_ONLY.match(re.sub(r"[\s\-*+>:：|·•,，.。`]", "", bare)):
                continue  # the line only carried a raw artifact / sandbox / download URL
            line = bare
        lead = _LEADING_DOWNLOAD.match(line)
        if lead:
            if not _CJK.search(line):
                continue  # an English "Download …" line: the file card is the download entry
            line = lead.group(1) + line[lead.end():]
        if _CJK.search(line):
            line = _PUBLISH_PHRASE.sub(_publish_to_chinese, line)
        if _ENGLISH_PUBLISH.match(line.strip().strip("`*_")):
            continue
        kept.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


def _with_answer(history: dict[str, Any], answer: str) -> dict[str, Any]:
    active = str(history.get("active_message_id") or "")
    messages = [dict(item) for item in history.get("messages") or []]
    target = next((item for item in messages if str(item.get("id") or "") == active and item.get("role") == "assistant"), None)
    if target is None:
        target = {"id": active, "role": "assistant", "content": "", "kind": "stored_reply"}
        messages.append(target)
    prefix = _MODEL_PREFIX.match(str(target.get("content") or ""))
    target["content"] = (prefix.group(0) if prefix else "") + answer
    return {**history, "messages": messages, "answer": target["content"], "answer_is_current": True}


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


def _parse_utc(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _newest_utc(values: list[Any]) -> datetime | None:
    parsed = [item for item in (_parse_utc(value) for value in values) if item is not None]
    return max(parsed) if parsed else None


def _planning_steps_finished(plan: Any) -> bool:
    """True only when every step is a completed object. Malformed plans are unfinished."""
    if not isinstance(plan, dict):
        return False
    steps = plan.get("steps")
    if not isinstance(steps, list) or not steps:
        return False
    for step in steps:
        if not isinstance(step, dict) or str(step.get("status") or "") != "completed":
            return False
    return True


def _automatic_retry_enabled() -> bool:
    return os.environ.get("EVOMIND_ASSISTANT_AUTO_RETRY", "").strip().casefold() in {"1", "true", "yes", "on"}


def _public_failover(value: Any) -> dict[str, Any] | None:
    """t238 C7e A1: only the fields the UI shows; never hosts, keys or raw errors."""
    if not isinstance(value, dict) or not value.get("notice"):
        return None
    return {"notice": str(value["notice"])[:120], "label": str(value.get("label") or "")[:60], "explicit": bool(value.get("explicit"))}


def _model_error(run: dict[str, Any], answer: str, metadata: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """t238 C7e A2: the failure kind for a run blocked on its model (rate_limited / unavailable / timeout / auth / ...)."""
    text = f"{answer or ''}\n{run.get('error_message') or ''}"
    match = re.search(r"LLMError/([A-Za-z0-9_]{1,80})", text)
    if run.get("status") not in {"blocked", "failed"} or not match:
        return None
    from .llm_backends import category_for_code
    metadata = metadata or {}
    explicit = bool(metadata.get("platform_model_id") or metadata.get("model_profile"))
    return {"kind": category_for_code(match.group(1)), "code": match.group(1), "explicit": explicit,
            "platform_model_id": str(metadata.get("platform_model_id") or "") or None}


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


SAFE_NAME_MAX_BYTES = 180


def _truncate_utf8(value: str, max_bytes: int) -> str:
    """Longest prefix of `value` whose UTF-8 encoding fits `max_bytes` (never splits a character)."""
    return value.encode("utf-8")[:max(0, max_bytes)].decode("utf-8", errors="ignore")


def _safe_name(value: str) -> str:
    """A filesystem-safe display name of at most SAFE_NAME_MAX_BYTES UTF-8 bytes (t238 C7b N4).

    NAME_MAX (ext4) is 255 bytes and NTFS 255 UTF-16 units; a CJK character is 3 UTF-8 bytes,
    so the cap is in bytes. A truncated name keeps its extension and gains `_<8 hex>` of the
    full name, so two long names sharing a prefix never map to the same name.
    """
    name = Path(str(value or "attachment.bin").replace("\x00", "")).name.strip()
    name = re.sub(r"[^A-Za-z0-9._()\-\u3400-\u9fff]+", "_", name).strip(" ._") or "attachment.bin"
    if len(name.encode("utf-8")) <= SAFE_NAME_MAX_BYTES:
        return name
    tag = "_" + hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
    suffix = Path(name).suffix
    if not suffix or len(suffix.encode("utf-8")) > 20 or suffix == name:
        suffix = ""
    stem = name[: len(name) - len(suffix)] if suffix else name
    budget = SAFE_NAME_MAX_BYTES - len(tag) - len(suffix.encode("utf-8"))
    stem = _truncate_utf8(stem, budget).rstrip(" ._") or "attachment"
    return stem + tag + suffix


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


def _media_type(path: Path, declared: str = "", name: str = "") -> str:
    """Derive the served type from bytes; names and browser declarations are hints only.

    `name` overrides the file name used for suffix hints (e.g. sniffing a tmp file).
    """
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
        except (OSError, zipfile.BadZipFile, ValueError, RuntimeError, NotImplementedError, EOFError):
            # ValueError covers UnicodeDecodeError from a UTF-8-flagged entry with an invalid name.
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
    suffix = Path(name).suffix.lower() if name else path.suffix.lower()
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
        # Injectable for tests: returns an object with .total and .free (shutil.disk_usage shape).
        self.disk_usage = shutil.disk_usage
        self._last_upload_gc = 0.0
        self._last_run_input_gc = float("-inf")
        self._last_chunk_low_water = float("-inf")
        self._last_chunk_free: int | None = None
        self._chunk_bytes_since_check = 0
        self._chunk_cleanup_pending: set[str] = set()
        # Run supervision v1 (dsh-superagent port): slots, attempt ledger, start priority and recovery reason.
        from .run_supervision import RunLedger, RunSlots
        self.slots = RunSlots()
        self.ledger = RunLedger(runtime.super_agent.store)
        self._start_priority: dict[str, str] = {}
        self._resume_reasons: dict[str, str] = {}
        self._recover_uploads()

    def _recover_uploads(self) -> None:
        """Service start: un-stick crashed completes, sweep abandoned uploads, rebuild the quota cache."""
        try:
            for upload_id in self.store.reset_completing_uploads():
                logger.warning("upload %s was completing at shutdown; reset to uploading", upload_id)
            self._backfill_upload_reservations()
            self._finish_crashed_deletes()
            self.gc_abandoned_uploads(sweep_chunks=True)
            self.store.reconcile_tenant_usage()
        except Exception:
            logger.exception("upload recovery failed at service start")
        self._maybe_gc_run_inputs()

    def _finish_crashed_deletes(self) -> None:
        """t238 C7b D6: an upload left in `deleting` (crash between file and row removal) is finished."""
        from .user_data import finish_delete
        for upload in self.store.list_uploads_with_status("deleting"):
            upload_id = str(upload["id"])
            if _SAFE_ID.fullmatch(upload_id):
                try:
                    finish_delete(self.runtime, upload_id, actor="recovery", service=self)
                except UploadLimitError:
                    logger.warning("upload %s: interrupted delete refused, a non-terminal run uses it", upload_id)
                    continue
                logger.warning("upload %s: finished an interrupted delete", upload_id)
        from .user_data import forget_deleted_owner_rows
        if dropped := forget_deleted_owner_rows(self.runtime):
            logger.warning("dropped %d owner rows of deleted uploads", dropped)

    def _backfill_upload_reservations(self) -> None:
        legacy = self.store.uploads_without_reservation()
        if not legacy:
            return
        from .tenant_access import AccessStore
        owners = AccessStore(self.root).owners_by_kind("upload")
        for upload in legacy:
            owner = owners.get(str(upload["id"]))
            key = owner.tenant_id if owner is not None and owner.tenant_id != "local" else OPERATOR_USAGE_KEY
            self.store.add_upload_reservation(str(upload["id"]), key, int(upload["total_bytes"]))

    @staticmethod
    def _quota_limits(tenant_key: str) -> tuple[int, int]:
        if tenant_key == OPERATOR_USAGE_KEY:
            return 0, 0
        return (
            int(_env_number("EVOMIND_TENANT_QUOTA_BYTES", DEFAULT_TENANT_QUOTA_BYTES)),
            int(_env_number("EVOMIND_TENANT_QUOTA_FILES", DEFAULT_TENANT_QUOTA_FILES)),
        )

    def _check_disk_low_water(self) -> None:
        percent = min(100.0, _env_number("EVOMIND_DISK_LOW_WATER_PCT", DEFAULT_DISK_LOW_WATER_PCT))
        if percent <= 0:
            return
        usage = self.disk_usage(self.upload_root)
        total = int(getattr(usage, "total", 0) or 0)
        if total > 0 and int(usage.free) * 100 < percent * total:
            logger.warning("upload rejected: uploads volume free %s of %s is below %.1f%%", usage.free, total, percent)
            raise UploadLimitError(413, "disk_low_water", DISK_LOW_WATER_MESSAGE)

    # t238 C7b m3: put_chunk re-checks the low-water at most this often (or sooner when free space
    # falls below the chunk about to land); complete_upload always checks before assembling.
    LOW_WATER_CHUNK_INTERVAL_SECONDS = 10.0
    COMPLETE_SAFETY_MARGIN_BYTES = 64 * 1024 * 1024

    def _check_chunk_low_water(self, chunk_bytes: int) -> None:
        now = time.monotonic()
        cached = self._last_chunk_free
        due = now - self._last_chunk_low_water >= self.LOW_WATER_CHUNK_INTERVAL_SECONDS
        if not due and cached is not None and cached - chunk_bytes >= 0 and self._chunk_bytes_since_check + chunk_bytes <= cached:
            self._chunk_bytes_since_check += chunk_bytes
            return
        self._last_chunk_low_water, self._chunk_bytes_since_check = now, chunk_bytes
        usage = self.disk_usage(self.upload_root)
        self._last_chunk_free = int(usage.free)
        self._check_disk_low_water()
        if int(usage.free) < chunk_bytes:
            raise UploadLimitError(413, "disk_low_water", DISK_LOW_WATER_MESSAGE)

    def _check_complete_free_space(self, total_bytes: int) -> None:
        self._check_disk_low_water()
        if int(self.disk_usage(self.upload_root).free) < int(total_bytes) + self.COMPLETE_SAFETY_MARGIN_BYTES:
            logger.warning("upload complete rejected: free space is below final size + margin")
            raise UploadLimitError(413, "disk_low_water", DISK_LOW_WATER_MESSAGE)

    def gc_abandoned_uploads(self, *, now: datetime | None = None, sweep_chunks: bool = False) -> list[str]:
        """Delete incomplete uploads idle past the TTL (files + rows + reservation)."""
        hours = _env_number("EVOMIND_UPLOAD_ABANDON_TTL_HOURS", DEFAULT_UPLOAD_ABANDON_TTL_HOURS)
        moment = now or datetime.now(timezone.utc)
        cutoff = datetime.fromtimestamp(moment.timestamp() - hours * 3600, timezone.utc).isoformat(timespec="milliseconds")
        removed: list[str] = []
        for upload in self.store.list_stale_uploads(cutoff):
            upload_id = str(upload["id"])
            if not _SAFE_ID.fullmatch(upload_id):
                continue
            with self._lock:
                if not self.store.delete_abandoned_upload(upload_id, cutoff):
                    continue
            shutil.rmtree(self.upload_root / upload_id, ignore_errors=True)
            removed.append(upload_id)
        if removed:
            logger.info("removed %d abandoned uploads", len(removed))
        self._sweep_orphan_claim_tmps()
        pending = set(self._chunk_cleanup_pending)
        if sweep_chunks:
            pending |= {path.parent.name for path in self.upload_root.glob("*/chunks") if (path.parent / "final").is_dir()}
        for upload_id in pending:
            upload = self.store.get_upload(upload_id) if _SAFE_ID.fullmatch(upload_id) else None
            if upload and upload["status"] == "complete":
                self._remove_chunk_files(upload_id)
            else:
                self._chunk_cleanup_pending.discard(upload_id)
        return removed

    def _sweep_orphan_claim_tmps(self) -> int:
        """Remove `final/.claim_<32hex>.tmp` (and C7a's `<safe_name>.claim_<32hex>.tmp`) orphans.

        Only this module's own tmp shapes for a known upload are ever deleted: a legacy tmp's
        prefix must be the upload's safe_name, and the committed final (`safe_name`,
        `final_path`) and any path an attachment row references are always kept, whatever
        their names look like. The file list is taken first: a tmp named for claim X only
        exists after X was set, so if X is not the current claim at check time it can never
        become current again.
        """
        removed = 0
        for path in list(self.upload_root.glob("*/final/*.claim_*.tmp")):
            upload_id = path.parent.parent.name
            if not _SAFE_ID.fullmatch(upload_id):
                continue
            current = _CLAIM_TMP.fullmatch(path.name)
            legacy = None if current else _LEGACY_CLAIM_TMP.fullmatch(path.name)
            if not current and not legacy:
                continue
            with self._lock:
                upload = self.store.get_upload(upload_id)
                if not upload or path.name == str(upload["safe_name"]):
                    continue
                if legacy and legacy.group(1) != str(upload["safe_name"]):
                    continue
                claim = (current or legacy).group(1 if current else 2)
                if upload["status"] == "completing" and upload.get("claim_id") == claim:
                    continue
                kept = {str(upload.get("final_path") or "")} | {
                    str(row.get("path") or "") for row in self.store.list_attachments_by_upload_id(upload_id)
                }
                if str(path) in kept or any(value and Path(value) == path for value in kept):
                    continue
                try:
                    path.unlink(missing_ok=True)
                    removed += 1
                except OSError:
                    logger.warning("could not remove orphan upload tmp %s; will retry at GC", path.name)
        if removed:
            logger.info("removed %d orphan upload tmp files", removed)
        return removed

    def _remove_chunk_files(self, upload_id: str) -> bool:
        """Chunks are redundant once the verified final is committed; a failure is retried at GC."""
        chunks = self.upload_root / upload_id / "chunks"
        try:
            if chunks.exists():
                shutil.rmtree(chunks)
            self._chunk_cleanup_pending.discard(upload_id)
            return True
        except OSError:
            logger.warning("could not remove chunks of completed upload %s; will retry at GC", upload_id, exc_info=True)
            self._chunk_cleanup_pending.add(upload_id)
            return False

    def gc_run_inputs(self, *, now: datetime | None = None) -> list[str]:
        """Delete `inputs/` copies of attachments for terminal runs past the retention (t238 C7b m6).

        Only a regular file directly in `<task_root>/<run>/inputs/` whose library final still exists
        is removed (a re-run re-materializes it from that final, exactly like today); symlinks,
        junctions, non-terminal runs and anything outside inputs/ are never touched.
        `EVOMIND_RUN_INPUT_GC_DRY_RUN=1` only logs.
        """
        days = _env_number("EVOMIND_RUN_INPUT_RETENTION_DAYS", DEFAULT_RUN_INPUT_RETENTION_DAYS)
        if days <= 0:
            return []
        dry_run = os.environ.get("EVOMIND_RUN_INPUT_GC_DRY_RUN", "") == "1"
        cutoff = ((now or datetime.now(timezone.utc)) - timedelta(days=days)).isoformat()
        removed: list[str] = []
        for status in sorted(TERMINAL_RUN_STATES):
            for run in self.store.list_assistant_runs_with_status(status):
                run_id = str(run.get("id") or "")
                finished = str(run.get("completed_at") or "") or str(run.get("updated_at") or "")
                if not _SAFE_ID.fullmatch(run_id) or not finished or finished >= cutoff:
                    continue
                inputs = self.task_root / run_id / "inputs"
                if inputs.is_symlink() or not inputs.is_dir() or _is_reparse_point(inputs):
                    continue
                for row in self.store.list_attachments(run_id):
                    path = Path(str(row.get("path") or ""))
                    if path.parent != inputs or path.is_symlink() or _is_reparse_point(path) or not path.is_file():
                        continue
                    library = self._library_final(str(row.get("upload_id") or ""), str(row.get("sha256") or ""))
                    if library is None:
                        continue  # the only remaining copy: keep it
                    removed.append(f"{run_id}/inputs/{path.name}")
                    if not dry_run:
                        path.unlink()
        if removed:
            logger.info("run-input GC %s %d copies", "would remove" if dry_run else "removed", len(removed))
        return removed

    def _library_final(self, upload_id: str, sha256: str) -> dict[str, Any] | None:
        for row in self.store.list_attachments_by_upload_id(upload_id) if upload_id else []:
            if str(row.get("run_id") or "") or row.get("sha256") != sha256:
                continue
            path = Path(str(row.get("path") or ""))
            if path.is_file() and not path.is_symlink() and path.stat().st_size == int(row.get("bytes") or -1):
                return row
        return None

    def _maybe_gc_run_inputs(self) -> None:
        if time.monotonic() - self._last_run_input_gc < RUN_INPUT_GC_INTERVAL_SECONDS:
            return
        self._last_run_input_gc = time.monotonic()
        try:
            self.gc_run_inputs()
        except Exception:
            logger.exception("run-input GC failed")

    def _maybe_gc_uploads(self) -> None:
        if time.monotonic() - self._last_upload_gc < UPLOAD_GC_INTERVAL_SECONDS:
            return
        self._last_upload_gc = time.monotonic()
        try:
            self.gc_abandoned_uploads()
        except Exception:
            logger.exception("lazy upload GC failed")

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
        self._maybe_gc_uploads()
        self._maybe_gc_run_inputs()
        self._check_disk_low_water()
        from .tenant_access import current_principal
        principal = current_principal.get()
        tenant_key = principal.tenant_id if principal is not None and principal.tenant_id != "local" else OPERATOR_USAGE_KEY
        max_bytes, max_files = self._quota_limits(tenant_key)
        upload_id = new_id("upload")
        safe_name = _safe_name(name)
        created = utc_now()
        accepted, usage = self.store.reserve_upload(
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
            },
            tenant_key,
            max_bytes=max_bytes,
            max_files=max_files,
        )
        if not accepted:
            raise UploadLimitError(413, "quota_exceeded", _quota_message(max_bytes, max_files, usage, adding_bytes=total))
        row = self.store.get_upload(upload_id) or {}
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
        if upload["status"] != "uploading":
            raise ValueError("upload is not writable")
        self._check_chunk_low_water(len(data))  # nothing written yet: the upload stays resumable
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

    # Stale threshold after which a replayed complete may take over a `completing` claim
    # whose worker died without a service restart (start-up recovery resets all claims).
    # A live worker heartbeats its claim, so it never reaches this age.
    COMPLETING_STALE_SECONDS = 900.0
    COMPLETING_HEARTBEAT_SECONDS = 5.0

    def complete_upload(self, upload_id: str) -> dict[str, Any]:
        """Assemble and verify an upload. The service lock covers only claim and commit.

        Concat + SHA-256 + fsync run unlocked (D3). Each complete owns a claim token stored on
        the row; heartbeat, commit and the failure reset are compare-and-set on that token, and
        the tmp file is per claim, so a worker that lost its claim can never commit, reset the
        new owner, share its inode or delete the chunks.
        """
        if not _SAFE_ID.fullmatch(upload_id):
            raise ValueError("upload id is invalid")
        with self._lock:
            upload = self.store.get_upload(upload_id)
            if upload and upload["status"] == "uploading":
                # t238 C7b m3: before claiming, so a rejection leaves the upload `uploading` with its chunks.
                self._check_complete_free_space(int(upload["total_bytes"]))
            upload, chunks, claim, replay = self._claim_upload_locked(upload_id)
        if replay is not None:
            self._remove_chunk_files(upload_id)
            return replay
        try:
            temporary, actual_sha, written = self._assemble_upload(upload, chunks, claim)
        except BaseException as exc:
            return self._abandon_claim(upload_id, claim, None, exc)
        final_path = temporary.parent / str(upload["safe_name"])
        try:
            detected_media_type = _media_type(temporary, str(upload["media_type"]), name=final_path.name)
        except BaseException as exc:
            return self._abandon_claim(upload_id, claim, temporary, exc)
        attachment = {
            "id": new_id("attachment"),
            "upload_id": upload_id,
            "run_id": "",
            "name": str(upload["safe_name"]),
            "path": str(final_path),
            "media_type": detected_media_type,
            "bytes": written,
            "sha256": actual_sha,
            "created_at": utc_now(),
        }
        with self._lock:
            # Every claim transition in this process happens under the service lock, so the
            # owner check, the rename onto the final name and the commit are one step.
            if not self.store.upload_claim_is(upload_id, claim):
                temporary.unlink(missing_ok=True)
                return self._lost_claim_result(upload_id)
            try:
                _retry_transient_file_op(lambda: os.replace(temporary, final_path))
                committed = self.store.commit_completed_upload(
                    upload_id,
                    claim,
                    final_path=str(final_path),
                    final_sha256=actual_sha,
                    media_type=attachment["media_type"],
                    attachment=attachment,
                )
            except BaseException as exc:
                # Nothing committed: free the claim so an immediate retry works (chunks are kept).
                return self._abandon_claim(upload_id, claim, temporary, exc)
            if committed is None:
                return self._lost_claim_result(upload_id)
        self._remove_chunk_files(upload_id)
        return {"upload": committed, "attachment": attachment, "replayed": False, "trainability": self._trainability(attachment)}

    def _abandon_claim(self, upload_id: str, claim: str, temporary: Path | None, exc: BaseException) -> dict[str, Any]:
        """Failure before commit: drop this claim's own tmp, release only this claim, re-raise.

        Never deletes chunks. If the claim was already lost, an ordinary error yields the
        winner's result instead; KeyboardInterrupt/SystemExit always propagate.
        """
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        released = self.store.release_upload_claim(upload_id, claim)
        if released or not isinstance(exc, Exception):
            raise exc
        return self._lost_claim_result(upload_id)

    def _lost_claim_result(self, upload_id: str) -> dict[str, Any]:
        """Another worker owns (or finished) this upload: replay its result, never touch its files."""
        current = self.store.get_upload(upload_id)
        existing = self.store.get_attachment_by_upload_id(upload_id)
        if current and current["status"] == "complete" and existing:
            return {"upload": current, "attachment": existing, "replayed": True, "trainability": self._trainability(existing)}
        raise UploadLimitError(409, "upload_completing", UPLOAD_COMPLETING_MESSAGE)

    def _claim_upload_locked(
        self, upload_id: str
    ) -> tuple[dict[str, Any], list[dict[str, Any]], str, dict[str, Any] | None]:
        upload = self.store.get_upload(upload_id)
        if not upload:
            raise KeyError(upload_id)
        existing = self.store.get_attachment_by_upload_id(upload_id)
        if upload["status"] == "complete" and existing:
            return upload, [], "", {"upload": upload, "attachment": existing, "replayed": True, "trainability": self._trainability(existing)}
        takeover = upload["status"] == "completing"
        if takeover and not self._completing_is_stale(upload):
            raise UploadLimitError(409, "upload_completing", UPLOAD_COMPLETING_MESSAGE)
        if not takeover and upload["status"] != "uploading":
            raise ValueError("upload is not writable")
        chunks = self.store.list_upload_chunks(upload_id)
        expected_count = int(upload["chunk_count"])
        if len(chunks) != expected_count or [int(item["chunk_index"]) for item in chunks] != list(range(expected_count)):
            raise ValueError("upload chunks are incomplete")
        claim = new_id("claim")
        if takeover:
            claimed = self.store.takeover_upload_claim(upload_id, upload.get("claim_id"), str(upload["updated_at"]), claim)
            if claimed:
                logger.warning("upload %s: took over a stale completing claim", upload_id)
        else:
            claimed = self.store.claim_upload(upload_id, claim)
        if not claimed:
            raise UploadLimitError(409, "upload_completing", UPLOAD_COMPLETING_MESSAGE)
        return upload, chunks, claim, None

    def _completing_is_stale(self, upload: dict[str, Any]) -> bool:
        try:
            updated = datetime.fromisoformat(str(upload["updated_at"]))
        except (KeyError, ValueError):
            return True
        return (datetime.now(timezone.utc) - updated).total_seconds() > self.COMPLETING_STALE_SECONDS

    def _assemble_upload(self, upload: dict[str, Any], chunks: list[dict[str, Any]], claim: str) -> tuple[Path, str, int]:
        """Write and verify `final/.<claim>.tmp`; the caller renames it only while it still owns the claim.

        The tmp name is independent of the filename (43 bytes), so a long final name never
        overflows NAME_MAX here (t238 C7b N4).
        """
        upload_id = str(upload["id"])
        final_dir = self.upload_root / upload_id / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        temporary = final_dir / f".{claim}.tmp"
        digest = hashlib.sha256()
        written = 0
        last_beat = time.monotonic()
        try:
            with temporary.open("xb") as output:
                for item in chunks:
                    part = Path(str(item["path"]))
                    with part.open("rb") as source:
                        for block in iter(lambda: source.read(1024 * 1024), b""):
                            output.write(block)
                            digest.update(block)
                            written += len(block)
                            if time.monotonic() - last_beat >= self.COMPLETING_HEARTBEAT_SECONDS:
                                last_beat = time.monotonic()
                                if not self.store.heartbeat_upload_claim(upload_id, claim):
                                    raise UploadLimitError(409, "upload_completing", UPLOAD_COMPLETING_MESSAGE)
                output.flush()
                os.fsync(output.fileno())
            actual_sha = digest.hexdigest()
            if written != int(upload["total_bytes"]):
                raise ValueError("final upload byte length mismatch")
            if upload["expected_sha256"] and upload["expected_sha256"] != actual_sha:
                raise ValueError("final upload sha256 mismatch")
        except BaseException:
            temporary.unlink(missing_ok=True)  # only this claim's own tmp
            raise
        return temporary, actual_sha, written

    @staticmethod
    def _trainability(attachment: dict[str, Any]) -> dict[str, Any]:
        """Advisory: a precheck failure degrades to `trainable: None` and never fails complete or replay."""
        try:
            return upload_trainability(Path(str(attachment["path"])), int(attachment["bytes"]))
        except Exception:
            logger.warning("trainability precheck failed for %s", attachment.get("id"), exc_info=True)
            return {"trainable": None, "reasons": ["precheck_unavailable"], "limits": {}, "bytes": int(attachment.get("bytes") or 0), "files": 1, "message": ""}

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
            for position, attachment_id in enumerate(candidate_ids):
                source = self.store.get_attachment(attachment_id)
                if not source:
                    valid = False
                    break
                if str(source.get("run_id") or "") and not Path(str(source["path"])).is_file():
                    # t238 C7b m6: the run-input copy was GC'd; re-materialize from the library final.
                    library = self._library_final(str(source.get("upload_id") or ""), str(source.get("sha256") or ""))
                    if library is None:
                        valid = False
                        break
                    source = library
                    candidate_ids[position] = str(library["id"])
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

    def _bindable_upload(self, source: dict[str, Any], attachment_id: str) -> bool:
        """t238 C7b D6-M1: a library file whose upload is not `complete` (deleting/uploading) is never bound.

        Returns whether an upload row backs the source (legacy rows without one stay bindable). Call under `self._lock`.
        """
        upload = self.store.get_upload(str(source.get("upload_id") or ""))
        if upload is not None and upload["status"] != "complete":
            raise ValueError(f"attachment not found: {attachment_id}")
        return upload is not None

    def _bind_attachment_ids(self, run_id: str, root: Path, attachment_ids: list[str]) -> list[dict[str, Any]]:
        bound: list[dict[str, Any]] = []
        for attachment_id in list(dict.fromkeys(attachment_ids))[:50]:
            source = self.store.get_attachment(str(attachment_id))
            if not source:
                raise ValueError(f"attachment not found: {attachment_id}")
            if str(source.get("run_id") or "") and not Path(str(source["path"])).is_file():
                # t238 C7b m6-m1: the run-input copy was GC'd; bind the library final instead.
                library = self._library_final(str(source.get("upload_id") or ""), str(source.get("sha256") or ""))
                if library is None:
                    raise ValueError(f"attachment not found: {attachment_id}")
                source = library
            # The status check and the row write each run under the lock that delete takes for its
            # in-use check + `deleting` CAS; the copy itself runs outside it so other runs do not stall.
            with self._lock:
                backed = self._bindable_upload(source, str(attachment_id))
            source_path = Path(str(source["path"])).resolve(strict=True)
            target = root / "inputs" / _safe_name(str(source["name"]))
            if target.exists():
                target = target.with_name(f"{target.stem}_{len(bound) + 1}{target.suffix}")
            # Inputs are versioned copies. A hard link lets one Run overwrite
            # the library file (and another Run's input) through the same inode,
            # so share blocks only via copy-on-write (reflink), else copy.
            _clone_or_copy(source_path, target)
            if _sha256_file(target) != source["sha256"]:
                target.unlink(missing_ok=True)
                raise ValueError("attachment copy verification failed")
            clone = {
                **source, "id": new_id("attachment"), "run_id": run_id,
                "path": str(target), "created_at": utc_now(),
            }
            with self._lock:
                try:
                    # Re-check: a delete may have started (or finished) while the file was copied.
                    if self._bindable_upload(source, str(attachment_id)) != backed:
                        raise ValueError(f"attachment not found: {attachment_id}")
                except ValueError:
                    target.unlink(missing_ok=True)
                    raise
                self.store.put_attachment(clone)
            bound.append(clone)
        return bound

    def _propose_experience_plan(self, run_id: str, prompt: str = "") -> None:
        """t237 C4: read-only retrieval (policy a) → plan card → always `completed` (never `blocked`).

        Invokes personal-allowed research tools only. Write/train/HPC are never offered or started.
        C4b: `needs_compute` follows the prompt's (negation-aware) ask, so a plan-only prompt
        such as the live §2.5 smoke stays completed after confirm.
        """
        from .experience_entry import answer_for, detect_compute_intent, retrieve_plan
        plan = retrieve_plan(self.runtime, run_id)
        plan = {**plan, "needs_compute": bool(plan.get("needs_compute")) or detect_compute_intent(prompt)}
        session = self.store.get_session(run_id) or {}
        self.store.update_session(run_id, metadata_json={**(session.get("metadata") or {}), "plan_proposed": plan})
        self.store.append_event(run_id, "plan_proposed", plan)
        self.store.add_turn(run_id, "assistant", answer_for(plan))
        now = utc_now()
        self.store.update_assistant_run(run_id, status=SessionStatus.COMPLETED.value, completed_at=now, updated_at=now)
        self.store.update_session(run_id, status=SessionStatus.COMPLETED.value)

    def _bind_project_files(self, run_id: str, root: Path, snapshot: Any, explicit_ids: list[str]) -> tuple[dict[str, str], list[dict[str, str]]]:
        """t237 C3.5: copy the snapshotted project files into inputs/ like attachments (M4 retrieval).

        Returns ({library file id: run attachment id}, unreadable [{name, error_class}]). A file that cannot be
        copied is reported, never fatal; project files never turn a question into an artifact contract.
        """
        mapping: dict[str, str] = {}
        unreadable: list[dict[str, str]] = []
        for item in (snapshot.get("files") or []) if isinstance(snapshot, dict) else []:
            file_id = str(item.get("id") or "") if isinstance(item, dict) else ""
            if not file_id or file_id in explicit_ids or file_id in mapping:
                continue
            try:
                mapping[file_id] = self._bind_attachment_ids(run_id, root, [file_id])[0]["id"]
            except (OSError, ValueError):
                unreadable.append({"name": str(item.get("name") or "")[:200], "error_class": "file_unavailable"})
        return mapping, unreadable

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
        platform_model_id: str = "",
        reasoning_effort: str = "",
        intent: str = "",
        memory_mode: str = "on",
        fallback_model_id: str = "",
    ) -> dict[str, Any]:
        objective = str(prompt or "").strip()
        if memory_mode not in {"on", "off", "incognito"}:
            raise ValueError("invalid_memory_mode")
        from .experience_entry import valid_intent
        intent = valid_intent(intent)
        if platform_model_id and model_profile_id:
            raise ValueError('invalid_platform_model')
        if platform_model_id:
            from .model_router import platform_transport_for
            if platform_transport_for(platform_model_id) is None:
                raise ValueError('invalid_platform_model')
        if reasoning_effort and reasoning_effort not in {'low', 'medium', 'high'}:
            raise ValueError('invalid_reasoning_effort')
        if fallback_model_id:  # t238 C7e A1: only for an explicit pick, only a listed model
            from .llm_backends import visible_fallback_model_ids
            if not (platform_model_id or model_profile_id) or fallback_model_id == platform_model_id or fallback_model_id not in visible_fallback_model_ids():
                raise ValueError('invalid_fallback_model')
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
        project_snapshot = None
        if project_id:
            from .tenant_access import current_principal
            from .user_projects import ProjectStore
            project_snapshot = ProjectStore(self.runtime.runtime_root).run_snapshot(self.runtime, current_principal.get(), project_id)
        run_id = _reserved_run_id or new_id("run")
        if not _SAFE_ID.fullmatch(run_id) or self.store.get_session(run_id):
            raise ValueError("reserved Run identity is invalid or already in use")
        root = self.task_root / run_id
        created_root = not root.exists()
        for name in ("inputs", "work", "outputs", "logs", "evidence"):
            (root / name).mkdir(parents=True, exist_ok=True)
        try:
            return self._create_run_rows(
                run_id, root, objective=objective, conversation=conversation, attachment_ids=attachment_ids,
                managed_hpc_identity=managed_hpc_identity, secret_refs=secret_refs, task=task, start=start,
                project_id=project_id, project_snapshot=project_snapshot, user_task_id=user_task_id,
                profile_binding=profile_binding, platform_model_id=platform_model_id,
                reasoning_effort=reasoning_effort, intent=intent, memory_mode=memory_mode, fallback_model_id=fallback_model_id)
        except BaseException:
            if self.store.get_assistant_run(run_id) is None:
                self._rollback_unborn_run(run_id, root, created_root)
            raise

    def _rollback_unborn_run(self, run_id: str, root: Path, created_root: bool) -> None:
        """t238 C7b N-M1: a create_run that failed before its run row leaves no clone row or input copy.

        While create_run is live its clones (run row not yet written) still block delete (D6-M1);
        once it fails they go, so the library file becomes deletable again.
        """
        with self._lock:
            for clone in self.store.list_attachments(run_id):
                from .user_data import _remove_run_input
                try:
                    _remove_run_input(self.task_root, run_id, Path(str(clone["path"])))
                except OSError:
                    logger.warning("run %s: could not remove an unborn run's input copy", run_id, exc_info=True)
            self.store.delete_run_attachments(run_id)
        if created_root and root.is_dir() and not root.is_symlink() and self.store.get_session(run_id) is None:
            shutil.rmtree(root, ignore_errors=True)

    def _create_run_rows(
        self, run_id: str, root: Path, *, objective: str, conversation: str, attachment_ids, managed_hpc_identity,
        secret_refs, task: str, start: bool, project_id: str, project_snapshot, user_task_id: str, profile_binding,
        platform_model_id: str, reasoning_effort: str, intent: str, memory_mode: str, fallback_model_id: str = "",
    ) -> dict[str, Any]:
        resolved_attachment_ids = list(dict.fromkeys(str(value) for value in (attachment_ids or []) if str(value)))[:50]
        attachment_context = {"mode": "explicit" if resolved_attachment_ids else "none", "source_run_id": ""}
        bound = self._bind_attachment_ids(run_id, root, resolved_attachment_ids)
        project_file_ids, unreadable_project_files = self._bind_project_files(run_id, root, project_snapshot, resolved_attachment_ids)
        artifact_requirements = _artifact_requirements_from_prompt(objective)
        delivery = classify_delivery_intent(objective, bound)
        plan = {
            "schema": "evomind.assistant.plan.v1",
            # An experience run ends at its plan card (spec 06); it never owes a file.
            "require_artifact": delivery["require_artifact"] and not intent,
            "delivery_mode": "inline" if intent else delivery["delivery_mode"],
            "requested_files": [] if intent else delivery["requested_files"],
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
                **({"project_snapshot": project_snapshot} if project_snapshot else {}),
                **({"project_file_ids": project_file_ids} if project_file_ids else {}),
                **({"intent": intent} if intent else {}),
                "user_task_id": user_task_id,
                "memory_mode": memory_mode,
                "model_profile": profile_binding,
                "attachment_context": attachment_context,
                **({"platform_model_id": platform_model_id} if platform_model_id else {}),
                **({"fallback_model_id": fallback_model_id} if fallback_model_id else {}),
                **({"reasoning_effort": reasoning_effort} if reasoning_effort else {}),
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
        if unreadable_project_files:
            self.store.append_event(run_id, "project_files_unreadable", {"files": unreadable_project_files})
        if intent == "experience":
            self._propose_experience_plan(run_id, objective)
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
        if status == "completed":
            label = next((str(item.get("label") or "") for item in steps if item.get("id") == step_id), "")
            self.store.append_event(run_id, "step_completed", {"run_id": run_id, "step_id": step_id, "label": label})
        return plan

    def _activity(self, run_id: str, status: str) -> dict[str, Any]:
        """Recent tool rows and the current action, rebuilt from durable records (C6a refresh)."""
        rows: list[dict[str, Any]] = []
        running: dict[str, Any] | None = None
        for call in self.store.list_tool_calls(run_id, limit=2000):
            state = str(call.get("status") or "")
            state = {"waiting_approval": "waiting", "denied": "failed", "rejected": "failed"}.get(state, state)
            if state not in {"running", "completed", "failed", "waiting"}:
                continue
            started = str(call.get("started_at") or call.get("created_at") or "")
            row = {"kind": "tool", "label": tool_label(str(call.get("tool_name") or "")), "status": state,
                   "started_at": started,
                   "duration_ms": duration_ms(started, call.get("completed_at")) if state in {"completed", "failed"} else None}
            if state == "failed":
                result = call.get("result") if isinstance(call.get("result"), dict) else {}
                row["error"] = error_label(str(result.get("error") or ""))
            if state == "running":
                running = row
            rows.append(row)
        for event in self.store.list_events(run_id, 0, 2000):
            if event.get("event_type") != "user_steered":
                continue
            payload = event.get("payload") or {}
            summary = str(payload.get("summary") or "你补充了…")[:40]
            rows.append({
                "kind": "steer",
                "label": summary,
                "status": "completed",
                "started_at": str(event.get("created_at") or ""),
                "duration_ms": None,
            })
        current = None
        if status not in TERMINAL_RUN_STATES and status not in {"waiting_approval", "paused"}:
            step = self.store.latest_event(run_id, "step_started")
            step_payload = (step or {}).get("payload") or {}
            position = step_position(str(step_payload.get("step_id") or ""))
            if running:
                current = {key: running[key] for key in ("kind", "label", "status", "started_at")} | position
            elif step:
                current = {"kind": "step", "label": safe_text(step_payload.get("label"), 40), "status": "running",
                           "started_at": str(step.get("created_at") or "")} | position
        return {"activity": rows[-ACTIVITY_LIMIT:], "activity_total": len(rows), "current_activity": current}

    def _conversation_context(self, run: dict[str, Any]) -> str:
        from .tenant_access import AccessStore
        access = AccessStore(self.root)
        owner = access.owner_for("session", str(run["session_id"]))
        project = self._run_project_id(run)
        rows = self.store.list_assistant_runs(str(run["conversation_id"]), 50)
        history: list[dict[str, str]] = []
        for row in reversed(rows):
            if row["id"] == run["id"]:
                continue
            if access.owner_for("session", str(row["session_id"])) != owner:
                continue
            # STM is per project (C3 STM_SCOPE=PROJECT); "" is its own no-project bucket.
            if self._run_project_id(row) != project:
                continue
            turns = self.store.list_turns(str(row["session_id"]))
            answer = next((str(item.get("content") or "") for item in reversed(turns) if item.get("role") == "assistant"), "")
            history.append({"user": str(row["prompt"])[:2000], "assistant": answer[:3000]})
        if len(history) <= 6:
            return json.dumps(history, ensure_ascii=False)
        # Older same-project turns stay answerable as a compact summary; recent 6 stay verbatim.
        earlier = [{"user": item["user"][:300], "assistant": item["assistant"][:300]} for item in history[:-6][-20:]]
        return json.dumps({"earlier_summary": earlier, "recent": history[-6:]}, ensure_ascii=False)

    def _run_project_id(self, run: dict[str, Any]) -> str:
        session = self.runtime.get_session(str(run["session_id"])) or {}
        metadata = session.get("metadata") if isinstance(session.get("metadata"), dict) else {}
        return str(metadata.get("project_id") or "")

    @staticmethod
    def _project_prompt_lines(snapshot: Any) -> list[str]:
        """Project context bound at run creation; a later edit of the project never changes this run."""
        if not isinstance(snapshot, dict):
            return []
        instructions = str(snapshot.get("instructions") or "")
        if hashlib.sha256(instructions.encode()).hexdigest() != snapshot.get("instructions_sha256"):
            raise ValueError("project_instructions_snapshot_changed")
        files = [item for item in snapshot.get("files") or [] if isinstance(item, dict)]
        return [
            "Project instructions (set by the user for every conversation in this project; the objective below still applies):",
            *(instructions.splitlines() if instructions.strip() else ["none"]),
            "Project files (names only; available inputs of this project):",
            *([f"- {item.get('name')} ({int(item.get('bytes') or 0)} bytes)" for item in files] or ["- none"]),
        ]

    def _memory_prompt_lines(self, metadata: dict[str, Any], guidance: Any, *, session_id: str = "", query: str = "") -> list[str]:
        """t237 C3.4 / C6c: the owner's live long-term memories, read fresh per prompt.

        Pinned first, then the rows most related to ``query`` (cap MEMORY_PROMPT_TOP_K). ``scope='project'`` rows join
        only their project's runs. ``memory_mode`` off / incognito ⇒ nothing read (L16). Injected ids are recorded on
        the session (snapshot ``memory_used``) and counted (``last_used_at`` / ``use_count``).
        """
        from .user_memory import principal_from_metadata
        principal = principal_from_metadata(metadata)
        if principal is None or str(metadata.get("memory_mode") or "on") != "on":
            return []
        store = UserMemoryStore(self.runtime.runtime_root)
        if not store.memory_enabled(principal):
            return []
        rows = store.select_for_prompt(principal, query, project_id=str(metadata.get("project_id") or ""), limit=MEMORY_PROMPT_TOP_K)
        if rows and session_id:
            store.mark_used([row["id"] for row in rows])
            self._record_memory_used(session_id, rows)
        return [
            "Long-term user memory (facts the user let you keep; apply them when relevant, the objective below still wins):",
            *([f"- {' '.join(row['text'].split())}" for row in rows] or ["- none"]),
            *guidance(("memory_remember",),
                      "When the user reveals a durable fact about themselves (preferences, background, tools, training or work experience and its outcome), "
                      "call memory_remember once with one short plain-text sentence. Do not store secrets, credentials, one-off task details or facts already listed above."),
        ]

    def _record_memory_used(self, session_id: str, rows: list[dict[str, Any]]) -> None:
        """Union of memories injected into this run's prompts ({id, short text}); the answer shows them as 用到的记忆."""
        session = self.store.get_session(session_id) or {}
        metadata = dict(session.get("metadata") or {})
        used = [item for item in (metadata.get("memory_used") or []) if isinstance(item, dict)]
        known = {item.get("id") for item in used}
        added = [{"id": row["id"], "text": " ".join(row["text"].split())[:120]} for row in rows if row["id"] not in known]
        if added:
            metadata["memory_used"] = (used + added)[:MEMORY_PROMPT_TOP_K * 2]
            self.store.update_session(session_id, metadata_json=metadata)

    def _execution_prompt(self, run: dict[str, Any], *, resume: bool, recovery_reason: str = "") -> str:
        from .run_supervision import recovery_note

        recovery_line = recovery_note(recovery_reason) if resume else ""
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
        from .personal_tool_boundary import offered, session_tenant
        tenant_id = session_tenant(session)

        def guidance(tools: tuple[str, ...], *lines: str) -> list[str]:
            # Never instruct a personal session to call a tool it is not offered.
            return list(lines) if all(offered(tenant_id, name) for name in tools) else []

        enforced_tool_scope = [
            str(value)
            for value in (metadata.get("run_allowed_tool_names") or [])
            if str(value) and offered(tenant_id, str(value))
        ]
        attachment_lines = [f"- inputs/{Path(str(item['path'])).name} ({item['bytes']} bytes, sha256={item['sha256']})" for item in attachments]
        project_lines = self._project_prompt_lines(metadata.get("project_snapshot"))
        memory_lines = self._memory_prompt_lines(metadata, guidance, session_id=str(run["session_id"]), query=str(run.get("prompt") or ""))
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
                "Read inputs only from inputs/. Put working files in work/.",
                # C6-answer SPEC §4.3: decide file need first; the answer itself is Markdown in the reply.
                (
                    "This objective asks for a file: put each final user-facing file in outputs/final/ and call artifact_publish for it. "
                    "Do not treat a file write as delivery until it is verified and published. Still reply with a short Markdown summary "
                    "(conclusion and key points) above the files. The file card is the only download entry: do not put download links, "
                    "file URLs or \"Download\" lines in the reply, do not start a line with \"Download\", do not paste artifact, sandbox: or "
                    "?download=1 URLs, and do not write English phrases such as \"verified and published\". "
                    "Write the reply in Chinese."
                    if plan.get("require_artifact")
                    else "Answer directly in Markdown in your reply; do not create a file just to answer. Only if the result is a standalone, "
                    "reusable, longer artifact (document, dataset, or code over 30 lines) put it in outputs/final/ and call artifact_publish, "
                    "and still give a short Markdown summary in the reply."
                ),
                "Workspace-local file writes, commands and tests are allowed. Stop at exact approval for external or irreversible effects.",
                *guidance(("hpc_execute_solution",), "Hotfix D8: training-run tuning: this host has no GPU and no torch/GPU stack. Never execute training, fine-tuning, inference or heavy data scripts on the local interpreter; run them on the managed HPC allocation through hpc_execute_solution (or managed_tensor_train), which enforces the 5/5 identity gate. A local 'ModuleNotFoundError: No module named torch' is expected and is NOT a valid smoke test - go straight to the managed HPC execution."),
                *guidance(("capability_discover",), "Hotfix D8: training-run tuning: keep the per-turn tool surface small. Do not call capability_discover unless you genuinely need capabilities you do not already have: it expands the visible tool list and makes every later turn much slower. Reuse settled receipts instead of repeating the same status reads in one run."),
                "Hotfix D9: long-run tuning: training, inference and preprocessing scripts must use a parallel, cache-aware data path. Decode/resize must run through worker processes or threads (for example torch DataLoader with num_workers >= 8, or a thread/process pool) and must reuse any cached, pre-scaled artifact that already exists under the managed data root. A single-threaded Python loop over hundreds of thousands of images is not acceptable: report the decode throughput (images/second) and the worker count in the run report.",
                "When an action needs approval, call the exact tool with its final arguments so the runtime can create the approval card; do not ask for approval only in prose.",
                "If a tool fails, inspect its error and logs, correct the narrow cause, then retry with the same objective.",
                *guidance(("competition_data_prepare", "competition_data_status"), "For competition or benchmark data, use competition_data_prepare and competition_data_status. Do not invent an ad-hoc downloader, guess a prior Run path, or treat exit code 0 as data readiness."),
                *guidance(("competition_data_prepare", "competition_data_status"), "For the six-source workflow, the fixed IDs are cure_bench, e2lmc, mindgames, ariel_2025, weather4cast, and open_polymer. Call competition_data_status for all six first, then competition_data_prepare only for items that are not ready. Do not call kaggle_list as a prerequisite: the official URLs and adapter slugs are already pinned, and a live listing failure must not block these six adapters. Continue independent sources after one source reaches a human, credential, network, or storage gate. Derive the summary mechanically: FULL_DATA_READY only for 6/6 ready, PARTIAL for 1-5/6 ready, and BLOCKED for 0/6 ready or a failed status audit."),
                *guidance(("competition_data_prepare", "competition_data_status"), "Sources whose access is local_prepared_mlebench (for example histopathologic_cancer, the Kaggle histopathologic-cancer-detection archive) are managed competition sources too: the official archive is already prepared on the shared GPU data root, so call competition_data_status, then competition_data_prepare to publish the persistent pointer (it symlinks $ROOT/prepared and records counts, label rows and a listing hash), and train from the prepared public/ layout. Never route a local_prepared source to the managed Kaggle connector and never report it as unavailable; a missing prepared layout is SOURCE_UNAVAILABLE, not an authorization problem."),
                "Competition credentials must come only from a tenant-bound controlled secret reference or the already-configured server-managed Kaggle connector. Never request or place passwords, tokens, cookies, or access codes in the prompt or tool arguments. The managed Kaggle connector authorizes only the requested read/download action; it never authorizes Join, rule acceptance, account creation, or submission. Weather4cast SFTP still requires its tenant-bound one-time controlled secret.",
                "For a new, replaced, reclaimed, or restored HPC allocation, the secure enrollment surface is Workbench > Data / Kaggle > AIMSLAB 60-second quick replacement. Direct the user to that browser-memory encrypted form and stop at its credential-entry gate. Do not claim the enrollment surface is unavailable merely because no enrollment tool schema is exposed to the model, and never request HPC or proxy credentials in the prompt.",
                *guidance(("verified_context",), "When a selected research task is present, call verified_context before judging whether prior evidence exists; an empty task inputs/ directory is not proof that historical run evidence is absent."),
                f"Resume mode: {str(resume).lower()}",
                *([recovery_line] if recovery_line else []),
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
                *project_lines,
                *memory_lines,
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

    def start(self, run_id: str, *, resume: bool = False, priority: str = "normal", reason: str = "") -> bool:
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
            # Run supervision v1: a person's resume / an approved action may take the reserved slot.
            self._start_priority[run_id] = priority
            if reason:
                self._resume_reasons[run_id] = reason
            thread = threading.Thread(
                target=self._run_thread,
                args=(run_id, resume),
                name=f"evomind-assistant-{run_id[-12:]}",
                daemon=True,
            )
            self._threads[run_id] = thread
            thread.start()
            return True

    def _start_or_defer(self, run_id: str, *, resume: bool, priority: str = "human") -> bool:
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
                            self.start(run_id, resume=resume, priority=priority)
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
            return self.start(run_id, resume=resume, priority=priority)
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
        with self._lock:
            priority = self._start_priority.pop(run_id, "normal")
            reason = self._resume_reasons.pop(run_id, "")
        if not self._acquire_slot(run_id, priority):
            return
        try:
            self._execute_in_slot(run_id, run, resume=resume, reason=reason)
        finally:
            self.slots.release(run_id)
            self._close_open_attempt(run_id)

    _REPORT_FOR_STATUS = {"completed": "ready", "cancelled": "dropped", "waiting_approval": "waiting", "paused": "waiting",
                          "blocked": "blocked", "failed": "failed"}

    def _close_open_attempt(self, run_id: str) -> None:
        """An attempt still open when its worker returns ends with the report its run status implies."""
        try:
            latest = self.ledger.latest(run_id) or {}
            if latest.get("status") != "running":
                return
            current = self.store.get_assistant_run(run_id) or {}
            report = self._REPORT_FOR_STATUS.get(str(current.get("status") or ""))
            if report:
                self.ledger.report(str(latest["attempt_id"]), report, closed_by="worker_exit")
        except Exception:  # noqa: BLE001
            logger.exception("closing the open attempt of %s failed", run_id)

    def _acquire_slot(self, run_id: str, priority: str) -> bool:
        """Run supervision v1: wait (status queued, with a position) until a model slot is free."""
        from .personal_tool_boundary import session_tenant

        tenant = session_tenant(self.store.get_session(run_id) or {})

        def aborted() -> bool:
            if self._stopping.is_set():
                return True
            current = self.store.get_assistant_run(run_id)
            return not current or current["status"] == "cancelled" or self.runtime.user_pause_requested(run_id)

        def queued(position: int) -> None:
            self.store.update_assistant_run(run_id, status="queued")
            self.store.append_event(run_id, "run_queued", {"run_id": run_id, "position": position})

        return self.slots.acquire(run_id, tenant, priority=priority, should_abort=aborted, on_queued=queued)

    def _ledger_safe(self, method: str, *args: Any, **kwargs: Any) -> Any:
        try:
            return getattr(self.ledger, method)(*args, **kwargs)
        except Exception:  # noqa: BLE001 - the ledger records a run; it never stops one
            logger.exception("run ledger %s failed", method)
            return ""

    def _set_attention(self, run_id: str, card: dict[str, Any] | None) -> None:
        session = self.store.get_session(run_id)
        if not session:
            return
        metadata = dict(session.get("metadata") or {})
        if card is None:
            if "attention" not in metadata:
                return
            metadata.pop("attention", None)
        else:
            metadata["attention"] = card
            self.store.append_event(run_id, "attention_required", {"run_id": run_id, "kind": card.get("kind"),
                                                                    "options": [item.get("action") for item in card.get("options") or []]})
        self.store.update_session(run_id, metadata_json=metadata)

    def _interruptible_sleep(self, run_id: str, seconds: float) -> bool:
        """Sleep in short steps; False when the run was cancelled, paused or the service is stopping."""
        deadline = time.monotonic() + max(0.0, seconds)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return True
            if self._stopping.wait(min(0.5, remaining)):
                return False
            current = self.store.get_assistant_run(run_id)
            if not current or current["status"] == "cancelled" or self.runtime.user_pause_requested(run_id):
                return False

    def _execute_in_slot(self, run_id: str, run: dict[str, Any], *, resume: bool, reason: str) -> None:
        from .run_supervision import ModelTurnFailed, model_retry_decision, model_unavailable_card

        attempts = int(run.get("retry_count") or 0)
        model_retries = 0
        attempt_reason = reason or ("resume" if resume else "start")
        # C6-answer H_a21: the process-row clock starts when execution starts, never at creation or while queued.
        self.store.append_event(run_id, "run_started", {"run_id": run_id, "started_at": utc_now()})
        while attempts < 3:
            if self._stopping.is_set():
                return
            attempt_id = ""
            try:
                current_run = self.store.get_assistant_run(run_id)
                if not current_run or current_run["status"] == "cancelled":
                    return
                if self.runtime.user_pause_requested(run_id):
                    self.runtime.settle_user_pause(run_id)
                    return
                attempt_id = self._ledger_safe("begin_attempt", run_id, reason=attempt_reason,
                                               model=str(current_run.get("model") or ""))
                self.store.update_assistant_run(run_id, status=SessionStatus.PLANNING.value, error_class="", error_message="")
                self.runtime.store.update_session(run_id, status=SessionStatus.PLANNING.value)
                self.store.append_event(run_id, "step_started", {"run_id": run_id, "step_id": "understand", "label": "理解目标与输入"})
                self._set_step(run_id, "understand", "completed")
                self.store.update_assistant_run(run_id, status=SessionStatus.RUNNING.value)
                self.runtime.store.update_session(run_id, status=SessionStatus.RUNNING.value)
                self.store.append_event(run_id, "step_started", {"run_id": run_id, "step_id": "execute", "label": "调用工具执行任务"})
                result = self.runtime.message(
                    run_id, self._execution_prompt(run, resume=resume or attempts > 0 or model_retries > 0, recovery_reason=attempt_reason),
                    max_steps=48)
                if self.runtime.user_pause_requested(run_id):
                    self.runtime.settle_user_pause(run_id)
                    return
                current_run = self.store.get_assistant_run(run_id)
                if not current_run or current_run["status"] == "cancelled":
                    return
                model = result.get("model_execution") if isinstance(result.get("model_execution"), dict) else {}
                provider = str(model.get("provider") or "")
                model_name = str(model.get("model") or "")
                if not provider and not model_name:
                    # A model.response recorded during the call is the served identity.
                    # The router decision is only a stand-in when no call returned.
                    served = self.store.get_assistant_run(run_id) or {}
                    provider = str(served.get("model_provider") or "")
                    model_name = str(served.get("model") or "")
                if not provider and not model_name:
                    routed = result.get("model") if isinstance(result.get("model"), dict) else {}
                    provider = str(routed.get("provider") or "")
                    model_name = str(routed.get("model") or "")
                self.store.update_assistant_run(run_id, model_provider=provider, model=model_name)
                self.store.append_event(
                    run_id,
                    "model_observed",
                    {"run_id": run_id, "provider": provider, "model": model_name, "tool_calls": model.get("native_tool_calls", 0)},
                )
                status = str(result.get("status") or "failed")
                if status == "waiting_approval":
                    self._ledger_safe("report", attempt_id, "waiting", detail="approval") if attempt_id else None
                    self._set_step(run_id, "execute", "waiting_approval")
                    self.store.update_assistant_run(run_id, status=SessionStatus.WAITING_APPROVAL.value)
                    approval = result.get("approval") if isinstance(result.get("approval"), dict) else {}
                    self.store.append_event(run_id, "approval_required", {"run_id": run_id, "approval": approval})
                    return
                if status != "completed":
                    raise ModelTurnFailed(
                        str(result.get("text") or f"runtime returned {status}"),
                        code=str(result.get("error_code") or ""),
                        retry_after_seconds=result.get("retry_after_seconds") if isinstance(result.get("retry_after_seconds"), (int, float)) else None,
                    )
                self._set_step(run_id, "execute", "completed")
                self.store.update_assistant_run(run_id, status=SessionStatus.VERIFYING.value)
                self.runtime.store.update_session(run_id, status=SessionStatus.VERIFYING.value)
                self.store.append_event(run_id, "step_started", {"run_id": run_id, "step_id": "verify", "label": "验证结果与文件"})
                answer = str(result.get("text") or "").strip()
                if provider and model_name:
                    observed_label = f"{provider} / {model_name}"
                    answer = f"**本轮实际模型**：`{observed_label}`（来自模型输出）\n\n{answer}".strip()
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
                if attempt_id:
                    self._ledger_safe("report", attempt_id, "ready", artifacts=len(artifacts), model=model_name)
                    self._ledger_safe("receipt", run_id, "deliver", True, artifacts=len(artifacts), attempt_id=attempt_id)
                self._set_attention(run_id, None)
                try:
                    self.drain_queued_messages(run_id)
                except Exception:
                    logger.exception("drain_queued_messages failed for %s", run_id)
                return
            except Exception as exc:
                if self.runtime.user_pause_requested(run_id):
                    self.store.append_event(run_id, "run_paused", {"error_type": type(exc).__name__, "budget_reset": False})
                    self.runtime.settle_user_pause(run_id)
                    return
                retry = model_retry_decision(
                    str(exc), model_retries,
                    code=getattr(exc, "code", "") if isinstance(exc, ModelTurnFailed) else "",
                    retry_after_seconds=getattr(exc, "retry_after_seconds", None),
                )
                if retry is not None:
                    served = self.store.get_assistant_run(run_id) or {}
                    if attempt_id:
                        self._ledger_safe("report", attempt_id, "failed" if retry.retry else "blocked",
                                          code=retry.code, category=retry.category)
                    self._ledger_safe("failure", run_id, signature=f"model:{retry.code}", category=f"model_{retry.category}",
                                      model=str(served.get("model") or ""), retry=retry.retry)
                if retry is not None and retry.retry:
                    # Run supervision v1: a passing model failure is tried again on the same model (never another one).
                    model_retries += 1
                    self.store.update_assistant_run(run_id, status=SessionStatus.RECOVERING.value,
                                                    error_class="model_retry", error_message=str(exc)[:1000])
                    self.store.append_event(run_id, "retry_scheduled", {
                        "run_id": run_id, "attempt": model_retries, "delay_seconds": retry.delay_seconds,
                        "error_class": f"model_{retry.category}", "error_type": type(exc).__name__, "kind": "model"})
                    if not self._interruptible_sleep(run_id, retry.delay_seconds):
                        if self.runtime.user_pause_requested(run_id):
                            self.runtime.settle_user_pause(run_id)
                        return
                    resume = True
                    attempt_reason = "model_retry"
                    run = self.store.get_assistant_run(run_id) or run
                    continue
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
                    if attempt_id and retry is None:
                        self._ledger_safe("report", attempt_id, "failed", code=failure_class)
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
                if attempt_id and retry is None:
                    self._ledger_safe("report", attempt_id, "failed" if terminal else "blocked", code=failure_class)
                if retry is not None:
                    session_meta = (self.store.get_session(run_id) or {}).get("metadata") or {}
                    served = self.store.get_assistant_run(run_id) or {}
                    self._set_attention(run_id, model_unavailable_card(
                        code=retry.code, category=retry.category, retries=model_retries,
                        platform_model_id=str(session_meta.get("platform_model_id") or ""),
                        provider=str(served.get("model_provider") or "")))
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
        plan = run.get("plan") if isinstance(run.get("plan"), dict) else {}
        expected_names = {
            str(name).casefold()
            for name in [*self._artifact_requirements(run)["names"], *(plan.get("requested_files") or [])]
        }
        for candidate in sorted(outputs.rglob("*")):
            if (
                ".evomind_previews" in candidate.parts
                or _ARTIFACT_SNAPSHOT_DIR in candidate.parts
                or not candidate.is_file()
                or candidate.is_symlink()
            ):
                continue
            resolved = candidate.resolve(strict=True)
            relative = resolved.relative_to(outputs)
            if str(resolved) in published_paths:
                continue
            # SPEC §4.4 (H_a4): drafts and input copies stay unpublished. Only a file the user named as a deliverable,
            # or a whitelisted type the model placed under outputs/final/, is auto-published.
            final_file = relative.parts[0] == "final" and resolved.suffix.casefold() in _AUTO_FINALIZE_SUFFIXES
            if not final_file and resolved.name.casefold() not in expected_names:
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
        # SPEC §4.4 (H_a3): no deliverable ⇒ no file. The answer is never wrapped into a result.md fallback.
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

    def _training_outcome(self, session_id: str) -> str:
        """t238 C7c-FIX4: ``failed`` when this Run asked for a managed training and none of them succeeded, so a reply that
        merely explains the failure is not reported as 已完成; ``succeeded`` when one did; empty when none was asked."""
        try:
            calls = [item for item in self.store.list_tool_calls(session_id, limit=2000) if item.get("tool_name") == "managed_tensor_train"]
        except Exception:  # noqa: BLE001 - presentation hint only
            return ""
        settled = [item for item in calls if str(item.get("status") or "") in {"completed", "failed"}]
        if not settled:
            return ""
        if any(isinstance(item.get("result"), dict) and item["result"].get("ok") is True for item in settled):
            return "succeeded"
        return "failed"

    def snapshot(self, run_id: str) -> dict[str, Any]:
        run = self.store.get_assistant_run(run_id)
        if not run:
            raise KeyError(run_id)
        if str(run.get("status") or "") == "planning":
            try:
                self.close_idle_planning(run_id=run_id)
            except Exception:
                logger.exception("close_idle_planning failed for snapshot %s", run_id)
            else:
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
        plan = run.get("plan") if isinstance(run.get("plan"), dict) else {}
        delivery_mode = _delivery_mode(plan, projected_deliverables)
        if projected_deliverables and not streaming_text:
            cleaned = strip_delivery_boilerplate(_MODEL_PREFIX.sub("", answer, count=1), projected_deliverables)
            if cleaned != _MODEL_PREFIX.sub("", answer, count=1).strip():
                message_history = _with_answer(message_history, cleaned)
                answer = message_history["answer"]
        if effective_run["status"] == "completed" and projected_deliverables and not _visible_answer(answer):
            # SPEC §4.4 (H_a5): a run that delivered files always carries a short inline line above them.
            message_history = _with_answer(message_history, _files_summary(projected_deliverables))
            answer = message_history["answer"]
        started = self.store.latest_event(str(run["session_id"]), "run_started")
        started_at = "" if effective_run["status"] == "queued" or not started else str(
            (started.get("payload") or {}).get("started_at") or started.get("created_at") or ""
        )
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
            # t238 C7e A1/A2: visible switch notice, the user's pre-selected alternate, and the failure kind.
            "model_failover": _public_failover(session_metadata.get("model_failover")),
            "fallback_model_id": str(session_metadata.get("fallback_model_id") or "") or None,
            "model_error": _model_error(effective_run, answer, session_metadata),
            **self._supervision_fields(run_id, effective_run, session_metadata, active_tools),
            "answer": answer,
            "started_at": started_at,
            "delivery_mode": delivery_mode,
            "partial_answer": (str(answer).strip() or None) if effective_run["status"] == "cancelled" else None,
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
            "training_outcome": self._training_outcome(str(run["session_id"])),
            "completion_repair_required": completion_repair_required,
            "missing_expected_artifacts": missing_expected_artifacts,
            "unresolved_tool_names": unresolved_tool_names,
            "last_event_seq": self.store.latest_event_seq(str(run["session_id"])),
            "terminal": effective_run["status"] in TERMINAL_RUN_STATES,
            "progress": self.runtime.progress.read(run_id, status=str(effective_run["status"])),
            **self._activity(str(run["session_id"]), str(effective_run["status"])),
            "memory_saved": UserMemoryStore(self.runtime.runtime_root).saved_by_run(run_id),
            "memory_used": [item for item in (session_metadata.get("memory_used") or []) if isinstance(item, dict)],
            "memory_mode": str(session_metadata.get("memory_mode") or "on"),
            "queued_messages": [
                item for item in ((session_metadata.get("c6b_inbox") or {}).get("queued") or [])
                if isinstance(item, dict) and item.get("status") == "pending"
            ] if isinstance(session_metadata.get("c6b_inbox"), dict) else [],
            "pending_steer": [
                item for item in ((session_metadata.get("c6b_inbox") or {}).get("pending_steer") or [])
                if isinstance(item, dict) and item.get("status") == "pending"
            ] if isinstance(session_metadata.get("c6b_inbox"), dict) else [],
            "reasoning_summary": str(session_metadata.get("reasoning_summary") or "") or None,
            "intent": str(session_metadata.get("intent") or ""),
            **({"plan_proposed": session_metadata.get("plan_proposed"), "plan_decision": PlanDecisions(self.runtime.runtime_root).get(run_id)}
               if session_metadata.get("intent") == "experience" else {}),
        }

    def list_runs(self, conversation_id: str = "", limit: int = 50) -> list[dict[str, Any]]:
        return [self.snapshot(str(item["id"])) for item in self.store.list_assistant_runs(conversation_id, limit)]

    def _supervision_fields(self, run_id: str, run: dict[str, Any], metadata: dict[str, Any],
                            active_tools: list[dict[str, Any]]) -> dict[str, Any]:
        """Run supervision v1: one ``stage`` (dsh priority order), the open ``attention`` card and the queue position."""
        from .run_supervision import GPU_SLOT_RELEASE_TOOLS, STAGE_LABELS, approval_card, derive_stage, public_attention

        status = str(run.get("status") or "")
        attention = public_attention(status, metadata)
        error_class = str(run.get("error_class") or "")
        if attention is None and status == "blocked" and error_class in {"approval_rejected", "approval_expired"}:
            attention = approval_card(error_class)
        position = self.slots.position(run_id)
        slot_state = self.slots.state(run_id)
        gpu_running = any(str(item.get("tool_name") or "") in GPU_SLOT_RELEASE_TOOLS for item in active_tools)
        stage = derive_stage(status, attention=attention, queue_position=position, slot_state=slot_state,
                             gpu_tool_running=gpu_running)
        try:
            latest = self.ledger.latest(run_id) or {}
        except Exception:  # noqa: BLE001 - supervision fields never break the run view
            latest = {}
        return {
            "stage": stage,
            "stage_label": STAGE_LABELS.get(stage, ""),
            "attention": attention,
            "queue_position": position,
            "supervision": {"attempts": latest.get("number"), "last_report": latest.get("report")} if latest else None,
        }

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
                payload = {"run_id": run_id, "tool": payload.get("tool_name"), "tool_call_id": payload.get("id"), "status": "running",
                           "label": tool_label(str(payload.get("tool_name") or "")),
                           "started_at": str(payload.get("started_at") or item.get("created_at") or "")}
            elif event_type == "tool_completed" and str(item["event_type"]).startswith("tool."):
                completed_call = self.store.get_tool_call(str(payload.get("tool_call_id") or "")) or {}
                payload = {
                    "tool": completed_call.get("tool_name"),
                    "run_id": run_id,
                    "tool_call_id": payload.get("tool_call_id"),
                    "status": "completed" if payload.get("ok") else "failed",
                    "summary": payload.get("summary"),
                    "error": payload.get("error"),
                    "label": tool_label(str(completed_call.get("tool_name") or "")),
                    "duration_ms": duration_ms(completed_call.get("started_at") or completed_call.get("created_at"), completed_call.get("completed_at")),
                    **({"error_label": error_label(str(payload.get("error") or ""))} if not payload.get("ok") else {}),
                }
            elif event_type in {"step_started", "step_completed"}:
                payload = {**payload, **step_position(str(payload.get("step_id") or ""))}
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


    def _inbox(self, session_id: str) -> dict[str, Any]:
        session = self.store.get_session(session_id) or {}
        metadata = session.get("metadata") if isinstance(session.get("metadata"), dict) else {}
        inbox = metadata.get("c6b_inbox")
        if not isinstance(inbox, dict):
            inbox = {"queued": [], "pending_steer": [], "idempotency": {}}
        inbox.setdefault("queued", [])
        inbox.setdefault("pending_steer", [])
        inbox.setdefault("idempotency", {})
        return inbox

    def _save_inbox(self, session_id: str, inbox: dict[str, Any]) -> None:
        session = self.store.get_session(session_id) or {}
        metadata = dict(session.get("metadata") or {})
        metadata["c6b_inbox"] = {
            "queued": list(inbox.get("queued") or []),
            "pending_steer": list(inbox.get("pending_steer") or []),
            "idempotency": dict(inbox.get("idempotency") or {}),
        }
        self.store.update_session(session_id, metadata_json=metadata)

    @staticmethod
    def _steer_summary(text: str) -> str:
        import re as _re
        cleaned = _re.sub(
            r"(?i)\b(?:HPC|Pezayo|lt4net|tool_[A-Za-z0-9_-]+|call_[A-Za-z0-9_-]+)\b|/workspace/\S+",
            "",
            text,
        )
        cleaned = _re.sub(r"\s+", " ", cleaned).strip()[:24]
        summary = f"你补充了：{cleaned}" if cleaned else "你补充了…"
        return summary[:40]

    def post_message(
        self,
        run_id: str,
        *,
        mode: str,
        text: str,
        idempotency_key: str = "",
    ) -> dict[str, Any]:
        """C6b2: queue (after round) or steer (next tool boundary) while running."""
        run = self.store.get_assistant_run(run_id)
        if not run:
            raise KeyError(run_id)
        command = str(mode or "").strip().lower()
        if command not in {"queue", "steer", "cancel_queue"}:
            raise ValueError("unsupported message mode")
        body = str(text or "").strip()
        if command != "cancel_queue" and not body:
            raise ValueError("message text is required and must be non-empty")
        if len(body) > 4_000:
            raise ValueError("message text must be at most 4000 characters")
        session_id = str(run["session_id"])
        key = str(idempotency_key or "").strip()
        inbox = self._inbox(session_id)
        if command == "cancel_queue":
            if not key:
                raise ValueError("cancel_queue requires idempotency_key")
            inbox["queued"] = [
                item for item in (inbox.get("queued") or [])
                if str(item.get("idempotency_key") or "") != key
            ]
            self._save_inbox(session_id, inbox)
            return self.snapshot(run_id)
        if key and key in (inbox.get("idempotency") or {}):
            return self.snapshot(run_id)
        status = str(run.get("status") or "")
        if status in TERMINAL_RUN_STATES and command == "steer":
            raise ValueError("steer requires a non-terminal run")
        item = {
            "text": body,
            "idempotency_key": key or new_id("msg"),
            "status": "pending",
            "created_at": utc_now(),
            "mode": command,
        }
        if command == "queue":
            inbox["queued"].append(item)
            if key:
                inbox["idempotency"][key] = {"mode": "queue"}
            self._save_inbox(session_id, inbox)
            self.store.append_event(session_id, "message_queued", {"text": body[:80], "idempotency_key": item["idempotency_key"]})
            return self.snapshot(run_id)
        # steer
        summary = self._steer_summary(body)
        item["summary"] = summary
        inbox["pending_steer"].append(item)
        if key:
            inbox["idempotency"][key] = {"mode": "steer"}
        self._save_inbox(session_id, inbox)
        self.store.append_event(session_id, "user_steered", {"summary": summary, "idempotency_key": item["idempotency_key"]})
        return self.snapshot(run_id)

    def apply_pending_steer(self, run_id: str) -> dict[str, Any] | None:
        """Pop one pending steer at the next tool/model boundary (L11)."""
        run = self.store.get_assistant_run(run_id)
        if not run:
            return None
        session_id = str(run["session_id"])
        inbox = self._inbox(session_id)
        pending = list(inbox.get("pending_steer") or [])
        if not pending:
            return None
        item = pending.pop(0)
        item["status"] = "applied"
        inbox["pending_steer"] = pending
        inbox.setdefault("applied_steer", [])
        inbox["applied_steer"] = [*(inbox.get("applied_steer") or []), item][-20:]
        self._save_inbox(session_id, inbox)
        return item

    def drain_queued_messages(self, run_id: str) -> list[dict[str, Any]]:
        """After a round completes, start the next queued prompt (one at a time, L10)."""
        run = self.store.get_assistant_run(run_id)
        if not run:
            return []
        if str(run.get("status") or "") != "completed":
            return []
        session_id = str(run["session_id"])
        session = self.store.get_session(session_id) or {}
        metadata = session.get("metadata") if isinstance(session.get("metadata"), dict) else {}
        inbox = self._inbox(session_id)
        pending = [item for item in (inbox.get("queued") or []) if item.get("status") == "pending"]
        if not pending:
            return []
        item = pending[0]
        remainder = [row for row in pending[1:]]
        # mark drained on parent before spawning; leftover pending moves to the child
        for row in inbox["queued"]:
            if row.get("idempotency_key") == item.get("idempotency_key"):
                row["status"] = "drained"
            elif row.get("status") == "pending":
                row["status"] = "moved"
        self._save_inbox(session_id, inbox)
        profile = metadata.get("model_profile") if isinstance(metadata.get("model_profile"), dict) else {}
        memory_mode = str(metadata.get("memory_mode") or "on")
        # t237 C6d M1: drain runs on the worker thread (no request principal). Create the follow-up as the
        # parent's owner so it stays owner-bound, and keep the parent's memory_mode (paused memory stays paused).
        from .tenant_access import AccessStore, current_principal
        from .user_memory import principal_from_metadata
        owner = principal_from_metadata(metadata)
        token = current_principal.set(owner) if owner is not None else None
        try:
            child = self.create_run(
                prompt=str(item.get("text") or ""),
                conversation_id=str(run.get("conversation_id") or ""),
                user_task_id=str(metadata.get("user_task_id") or ""),
                project_id=str(metadata.get("project_id") or ""),
                model_profile_id=str(profile.get("id") or ""),
                model_profile_version=int(profile.get("version") or 0) if profile.get("id") else 0,
                platform_model_id=str(metadata.get("platform_model_id") or ""),
                fallback_model_id=str(metadata.get("fallback_model_id") or ""),
                reasoning_effort=str(metadata.get("reasoning_effort") or ""),
                memory_mode=memory_mode if memory_mode in {"on", "off", "incognito"} else "off",
                start=False,
            )
            if owner is not None:
                AccessStore(self.runtime.runtime_root).created_run(self.runtime, child, owner)
        finally:
            if token is not None:
                current_principal.reset(token)
        if remainder:
            child_inbox = self._inbox(str(child["session_id"]))
            child_inbox["queued"] = [
                {**row, "status": "pending"} for row in remainder
            ]
            self._save_inbox(str(child["session_id"]), child_inbox)
        self.start(str(child["id"]))
        self.store.append_event(
            session_id,
            "message_drained",
            {"child_run_id": child["id"], "idempotency_key": item.get("idempotency_key"), "text": str(item.get("text") or "")[:80]},
        )
        return [self.snapshot(str(child["id"]))]

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
            # C6b L9: keep produced stream text + files; surface them as partial_answer.
            session_id = str(run["session_id"])
            partial_event = self.store.latest_event(session_id, "assistant.stream")
            payload = (partial_event or {}).get("payload") or {}
            stream_text = ""
            if payload.get("phase") == "streaming":
                stream_text = str(payload.get("text") or "").strip()
            existing = ""
            for turn in reversed(self.store.list_turns(session_id)):
                if turn.get("role") == "assistant" and isinstance(turn.get("content"), str) and turn["content"].strip():
                    existing = turn["content"].strip()
                    break
            partial_text = stream_text or existing
            if stream_text and stream_text != existing:
                self.store.add_turn(session_id, "assistant", stream_text)
                self.store.append_event(session_id, "assistant.stream", {"text": stream_text, "phase": "cancelled_partial"})
            self.runtime.cancel(run_id)
            self.store.update_assistant_run(run_id, status="cancelled", completed_at=utc_now())
            snap = self.snapshot(run_id)
            snap["partial_answer"] = partial_text or None
            if partial_text and not str(snap.get("answer") or "").strip():
                snap["answer"] = partial_text
            return snap
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
            self._clear_supervision_holds(run_id)
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
                self.start(run_id, resume=True, priority="human")
            return self.snapshot(run_id)

    def _clear_supervision_holds(self, run_id: str) -> None:
        """A person's resume answers the open card and resets the tool breaker (counts start again from now)."""
        session = self.store.get_session(run_id)
        if not session:
            return
        metadata = dict(session.get("metadata") or {})
        had = bool(metadata.pop("attention", None)) | bool(metadata.pop("tool_circuit", None))
        metadata["breaker_reset_at"] = utc_now()
        self.store.update_session(run_id, metadata_json=metadata)
        if had:
            self.store.append_event(run_id, "attention_resolved", {"run_id": run_id, "by": "person"})

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

    def close_idle_planning(self, *, idle_seconds: int = 1800, run_id: str = "") -> dict[str, Any]:
        """Close planning rows that have no live worker, child, or recent activity.

        A finished plan becomes completed only when its delivery gate is met.
        A fresh confirmation stays a confirmation. A running tool or child run
        is left open. Nothing is cancelled.
        """
        now = datetime.now(timezone.utc)
        if run_id:
            row = self.store.get_assistant_run(run_id)
            rows = [row] if row and str(row.get("status") or "") == "planning" else []
        else:
            rows = self.store.list_assistant_runs_with_status("planning")
        closed: list[dict[str, Any]] = []
        skipped: dict[str, int] = {}

        def skip(reason: str) -> None:
            skipped[reason] = skipped.get(reason, 0) + 1

        for run in rows:
            if not isinstance(run, dict):
                continue
            try:
                outcome = self._close_one_idle_planning(run, now=now, idle_seconds=idle_seconds)
            except Exception:
                logger.exception("close_idle_planning skipped run %s", run.get("id"))
                skip("close_failed")
                continue
            if outcome is None:
                continue
            if outcome.get("skipped"):
                skip(str(outcome["skipped"]))
                continue
            closed.append(outcome)
        return {
            "checked_at": now.isoformat(),
            "closed_count": len(closed),
            "closed": closed,
            "skipped": dict(sorted(skipped.items())),
        }

    def _local_worker_alive(self, run_id: str) -> bool:
        for bucket in (self._threads, self._resume_waiters):
            worker = bucket.get(run_id)
            if worker is not None and worker.is_alive():
                return True
        return False

    def _fresh_session_approval(self, session_id: str, now: datetime) -> dict[str, Any] | None:
        for approval in self.store.list_session_approvals(session_id, status="pending"):
            deadline = _parse_utc(approval.get("expires_at"))
            if deadline is None or deadline > now:
                return approval
        return None

    def _planning_delivery_blocked(self, run: dict[str, Any]) -> bool:
        plan = run.get("plan")
        if not isinstance(plan, dict) or not plan.get("require_artifact"):
            return False
        try:
            if self._missing_expected_artifacts(run):
                return True
            return not self.store.list_deliverables(str(run.get("id") or ""))
        except Exception:
            logger.exception("planning delivery check failed for %s", run.get("id"))
            return True

    def _close_one_idle_planning(
        self,
        run: dict[str, Any],
        *,
        now: datetime,
        idle_seconds: int,
    ) -> dict[str, Any] | None:
        current_id = str(run.get("id") or "")
        if not current_id or str(run.get("status") or "") != "planning":
            return None
        # A missing or unreadable stamp is not evidence of idleness.
        if _parse_utc(run.get("updated_at")) is None:
            return {"skipped": "missing_timestamp"}
        observed_updated_at = str(run.get("updated_at"))
        with self._lock:
            if self._local_worker_alive(current_id):
                return {"skipped": "worker_alive"}
            current = self.store.get_assistant_run(current_id)
            if not current or str(current.get("status") or "") != "planning":
                return {"skipped": "already_settled"}
            if str(current.get("updated_at") or "") != observed_updated_at:
                return {"skipped": "already_settled"}
            run_updated = _parse_utc(current.get("updated_at"))
            if run_updated is None:
                return {"skipped": "missing_timestamp"}
            if (now - run_updated).total_seconds() < idle_seconds:
                return {"skipped": "recent_activity"}
            session = self.store.get_session(current_id) or {}
            metadata = session.get("metadata") if isinstance(session.get("metadata"), dict) else {}
            if metadata.get("user_pause_requested"):
                return {"skipped": "user_pause"}
            calls = self.store.list_tool_calls(current_id, limit=2000)
            if any(str(call.get("status") or "") == "running" for call in calls):
                return {"skipped": "tool_call_active"}
            stamps: list[Any] = [
                current.get("updated_at"),
                session.get("updated_at"),
                self.store.latest_event_created_at(current_id),
            ]
            for call in calls:
                stamps.extend(call.get(key) for key in ("started_at", "created_at", "completed_at"))
            newest = _newest_utc(stamps)
            if newest is None:
                return {"skipped": "missing_timestamp"}
            if (now - newest).total_seconds() < idle_seconds:
                return {"skipped": "recent_activity"}
            if self.store.has_open_child_run(current_id):
                return {"skipped": "child_run"}
            fresh_approval = self._fresh_session_approval(current_id, now)
            finished = _planning_steps_finished(current.get("plan"))
            if fresh_approval is not None:
                target, error_class, reason = "waiting_approval", "", "planning_needs_confirmation"
            elif finished and not self._planning_delivery_blocked(current):
                target, error_class, reason = "completed", "", "planning_already_finished"
            elif finished:
                target, error_class, reason = "blocked", "incomplete_deliverables", "planning_artifacts_missing"
            else:
                target, error_class, reason = "blocked", "planning_idle", "planning_idle"
            fields: dict[str, Any] = {"status": target, "error_class": error_class, "error_message": ""}
            if target == "completed":
                fields["completed_at"] = utc_now()
            changed = self.store.settle_planning_if_unchanged(current_id, observed_updated_at, **fields)
            if not changed:
                return {"skipped": "already_settled"}
            try:
                self.runtime.store.update_session(current_id, status=target)
            except KeyError:
                pass
            if target == "completed":
                event = "run_completed"
            elif target == "blocked":
                event = "run_blocked"
            else:
                event = "planning_closed"
            self.store.append_event(current_id, event, {
                "run_id": current_id,
                "previous_status": "planning",
                "status": target,
                "reason": reason,
                "recoverable": target != "completed",
            })
            return {"run_id": current_id, "from": "planning", "to": target, "reason": reason}

    def recover_incomplete(self, max_age_seconds: float | None = None) -> list[str]:
        """Resume interrupted runs (oldest first) and start the background run supervision.

        ``max_age_seconds`` is the production entry's recovery window (EVOMIND_RECOVER_WINDOW_HOURS);
        without it the window is 7 days. Older stragglers are left to the stall supervisor.
        """
        if max_age_seconds is not None:  # the production entry; tests and the legacy serve() start no monitor here
            from .run_supervision import ensure_background_supervision

            try:
                ensure_background_supervision(self.runtime)
            except Exception:
                logger.exception("background run supervision failed to start")
        try:
            self.close_idle_planning()
        except Exception:
            logger.exception("close_idle_planning failed during recover_incomplete")
        recovered: list[str] = []
        # Run supervision v1: every active run inside the window, oldest first (dsh 003 "Restart"),
        # not only the newest 200; older stragglers are left to the stall supervisor.
        window = 7 * 86400.0 if max_age_seconds is None else max(0.0, float(max_age_seconds))
        since = (datetime.now(timezone.utc) - timedelta(seconds=window)).isoformat()
        candidates = self.store.list_assistant_runs_by_status(
            ("queued", "planning", "running", "verifying", "recovering", "pausing", "paused",
             SessionStatus.WAITING_APPROVAL.value), since=since, limit=500)
        for run in candidates:
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
                if self.start(str(run["id"]), resume=True, reason="host_restarted"):
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
