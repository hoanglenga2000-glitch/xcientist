"""t238 C7b D6: a tenant deletes or exports its own uploaded data (資料).

Ownership is the server-derived principal only: a non-owner gets 404 (no existence leak); the
operator (no principal: service token / verified local automation) may delete any upload.
Delete is crash-safe: mark `deleting` → remove files → drop bindings → rows + usage + deletion_log
in one transaction; start-up recovery finishes any upload left in `deleting`. Copies training made
of the data go too (see tenant_purge); if one cannot be removed yet the result says `partial`.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO

from . import tenant_purge
from .tenant_access import AccessError, Principal

ATTACHMENT_IN_USE_MESSAGE = "任务「{task}」正在运行，结束后才能删除这个文件。"
EXPORT_BUSY_MESSAGE = "现在导出的人较多，请稍后再试。"
DELETE_CONFIRM_MESSAGE = "请确认要删除你上传的全部资料。"
UNTITLED_TASK = "未命名任务"
MAX_EXPORTS = 2
_EXPORTS = threading.BoundedSemaphore(MAX_EXPORTS)
_UNSAFE_ZIP_NAME = re.compile(r'[\x00-\x1f\x7f/\\:*?"<>|]+')


def _owner_check(runtime, access, principal: Principal | None, attachment: dict | None) -> dict:
    """The library attachment (run_id '') of an owned upload, else 404."""
    if not attachment or str(attachment.get("run_id") or ""):
        raise AccessError()
    if principal is not None and not (
        access.owns("attachment", str(attachment["id"]), principal)
        or access.owns("upload", str(attachment["upload_id"]), principal)
    ):
        raise AccessError()
    return attachment


def library_attachment_for_upload(runtime, upload_id: str) -> dict | None:
    for row in runtime.store.list_attachments_by_upload_id(upload_id):
        if not str(row.get("run_id") or ""):
            return row
    return None


def _task_titles(user_tasks, run_ids: list[str]) -> dict[str, tuple[str, str]]:
    """run_id → (task_id, title) for runs linked to a user task (one query per call)."""
    if not run_ids:
        return {}
    marks = ",".join("?" for _ in run_ids)
    with user_tasks.connect() as connection:
        rows = connection.execute(
            f"SELECT r.run_id, t.id, t.title FROM user_task_runs r JOIN user_tasks t ON t.id=r.task_id WHERE r.run_id IN ({marks})",
            run_ids,
        ).fetchall()
    return {str(row[0]): (str(row[1]), str(row[2] or "")) for row in rows}


def referenced_by(runtime, access, principal: Principal | None, user_tasks, upload_id: str, attachment_id: str) -> list[dict]:
    """Runs whose inputs/ hold a copy of this upload, and tasks that selected the file.

    Only runs/tasks the caller owns are named; another tenant's run never appears.
    """
    from .assistant_runs import TERMINAL_RUN_STATES

    runs: dict[str, bool] = {}
    for row in runtime.store.list_attachments_by_upload_id(upload_id):
        run_id = str(row.get("run_id") or "")
        if not run_id or run_id in runs:
            continue
        if principal is not None and not access.owns_run(runtime, run_id, principal):
            continue
        run = runtime.store.get_assistant_run(run_id)
        if run is not None:
            runs[run_id] = str(run.get("status") or "") not in TERMINAL_RUN_STATES
    titles = _task_titles(user_tasks, list(runs))
    items: dict[str, dict] = {}
    for run_id, active in runs.items():
        task_id, title = titles.get(run_id, ("", ""))
        key = task_id or run_id
        current = items.get(key)
        items[key] = {"task_id": task_id, "task_title": title or UNTITLED_TASK, "run_id": run_id,
                      "active": active or bool(current and current["active"])}
    with user_tasks.connect() as connection:
        selected = connection.execute(
            "SELECT t.id, t.title, t.tenant, t.owner FROM user_task_files f JOIN user_tasks t ON t.id=f.task_id WHERE f.file_id=?",
            (attachment_id,),
        ).fetchall()
    for task_id, title, tenant, owner in selected:
        if principal is not None and (tenant, owner) != (principal.tenant_id, principal.owner_id):
            continue
        items.setdefault(str(task_id), {"task_id": str(task_id), "task_title": str(title or "") or UNTITLED_TASK,
                                        "run_id": "", "active": False})
    return sorted(items.values(), key=lambda item: (not item["active"], item["task_title"]))


def usage(runtime, access, principal, user_tasks, attachment_id: str) -> dict:
    attachment = _owner_check(runtime, access, principal, runtime.store.get_attachment(attachment_id))
    return {"id": attachment_id, "referenced_by": referenced_by(
        runtime, access, principal, user_tasks, str(attachment["upload_id"]), attachment_id)}


def _actor(principal: Principal | None) -> str:
    return "operator" if principal is None else f"{principal.tenant_id}:{principal.owner_id}"


def _cleanup_fields(count: int) -> dict:
    return {"cleanup_pending": count, "message": tenant_purge.CLEANUP_PENDING_MESSAGE} if count else {}


def delete_upload(runtime, access, principal, user_tasks, *, upload_id: str = "", attachment_id: str = "",
                  purge: "tenant_purge.PurgeSession | None" = None) -> dict:
    """Delete one owned upload; 404 for a non-owner, 409 when an active run uses it."""
    from .assistant_runs import UploadLimitError, UPLOAD_COMPLETING_MESSAGE, _SAFE_ID

    if attachment_id:
        attachment = runtime.store.get_attachment(attachment_id)
        upload_id = str((attachment or {}).get("upload_id") or "")
    if not _SAFE_ID.fullmatch(upload_id or "-"):
        raise AccessError()
    if principal is not None and not (
        access.owns("upload", upload_id, principal) or (attachment_id and access.owns("attachment", attachment_id, principal))
        or access.owns("deleted:upload", upload_id, principal)  # repeat delete → already_deleted (H2)
    ):
        raise AccessError()
    upload = runtime.store.get_upload(upload_id)
    if upload is None:
        logged = runtime.store.deletion_logged(upload_id)
        if logged and (principal is None or logged["tenant_id"] == principal.tenant_id):
            left = tenant_purge.pending(runtime.runtime_root, str(logged["tenant_id"]))
            return {"upload_id": upload_id, "status": "already_deleted", **_cleanup_fields(len(left))}
        raise AccessError()
    if upload["status"] == "completing":
        raise UploadLimitError(409, "upload_completing", UPLOAD_COMPLETING_MESSAGE)
    library = library_attachment_for_upload(runtime, upload_id)
    service = runtime.assistant
    with service._lock:
        active = [item for item in referenced_by(runtime, access, None, user_tasks, upload_id, str((library or {}).get("id") or ""))
                  if item["active"]]
        if active:
            title = active[0]["task_title"] if principal is None or access.owns_run(runtime, active[0]["run_id"], principal) else UNTITLED_TASK
            raise UploadLimitError(409, "attachment_in_use", ATTACHMENT_IN_USE_MESSAGE.format(task=title))
        status = runtime.store.mark_upload_deleting(upload_id)
    if status != "deleting":
        raise AccessError()
    return finish_delete(runtime, upload_id, actor=_actor(principal), user_tasks=user_tasks, purge=purge)


def finish_delete(runtime, upload_id: str, *, actor: str, user_tasks=None, service=None,
                  purge: "tenant_purge.PurgeSession | None" = None) -> dict:
    """Idempotent tail of a delete (also run by start-up recovery for a crashed delete)."""
    from .tenant_access import AccessStore
    from .user_projects import ProjectStore
    from .user_tasks import UserTasks

    recovering = service is not None  # start-up recovery: before the port binds, so nothing remote here
    service = service or runtime.assistant  # start-up recovery runs before runtime.assistant is set
    upload = runtime.store.get_upload(upload_id) or {}
    rows = runtime.store.list_attachments_by_upload_id(upload_id)
    library = next((row for row in rows if not str(row.get("run_id") or "")), None)
    tenant = runtime.store.upload_tenant(upload_id)
    if not tenant:
        owner = AccessStore(runtime.runtime_root).owner_for("upload", upload_id)
        tenant = owner.tenant_id if owner is not None else "__operator__"
    from .assistant_runs import TERMINAL_RUN_STATES, UploadLimitError

    # t238 C7b D6-M1: a clone whose run is non-terminal (or still being created, no run row yet) means an
    # active run reads this file. Refuse before touching anything. While the library final is intact the
    # upload goes back to `complete`; otherwise it stays `deleting` (unbindable) and start-up retries.
    for row in rows:
        run_id = str(row.get("run_id") or "")
        run = runtime.store.get_assistant_run(run_id) if run_id else None
        if run_id and (run is None or str(run.get("status") or "") not in TERMINAL_RUN_STATES):
            if library is not None and Path(str(library["path"])).is_file():
                runtime.store.unmark_upload_deleting(upload_id)
            raise UploadLimitError(409, "attachment_in_use", ATTACHMENT_IN_USE_MESSAGE.format(task=UNTITLED_TASK))
    # t238 C7c FIX1 H5: before the rows go, a crash here is finished by start-up recovery.
    session = purge or tenant_purge.PurgeSession(runtime.runtime_root, service.task_root, defer_remote=recovering)
    try:
        left = session.purge(sorted({str(row.get("run_id") or "") for row in rows} - {""}), tenant_id=tenant, actor=actor)
    finally:
        if purge is None:
            session.close()
    files = 0
    for row in rows:
        run_id = str(row.get("run_id") or "")
        if run_id:
            files += int(_remove_run_input(service.task_root, run_id, Path(str(row["path"]))))
    upload_dir = service.upload_root / upload_id
    if upload_dir.is_dir() and not upload_dir.is_symlink():
        files += sum(1 for path in upload_dir.rglob("*") if path.is_file())
        import shutil
        shutil.rmtree(upload_dir)
    if library is not None:
        file_id = str(library["id"])
        tasks = user_tasks or UserTasks(runtime.runtime_root)
        with tasks.connect() as connection:
            connection.execute("DELETE FROM user_task_files WHERE file_id=?", (file_id,))
        with ProjectStore(runtime.runtime_root).connect() as connection:
            connection.execute("DELETE FROM project_files WHERE file_id=?", (file_id,))
    name = str(upload.get("original_name") or (library or {}).get("name") or "")
    runtime.store.finish_upload_delete(upload_id, {
        "tenant_id": tenant, "attachment_id": str((library or {}).get("id") or ""),
        "bytes": int((library or {}).get("bytes") or upload.get("total_bytes") or 0), "files": files,
        "actor": actor, "name_sha256": hashlib.sha256(name.encode("utf-8")).hexdigest(),
    })
    # t238 C7b H2: the ownership rows go last (the tenant fallback above still needs them). A crash
    # before this line leaves only stale owner rows: start-up `forget_deleted_owner_rows` drops them via
    # deletion_log, and the catalog skips + heals any it meets meanwhile.
    AccessStore(runtime.runtime_root).forget_many(
        [("upload", upload_id)] + [("attachment", str(row["id"])) for row in rows])
    return {"upload_id": upload_id, "status": "partial" if left else "deleted", "files_removed": files, **_cleanup_fields(len(left))}


def forget_deleted_owner_rows(runtime) -> int:
    """Start-up repair (t238 C7b H2): drop owner rows of uploads/attachments recorded in deletion_log."""
    from .tenant_access import AccessStore

    access = AccessStore(runtime.runtime_root)
    uploads, attachments = set(access.owners_by_kind("upload")), set(access.owners_by_kind("attachment"))
    stale = []
    for upload_id, attachment_id in runtime.store.deleted_upload_ids():
        if upload_id in uploads and runtime.store.get_upload(upload_id) is None:
            stale.append(("upload", upload_id))
        if attachment_id in attachments and runtime.store.get_attachment(attachment_id) is None:
            stale.append(("attachment", attachment_id))
    return access.forget_many(stale)


def _remove_run_input(task_root: Path, run_id: str, path: Path) -> bool:
    """Unlink a run-input copy only when it is a regular file directly under `<run>/inputs/`."""
    inputs = (task_root / run_id / "inputs")
    try:
        if path.parent.resolve(strict=True) != inputs.resolve(strict=True):
            return False
    except (OSError, RuntimeError):
        return False
    if path.is_symlink() or not path.is_file():
        return False
    path.unlink()
    return True


def delete_all(runtime, access, principal, user_tasks, body: dict) -> dict:
    from .assistant_runs import UploadLimitError

    if principal is None:
        raise AccessError(403, "tenant_principal_required")
    if body != {"confirm": "DELETE"}:
        raise UploadLimitError(400, "delete_confirm_required", DELETE_CONFIRM_MESSAGE)
    deleted, in_use = [], []
    session = tenant_purge.PurgeSession(runtime.runtime_root, runtime.assistant.task_root)
    try:
        for upload_id in access.resources("upload", principal):
            if runtime.store.get_upload(upload_id) is None:
                continue
            try:
                result = delete_upload(runtime, access, principal, user_tasks, upload_id=upload_id, purge=session)
            except UploadLimitError as exc:
                in_use.append({"upload_id": upload_id, "error": exc.code, "message": exc.message})
                continue
            except AccessError:
                continue
            if result["status"] in {"deleted", "partial"}:
                deleted.append(upload_id)
    finally:
        session.close()
    left = tenant_purge.pending(runtime.runtime_root, principal.tenant_id)
    return {"deleted": deleted, "skipped": in_use, "usage": runtime.store.get_tenant_usage(principal.tenant_id),
            **_cleanup_fields(len(left))}


# ---------------------------------------------------------------- export


def _zip_name(raw: str, used: set[str]) -> str:
    name = _UNSAFE_ZIP_NAME.sub("_", Path(str(raw or "")).name).strip(" .") or "attachment.bin"
    stem, suffix = (name[: -len(Path(name).suffix)], Path(name).suffix) if Path(name).suffix else (name, "")
    candidate, n = name, 1
    while candidate.lower() in used or candidate == "manifest.json":
        n += 1
        candidate = f"{stem} ({n}){suffix}"
    used.add(candidate.lower())
    return f"files/{candidate}"


def export_items(runtime, access, principal) -> list[tuple[dict, dict]]:
    """(upload, library attachment) pairs the caller owns, oldest first."""
    if principal is None:
        raise AccessError(403, "tenant_principal_required")
    items = []
    for attachment_id in reversed(access.resources("attachment", principal)):
        attachment = runtime.store.get_attachment(attachment_id)
        if not attachment or str(attachment.get("run_id") or ""):
            continue
        upload = runtime.store.get_upload(str(attachment["upload_id"]))
        if upload and upload["status"] == "complete":
            items.append((upload, attachment))
    return items


def acquire_export_slot():
    from .assistant_runs import UploadLimitError
    if not _EXPORTS.acquire(blocking=False):
        raise UploadLimitError(429, "export_busy", EXPORT_BUSY_MESSAGE)
    return _EXPORTS


def export_filename(now: datetime | None = None) -> str:
    return f"evomind-export-{(now or datetime.now(timezone.utc)).strftime('%Y%m%d')}.zip"


def write_export(runtime, items: list[tuple[dict, dict]], output: BinaryIO) -> dict:
    """Stream a ZIP64 archive of the files + manifest.json to `output` (never buffered whole)."""
    used: set[str] = set()
    entries = []
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for upload, attachment in items:
            path = Path(str(attachment["path"]))
            if path.is_symlink() or not path.is_file():
                continue
            arcname = _zip_name(str(upload.get("original_name") or attachment["name"]), used)
            digest, size = hashlib.sha256(), 0
            info = zipfile.ZipInfo(arcname, date_time=datetime.now(timezone.utc).timetuple()[:6])
            with path.open("rb") as source, archive.open(info, "w", force_zip64=True) as target:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    target.write(block)
                    digest.update(block)
                    size += len(block)
            trainability = runtime.assistant._trainability(attachment)
            entries.append({
                "upload_id": str(upload["id"]), "attachment_id": str(attachment["id"]), "name": arcname[len("files/"):],
                "bytes": size, "sha256": digest.hexdigest(), "sha256_matches_upload": digest.hexdigest() == attachment["sha256"],
                "created_at": str(attachment.get("created_at") or ""),
                "trainability": {"trainable": trainability.get("trainable"), "reasons": list(trainability.get("reasons") or [])},
            })
        manifest = {"schema": "evomind.tenant_export.v1", "exported_at": datetime.now(timezone.utc).isoformat(),
                    "file_count": len(entries), "files": entries}
        archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest
