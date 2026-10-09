"""Private upload catalog; no global directory scanning or owner inference."""
from __future__ import annotations

import hashlib
from pathlib import Path
from .tenant_access import AccessError


def metadata(runtime, access, principal, file_id):
    if principal is None:
        raise AccessError(403, 'file_principal_required')
    access.require('attachment', file_id, principal)
    row = runtime.store.get_attachment(file_id)
    if not row:
        raise AccessError()
    return {key: row[key] for key in ('id', 'name', 'bytes', 'sha256', 'media_type', 'created_at')}


def catalog(runtime, access, principal):
    if principal is None:
        raise AccessError(403, 'file_principal_required')
    return [metadata(runtime, access, principal, file_id) for file_id in access.resources('attachment', principal)[:1000]]


def open_verified(runtime, access, principal, file_id):
    item = metadata(runtime, access, principal, file_id)
    path = Path(runtime.store.get_attachment(file_id)['path'])
    if not path.resolve(strict=True).is_relative_to(runtime.assistant.root.resolve()):
        raise AccessError(409, 'file_location_changed')
    for parent in (path, *path.parents):
        if parent.is_symlink() or getattr(parent.lstat(), 'st_file_attributes', 0) & 1024:
            raise AccessError(409, 'file_alias_rejected')
        if parent == runtime.assistant.root:
            break
    handle = path.open('rb')
    try:
        digest = hashlib.sha256()
        size = 0
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
            size += len(block)
        if size != item['bytes'] or digest.hexdigest() != item['sha256']:
            raise AccessError(409, 'file_changed_since_upload')
        handle.seek(0)
        return item, handle
    except Exception:
        handle.close()
        raise
