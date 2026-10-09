from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

_SECRET_REF = re.compile(r"^rs_[A-Za-z0-9_-]{43}$")
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_SAFE_RUN = re.compile(r"^[A-Za-z0-9_.-]{1,160}$")
_PURPOSES = {"weather4cast_sftp", "kaggle_api"}
_LOCK = threading.RLock()
_POWERSHELL7_ENV = "EVOMIND_PWSH_PATH"
_CONSUMER_ERROR_MARKER = re.compile(rb"RUN_SECRET_CONSUMER_ERROR:([A-Za-z][A-Za-z0-9_.-]{0,64})")
_CONSUMER_ERROR_TYPES = frozenset({
    "ArgumentException",
    "ArgumentNullException",
    "AuthenticationException",
    "CryptographicException",
    "FormatException",
    "InvalidOperationException",
    "IOException",
    "JsonReaderException",
    "NotSupportedException",
    "ObjectDisposedException",
    "PlatformNotSupportedException",
    "RuntimeException",
    "UnauthorizedAccessException",
})
_CONSUMER_HOST_ERROR_TYPES = frozenset({
    "CommandNotFoundException",
    "FileNotFoundException",
    "MethodInvocationException",
    "ParameterBindingException",
    "PSInvalidOperationException",
    "RuntimeException",
    "SecurityException",
    "TypeInitializationException",
    "UnauthorizedAccessException",
})
_CONSUMER_STAGE_TYPES = frozenset({
    "Stage_identity",
    "Stage_metadata",
    "Stage_metadata_json",
    "Stage_metadata_contract",
    "Stage_metadata_expiry",
    "Stage_envelope",
    "Stage_envelope_read",
    "Stage_envelope_json",
    "Stage_envelope_contract",
    "Stage_envelope_empty",
    "Stage_envelope_nonobject",
    "Stage_envelope_truncated",
    "Stage_envelope_parse",
    "Stage_key",
    "Stage_rsa",
    "Stage_oaep",
    "Stage_gcm",
    "Stage_emit",
})


class RunSecretConsumerProcessExit(RuntimeError):
    """A consumer exited non-zero with a fixed, non-secret error marker."""

    def __init__(self, error_type: str) -> None:
        self.error_type = error_type if error_type in (_CONSUMER_ERROR_TYPES | _CONSUMER_STAGE_TYPES) else "Unknown"
        super().__init__(f"consumer_process_exit:{self.error_type}")


class RunSecretConsumerInvalidBase64(RuntimeError):
    """A successful consumer process returned a non-base64 stdout payload."""

    def __init__(self) -> None:
        super().__init__("consumer_invalid_base64")


class RunSecretConsumerHostExit(RuntimeError):
    """PowerShell failed before or around invoking the consumer script."""

    def __init__(self, error_type: str) -> None:
        safe_type = error_type if error_type in _CONSUMER_HOST_ERROR_TYPES else "Unknown"
        self.error_type = safe_type
        super().__init__(f"consumer_host_exit:{safe_type}")

_CONSUMER_PS1 = r"""
param(
  [Parameter(Mandatory=$true)][string]$EnvelopePath,
  [Parameter(Mandatory=$true)][string]$MetadataPath,
  [Parameter(Mandatory=$true)][string]$StateRoot
)
$ErrorActionPreference='Stop'
$stage='identity'
$plain=$null;$aesKey=$null;$aad=$null;$rsa=$null;$aes=$null
try {
  $stage='identity'
  $identity=[Security.Principal.WindowsIdentity]::GetCurrent().Name
  if($env:EVOMIND_ALLOW_RUN_SECRET_TEST -ne '1' -and $identity -notmatch '(?i)(^|\\)EvoMindSvc$'){throw 'EVOMINDSVC_CONTEXT_REQUIRED'}
  $stage='metadata_read'
  $metadataRaw=Get-Content -LiteralPath $MetadataPath -Raw -Encoding UTF8
  $metadata=$metadataRaw|ConvertFrom-Json
  $stage='metadata_contract'
  if([string]$metadata.schema -cne 'evomind.run_secret_claimed.v1' -or [string]$metadata.state -cne 'claimed'){throw 'RUN_SECRET_METADATA_INVALID'}
  $stage='metadata_expiry'
  $expiresMatch=[regex]::Match($metadataRaw,'"expires_at_utc"\s*:\s*"(?<value>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,7})?Z)"')
  if(-not $expiresMatch.Success){throw 'RUN_SECRET_EXPIRY_FORMAT_INVALID'}
  $expiresText=$expiresMatch.Groups['value'].Value
  $expiresAt=[DateTimeOffset]::Parse($expiresText,[Globalization.CultureInfo]::InvariantCulture,[Globalization.DateTimeStyles]::RoundtripKind)
  if($expiresAt -le [DateTimeOffset]::UtcNow){throw 'RUN_SECRET_EXPIRED'}
  $stage='envelope_read'
  $envelopeText=Get-Content -LiteralPath $EnvelopePath -Raw -Encoding UTF8
  $stage='envelope_json'
  try {
    $envelope=$envelopeText|ConvertFrom-Json
  } catch {
    if([string]::IsNullOrWhiteSpace($envelopeText)) {
      $stage='envelope_empty'
    } elseif(-not $envelopeText.TrimStart().StartsWith('{')) {
      $stage='envelope_nonobject'
    } elseif(-not $envelopeText.TrimEnd().EndsWith('}')) {
      $stage='envelope_truncated'
    } else {
      $stage='envelope_parse'
    }
    throw
  }
  $stage='envelope_contract'
  if([string]$envelope.schema -cne 'evomind.hybrid_run_secret_envelope.v1' -or [string]$envelope.algorithm -cne 'RSA-OAEP-256+A256GCM'){throw 'RUN_SECRET_ENVELOPE_INVALID'}
  $stage='key'
  $keyRecord=Get-Content -LiteralPath (Join-Path $StateRoot 'enrollment-keys\current-public-key.json') -Raw -Encoding UTF8|ConvertFrom-Json
  if([string]$keyRecord.key_id -cne [string]$metadata.key_id -or [string]$envelope.key_id -cne [string]$metadata.key_id -or [string]$keyRecord.algorithm -cne 'RSA-OAEP-256+A256GCM'){throw 'RUN_SECRET_KEY_INVALID'}
  $stage='rsa'
  $cert=Get-Item ("Cert:\LocalMachine\My\{0}" -f [string]$keyRecord.certificate_thumbprint)
  $observedKeyId=([Security.Cryptography.SHA256]::Create().ComputeHash($cert.RawData)|ForEach-Object ToString x2)-join''
  if($observedKeyId -cne [string]$metadata.key_id){throw 'RUN_SECRET_CERTIFICATE_KEY_ID_MISMATCH'}
  $rsa=[Security.Cryptography.X509Certificates.RSACertificateExtensions]::GetRSAPrivateKey($cert)
  if(-not $rsa){throw 'RUN_SECRET_PRIVATE_KEY_UNAVAILABLE'}
  $stage='oaep'
  $aesKey=$rsa.Decrypt([Convert]::FromBase64String([string]$envelope.wrapped_key),[Security.Cryptography.RSAEncryptionPadding]::OaepSHA256)
  if($aesKey.Length -ne 32){throw 'RUN_SECRET_AES_KEY_INVALID'}
  $stage='gcm'
  $iv=[Convert]::FromBase64String([string]$envelope.iv);$cipher=[Convert]::FromBase64String([string]$envelope.ciphertext);$tag=[Convert]::FromBase64String([string]$envelope.tag)
  if($iv.Length -ne 12 -or $tag.Length -ne 16){throw 'RUN_SECRET_GCM_PARAMETERS_INVALID'}
  $aadText="{0}|{1}|{2}|{3}|{4}|{5}|{6}|{7}|{8}" -f $metadata.tenant_id,$metadata.owner_principal_id,$metadata.secret_id,$metadata.purpose,$metadata.job_id,$metadata.allocation_generation,$metadata.profile_instance_id,$metadata.key_id,$expiresText
  $aad=[Text.Encoding]::UTF8.GetBytes($aadText);$plain=New-Object byte[] $cipher.Length
  $aes=[Security.Cryptography.AesGcm]::new($aesKey);$aes.Decrypt($iv,$cipher,$tag,$plain,$aad)
  $stage='emit'
  [Console]::Out.Write([Convert]::ToBase64String($plain))
}
catch {
  # Keep consumer failures diagnosable without returning stderr or plaintext.
  [Console]::Error.WriteLine("RUN_SECRET_CONSUMER_ERROR:Stage_" + $stage)
  exit 86
} finally {
  if($aes){$aes.Dispose()};if($rsa){$rsa.Dispose()}
  if($plain){[Array]::Clear($plain,0,$plain.Length)};if($aesKey){[Array]::Clear($aesKey,0,$aesKey.Length)};if($aad){[Array]::Clear($aad,0,$aad.Length)}
}
""".strip()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_time(value: Any) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _resolve_powershell7() -> str:
    """Resolve a PowerShell 7 executable without a PowerShell 5.1 fallback.

    The consumer uses .NET AesGcm and RSA OAEP-SHA256. Windows PowerShell
    5.1 can exist on PATH while lacking those APIs, so selecting it would
    consume a one-time envelope and only then fail.
    """

    candidates: list[str] = []
    configured = str(os.environ.get(_POWERSHELL7_ENV) or "").strip()
    if configured:
        candidates.append(configured)
    for name in ("pwsh", "pwsh.exe"):
        located = shutil.which(name)
        if located:
            candidates.append(located)
    for variable in ("ProgramW6432", "ProgramFiles", "ProgramFiles(x86)"):
        root = str(os.environ.get(variable) or "").strip()
        if root:
            candidates.append(str(Path(root) / "PowerShell" / "7" / "pwsh.exe"))

    seen: set[str] = set()
    for raw in candidates:
        candidate = str(Path(raw).expanduser())
        key = candidate.casefold()
        if key in seen:
            continue
        seen.add(key)
        if Path(candidate).is_file():
            return candidate
    raise RuntimeError("run secret consumer requires PowerShell 7 (pwsh)")


def _consumer_error_type(stderr: bytes) -> str:
    match = _CONSUMER_ERROR_MARKER.search(stderr or b"")
    if not match:
        return "Unknown"
    try:
        value = match.group(1).decode("ascii")
    except UnicodeDecodeError:
        return "Unknown"
    return value if value in (_CONSUMER_ERROR_TYPES | _CONSUMER_STAGE_TYPES) else "Unknown"


def _consumer_host_error_type(stderr: bytes) -> str:
    match = _CONSUMER_ERROR_MARKER.search(stderr or b"")
    if not match:
        return "Unknown"
    try:
        value = match.group(1).decode("ascii")
    except UnicodeDecodeError:
        return "Unknown"
    return value if value in _CONSUMER_HOST_ERROR_TYPES else "Unknown"


def _powershell_single_quote(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _ref_hash(secret_ref: str) -> str:
    return hashlib.sha256(secret_ref.encode("utf-8")).hexdigest()


def _identity_matches(record: dict[str, Any], identity: dict[str, Any]) -> bool:
    return all(
        record.get(name) == identity.get(name)
        for name in (
            "tenant_id",
            "owner_principal_id",
            "job_id",
            "allocation_generation",
            "profile_instance_id",
            "allocation_binding_id",
            "credential_profile",
        )
    )


class RunSecretStore:
    def __init__(
        self,
        runtime_root: str | Path,
        *,
        consumer: Callable[[Path, Path], bytes] | None = None,
        now: Callable[[], datetime] = _now,
    ) -> None:
        self.runtime_root = Path(runtime_root).resolve()
        self.root = self.runtime_root / "run-secrets"
        self.pending = self.root / "pending"
        self.claimed = self.root / "claimed"
        self.consumed = self.root / "consumed"
        self._consumer = consumer or self._consume_with_service_certificate
        self._now = now

    def _pending_paths(self, secret_ref: str) -> tuple[Path, Path]:
        digest = _ref_hash(secret_ref)
        return self.pending / f"{digest}.json", self.pending / f"{digest}.envelope"

    def available_purposes_for_run(self, run_id: str) -> dict[str, int]:
        """Return only counts of live claimed secrets, never refs or values."""

        if not _SAFE_RUN.fullmatch(run_id):
            return {}
        counts: dict[str, int] = {}
        root = self.claimed / run_id
        with _LOCK:
            for metadata_path in sorted(root.glob("*.json")) if root.is_dir() else []:
                try:
                    record = json.loads(metadata_path.read_text(encoding="utf-8"))
                    envelope_path = metadata_path.with_suffix(".envelope")
                    purpose = str(record.get("purpose") or "")
                    if (
                        record.get("schema") == "evomind.run_secret_claimed.v1"
                        and record.get("state") == "claimed"
                        and purpose in _PURPOSES
                        and envelope_path.is_file()
                        and _parse_time(record.get("expires_at_utc")) > self._now()
                        and hashlib.sha256(envelope_path.read_bytes()).hexdigest() == record.get("envelope_sha256")
                    ):
                        counts[purpose] = counts.get(purpose, 0) + 1
                except (OSError, TypeError, ValueError, json.JSONDecodeError):
                    continue
        return counts

    def _claimed_paths(self, run_id: str, secret_ref: str) -> tuple[Path, Path]:
        digest = _ref_hash(secret_ref)
        root = self.claimed / run_id
        return root / f"{digest}.json", root / f"{digest}.envelope"

    def claimed_ref_is_live(
        self,
        run_id: str,
        secret_ref: str,
        *,
        managed_hpc_identity: dict[str, Any],
        purpose: str,
    ) -> bool:
        """Verify an existing same-Run claim without exposing its reference."""

        if not _SAFE_RUN.fullmatch(run_id) or not _SECRET_REF.fullmatch(secret_ref) or purpose not in _PURPOSES:
            return False
        metadata_path, envelope_path = self._claimed_paths(run_id, secret_ref)
        with _LOCK:
            try:
                record = json.loads(metadata_path.read_text(encoding="utf-8"))
                return bool(
                    isinstance(record, dict)
                    and record.get("schema") == "evomind.run_secret_claimed.v1"
                    and record.get("state") == "claimed"
                    and record.get("run_id") == run_id
                    and record.get("purpose") == purpose
                    and record.get("secret_ref_sha256") == _ref_hash(secret_ref)
                    and _identity_matches(record, managed_hpc_identity)
                    and _parse_time(record.get("expires_at_utc")) > self._now()
                    and envelope_path.is_file()
                    and hashlib.sha256(envelope_path.read_bytes()).hexdigest() == record.get("envelope_sha256")
                )
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                return False

    def _validate_pending(
        self,
        secret_ref: str,
        managed_hpc_identity: dict[str, Any],
    ) -> tuple[dict[str, Any], Path, Path]:
        if not _SECRET_REF.fullmatch(secret_ref):
            raise ValueError("run secret reference is invalid")
        metadata_path, envelope_path = self._pending_paths(secret_ref)
        record = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            not isinstance(record, dict)
            or record.get("schema") != "evomind.run_secret_pending.v1"
            or record.get("state") != "pending"
            or record.get("secret_ref_sha256") != _ref_hash(secret_ref)
            or record.get("purpose") not in _PURPOSES
            or not _identity_matches(record, managed_hpc_identity)
            or _parse_time(record.get("expires_at_utc")) <= self._now()
        ):
            raise ValueError("run secret reference is mismatched or expired")
        envelope_hash = hashlib.sha256(envelope_path.read_bytes()).hexdigest()
        if not _SHA256.fullmatch(str(record.get("envelope_sha256") or "")) or envelope_hash != record["envelope_sha256"]:
            raise ValueError("run secret envelope hash mismatch")
        return record, metadata_path, envelope_path

    def claim_refs(
        self,
        run_id: str,
        secret_refs: list[str] | None,
        *,
        managed_hpc_identity: dict[str, Any],
    ) -> list[str]:
        if not _SAFE_RUN.fullmatch(run_id):
            raise ValueError("run id is invalid")
        refs = list(dict.fromkeys(str(value) for value in (secret_refs or []) if str(value)))
        if len(refs) > 4:
            raise ValueError("too many run secret references")
        if not refs:
            return []
        with _LOCK:
            validated = [self._validate_pending(secret_ref, managed_hpc_identity) for secret_ref in refs]
            moved: list[tuple[Path, Path, Path, Path]] = []
            try:
                for secret_ref, (record, pending_meta, pending_envelope) in zip(refs, validated):
                    claimed_meta, claimed_envelope = self._claimed_paths(run_id, secret_ref)
                    if claimed_meta.exists() or claimed_envelope.exists():
                        raise ValueError("run secret reference was already claimed")
                    claimed_meta.parent.mkdir(parents=True, exist_ok=True)
                    moved.append((pending_meta, pending_envelope, claimed_meta, claimed_envelope))
                    next_record = {
                        **record,
                        "schema": "evomind.run_secret_claimed.v1",
                        "state": "claimed",
                        "run_id": run_id,
                        "claimed_at_utc": self._now().isoformat(),
                    }
                    _atomic_json(claimed_meta, next_record)
                    os.replace(pending_envelope, claimed_envelope)
                    pending_meta.unlink()
            except Exception:
                for pending_meta, pending_envelope, claimed_meta, claimed_envelope in reversed(moved):
                    if claimed_envelope.exists():
                        os.replace(claimed_envelope, pending_envelope)
                    if claimed_meta.exists():
                        previous = json.loads(claimed_meta.read_text(encoding="utf-8"))
                        previous.update({"schema": "evomind.run_secret_pending.v1", "state": "pending"})
                        previous.pop("run_id", None)
                        previous.pop("claimed_at_utc", None)
                        _atomic_json(pending_meta, previous)
                        claimed_meta.unlink(missing_ok=True)
                raise
        return refs

    def _consume_with_service_certificate(self, envelope_path: Path, metadata_path: Path) -> bytes:
        executable = _resolve_powershell7()
        consumer_path = self.root / "consumer" / "Consume-RunSecretEnvelope.ps1"
        expected_hash = hashlib.sha256((_CONSUMER_PS1 + "\n").encode("utf-8")).hexdigest()
        if not consumer_path.is_file() or hashlib.sha256(consumer_path.read_bytes()).hexdigest() != expected_hash:
            consumer_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = consumer_path.with_suffix(".ps1.tmp")
            temporary.write_text(_CONSUMER_PS1 + "\n", encoding="utf-8", newline="\n")
            os.replace(temporary, consumer_path)
        state_root = Path(os.environ.get("WORKSTATION_BYOA_STATE_ROOT") or r"C:\ProgramData\EvoMind\byoa")
        invocation = (
            "$ErrorActionPreference='Stop';"
            "try { "
            f"& {_powershell_single_quote(str(consumer_path))}"
            f" -EnvelopePath {_powershell_single_quote(str(envelope_path))}"
            f" -MetadataPath {_powershell_single_quote(str(metadata_path))}"
            f" -StateRoot {_powershell_single_quote(str(state_root))};"
            "if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) { exit $LASTEXITCODE }"
            "} catch { "
            "[Console]::Error.WriteLine('RUN_SECRET_CONSUMER_ERROR:' + $_.Exception.GetType().Name);"
            "exit 87 }"
        )
        completed = subprocess.run(
            [executable, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", invocation],
            capture_output=True,
            timeout=30,
            check=False,
        )
        if completed.returncode != 0:
            marker_type = _consumer_error_type(completed.stderr)
            if marker_type == "Unknown":
                raise RunSecretConsumerHostExit(_consumer_host_error_type(completed.stderr))
            raise RunSecretConsumerProcessExit(marker_type)
        try:
            return base64.b64decode(completed.stdout.strip(), validate=True)
        except ValueError as exc:
            raise RunSecretConsumerInvalidBase64() from exc

    def consume_for_run(
        self,
        run_id: str,
        *,
        purpose: str,
        managed_hpc_identity: dict[str, Any],
    ) -> dict[str, Any]:
        if not _SAFE_RUN.fullmatch(run_id) or purpose not in _PURPOSES:
            raise ValueError("run secret consumption request is invalid")
        with _LOCK:
            records: list[tuple[dict[str, Any], Path, Path]] = []
            root = self.claimed / run_id
            for metadata_path in sorted(root.glob("*.json")) if root.is_dir() else []:
                record = json.loads(metadata_path.read_text(encoding="utf-8"))
                envelope_path = metadata_path.with_suffix(".envelope")
                if (
                    record.get("purpose") == purpose
                    and record.get("state") == "claimed"
                    and envelope_path.is_file()
                    and _parse_time(record.get("expires_at_utc")) > self._now()
                    and hashlib.sha256(envelope_path.read_bytes()).hexdigest() == record.get("envelope_sha256")
                ):
                    records.append((record, metadata_path, envelope_path))
            if purpose == "kaggle_api":
                if not records:
                    raise FileNotFoundError("at least one controlled Kaggle secret is required")
                # The browser creates one independently consumable reference
                # per official Kaggle competition.  Consume the oldest one for
                # this adapter invocation; the next invocation consumes the
                # next reference from the same Run.
                records = records[:1]
            elif not records:
                raise FileNotFoundError("at least one controlled Weather4cast secret is required")
            else:
                # A blocked Run may retain more than one unconsumed reference
                # after a browser retry. Consume the newest valid reference
                # deterministically; older references remain independently
                # auditable and are never silently reused.
                records.sort(
                    key=lambda item: _parse_time(item[0].get("claimed_at_utc")),
                    reverse=True,
                )
                records = records[:1]
            record, metadata_path, envelope_path = records[0]
            if (
                record.get("schema") != "evomind.run_secret_claimed.v1"
                or record.get("run_id") != run_id
                or not _identity_matches(record, managed_hpc_identity)
                or _parse_time(record.get("expires_at_utc")) <= self._now()
                or hashlib.sha256(envelope_path.read_bytes()).hexdigest() != record.get("envelope_sha256")
            ):
                raise ValueError("claimed run secret identity is invalid")
            status = "failed"
            error_class = "run_secret_consumer_failed"
            try:
                plaintext = self._consumer(envelope_path, metadata_path)
                payload = json.loads(plaintext.decode("utf-8"))
                if not isinstance(payload, dict) or payload.get("schema") != "evomind.run_secret_payload.v1" or payload.get("purpose") != purpose:
                    raise ValueError("run secret payload contract mismatch")
                aad_text = "|".join(
                    str(record.get(name) or "")
                    for name in (
                        "tenant_id",
                        "owner_principal_id",
                        "secret_id",
                        "purpose",
                        "job_id",
                        "allocation_generation",
                        "profile_instance_id",
                        "key_id",
                        "expires_at_utc",
                    )
                )
                if payload.get("binding_sha256") != hashlib.sha256(aad_text.encode("utf-8")).hexdigest():
                    raise ValueError("run secret payload binding mismatch")
                allowed = {
                    "weather4cast_sftp": {"schema", "purpose", "binding_sha256", "host", "port", "username", "password", "remote_path", "target_subdir"},
                    "kaggle_api": {"schema", "purpose", "binding_sha256", "username", "token"},
                }[purpose]
                if not set(payload).issubset(allowed):
                    raise ValueError("run secret payload contains unsupported fields")
                status, error_class = "consumed", ""
                return payload
            except Exception as exc:
                if isinstance(exc, RunSecretConsumerProcessExit):
                    error_class = str(exc)
                elif isinstance(exc, RunSecretConsumerHostExit):
                    error_class = str(exc)
                elif isinstance(exc, RunSecretConsumerInvalidBase64):
                    error_class = str(exc)
                else:
                    error_class = type(exc).__name__
                raise
            finally:
                tombstone = {
                    "schema": "evomind.run_secret_consumed_tombstone.v1",
                    "run_id": run_id,
                    "secret_id": record.get("secret_id"),
                    "purpose": purpose,
                    "envelope_sha256": record.get("envelope_sha256"),
                    "status": status,
                    "error_class": error_class,
                    "consumed_at_utc": self._now().isoformat(),
                    "secret_values_logged": False,
                }
                _atomic_json(self.consumed / run_id / f"{record['secret_ref_sha256']}.json", tombstone)
                envelope_path.unlink(missing_ok=True)
                metadata_path.unlink(missing_ok=True)


__all__ = ["RunSecretStore"]
