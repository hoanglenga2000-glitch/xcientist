"""Private, Run-scoped Windows DPAPI storage for short-lived public asset URLs."""
from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path
import re
import subprocess

MAX_BYTES = 64 * 1024


def descriptor_path(workspace: Path, digest: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise ValueError("model_descriptor_digest_invalid")
    root = Path(workspace).resolve(strict=True)
    path = root / "work" / ".model-descriptors" / (digest + ".dpapi")
    current = root
    for component in path.relative_to(root).parts:
        current = current / component
        if current.is_symlink() or (current.exists() and getattr(current.lstat(), "st_file_attributes", 0) & 1024):
            raise ValueError("model_descriptor_reparse_path_rejected")
    path.resolve(strict=False).relative_to(root)
    return path


def _powershell(script: str, path: Path, payload: bytes = b"") -> bytes:
    if os.name != "nt":
        raise ValueError("model_descriptor_store_requires_windows")
    environment = {**os.environ, "EVOMIND_MODEL_DESCRIPTOR_BLOB": str(path)}
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
        input=payload, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
        env=environment, creationflags=subprocess.CREATE_NO_WINDOW,
    )
    if result.returncode or len(result.stdout) > MAX_BYTES:
        raise ValueError("model_descriptor_protection_failed")
    return result.stdout


def _unprotect(path: Path) -> bytes:
    return _powershell(r"""
$ErrorActionPreference='Stop';$ProgressPreference='SilentlyContinue'
Add-Type -AssemblyName System.Security
$cipher=[IO.File]::ReadAllBytes($env:EVOMIND_MODEL_DESCRIPTOR_BLOB)
$plain=[Security.Cryptography.ProtectedData]::Unprotect($cipher,$null,[Security.Cryptography.DataProtectionScope]::LocalMachine)
[Console]::OpenStandardOutput().Write($plain,0,$plain.Length)
""", path)


def load_descriptor(workspace: Path, digest: str) -> bytes:
    from .managed_model_download import validate_cdn_descriptor
    path = descriptor_path(workspace, digest)
    if not path.is_file() or path.stat().st_size > MAX_BYTES + 4096:
        raise ValueError("model_descriptor_blob_missing_or_oversized")
    payload = _unprotect(path)
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError("model_descriptor_hash_mismatch")
    validate_cdn_descriptor(payload, minimum_remaining=1200)
    return payload


def seal_descriptor(workspace: Path, payload: bytes) -> dict:
    from .managed_model_download import validate_cdn_descriptor
    descriptor = validate_cdn_descriptor(payload, minimum_remaining=1300)
    digest = hashlib.sha256(payload).hexdigest()
    path = descriptor_path(workspace, digest)
    if path.exists():
        load_descriptor(workspace, digest)
        return {"sha256": digest, "reused": True, "expires_at": descriptor["expires_at"]}
    _powershell(r"""
$ErrorActionPreference='Stop';$ProgressPreference='SilentlyContinue'
Add-Type -AssemblyName System.Security
$path=$env:EVOMIND_MODEL_DESCRIPTOR_BLOB
$folder=[IO.Path]::GetDirectoryName($path)
$current=[Security.Principal.WindowsIdentity]::GetCurrent().User
$system=[Security.Principal.SecurityIdentifier]::new('S-1-5-18')
$admins=[Security.Principal.SecurityIdentifier]::new('S-1-5-32-544')
$service=([Security.Principal.NTAccount]::new($env:COMPUTERNAME,'EvoMindSvc')).Translate([Security.Principal.SecurityIdentifier])
$allowed=@($current.Value,$system.Value,$admins.Value,$service.Value)
if(-not (Test-Path -LiteralPath $folder)){
 [IO.Directory]::CreateDirectory($folder)|Out-Null
 $acl=[Security.AccessControl.DirectorySecurity]::new()
 $acl.SetAccessRuleProtection($true,$false)
 foreach($sid in @($current,$system,$admins,$service)){
  $rule=[Security.AccessControl.FileSystemAccessRule]::new($sid,'FullControl','ContainerInherit,ObjectInherit','None','Allow')
  $acl.AddAccessRule($rule)
 }
 Set-Acl -LiteralPath $folder -AclObject $acl
}
$item=Get-Item -LiteralPath $folder
if(($item.Attributes -band [IO.FileAttributes]::ReparsePoint)-ne 0){throw 'private_folder_reparse'}
$acl=Get-Acl -LiteralPath $folder
if(-not $acl.AreAccessRulesProtected){throw 'private_folder_inheritance'}
foreach($rule in $acl.Access){
 $sid=$rule.IdentityReference.Translate([Security.Principal.SecurityIdentifier]).Value
 if($rule.AccessControlType -eq 'Allow' -and $sid -notin $allowed){throw 'private_folder_broad_acl'}
}
$buffer=[IO.MemoryStream]::new()
[Console]::OpenStandardInput().CopyTo($buffer)
$plain=$buffer.ToArray()
if($plain.Length -gt 65536){throw 'descriptor_size_limit'}
$cipher=[Security.Cryptography.ProtectedData]::Protect($plain,$null,[Security.Cryptography.DataProtectionScope]::LocalMachine)
$stream=[IO.File]::Open($path,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
try{$stream.Write($cipher,0,$cipher.Length);$stream.Flush($true)}finally{$stream.Dispose()}
""", path, payload)
    load_descriptor(workspace, digest)
    return {"sha256": digest, "reused": False, "expires_at": descriptor["expires_at"], "encrypted_at_rest": True}


def remove_descriptor(workspace: Path, digest: str) -> None:
    path = descriptor_path(workspace, digest)
    try:
        path.unlink()
    except FileNotFoundError:
        pass
