from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ProviderStatus:
    name: str
    state: str
    configured: bool
    notes: str = ""


@dataclass(slots=True)
class ConnectorStatus:
    code_agent: ProviderStatus
    python_runner: ProviderStatus
    gpu: ProviderStatus
    kaggle: ProviderStatus
    llm: ProviderStatus
    storage: ProviderStatus
    env_keys: dict[str, str]


@dataclass(slots=True)
class CredentialStatus:
    configured: bool
    provider: str
    message: str


@dataclass(slots=True)
class DownloadResult:
    success: bool
    target_dir: Path
    files: list[Path] = field(default_factory=list)
    message: str = ""


@dataclass(slots=True)
class SubmissionResult:
    success: bool
    gated: bool
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)
