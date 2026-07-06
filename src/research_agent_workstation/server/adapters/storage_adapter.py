from __future__ import annotations

import json
from abc import abstractmethod
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from .base import Adapter


class StorageAdapter(Adapter):
    provider = "storage"

    @abstractmethod
    def write_json(self, path: Path, payload: Any) -> Path:
        raise NotImplementedError

    @abstractmethod
    def read_json(self, path: Path) -> Any:
        raise NotImplementedError

    @abstractmethod
    def ensure_dir(self, path: Path) -> Path:
        raise NotImplementedError


class LocalStorageAdapter(StorageAdapter):
    provider = "local_workspace"

    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root

    def ensure_dir(self, path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_json(self, path: Path, payload: Any) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def read_json(self, path: Path) -> Any:
        return json.loads(path.read_text(encoding="utf-8"))

    def _jsonable(self, value: Any) -> Any:
        if is_dataclass(value):
            return {k: self._jsonable(v) for k, v in asdict(value).items()}
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, list):
            return [self._jsonable(item) for item in value]
        if isinstance(value, dict):
            return {str(k): self._jsonable(v) for k, v in value.items()}
        return value
