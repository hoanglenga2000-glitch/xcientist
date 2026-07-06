from __future__ import annotations

import subprocess
import sys
from abc import abstractmethod
from pathlib import Path
from uuid import uuid4

from .base import Adapter
from ..schemas.evidence import ArtifactManifest, EvidenceRecord
from ..schemas.run import RunResult


class PythonRunnerAdapter(Adapter):
    provider = "python_runner"

    @abstractmethod
    def run_script(self, script_path: Path, args: list[str], cwd: Path, env: dict[str, str] | None = None) -> RunResult:
        raise NotImplementedError

    @abstractmethod
    def stream_logs(self, run_id: str) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def stop_run(self, run_id: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def collect_artifacts(self, run_id: str, artifact_dir: Path) -> ArtifactManifest:
        raise NotImplementedError


class LocalPythonRunnerAdapter(PythonRunnerAdapter):
    provider = "local"

    def __init__(self, log_root: Path) -> None:
        self.log_root = log_root
        self.log_root.mkdir(parents=True, exist_ok=True)

    def run_script(self, script_path: Path, args: list[str], cwd: Path, env: dict[str, str] | None = None) -> RunResult:
        run_id = f"run_{uuid4().hex[:10]}"
        run_dir = self.log_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = run_dir / "train_log.txt"
        stderr_path = run_dir / "error_log.txt"
        command = [sys.executable, str(script_path), *args]
        completed = subprocess.run(command, cwd=str(cwd), env=env, text=True, capture_output=True, check=False)
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        (run_dir / "return_code.txt").write_text(str(completed.returncode), encoding="utf-8")
        return RunResult(run_id, command, cwd, completed.returncode, stdout_path, stderr_path)

    def stream_logs(self, run_id: str) -> list[str]:
        path = self.log_root / run_id / "train_log.txt"
        return path.read_text(encoding="utf-8", errors="ignore").splitlines() if path.exists() else []

    def stop_run(self, run_id: str) -> bool:
        return True

    def collect_artifacts(self, run_id: str, artifact_dir: Path) -> ArtifactManifest:
        records = [
            EvidenceRecord(path.name, "unknown", path, "file", self.provider)
            for path in artifact_dir.rglob("*")
            if path.is_file()
        ]
        return ArtifactManifest("unknown", run_id, records)
