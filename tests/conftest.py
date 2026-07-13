"""Shared pytest fixtures and path setup for the research workstation tests."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (SRC, ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

class LocalSubprocessRunner:
    """Test-only runner; this module is outside every release package."""

    def __init__(self, workdir: str | Path, *, timeout: int = 900, python_exe: str | None = None) -> None:
        self.workdir = Path(workdir)
        self.timeout = timeout
        self.python_exe = python_exe or sys.executable

    def run(self, code: str, *, data_dir: str, out_dir: str, exp_id: str):
        from research_os.evolution_loop import RunResult, _parse_cv_score

        script_dir = self.workdir / exp_id
        script_dir.mkdir(parents=True, exist_ok=True)
        script_path = script_dir / "solution.py"
        script_path.write_text(code, encoding="utf-8")
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        try:
            proc = subprocess.run(
                [self.python_exe, str(script_path), "--data-dir", data_dir, "--out-dir", out_dir],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            return RunResult(False, None, error=f"timeout after {self.timeout}s", out_dir=out_dir)
        combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
        score = _parse_cv_score(proc.stdout or "")
        metrics_path = Path(out_dir) / "metrics.json"
        if score is None and metrics_path.exists():
            try:
                score = float(json.loads(metrics_path.read_text(encoding="utf-8")).get("cv_score"))
            except (ValueError, TypeError, json.JSONDecodeError):
                score = None
        artifacts = [str(path) for path in Path(out_dir).glob("*") if path.is_file()]
        if proc.returncode != 0 or score is None:
            return RunResult(
                False,
                score,
                stdout_tail=combined[-1500:],
                error=(proc.stderr or "no CV_SCORE emitted")[-1500:],
                out_dir=out_dir,
                artifacts=artifacts,
            )
        return RunResult(True, score, stdout_tail=combined[-800:], out_dir=out_dir, artifacts=artifacts)
