"""Test-only candidate runner; production local execution remains disabled."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from research_os.evolution_loop import RunResult, _parse_cv_score


class TestSubprocessRunner:
    __test__ = False

    def __init__(self, workdir, *, timeout=120, python_exe=None):
        self.workdir = Path(workdir)
        self.timeout = timeout
        self.python_exe = python_exe or sys.executable

    def run(self, code, *, data_dir, out_dir, exp_id):
        script_dir = self.workdir / exp_id
        script_dir.mkdir(parents=True, exist_ok=True)
        script_path = script_dir / "solution.py"
        script_path.write_text(code, encoding="utf-8")
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        try:
            proc = subprocess.run([self.python_exe, str(script_path), "--data-dir", data_dir, "--out-dir", out_dir], capture_output=True, text=True, timeout=self.timeout)
        except subprocess.TimeoutExpired:
            return RunResult(False, None, error=f"timeout after {self.timeout}s", out_dir=out_dir, exit_code=124)
        score = _parse_cv_score(proc.stdout or "")
        metrics_path = Path(out_dir) / "metrics.json"
        if score is None and metrics_path.exists():
            try:
                score = float(json.loads(metrics_path.read_text(encoding="utf-8"))["cv_score"])
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
                score = None
        artifacts = [str(path) for path in Path(out_dir).glob("*") if path.is_file()]
        combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
        if proc.returncode != 0 or score is None:
            return RunResult(False, score, stdout_tail=combined[-1500:], error=(proc.stderr or "no CV_SCORE emitted")[-1500:], out_dir=out_dir, artifacts=artifacts, exit_code=proc.returncode)
        return RunResult(True, score, stdout_tail=combined[-800:], out_dir=out_dir, artifacts=artifacts, exit_code=0, evaluator_version="unit-test-v1", environment_hash="0" * 64)
