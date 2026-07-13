r"""Auto-Repair Pipeline — error → classify → memory search → fix → retry.

When a training experiment fails, this pipeline:
  1. Classifies the error into a reusable failure pattern (timeout, oom, etc.)
  2. Searches retrospective memory for similar past failures and their fixes
  3. Generates a concrete repair strategy from the matched lessons
  4. If confidence is high enough, auto-retries with the repaired code
  5. Records whether the repair worked (for self-evolution tracking)

This is how EvoMind learns from its mistakes — each failure makes it stronger.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# Import from research_os for error classification (avoid circular imports)
try:
    from research_os.evolution_loop import _classify_failure, _clean_error_for_feedback
except ImportError:
    def _classify_failure(error: str) -> str:
        low = (error or "").lower()
        if "timeout" in low or "timed out" in low: return "timeout"
        if "out of memory" in low or "memoryerror" in low: return "oom"
        if "modulenotfound" in low or "importerror" in low: return "import_error"
        if "no such file" in low or "filenotfound" in low: return "file_not_found"
        if "keyerror" in low: return "key_error"
        if "valueerror" in low: return "value_error"
        if "cv_score" in low: return "contract_violation"
        if "syntaxerror" in low or "indentationerror" in low: return "syntax_error"
        return "runtime_error"

    def _clean_error_for_feedback(error: str, max_chars: int = 1200) -> str:
        if not error:
            return ""
        lines = [l.strip() for l in error.replace("\r", "\n").splitlines()
                if l.strip() and not ("%|" in l.strip())]
        return "\n".join(lines)[-max_chars:]


@dataclass
class RepairDiagnosis:
    """The result of analyzing a failed experiment."""
    exp_id: str
    failure_pattern: str          # e.g. "timeout", "oom", "syntax_error"
    cleaned_error: str            # progress-bar noise stripped
    matched_lessons: list[dict] = field(default_factory=list)
    repair_strategy: str = ""     # concrete fix suggestion
    confidence: float = 0.0       # 0.0–1.0 how confident we are in the repair
    auto_retry_ready: bool = False


@dataclass
class RepairOutcome:
    """The result of attempting a repair."""
    exp_id: str
    repair_applied: bool
    re_run_exp_id: str = ""       # ID of the retry experiment
    re_run_success: bool = False
    re_run_score: Optional[float] = None
    lesson_updated: bool = False


# ── Repair strategies per failure pattern ────────────────────────────

_REPAIR_TEMPLATES = {
    "timeout": (
        "The experiment ran too long. Reduce n_estimators, reduce training data size "
        "via subsampling, or increase the timeout budget. For GPU runs, check the HPC "
        "job scheduler's walltime limit."
    ),
    "oom": (
        "Out of memory. Reduce batch_size, reduce model complexity, use gradient "
        "accumulation, or switch to a lighter model family. For GPU, reduce "
        "max_features or use float16."
    ),
    "import_error": (
        "Missing Python package. Add the required import to the script, or install "
        "the package in the training environment. Common fix: add pip install "
        "command to the training script."
    ),
    "file_not_found": (
        "Required file is missing. Check the data directory path, ensure train.csv "
        "/ test.csv are in the expected location, or update --data-dir argument."
    ),
    "key_error": (
        "Referenced column doesn't exist in the dataset. Check the column name "
        "spelling, or use a try/except fallback. Run inspect_data to see available "
        "columns."
    ),
    "schema_mismatch": (
        "The code references a column that doesn't exist in the actual dataset. "
        "Run inspect_data to see the real column names, then update the code to "
        "match. Use .get() with defaults for optional columns."
    ),
    "value_error": (
        "Invalid value passed to a function. Check data types, ensure categorical "
        "values match expected ranges, or add input validation before the failing "
        "operation."
    ),
    "contract_violation": (
        "The training script didn't output CV_SCORE or save required artifacts. "
        "Ensure the script prints exactly 'CV_SCORE=<float>' and saves "
        "submission.csv + metrics.json to --out-dir."
    ),
    "syntax_error": (
        "Python syntax error in the generated code. Fix the syntax, ensure proper "
        "indentation, and re-run with the corrected script."
    ),
    "runtime_error": (
        "General runtime error. Review the error message, add defensive checks, "
        "and ensure the code handles edge cases (empty data, NaN values, etc.)."
    ),
}


def diagnose_failure(exp_id: str, error_text: str, *,
                     memory_library=None) -> RepairDiagnosis:
    """Analyze a failed experiment and propose a repair strategy.

    Args:
        exp_id: The experiment ID that failed.
        error_text: Raw error output from the runner.
        memory_library: Optional MemoryLibrary instance for searching past lessons.

    Returns:
        RepairDiagnosis with the failure classification and repair strategy.
    """
    cleaned = _clean_error_for_feedback(error_text)
    pattern = _classify_failure(error_text)

    diagnosis = RepairDiagnosis(
        exp_id=exp_id,
        failure_pattern=pattern,
        cleaned_error=cleaned[:600],
    )

    # Search memory for similar failures
    if memory_library is not None:
        try:
            records = memory_library.retrieve(failure_pattern=pattern, limit=5)
            if records:
                diagnosis.matched_lessons = records
        except Exception:
            pass

    # Build repair strategy from template + matched lessons
    template = _REPAIR_TEMPLATES.get(pattern, _REPAIR_TEMPLATES["runtime_error"])
    parts = [template]

    if diagnosis.matched_lessons:
        # Extract what worked from past similar failures
        fixes = []
        for r in diagnosis.matched_lessons[:3]:
            worked = r.get("what_worked", "")
            strategy = r.get("reusable_strategy", "")
            if worked: fixes.append(f"  • Previously fixed by: {worked}")
            if strategy: fixes.append(f"    Strategy: {strategy}")
        if fixes:
            parts.append("\nBased on past experience:")
            parts.extend(fixes)
            diagnosis.confidence = min(0.3 + 0.2 * len(fixes), 0.9)
    else:
        diagnosis.confidence = 0.15  # Template-only, low confidence

    diagnosis.repair_strategy = "\n".join(parts)
    diagnosis.auto_retry_ready = (
        diagnosis.confidence >= 0.4  # Need at least moderate confidence
        and pattern in ("syntax_error", "import_error", "contract_violation", "file_not_found")
    )

    return diagnosis


def build_repair_prompt(diagnosis: RepairDiagnosis, original_code: str) -> str:
    """Build a Diff-mode repair prompt for the LLM based on the diagnosis."""
    return (
        "The experiment FAILED. Here is the diagnosis:\n\n"
        f"FAILURE PATTERN: {diagnosis.failure_pattern}\n"
        f"ERROR (cleaned):\n{diagnosis.cleaned_error}\n\n"
        f"REPAIR STRATEGY:\n{diagnosis.repair_strategy}\n\n"
        f"ORIGINAL CODE:\n```python\n{original_code[:3000]}\n```\n\n"
        "Please generate a CORRECTED version of the code. Keep changes minimal — "
        "only fix what caused the error. Return the COMPLETE runnable script."
    )
