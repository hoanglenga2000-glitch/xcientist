"""Variation generator: the evolution engine's proposal step.

This is the piece the workstation was missing. Given the current best solution,
the CV history so far, and retrospective-memory lessons, it asks the LLM to
write the *next* candidate solution as a complete, runnable Python script that
satisfies a fixed I/O contract. Result-driven: it reads scores and lessons, so
each proposal is informed by what already worked or failed (true feedback loop),
not a hard-coded ``if n_prev`` ladder.

The generated script contract (enforced in the prompt and checked by the loop):
  * Reads ``train.csv`` / ``test.csv`` / ``sample_submission.csv`` from --data-dir.
  * Runs cross-validation and prints a final line ``CV_SCORE=<float>``.
  * Writes ``submission.csv`` and ``metrics.json`` (with ``cv_score``) to --out-dir.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from .llm_client import LLMClient, LLMResponse


# The contract has modality-independent steps (artifacts + determinism) and two
# modality-dependent knobs: the allowed library set and the compute budget. Deep
# modalities (image/multimodal/audio) need DL libs + GPU; tabular/text-default
# stay on the fast CPU tree/linear path. Keeping these split (rather than one
# tabular-only string) is what makes the engine general, not tabular-biased.
_CONTRACT_HEAD = """The script MUST:
1. Accept CLI args: --data-dir (has train.csv, test.csv, sample_submission.csv) and --out-dir.
2. Load the data, engineer features, and train with K-fold cross-validation.
3. Compute the out-of-fold CV score for the stated metric and direction.
4. Print exactly one line to stdout: CV_SCORE=<float>
5. Write <out-dir>/submission.csv matching sample_submission.csv columns/format.
6. Write <out-dir>/metrics.json containing at least {"cv_score": <float>, "metric": "<name>"}."""

_CONTRACT_TAIL = (
    "8. Be self-contained, deterministic (fixed seeds), and runnable as: "
    "python script.py --data-dir D --out-dir O"
)

_LIBS_TABULAR = (
    "7. Use only these libraries: pandas, numpy, scikit-learn, lightgbm, xgboost, catboost."
)
_LIBS_TEXT = (
    "7. Use pandas, numpy, and scikit-learn (TF-IDF + linear models) by default. "
    "You MAY use torch/transformers only if the run still fits the time budget."
)
_LIBS_DL = (
    "7. Use pandas, numpy, scikit-learn, and the deep-learning stack as needed: "
    "torch, torchvision, torchaudio, timm, transformers, Pillow (PIL), librosa. "
    "Pick libraries appropriate to the modality; do not restrict to trees."
)

_BUDGET_CPU = """COMPUTE BUDGET (strict):
  * The script MUST finish within about 4 minutes on a single CPU machine.
  * Use at most 5 CV folds.
  * Do NOT use exhaustive GridSearchCV/RandomizedSearchCV over large grids. Prefer
    fixed, well-chosen hyperparameters, or a tiny search (<= 6 candidates).
  * For gradient boosting, cap rounds (e.g., n_estimators/num_boost_round <= 2000)
    and use early stopping. Set n_jobs to a small number (<= 4).
  * Never loop training over many hyperparameter combinations blindly.
  * LARGE DATA: if the training set has more than ~200,000 rows, subsample to about
    150,000 rows for cross-validation (fixed seed) to stay within the time budget.
    Still generate predictions for the full test set."""

_BUDGET_GPU = """COMPUTE BUDGET (strict):
  * A single CUDA GPU is available; use device='cuda'. Target about 4-6 minutes
    on one GPU. Ignore any 'CPU only' assumption.
  * Keep it light: a small/pretrained backbone with a few epochs, mixed precision
    if convenient, and a sensible batch size. Do NOT train huge models from scratch.
  * Use a validation split or K-fold (<= 5) for the CV score; never leak.
  * LARGE DATA: cap epochs/steps and, if needed, subsample training rows (fixed
    seed) to stay in budget. Still predict for the full test set."""

_DL_MODALITIES = {"image", "multimodal", "audio"}


def _solution_contract(modality: str) -> str:
    """Return the modality-appropriate solution contract (libs + compute budget)."""
    m = (modality or "tabular").lower()
    if m in _DL_MODALITIES:
        libs, budget = _LIBS_DL, _BUDGET_GPU
    elif m == "text":
        libs, budget = _LIBS_TEXT, _BUDGET_CPU
    else:  # tabular / time_series and any unknown -> safe fast default
        libs, budget = _LIBS_TABULAR, _BUDGET_CPU
    return f"{_CONTRACT_HEAD}\n{libs}\n{_CONTRACT_TAIL}\n\n{budget}\n"


# Backward-compatible default (tabular) for any external reference.
SOLUTION_CONTRACT = _solution_contract("tabular")


@dataclass
class TaskContext:
    task_name: str
    modality: str            # tabular | image | text | time_series | multimodal
    task_type: str           # classification | regression
    metric: str              # e.g. accuracy, roc_auc, rmse, rmsle
    metric_direction: str    # maximize | minimize
    target_column: str = ""
    id_column: str = ""
    data_schema: str = ""    # short textual description of columns/dtypes
    n_train: int = 0
    n_test: int = 0
    extra_notes: str = ""


@dataclass
class RefSolution:
    """A high-scoring solution from another branch, fed into cross-branch and
    aggregation prompts so the LLM can borrow or fuse proven ideas."""
    exp_id: str
    score: Optional[float]
    branch_id: str = ""
    code: str = ""          # full code or a truncated excerpt
    note: str = ""          # short human hint, e.g. what made it strong


@dataclass
class VariationProposal:
    exp_id: str
    code: str
    hypothesis: str
    changes_summary: str
    applied_strategies: list[str] = field(default_factory=list)
    parent_exp_id: Optional[str] = None
    code_generation_mode: str = "Base"
    provider: str = ""
    model: str = ""
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    raw_response: str = ""


_SYSTEM_PREAMBLE = (
    "You are the variation generator inside a self-evolving machine-learning "
    "research engine. Your job each round is to write ONE complete Python script "
    "that is the next candidate solution for a Kaggle-style task. You improve on "
    "the current best solution using the cross-validation history and the lessons "
    "learned from past experiments. You are rigorous, avoid data leakage, and "
    "always honor the solution contract exactly.\n\n"
)


def _system_prompt(modality: str) -> str:
    """System prompt carrying the modality-appropriate contract, so the engine
    is never told 'CPU/tree-only' on an image/multimodal/audio task."""
    return _SYSTEM_PREAMBLE + _solution_contract(modality)


# Backward-compatible default (tabular) for any external reference.
_SYSTEM_PROMPT = _system_prompt("tabular")

_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


def _format_history(cv_history: list[dict[str, Any]]) -> str:
    if not cv_history:
        return "(no previous experiments; this is the baseline)"
    lines = []
    for item in cv_history[-8:]:  # cap context; recent history matters most
        lines.append(
            f"- {item.get('exp_id', '?')} [{item.get('branch', 'n/a')}] "
            f"cv={item.get('cv_score')} promoted={item.get('promoted')} "
            f"note={item.get('note', '')}"
        )
    return "\n".join(lines)


# A named failure pattern -> the concrete corrective action the proposer must
# take next time. This is what closes the loop: classification alone only NAMES
# the problem; these directives make the next proposal actually SOLVE it.
_CORRECTIVE_ACTION = {
    "timeout": ("PREVIOUS RUN TIMED OUT before emitting CV_SCORE. You MUST cut the "
                "compute: use <=3 CV folds, cap TF-IDF/features (e.g. max_features<=50000, "
                "SVD<=200 dims), subsample training rows (fixed seed) if >100k, and prefer a "
                "linear model over heavy DL. Emit CV_SCORE early enough to finish in budget."),
    "oom": ("PREVIOUS RUN WAS OOM-KILLED before emitting CV_SCORE. You MUST shrink the "
            "memory footprint: use sparse TF-IDF (do NOT .toarray() large matrices), cap "
            "max_features, reduce SVD dims, process folds without holding all vectorizers in "
            "memory, and subsample rows if very large. Free intermediates between folds."),
    "segfault": ("PREVIOUS RUN SEGFAULTED. Avoid the crashing native path: pin/reduce library "
                 "versions of the offending op, avoid mixed-precision if unstable, and prefer a "
                 "simpler estimator that reaches CV_SCORE reliably."),
}


def _format_lessons(lessons: list[dict[str, Any]]) -> str:
    if not lessons:
        return "(no retrospective memory yet)"
    lines = []
    directives: list[str] = []
    for item in lessons[:8]:
        worked = item.get("what_worked") or item.get("reusable_strategy") or ""
        failed = item.get("what_failed") or item.get("failure_pattern") or ""
        pattern = (item.get("failure_pattern") or "").strip().lower()
        piece = f"- task_type={item.get('task_type', '?')}"
        if worked:
            piece += f" | WORKED: {worked}"
        if failed:
            piece += f" | FAILED: {failed}"
        lines.append(piece)
        action = _CORRECTIVE_ACTION.get(pattern)
        if action and action not in directives:
            directives.append(action)
    out = "\n".join(lines)
    if directives:
        out += "\n\nREQUIRED CORRECTIVE ACTIONS (from past failures on this kind of task):\n"
        out += "\n".join(f"* {d}" for d in directives)
    return out


def _modality_guidance(modality: str) -> str:
    """Modality-specific library allowance + compute guidance.

    The base contract forbids heavy DL libs to keep tabular runs fast. Image
    (and optionally text) tasks need them, so we widen the allowance and point
    the model at the GPU here.
    """
    m = (modality or "tabular").lower()
    if m == "image":
        return (
            "MODALITY=IMAGE: You MAY additionally use torch, torchvision, and Pillow (PIL). "
            "A CUDA GPU is available (use device='cuda'). Load images from the id->file mapping "
            "in train.csv (files live directly under <data-dir>/train and <data-dir>/test). "
            "Use a small CNN or a lightweight pretrained backbone (e.g. resnet18) with a few "
            "epochs so the run fits ~4-6 minutes on one GPU. Normalize inputs; use a validation "
            "split or K-fold for the CV score. Ignore the 'CPU only' budget line for image tasks."
        )
    if m == "text":
        return (
            "MODALITY=TEXT: Prefer TF-IDF (word + char n-grams) with a linear model "
            "(LogisticRegression / SGD / Multinomial NB) for speed. torch/transformers are "
            "allowed only if the run still fits the time budget; default to the TF-IDF approach."
        )
    if m in {"time_series", "timeseries"}:
        return (
            "MODALITY=TIME_SERIES: Use forward-chaining CV (never shuffle). Build lag/rolling "
            "features from the time index. Respect temporal order in train/validation splits."
        )
    if m == "multimodal":
        return (
            "MODALITY=MULTIMODAL: The data mixes types (e.g. images/text plus tabular columns). "
            "A CUDA GPU is available (device='cuda'). Build one encoder per modality — a light "
            "pretrained backbone (e.g. resnet18 / a small transformer) for image/text and a "
            "tabular head (MLP or GBM features) for the structured columns — then FUSE them "
            "(concatenate embeddings into a joint head, or blend per-modality model probabilities). "
            "Use torch/torchvision/transformers as needed. Validate with a split or K-fold; never leak."
        )
    if m == "audio":
        return (
            "MODALITY=AUDIO: A CUDA GPU is available (device='cuda'). Convert waveforms to "
            "log-mel spectrograms (librosa or torchaudio) and treat them as images: a small CNN "
            "or pretrained backbone (e.g. resnet18) over the spectrogram works well. Normalize, "
            "keep epochs few, and use a validation split or K-fold for the CV score; never leak."
        )
    return (
        "MODALITY=TABULAR: Gradient-boosted trees (LightGBM/CatBoost/XGBoost) with clean K-fold "
        "CV are the strong default."
    )


def _format_references(refs: list["RefSolution"], *, with_code: bool) -> str:
    if not refs:
        return "(none)"
    lines = []
    for r in refs:
        head = f"── {r.exp_id} (branch={r.branch_id or 'n/a'}, cv={r.score})"
        if r.note:
            head += f": {r.note}"
        lines.append(head)
        if with_code and r.code:
            excerpt = r.code.strip()
            if len(excerpt) > 1600:  # keep prompts bounded (risk #2 mitigation)
                excerpt = excerpt[:1600] + "\n# ... (truncated)"
            lines += ["```python", excerpt, "```"]
    return "\n".join(lines)


def _build_user_prompt(
    context: TaskContext,
    *,
    mode: str,
    cv_history: list[dict[str, Any]],
    lessons: list[dict[str, Any]],
    strategies: list[str],
    best_code: Optional[str],
    expansion_type: str = "primary",
    reference_solutions: Optional[list["RefSolution"]] = None,
) -> str:
    parts = [
        f"TASK: {context.task_name}",
        f"modality={context.modality} | task_type={context.task_type} | "
        f"metric={context.metric} ({context.metric_direction})",
        f"target_column={context.target_column or 'unknown'} | id_column={context.id_column or 'none'}",
        f"n_train={context.n_train} | n_test={context.n_test}",
        "",
        "DATA SCHEMA:",
        context.data_schema or "(schema not provided; infer from the CSV files)",
        "",
        "CV HISTORY (most recent last):",
        _format_history(cv_history),
        "",
        "RETROSPECTIVE MEMORY LESSONS:",
        _format_lessons(lessons),
        "",
        f"RECOMMENDED STRATEGIES to consider: {', '.join(strategies) if strategies else '(none suggested)'}",
        "",
        f"CODE-GENERATION MODE: {mode}",
    ]
    parts += ["", _modality_guidance(context.modality)]
    if context.extra_notes:
        parts += ["", "NOTES:", context.extra_notes]

    refs = reference_solutions or []
    if expansion_type == "aggregation" and refs:
        # Fusion: don't anchor on one base; present the top solutions and fuse them.
        parts += [
            "",
            f"EXPANSION=AGGREGATION. The search has stalled; fuse the strongest "
            f"solutions found so far into ONE stronger solution. Prefer OOF stacking "
            f"or probability blending, or combine their best feature engineering.",
            "",
            "TOP SOLUTIONS TO FUSE:",
            _format_references(refs, with_code=True),
        ]
    elif expansion_type == "cross_branch" and refs and best_code:
        # Borrow proven ideas from other branches into the current branch's code.
        parts += [
            "",
            "CURRENT SOLUTION (this branch — improve it):",
            "```python", best_code.strip(), "```",
            "",
            "EXPANSION=CROSS_BRANCH. Other parallel branches produced these stronger "
            "or different-idea solutions. Borrow 1-2 of their most promising ideas and "
            "fold them into the current solution's structure. Do NOT copy wholesale:",
            _format_references(refs, with_code=True),
        ]
    elif mode == "Base" or not best_code:
        if (context.modality or "tabular").lower() in _DL_MODALITIES:
            parts += [
                "",
                "Write a strong BASELINE solution from scratch using the modality-appropriate "
                "approach described above (a light/pretrained neural model on the GPU). Keep it "
                "simple and fast, with clean validation and no leakage.",
            ]
        else:
            parts += [
                "",
                "Write a strong BASELINE solution from scratch. Prefer a gradient-boosted "
                "model (LightGBM or CatBoost) with sensible defaults and clean K-fold CV.",
            ]
    else:
        parts += [
            "",
            "CURRENT BEST SOLUTION (improve on this):",
            "```python",
            best_code.strip(),
            "```",
        ]
        if mode == "Diff":
            parts += [
                "",
                "The branch is STAGNATING or a recent attempt FAILED. Make a targeted, "
                "low-risk change: fix suspected issues, adjust CV, or apply one reliable "
                "strategy. Do NOT rewrite everything.",
            ]
        else:  # Stepwise
            parts += [
                "",
                "Extend the current best with ONE meaningful improvement (new features, a "
                "stronger/added model, or an ensemble) that plausibly raises the CV score. "
                "Keep what already works.",
            ]

    parts += [
        "",
        "Respond with a brief 1-2 sentence hypothesis, then a SINGLE ```python fenced "
        "code block containing the full runnable script. No prose after the code block.",
    ]
    return "\n".join(parts)


def _extract_code(text: str) -> str:
    matches = _CODE_BLOCK_RE.findall(text)
    if matches:
        # Use the longest block, which is the full script (not a snippet).
        return max(matches, key=len).strip()
    return ""


def _extract_hypothesis(text: str) -> str:
    before = _CODE_BLOCK_RE.split(text, maxsplit=1)[0].strip()
    return before[:500] if before else "(no explicit hypothesis provided)"


class VariationGenerator:
    """Turns task context + history + memory into the next runnable solution."""

    def __init__(self, client: Optional[LLMClient] = None, *, max_tokens: int = 8192) -> None:
        self.client = client or LLMClient()
        self.max_tokens = max_tokens

    def propose(
        self,
        context: TaskContext,
        *,
        exp_id: str,
        mode: str = "Base",
        cv_history: Optional[list[dict[str, Any]]] = None,
        lessons: Optional[list[dict[str, Any]]] = None,
        strategies: Optional[list[str]] = None,
        best_code: Optional[str] = None,
        parent_exp_id: Optional[str] = None,
        temperature: Optional[float] = None,
        expansion_type: str = "primary",
        reference_solutions: Optional[list[RefSolution]] = None,
    ) -> VariationProposal:
        user = _build_user_prompt(
            context,
            mode=mode,
            cv_history=cv_history or [],
            lessons=lessons or [],
            strategies=strategies or [],
            best_code=best_code,
            expansion_type=expansion_type,
            reference_solutions=reference_solutions or [],
        )
        response: LLMResponse = self.client.generate(
            user, system=_system_prompt(context.modality),
            max_tokens=self.max_tokens, temperature=temperature,
        )
        code = _extract_code(response.text)
        if not code:
            raise ValueError("LLM response contained no python code block")
        return VariationProposal(
            exp_id=exp_id,
            code=code,
            hypothesis=_extract_hypothesis(response.text),
            changes_summary=f"{mode}/{expansion_type} proposal for {context.task_name}",
            applied_strategies=list(strategies or []),
            parent_exp_id=parent_exp_id,
            code_generation_mode=mode,
            provider=response.provider,
            model=response.model,
            llm_input_tokens=response.input_tokens,
            llm_output_tokens=response.output_tokens,
            raw_response=response.text,
        )
