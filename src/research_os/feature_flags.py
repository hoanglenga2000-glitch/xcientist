"""MLEvolve P0 feature-flag scaffolding (DeepEvo W1).

Env vars ``EVOMIND_FF_*`` default **off** (false) except
``EVOMIND_FF_BENCHMARK_CLAIM_GATE``, which defaults **on** because
``build_benchmark_claim_gate`` / claim-audit blocking already exists and must
stay active in prod.

W1 wired read points. W2 gates MCGS UCT / C-decay / multi-branch behind
``EVOMIND_FF_MCGS_UCT``, ``EVOMIND_FF_MCGS_C_DECAY``, and
``EVOMIND_FF_MULTI_BRANCH`` (defaults off = linear / constant-C / single-path).
"""

from __future__ import annotations

import os
from typing import Any, Mapping

# Prefer ON: claim gate already present and always-on when invoked.
_DEFAULT_ON: frozenset[str] = frozenset({"EVOMIND_FF_BENCHMARK_CLAIM_GATE"})

# Canonical P0 flag names from DeepEvo checklist section B / C W1.
FLAG_NAMES: tuple[str, ...] = (
    "EVOMIND_FF_MCGS_UCT",
    "EVOMIND_FF_MCGS_C_DECAY",
    "EVOMIND_FF_MULTI_BRANCH",
    "EVOMIND_FF_RETRO_MEMORY",
    "EVOMIND_FF_MEMORY_VECTOR",
    "EVOMIND_FF_MEMORY_FAISS",  # optional stub
    "EVOMIND_FF_SEARCH_BUDGET_500",
    "EVOMIND_FF_STAGNATION_DETECT",
    "EVOMIND_FF_CROSS_BRANCH_FUSION",
    "EVOMIND_FF_DEBUG_AGENT",
    "EVOMIND_FF_CODE_REVIEW_AGENT",
    "EVOMIND_FF_RESEARCH_OS_HARNESS",
    "EVOMIND_FF_SINGLE_TASK_LOOP",
    "EVOMIND_FF_BENCHMARK_CLAIM_GATE",
)

_TRUTHY = frozenset({"1", "true", "yes", "on", "enabled"})
_FALSY = frozenset({"0", "false", "no", "off", "disabled", ""})


def _default_for(name: str) -> bool:
    return name in _DEFAULT_ON


def _parse_bool(raw: str | None, *, default: bool) -> bool:
    if raw is None:
        return default
    text = str(raw).strip().lower()
    if text in _TRUTHY:
        return True
    if text in _FALSY:
        return False
    return default


def is_enabled(name: str, *, env: Mapping[str, str] | None = None) -> bool:
    """Return whether ``name`` is enabled (env override, else typed default)."""
    if name not in FLAG_NAMES:
        raise KeyError(f"unknown EVOMIND feature flag: {name}")
    source = os.environ if env is None else env
    return _parse_bool(source.get(name), default=_default_for(name))


def flags_snapshot(*, env: Mapping[str, str] | None = None) -> dict[str, bool]:
    """Stable dict of every canonical flag → bool (for logs / artifacts)."""
    return {name: is_enabled(name, env=env) for name in FLAG_NAMES}


# ── typed getters (defaults false except claim gate) ─────────────────────────

def ff_mcgs_uct(*, env: Mapping[str, str] | None = None) -> bool:
    return is_enabled("EVOMIND_FF_MCGS_UCT", env=env)


def ff_mcgs_c_decay(*, env: Mapping[str, str] | None = None) -> bool:
    return is_enabled("EVOMIND_FF_MCGS_C_DECAY", env=env)


def ff_multi_branch(*, env: Mapping[str, str] | None = None) -> bool:
    return is_enabled("EVOMIND_FF_MULTI_BRANCH", env=env)


def ff_retro_memory(*, env: Mapping[str, str] | None = None) -> bool:
    return is_enabled("EVOMIND_FF_RETRO_MEMORY", env=env)


def ff_memory_vector(*, env: Mapping[str, str] | None = None) -> bool:
    return is_enabled("EVOMIND_FF_MEMORY_VECTOR", env=env)


def ff_memory_faiss(*, env: Mapping[str, str] | None = None) -> bool:
    """Optional FAISS backend stub — default off until env/wheels approved."""
    return is_enabled("EVOMIND_FF_MEMORY_FAISS", env=env)


def ff_search_budget_500(*, env: Mapping[str, str] | None = None) -> bool:
    return is_enabled("EVOMIND_FF_SEARCH_BUDGET_500", env=env)


def ff_stagnation_detect(*, env: Mapping[str, str] | None = None) -> bool:
    return is_enabled("EVOMIND_FF_STAGNATION_DETECT", env=env)


def ff_cross_branch_fusion(*, env: Mapping[str, str] | None = None) -> bool:
    return is_enabled("EVOMIND_FF_CROSS_BRANCH_FUSION", env=env)


def ff_debug_agent(*, env: Mapping[str, str] | None = None) -> bool:
    return is_enabled("EVOMIND_FF_DEBUG_AGENT", env=env)


def ff_code_review_agent(*, env: Mapping[str, str] | None = None) -> bool:
    return is_enabled("EVOMIND_FF_CODE_REVIEW_AGENT", env=env)


def ff_research_os_harness(*, env: Mapping[str, str] | None = None) -> bool:
    return is_enabled("EVOMIND_FF_RESEARCH_OS_HARNESS", env=env)


def ff_single_task_loop(*, env: Mapping[str, str] | None = None) -> bool:
    return is_enabled("EVOMIND_FF_SINGLE_TASK_LOOP", env=env)


def ff_benchmark_claim_gate(*, env: Mapping[str, str] | None = None) -> bool:
    """Prefer ON: claim blocking already present; keep active unless explicitly off."""
    return is_enabled("EVOMIND_FF_BENCHMARK_CLAIM_GATE", env=env)


def flag_metadata() -> dict[str, Any]:
    """Human-oriented notes for audits (not env-dependent)."""
    return {
        "schema": "academic_research_os.feature_flags.v1",
        "defaults_off_except": sorted(_DEFAULT_ON),
        "claim_gate_note": (
            "EVOMIND_FF_BENCHMARK_CLAIM_GATE defaults ON because "
            "build_benchmark_claim_gate / audit_claim blocking already exists; "
            "W1 does not weaken claim rejection when the flag is left at default."
        ),
        "already_partially_present": [
            "EVOMIND_FF_BENCHMARK_CLAIM_GATE",
            "EVOMIND_FF_MCGS_UCT",  # W2: gates UCT walk + visit backprop; loop auto-injects when on
            "EVOMIND_FF_MCGS_C_DECAY",  # W2: gated — piecewise only when flag on
            "EVOMIND_FF_RETRO_MEMORY",  # EvolutionLoop store I/O pre-existing; controller gated in W3
            "EVOMIND_FF_STAGNATION_DETECT",  # SearchGraph.detect_global_stagnation already used
        ],
        "w1_policy": (
            "Read-only scaffolding: flags are snapshotted at natural loci; "
            "default-off flags must not alter search/MCGS behavior. "
            "Already-on code paths stay on until a later week explicitly gates them."
        ),
        "w2_policy": (
            "P0-SG-01/02 + P0-BR-01: MCGSSelector.select/backpropagate and "
            "multi-branch diversify/concurrent drafts honor EVOMIND_FF_MCGS_UCT, "
            "EVOMIND_FF_MCGS_C_DECAY, EVOMIND_FF_MULTI_BRANCH. Defaults remain off "
            "in prod; enable only in tests or explicit env for harness runs."
        ),
        "w3_policy": (
            "P0-MEM-01/02: mlevolve_controller record/reload helpers and "
            "memory_reuse_records honor EVOMIND_FF_RETRO_MEMORY; TF-IDF k-NN "
            "honors EVOMIND_FF_MEMORY_VECTOR; EVOMIND_FF_MEMORY_FAISS is stub-only "
            "(not_configured → TF-IDF fallback, no FAISS wheels). EvolutionLoop "
            "store I/O stays pre-existing always-on; claim audit cites memory ids "
            "when RETRO_MEMORY on and refs present."
        ),
        "w4_policy": (
            "P0-BUD-01 / P0-ST-01 / P0-ST-02: SEARCH_BUDGET_500 arms up to 500 steps "
            "(+ optional hours) on the same BudgetLedger path (tests use small N); "
            "STAGNATION_DETECT gates graph detect_* and linear early-exit (off = "
            "prior stagnation_patience Diff behavior); CROSS_BRANCH_FUSION gates "
            "cross_branch/aggregation expansions, reference_edges, and audit "
            "cross_branch_references (off = no cross-branch fusion)."
        ),
        "w5_policy": (
            "P0-AG-01/02 + P0-HN-01/03: DEBUG_AGENT attaches stderr/trace into Diff "
            "prompts + debug_feedback.json (off = prior LAST RUN FAILED notes); "
            "CODE_REVIEW_AGENT writes review + quality gate and can block enqueue "
            "(off = skip); RESEARCH_OS_HARNESS runs local three-layer skeleton "
            "(search graph + claim audit, insufficient evidence ⇒ blocked); "
            "BENCHMARK_CLAIM_GATE stays default ON and refuses beat-AI2/MLEvolve "
            "overclaims without 75-task + budget evidence (label preliminary/partial/no)."
        ),
        "w6_policy": (
            "P0-HN-02: SINGLE_TASK_LOOP (default off) runs a CI/local CPU stub closed "
            "loop that writes metrics/OOF/submission_audit/claim_audit/"
            "rank_promotion_gate/benchmark_claim_gate for one run id; never enqueues "
            "managed_tensor_train or live HPC/GPU. Claim gate (W5) refuses 75-task "
            "MLEvolve-level claims on a single task. Flag off = no production change."
        ),
    }
