#!/usr/bin/env python3
"""Web-callable wrapper around the research_os evolution engine (engine A).

This is the thin, JSON-in/JSON-out entry point the workstation web app spawns to
run REAL evolution training through ``research_os.EvolutionLoop`` (opus-4-8 +
GPURunner or LocalSubprocessRunner). It deliberately REUSES the standalone
``run_evolution.py`` helpers (``_load_context`` / ``_strategies_for``) instead of
duplicating them, so both tracks stay in lockstep.

Contract (mirrors evolution_engine_cli.py conventions):
  * Input: a JSON file via --input, with keys:
      task_id            (required) workstation task id, e.g. "nyc_taxi"
      runner             "gpu" | "local_gpu" | "local" (default "gpu")
      iterations         int                        (default 8)
      mcgs               bool                       (default true)
      search_mode        "legacy_uct" | "experience_mcgs_v1"
      max_nodes          hard node cap              (default iterations)
      max_tokens         hard total-token cap       (default 2,000,000)
      max_wall_seconds   hard observed wall cap     (default 43,200)
      max_cost           optional estimated USD cap
      evolution_config   optional explicit path to a configs/evolution/*.json
      data_dir           optional local data dir (local runner only)
  * Output: exactly one JSON object on stdout:
      {ok, task_id, runner, summary_path, exp_dir, best_exp_id, best_cv_score,
       metric, metric_direction, n_iterations, n_promotions}
    On failure: {ok:false, error, decision}.

Never fabricates: it only reports what the engine actually wrote to disk.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (SRC, ROOT, ROOT / "scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

# Reuse the standalone engine helpers so the two tracks never drift apart.
from run_evolution import _load_context, _strategies_for  # type: ignore  # noqa: E402

from research_os.evolution_loop import EvolutionConfig, EvolutionLoop, LocalSubprocessRunner, RunResult  # noqa: E402
from research_os.independent_holdout import evaluate_independent_holdout  # noqa: E402
from research_os.llm_client import LLMClient  # noqa: E402
from research_os.retrospective_memory import RetrospectiveMemoryStore  # noqa: E402
from research_os.variation_generator import VariationGenerator  # noqa: E402

EVOLUTION_CONFIG_DIR = ROOT / "configs" / "evolution"

# task_id -> evolution config stem. Web task ids can differ from the JSON stems;
# unknown ids fall through to a direct "<task_id>.json" probe (no guessing beyond
# an exact filename match, so we never run the wrong task's config).
_ALIASES = {
    "nyc_taxi": "nyc_taxi",
    "new-york-city-taxi-fare-prediction": "nyc_taxi",
    "nomad2018": "nomad2018",
    "aerial_cactus": "aerial_cactus",
    "aerial-cactus-identification": "aerial_cactus",
    "champs": "champs",
    "leaf_classification": "leaf_classification",
    "spooky_author": "spooky_author",
    "tps_dec2021": "tps_dec2021",
    "tps_may2022": "tps_may2022",
    "ventilator": "ventilator",
}

_RUNNERS = {"gpu", "local_gpu", "local"}
_SEARCH_MODES = {"legacy_uct", "experience_mcgs_v1"}
_DEMO_TASK_ID = "evomind_demo_customer_churn"


def _bounded_int(value: object, *, name: str, default: int, minimum: int, maximum: int) -> int:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed < minimum or parsed > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return parsed


def _bounded_float(
    value: object,
    *,
    name: str,
    default: float | None,
    minimum: float,
    maximum: float,
    optional: bool = False,
) -> float | None:
    if value is None or value == "":
        return None if optional else default
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not (minimum <= parsed <= maximum):
        raise ValueError(f"{name} must be between {minimum:g} and {maximum:g}")
    return parsed


def _parse_run_contract(data_in: dict) -> dict:
    """Validate the JSON contract before creating a run directory or backend."""
    if str(data_in.get("task_id", "") or "") == _DEMO_TASK_ID:
        # The public demo is one reproducible evidence treatment, not a tunable
        # variant of the general runner.  Canonicalize before generic parsing so
        # stale UI values (including a disabled MCGS checkbox) cannot silently
        # turn it into a short legacy run that later overclaims the evidence.
        from research_os.demo_campaign import (
            DEMO_ITERATIONS,
            DEMO_MAX_COST_USD,
            DEMO_MAX_NODES,
            DEMO_MAX_TOKENS,
            DEMO_MAX_WALL_SECONDS,
            DEMO_RUNNER,
            DEMO_SEARCH_MODE,
        )

        return {
            "runner": DEMO_RUNNER,
            "search_mode": DEMO_SEARCH_MODE,
            "iterations": DEMO_ITERATIONS,
            "requested_iterations": DEMO_ITERATIONS,
            "max_nodes": DEMO_MAX_NODES,
            "max_tokens": DEMO_MAX_TOKENS,
            "max_wall_seconds": DEMO_MAX_WALL_SECONDS,
            "max_cost": DEMO_MAX_COST_USD,
            "mcgs": True,
        }
    runner = str(data_in.get("runner", "gpu") or "gpu").strip().lower()
    if runner not in _RUNNERS:
        raise ValueError(f"runner must be one of {sorted(_RUNNERS)}")
    default_search_mode = "experience_mcgs_v1" if os.environ.get("EXPERIENCE_MCGS_V1") == "1" else "legacy_uct"
    search_mode = str(data_in.get("search_mode", default_search_mode) or default_search_mode).strip().lower()
    if search_mode not in _SEARCH_MODES:
        raise ValueError(f"search_mode must be one of {sorted(_SEARCH_MODES)}")
    iterations = _bounded_int(
        data_in.get("iterations"), name="iterations", default=8, minimum=1, maximum=64
    )
    max_nodes = _bounded_int(
        data_in.get("max_nodes"), name="max_nodes", default=iterations, minimum=1, maximum=64
    )
    max_tokens = _bounded_int(
        data_in.get("max_tokens"), name="max_tokens", default=2_000_000, minimum=1, maximum=10_000_000
    )
    max_wall_seconds = _bounded_float(
        data_in.get("max_wall_seconds"),
        name="max_wall_seconds",
        default=43_200.0,
        minimum=1.0,
        maximum=43_200.0,
    )
    max_cost = _bounded_float(
        data_in.get("max_cost"),
        name="max_cost",
        default=None,
        minimum=0.0,
        maximum=100_000.0,
        optional=True,
    )
    use_mcgs = bool(data_in.get("mcgs", True))
    if search_mode == "experience_mcgs_v1" and not use_mcgs:
        raise ValueError("experience_mcgs_v1 requires mcgs=true")
    return {
        "runner": runner,
        "search_mode": search_mode,
        # The loop count is capped independently from the selector ledger so a
        # first/root node can never exceed max_nodes before it enters the board.
        "iterations": min(iterations, max_nodes),
        "requested_iterations": iterations,
        "max_nodes": max_nodes,
        "max_tokens": max_tokens,
        "max_wall_seconds": float(max_wall_seconds or 43_200.0),
        "max_cost": max_cost,
        "mcgs": use_mcgs,
    }


def _validate_local_gpu_candidate(code: str) -> list[str]:
    """Reject generated candidates that violate this task's evidence contract."""
    low = code.lower()
    violations: list[str] = []
    if "holdout_labels" in low:
        violations.append("candidate must not read retained holdout labels")
    if re.search(r"groupby\([^\n]*\)\s*\[\s*['\"]is_fraud['\"]\s*\]\s*\.mean\(", code, re.IGNORECASE):
        violations.append("full-frame target mean encoding is forbidden; use fold-safe or target-free encoding")
    compact = low.replace(" ", "")
    if "train_test_split" in low or "stratifiedkfold" in low or "shuffle=true" in compact:
        violations.append("credit-card task requires chronological validation, not shuffled/random splitting")
    if "oof_decision_threshold" not in low:
        violations.append("metrics.json must persist oof_decision_threshold")
    for metric_name in ("f1", "recall", "precision", "brier"):
        if metric_name not in low:
            violations.append(f"metrics.json must persist {metric_name}")
    if re.search(r"eval_metric\s*[:=]\s*['\"]AveragePrecision['\"]", code, re.IGNORECASE):
        violations.append("CatBoost eval_metric must be PRAUC, not AveragePrecision")
    return violations


class GuardedLocalGpuRunner(LocalSubprocessRunner):
    def run(self, code: str, *, data_dir: str, out_dir: str, exp_id: str) -> RunResult:
        violations = _validate_local_gpu_candidate(code)
        if violations:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            return RunResult(
                False,
                None,
                error="generated_candidate_contract_rejected: " + "; ".join(violations),
                out_dir=out_dir,
                artifacts=[],
            )
        return super().run(code, data_dir=data_dir, out_dir=out_dir, exp_id=exp_id)


def _available_memory_mib() -> int:
    if sys.platform != "win32":
        return 0

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(MemoryStatusEx)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return 0
    return int(status.ullAvailPhys // (1024 * 1024))


def _local_gpu_resource_gate(exp_root: Path) -> dict:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    row = (query.stdout or "").strip().splitlines()
    fields = [item.strip() for item in row[0].split(",")] if row else []
    gpu = {
        "name": fields[0] if len(fields) > 0 else "",
        "memory_total_mib": int(float(fields[1])) if len(fields) > 1 else 0,
        "memory_used_mib": int(float(fields[2])) if len(fields) > 2 else 0,
        "memory_free_mib": int(float(fields[3])) if len(fields) > 3 else 0,
        "utilization_percent": int(float(fields[4])) if len(fields) > 4 else 0,
        "temperature_c": int(float(fields[5])) if len(fields) > 5 else 0,
    }
    packages: dict[str, dict] = {}
    for name in ("pandas", "sklearn", "lightgbm", "xgboost", "catboost", "torch"):
        try:
            module = importlib.import_module(name)
            packages[name] = {"ready": True, "version": str(getattr(module, "__version__", ""))}
        except Exception as exc:  # dependency names only; no secret-bearing payloads
            packages[name] = {"ready": False, "error": type(exc).__name__}
    torch_cuda = False
    cuda_device = ""
    try:
        import torch

        torch_cuda = bool(torch.cuda.is_available())
        cuda_device = str(torch.cuda.get_device_name(0)) if torch_cuda else ""
    except Exception:
        pass
    xgboost_cuda = False
    try:
        import xgboost

        xgboost_cuda = bool(xgboost.build_info().get("USE_CUDA"))
    except Exception:
        pass
    available_memory_mib = _available_memory_mib()
    checks = {
        "nvidia_smi": query.returncode == 0 and bool(fields),
        "expected_gpu": "RTX 4060" in gpu["name"],
        "vram_at_least_7gib": gpu["memory_total_mib"] >= 7000,
        "vram_free_at_least_4gib": gpu["memory_free_mib"] >= 4096,
        "system_memory_free_at_least_3gib": available_memory_mib >= 3072,
        "required_packages": all(packages[name]["ready"] for name in packages),
        "torch_cuda": torch_cuda and "RTX 4060" in cuda_device,
        "xgboost_cuda_build": xgboost_cuda,
    }
    payload = {
        "schema": "evomind.local_gpu.resource_gate.v1",
        "status": "passed" if all(checks.values()) else "blocked",
        "runner": "local_gpu",
        "gpu": gpu,
        "system_memory_available_mib": available_memory_mib,
        "python": sys.executable,
        "packages": packages,
        "checks": checks,
        "official_submission": "disabled",
    }
    (exp_root / "local_resource_gate.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def resolve_evolution_config(task_id: str, explicit: str = "") -> Path | None:
    """Return the configs/evolution/*.json for a task_id, or None if none exists.

    Resolution order: explicit path -> alias map -> exact "<task_id>.json".
    Returns None (never a guess) when nothing matches, so the caller can report
    ``no_evolution_config`` instead of running an unintended task.
    """
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = ROOT / explicit
        return p if p.exists() else None
    stem = _ALIASES.get(task_id, task_id)
    candidate = EVOLUTION_CONFIG_DIR / f"{stem}.json"
    return candidate if candidate.exists() else None


def _fail(error: str, decision: str = "failed") -> int:
    print(json.dumps({"ok": False, "error": error, "decision": decision}, ensure_ascii=False))
    return 1


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Web-callable research_os evolution runner.")
    ap.add_argument("--input", required=True, help="JSON input file (task_id, runner, iterations, ...)")
    args = ap.parse_args()

    try:
        data_in = json.loads(Path(args.input).read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - report as JSON, never crash silently
        return _fail(f"bad input file: {type(exc).__name__}")

    task_id = str(data_in.get("task_id", "") or "")
    if not task_id:
        return _fail("task_id is required", "rejected")
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,200}", task_id) or ".." in task_id:
        return _fail("task_id contains unsupported characters", "rejected")
    try:
        run_contract = _parse_run_contract(data_in)
    except ValueError as exc:
        return _fail(str(exc), "rejected")
    runner_kind = str(run_contract["runner"])
    search_mode = str(run_contract["search_mode"])
    iterations = int(run_contract["iterations"])
    use_mcgs = bool(run_contract["mcgs"])
    explicit_cfg = str(data_in.get("evolution_config", "") or "")

    config_path = resolve_evolution_config(task_id, explicit_cfg)
    if config_path is None:
        return _fail(
            f"no evolution config for task_id '{task_id}'. Create configs/evolution/<name>.json "
            f"or pass evolution_config.", "no_evolution_config")

    ctx, cfg_data = _load_context(config_path)
    demo_campaign = bool(cfg_data.get("demo_campaign", False))
    demo_task = task_id == _DEMO_TASK_ID
    if demo_campaign or demo_task:
        from research_os.demo_campaign import (
            DEMO_ITERATIONS,
            DEMO_MAX_NODES,
            DEMO_RUNNER,
            DEMO_SEARCH_MODE,
            DEMO_TASK_ID,
        )

        if not demo_campaign or task_id != DEMO_TASK_ID or ctx.task_name != DEMO_TASK_ID:
            return _fail("demo campaign identity mismatch", "rejected")
        if (
            runner_kind != DEMO_RUNNER
            or iterations != DEMO_ITERATIONS
            or search_mode != DEMO_SEARCH_MODE
            or int(run_contract["max_nodes"]) != DEMO_MAX_NODES
            or not use_mcgs
        ):
            return _fail("demo campaign canonical run contract was not enforced", "rejected")
    orchestration_model = str(cfg_data.get("orchestration_model", "") or "")
    if orchestration_model:
        os.environ["OPENAI_MODEL"] = orchestration_model
    if runner_kind == "local_gpu":
        ctx.compute_backend = "local_gpu"
    strategies = _strategies_for(ctx, cfg_data)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    exp_root = ROOT / "experiments" / "evolution" / f"{ctx.task_name}_{runner_kind}_{stamp}"
    exp_root.mkdir(parents=True, exist_ok=True)
    memory = RetrospectiveMemoryStore(ROOT / "experiments" / "evolution" / "retrospective_memory.json")

    resource_gate = None
    if runner_kind == "gpu":
        from research_os.gpu_runner import GPURunner, GPURunnerConfig
        dirname = cfg_data.get("remote_data_dirname") or ctx.task_name
        runner = GPURunner(dirname, config=GPURunnerConfig())
        data_dir = cfg_data.get("gpu_data_dir") or dirname
    else:
        runner = GuardedLocalGpuRunner(exp_root / "runs") if runner_kind == "local_gpu" else LocalSubprocessRunner(exp_root / "runs")
        data_dir = str(data_in.get("data_dir", "") or cfg_data.get("local_data_dir", ""))
        if not data_dir:
            return _fail("local runner needs data_dir (or local_data_dir in the config)", "rejected")
        data_path = Path(data_dir)
        if not data_path.is_absolute():
            data_path = (ROOT / data_path).resolve()
        data_dir = str(data_path)
        if runner_kind == "local_gpu":
            resource_gate = _local_gpu_resource_gate(exp_root)
            if resource_gate.get("status") != "passed":
                return _fail("local RTX 4060 resource gate is blocked", "resource_gate_blocked")

    selector = None
    if use_mcgs:
        from research_os.mcgs_selector import MCGSSelector
        selector = MCGSSelector(
            total_steps=int(run_contract["max_nodes"]),
            search_mode=search_mode,
            max_total_tokens=int(run_contract["max_tokens"]),
            max_wall_seconds=float(run_contract["max_wall_seconds"]),
            max_cost_usd=run_contract["max_cost"],
        )

    generator = None
    demo_campaign_root = None
    if demo_campaign:
        from research_os.demo_campaign import DemoVariationGenerator, prepare_demo_dataset

        demo_campaign_root = Path(data_dir).parent
        prepare_demo_dataset(demo_campaign_root)
        generator = DemoVariationGenerator()
    elif runner_kind == "local_gpu":
        generator = VariationGenerator(
            client=LLMClient(primary="openai", fallback="openai", max_retries=1, timeout=240, temperature=0.2),
            max_tokens=8192,
        )

    events_path = exp_root / "evolution-events.jsonl"
    retrieval_path = exp_root / "retrieval-bundles.jsonl"
    events_path.write_text("", encoding="utf-8")
    retrieval_path.write_text("", encoding="utf-8")

    def record_event(event: dict) -> None:
        # Events contain only controller decisions and public-validation state;
        # prompts, source code, absolute paths, credentials and grader data are
        # deliberately absent from this projection.
        with events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        if event.get("type") == "select" and selector is not None:
            bundle = getattr(selector, "last_retrieval_bundle", None)
            if bundle is not None:
                retrieval_record = {
                    "seq": event.get("seq"),
                    "ts": event.get("ts"),
                    "selected_node_id": event.get("node_exp_id"),
                    **bundle.to_dict(),
                    # The selector resolves this after prompt_context() hydrates
                    # the lazy summary cache and overwrites the JSONL with the
                    # definitive per-card hit/miss record during export.
                    "cache_hit": None,
                    "cache_status": "lazy_summary_pending",
                }
                with retrieval_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(retrieval_record, ensure_ascii=False, sort_keys=True) + "\n")

    public_data_hashes = [
        str(item).lower()
        for item in (cfg_data.get("public_data_hashes") or [])
        if isinstance(item, str) and re.fullmatch(r"[0-9a-fA-F]{64}", item)
    ]
    dataset_contract_path = Path(data_dir) / "dataset_contract.json"
    if dataset_contract_path.is_file():
        public_data_hashes.append(hashlib.sha256(dataset_contract_path.read_bytes()).hexdigest())
    public_data_hashes = sorted(set(public_data_hashes))

    run_contract_path = exp_root / "search-run-contract.json"
    run_contract_path.write_text(
        json.dumps(
            {
                "schema": "evomind.experience_search_run.v1",
                "task_id": task_id,
                "search_mode": search_mode,
                "mcgs": use_mcgs,
                "runner": runner_kind,
                "demo_campaign": demo_campaign,
                "evidence_class": "real_local_cpu_execution" if demo_campaign else "research_execution",
                "requested_iterations": run_contract["requested_iterations"],
                "effective_iterations": iterations,
                "budgets": {
                    "max_nodes": run_contract["max_nodes"],
                    "max_tokens": run_contract["max_tokens"],
                    "max_wall_seconds": run_contract["max_wall_seconds"],
                    "max_cost": run_contract["max_cost"],
                },
                "official_submission": "disabled",
                "public_data_hashes": public_data_hashes,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    input_contract_path = exp_root / "input-contract.json"
    input_contract = {
        "schema": "evomind.demo.input_contract.v1" if demo_campaign else "evomind.evolution.input_contract.v1",
        "task_id": task_id,
        "runner": runner_kind,
        "iterations": iterations,
        "requested_iterations": run_contract["requested_iterations"],
        "mcgs": use_mcgs,
        "search_mode": search_mode,
        "max_nodes": run_contract["max_nodes"],
        "max_tokens": run_contract["max_tokens"],
        "max_wall_seconds": run_contract["max_wall_seconds"],
        "max_cost": run_contract["max_cost"],
        "demo_campaign": demo_campaign,
        "official_submission": "disabled",
        "created_at": datetime.now().astimezone().isoformat(),
    }
    _write_json_atomic(input_contract_path, input_contract)
    input_contract_sha256 = hashlib.sha256(input_contract_path.read_bytes()).hexdigest()

    loop = EvolutionLoop(
        ctx, data_dir=data_dir, work_dir=exp_root, runner=runner,
        generator=generator, memory=memory,
        config=EvolutionConfig(
            max_iterations=iterations,
            public_data_hashes=tuple(public_data_hashes),
        ),
        selector=selector, on_event=record_event,
    )

    try:
        summary = loop.run(strategies=strategies)
    except Exception as exc:  # noqa: BLE001 - engine errors reported as JSON
        return _fail(f"evolution loop failed: {type(exc).__name__}: {exc}", "training_failed")

    summary_path = exp_root / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if loop.best_code:
        (exp_root / "best_solution.py").write_text(loop.best_code, encoding="utf-8")
    loop.graph.export_json(exp_root / "search_graph.json")

    demo_evidence = None
    if demo_campaign and demo_campaign_root is not None and summary.get("best_exp_id"):
        from research_os.demo_campaign import freeze_and_review_demo_candidate

        try:
            demo_evidence = freeze_and_review_demo_candidate(exp_root, summary, demo_campaign_root)
        except Exception as exc:  # fail closed before a demo claim reaches stdout
            return _fail(f"demo evidence chain failed: {type(exc).__name__}: {exc}", "demo_evidence_failed")

    independent_review = None
    self_evolution_comparison = None
    holdout_setting = str(cfg_data.get("independent_holdout_labels", "") or "")
    if holdout_setting and summary.get("best_exp_id"):
        holdout_path = Path(holdout_setting)
        if not holdout_path.is_absolute():
            holdout_path = ROOT / holdout_path
        best_out = exp_root / str(summary["best_exp_id"]) / "out"
        review_path = exp_root / "independent_holdout_review.json"
        try:
            contract_path = Path(data_dir) / "dataset_contract.json"
            split_policy = "independent temporal holdout"
            if contract_path.is_file():
                contract_payload = json.loads(contract_path.read_text(encoding="utf-8"))
                split_policy = str(contract_payload.get("split_policy") or split_policy)
            independent_review = evaluate_independent_holdout(
                best_out / "submission.csv",
                holdout_path,
                best_out / "metrics.json",
                output_path=review_path,
                split_policy=split_policy,
            )
        except Exception as exc:  # reviewer failure is explicit evidence, never silent
            independent_review = {
                "schema": "evomind.independent_holdout_review.v1",
                "status": "rejected",
                "reviewer": "IndependentHoldoutReviewer",
                "error": f"{type(exc).__name__}: {exc}",
                "claim_audit": {"status": "rejected", "official_submission_performed": False},
            }
            review_path.write_text(
                json.dumps(independent_review, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

    if independent_review and independent_review.get("status") == "passed" and summary.get("best_exp_id"):
        promoted = [
            str(item.get("exp_id"))
            for item in list(summary.get("iterations") or [])
            if item.get("success") and item.get("promoted") and item.get("exp_id") != summary.get("best_exp_id")
        ]
        if promoted:
            previous_exp_id = promoted[-1]
            previous_out = exp_root / previous_exp_id / "out"
            previous_review_path = exp_root / "previous_holdout_review.json"
            try:
                previous_review = evaluate_independent_holdout(
                    previous_out / "submission.csv",
                    holdout_path,
                    previous_out / "metrics.json",
                    output_path=previous_review_path,
                    split_policy=split_policy,
                )
                before = dict(previous_review.get("metrics") or {})
                after = dict(independent_review.get("metrics") or {})
                comparable = ("pr_auc", "recall", "brier_score", "expected_calibration_error_15bin")
                deltas = {
                    key: float(after[key]) - float(before[key])
                    for key in comparable
                    if isinstance(before.get(key), (int, float)) and isinstance(after.get(key), (int, float))
                }
                self_evolution_comparison = {
                    "schema": "evomind.self_evolution_comparison.v1",
                    "status": "passed" if previous_review.get("status") == "passed" else "rejected",
                    "parent_exp_id": previous_exp_id,
                    "child_exp_id": str(summary["best_exp_id"]),
                    "change_scope": "minimal_stepwise_candidate_change; unchanged data hash and temporal holdout",
                    "rerun_scope": ["candidate_generation", "candidate_training", "independent_review"],
                    "before": before,
                    "after": after,
                    "delta": deltas,
                    "old_version_preserved": True,
                    "parent_evidence_sha256": previous_review.get("input"),
                    "child_evidence_sha256": independent_review.get("input"),
                    "official_submission_performed": False,
                }
                (exp_root / "self_evolution_comparison.json").write_text(
                    json.dumps(self_evolution_comparison, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            except Exception as exc:
                self_evolution_comparison = {
                    "schema": "evomind.self_evolution_comparison.v1",
                    "status": "rejected",
                    "error": f"{type(exc).__name__}: {exc}",
                    "old_version_preserved": True,
                    "official_submission_performed": False,
                }
                (exp_root / "self_evolution_comparison.json").write_text(
                    json.dumps(self_evolution_comparison, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )

    observed_generators = [
        {"provider": item.get("provider"), "model": item.get("model")}
        for item in list(summary.get("iterations") or [])
        if item.get("provider") or item.get("model")
    ]
    provenance_passed = not orchestration_model or (
        bool(observed_generators)
        and all(item.get("provider") == "openai" and item.get("model") == orchestration_model for item in observed_generators)
    )
    provenance = {
        "status": "passed" if provenance_passed else "rejected",
        "requested_model": orchestration_model or None,
        "observed_generators": observed_generators,
    }
    (exp_root / "model_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    rel = lambda p: str(p).replace(str(ROOT), "").lstrip("/\\").replace("\\", "/")  # noqa: E731
    terminal_reason = str(summary.get("terminal_reason") or "")
    successful_terminal_reasons = {
        "node_budget_exhausted",
        "token_budget_exhausted",
        "wall_clock_budget_exhausted",
        "cost_budget_exhausted",
    }
    terminal_ok = not terminal_reason or terminal_reason in successful_terminal_reasons
    demo_claim_ok = (
        not demo_campaign
        or (
            isinstance(demo_evidence, dict)
            and isinstance(demo_evidence.get("claim_audit"), dict)
            and demo_evidence["claim_audit"].get("status") == "passed"
        )
    )
    run_ok = terminal_ok and demo_claim_ok
    cli_receipt = {
        "schema": "evomind.evolution_cli_receipt.v1",
        "ok": run_ok,
        "task_id": task_id,
        "runner": runner_kind,
        "summary_path": rel(summary_path),
        "exp_dir": rel(exp_root),
        "best_exp_id": summary.get("best_exp_id"),
        "best_cv_score": summary.get("best_cv_score"),
        "metric": summary.get("metric"),
        "metric_direction": summary.get("metric_direction"),
        "n_iterations": summary.get("n_iterations"),
        "n_promotions": summary.get("n_promotions"),
        "terminal_reason": terminal_reason or None,
        "search_mode": search_mode,
        "budgets": {
            "max_nodes": run_contract["max_nodes"],
            "max_tokens": run_contract["max_tokens"],
            "max_wall_seconds": run_contract["max_wall_seconds"],
            "max_cost": run_contract["max_cost"],
        },
        "observability": {
            "run_contract": rel(run_contract_path),
            "experience_board": rel(exp_root / "experience-board.json") if (exp_root / "experience-board.json").is_file() else None,
            "selection_traces": rel(exp_root / "selection-traces.jsonl") if (exp_root / "selection-traces.jsonl").is_file() else None,
            "retrieval_bundles": rel(retrieval_path),
            "budget_ledger": rel(exp_root / "budget-ledger.json") if (exp_root / "budget-ledger.json").is_file() else None,
            "events": rel(events_path),
        },
        "resource_gate": resource_gate,
        "independent_review": independent_review,
        "orchestration_model": orchestration_model or None,
        "model_provenance": provenance,
        "self_evolution_comparison": self_evolution_comparison,
        "demo_campaign": demo_campaign,
        "demo_evidence": {
            "candidate_freeze_sha256": demo_evidence.get("freeze_sha256"),
            "independent_review_sha256": demo_evidence.get("independent_review_sha256"),
            "claim_audit_sha256": demo_evidence.get("claim_audit_sha256"),
            "independent_review": demo_evidence.get("review"),
            "claim_audit": demo_evidence.get("claim_audit"),
        } if demo_evidence else None,
        "input_contract_path": rel(input_contract_path),
        "input_contract_sha256": input_contract_sha256,
    }
    cli_receipt_path = exp_root / "cli-receipt.json"
    _write_json_atomic(cli_receipt_path, cli_receipt)
    print(json.dumps(cli_receipt, ensure_ascii=False))
    return 0 if run_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
