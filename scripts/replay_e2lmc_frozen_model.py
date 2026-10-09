"""Independently replay a frozen E2LMC model without training or selection."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import itertools
import json
import platform
import sys
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import rankdata


INPUT_HASHES = {
    "plot-data-combined-all.json": "afbea8052cc0b3c3c08a2b4d6000e0c1b960f9d9b4bdd6671d167abb01278d69",
    "official-5-Scoring.ipynb": "27fc483c51ce8dd2fc4753c55c9e66564e315af483ab2bdff5441ad4b9182b70",
    "04007736cb774386.json": "43905228a0e05c81239784ed31f94992a55ee75bd4653372c5a21fd9f8fff25a",
    "4ccde5db15994f98.json": "35b92174a2b3184734dabc40d9a7b2eee9d202b20d8da3e53031df4d75709670",
    "c62b55c6e9124ba1.json": "69ad5c6354d612ce79ce4cda1b34e89624ebff41bfae041305506e96f539ddc7",
    "d1778e19ae22434a.json": "e0b7b5ed2c37514278e369cee20d6016dfabcfc3ed125d052fbc54ec6fba0b4b",
}
PAPER_GROUPS = ("dense-500m-arch1", "dense-1b-arch1", "dense-3b-arch1")
PAPER_SQ = dict(zip(PAPER_GROUPS, (0.9354, 0.9814, 0.9601)))
FUNCTIONS = {"autocorrelation", "spearmancorrelation", "signal_quality_score", "convert_iter_to_gt"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def official_functions(notebook: dict):
    """Load only the four inspected mathematical definitions, not notebook I/O."""
    definitions = {}
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS:
                definitions[node.name] = node
    if set(definitions) != FUNCTIONS:
        raise ValueError("official_scorer_definition_set_changed")
    selected = ast.Module(body=list(definitions.values()), type_ignores=[])
    allowed_calls = {"len", "range", "int", "max", "sum", "abs"}
    allowed_numpy = {"array", "argsort", "sum", "corrcoef", "mean", "abs", "arange", "ones", "max", "cumsum", "searchsorted"}
    for node in ast.walk(selected):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.With, ast.AsyncWith, ast.Global, ast.Nonlocal)):
            raise ValueError("official_scorer_contains_unexpected_io")
        if isinstance(node, ast.Call):
            target = node.func
            ordinary = isinstance(target, ast.Name) and target.id in (allowed_calls | FUNCTIONS)
            numpy_call = isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "np" and target.attr in allowed_numpy
            append = isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "a_gt" and target.attr == "append"
            if not (ordinary or numpy_call or append):
                raise ValueError("official_scorer_call_not_allowlisted")
    namespace = {"np": np}
    exec(compile(selected, "official-5-Scoring:pure-functions", "exec"), namespace)
    return namespace


def correlation(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a, b = a - a.mean(), b - b.mean()
    denominator = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
    if denominator <= 0 or not np.isfinite(denominator):
        raise ValueError("degenerate_sequence")
    return float(np.dot(a, b) / denominator)


def sq(values, mode="official") -> dict:
    y = np.asarray(values, dtype=float)
    n = len(y)
    if n < 8 or not np.isfinite(y).all() or np.var(y) <= 0:
        raise ValueError("invalid_scoring_sequence")
    if mode == "legacy_average_ties":
        rho = correlation(np.arange(n), rankdata(y, method="average"))
    else:
        ordinal = np.argsort(np.argsort(y))
        rho = float(1 - 6 * np.sum((np.arange(n) - ordinal) ** 2) / (n * (n * n - 1)))
    lags = list(range(1, n // 4))
    autocorr = float(np.mean([abs(correlation(y[:-lag], y[lag:])) for lag in lags]))
    return {"sq": (max(rho, 0.0) + autocorr) / 2, "rho": rho, "autocorrelation": autocorr,
            "n": n, "lags": lags, "tied_observations": int(n - len(np.unique(y)))}


def tie_bounds(values) -> list[float]:
    y = np.asarray(values, dtype=float)
    n = len(y)
    low_rank = np.empty(n, dtype=int)
    high_rank = np.empty(n, dtype=int)
    start = 0
    for value in np.unique(y):
        indices = np.flatnonzero(y == value)
        ranks = np.arange(start, start + len(indices))
        high_rank[indices] = ranks
        low_rank[indices] = ranks[::-1]
        start += len(indices)
    ac = sq(y)["autocorrelation"]
    def score(ranks):
        rho = 1 - 6 * np.sum((np.arange(n) - ranks) ** 2) / (n * (n * n - 1))
        return float((max(rho, 0.0) + ac) / 2)
    return [score(low_rank), score(high_rank)]


def transform(values, state):
    knots, ranks = np.asarray(state["knots"], float), np.asarray(state["ranks"], float)
    if len(knots) != len(ranks) or not np.isfinite(knots).all() or not np.isfinite(ranks).all():
        raise ValueError("calibrator_shape_invalid")
    unique, inverse = np.unique(knots, return_inverse=True)
    if len(unique) < 2:
        raise ValueError("calibrator_degenerate")
    average_ranks = np.bincount(inverse, weights=ranks) / np.bincount(inverse)
    return np.interp(np.clip(values, state["lo"], state["hi"]), unique, average_ranks)


def exact_group_interval(deltas) -> list[float]:
    d = np.asarray(deltas, float)
    values = np.asarray([np.mean(d[list(indices)]) for indices in itertools.product(range(len(d)), repeat=len(d))])
    return np.quantile(values, [0.025, 0.975]).tolist()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "evomind.e2lmc_frozen_independent_replay.v1", "status": "failed",
              "source_run_id": "run_378142db3df44b5eaadad80d4b3ac73a", "model_generation": "Gen4",
              "training_repeated": False, "model_selection_performed": False, "new_holdout_evaluations": 0,
              "hidden_rc_sc_accessed": False, "gpu_used": False}
    try:
        for name, expected in INPUT_HASHES.items():
            if digest(args.input_dir / name) != expected:
                raise ValueError("input_hash_mismatch:" + name)
        def load(name):
            return json.loads((args.input_dir / name).read_text(encoding="utf-8"))
        data = load("plot-data-combined-all.json")["data"]
        weights = load("04007736cb774386.json")
        calibrators = load("4ccde5db15994f98.json")
        features = load("c62b55c6e9124ba1.json")["features"]
        historical = {row["outer_group"]: row for row in load("d1778e19ae22434a.json")["outer_folds"]}
        official = official_functions(load("official-5-Scoring.ipynb"))
        groups = sorted(weights)
        if set(groups) != set(data) or set(groups) != set(calibrators) or set(groups) != set(historical):
            raise ValueError("group_identity_mismatch")
        protocol = {"paper": "https://arxiv.org/html/2506.07731v1", "paper_tables": [3, 6],
                    "paper_sq_groups": list(PAPER_GROUPS), "paper_group_scores_rounded": PAPER_SQ,
                    "paper_sq_rounded": 0.959, "paper_total_score_rounded": 0.717,
                    "scorer_source": "https://colab.research.google.com/drive/1sH0Pe-HS2zJyFt0yxec7dyMk0KY2ZRlz",
                    "data_filter": "iteration modulo 4000 == 0; official ramp-up conversion <= 206 billion tokens",
                    "tie_rule": "official notebook ordinal double-argsort; no average-rank substitution",
                    "all_six_groups_also_reported": True, "evaluation_scope": "retrospective_consumed_public_development_data",
                    "selection_frozen_before_replay": True, "fresh_independent_gain_claim_allowed": False,
                    "compute_budget_parity_established": False, "rc_sc_and_total_score_computed": False,
                    "input_hashes": INPUT_HASHES}
        save(output / "protocol.json", protocol)
        prepared = {}
        for group in groups:
            iterations = np.asarray(data[group]["iteration"], dtype=int)
            indices = np.flatnonzero(iterations % 4000 == 0)
            tokens = np.asarray(official["convert_iter_to_gt"](iterations[indices]), dtype=float)
            keep = tokens <= 206
            indices, tokens = indices[keep], tokens[keep]
            if iterations[indices].tolist() != list(range(4000, 56001, 4000)):
                raise ValueError("official_support_changed:" + group)
            values = {feature: np.asarray(data[group][feature], dtype=float)[indices] for feature in features}
            if not all(np.isfinite(v).all() for v in values.values()):
                raise ValueError("nonfinite_feature_on_required_support:" + group)
            prepared[group] = {"iterations": iterations[indices], "tokens": tokens, "values": values}
        rows, predictions = [], []
        for group in groups:
            state, spec, sample = calibrators[group], weights[group], prepared[group]
            training_groups = [g for g in groups if g != group]
            if set(historical[group]["train_groups"]) != set(training_groups):
                raise ValueError("training_group_boundary_mismatch:" + group)
            for feature in features:
                pooled = np.concatenate([prepared[g]["values"][feature] for g in training_groups])
                c = state[feature]
                lo, hi = np.quantile(pooled, [c["winsor"], 1 - c["winsor"]])
                grid = np.linspace(0, 1, c["quantiles"])
                knots = np.quantile(np.clip(pooled, lo, hi), grid)
                if not (np.allclose([lo, hi], [c["lo"], c["hi"]], rtol=0, atol=1e-12)
                        and np.allclose(knots, c["knots"], rtol=0, atol=1e-12)
                        and np.allclose(grid, c["ranks"], rtol=0, atol=1e-12)):
                    raise ValueError("frozen_calibrator_training_partition_mismatch:" + group + ":" + feature)
            if abs(spec["w_mmluvar"] + sum(spec["other"].values()) - 1) > 1e-12 or min([spec["w_mmluvar"], *spec["other"].values()]) < 0:
                raise ValueError("frozen_weight_contract_invalid")
            baseline = sample["values"]["mmluvar"]
            candidate = spec["w_mmluvar"] * transform(baseline, state["mmluvar"])
            for feature, weight in spec["other"].items():
                candidate += weight * transform(sample["values"][feature], state[feature])
            b, c = sq(baseline), sq(candidate)
            legacy = sq(candidate, "legacy_average_ties")
            for values, result in [(baseline, b), (candidate, c)]:
                reference = official["signal_quality_score"](np.asarray(values))
                if not np.allclose([max(result["rho"], 0), result["autocorrelation"], result["sq"]], reference, rtol=0, atol=1e-12):
                    raise ValueError("official_reference_parity_failed")
            if abs(b["sq"] - historical[group]["mmluvar_sq"]) > 1e-12 or abs(legacy["sq"] - historical[group]["candidate_sq"]) > 1e-12:
                raise ValueError("historical_prediction_replay_mismatch:" + group)
            if group in PAPER_SQ and abs(b["sq"] - PAPER_SQ[group]) > 0.00005:
                raise ValueError("paper_table_baseline_anchor_failed:" + group)
            row = {"group": group, "in_paper_sq_cohort": group in PAPER_GROUPS,
                   "n": len(baseline), "baseline_sq": b["sq"], "candidate_sq": c["sq"],
                   "legacy_candidate_sq": legacy["sq"], "paired_delta": c["sq"] - b["sq"],
                   "tied_observations": c["tied_observations"], "candidate_tie_order_bounds": tie_bounds(candidate),
                   "active_input_benchmarks": ["mmluvar", *[k for k, value in spec["other"].items() if value > 0]],
                   "training_groups_verified": training_groups,
                   "support_sha256": hashlib.sha256(json.dumps(sample["iterations"].tolist()).encode()).hexdigest()}
            rows.append(row)
            for iteration, tokens, base_value, value in zip(sample["iterations"], sample["tokens"], baseline, candidate):
                predictions.append({"group": group, "iteration": int(iteration), "billion_tokens": float(tokens),
                                    "baseline_value": float(base_value), "candidate_value": float(value)})
        aggregates = {}
        for name, selected in [("paper_arch1_three_groups", [r for r in rows if r["in_paper_sq_cohort"]]), ("all_six_public_groups", rows)]:
            baseline = float(np.mean([r["baseline_sq"] for r in selected]))
            candidate = float(np.mean([r["candidate_sq"] for r in selected]))
            aggregates[name] = {"groups": [r["group"] for r in selected], "baseline_sq": baseline,
                                "candidate_sq": candidate, "paired_delta": candidate - baseline,
                                "descriptive_paired_group_bootstrap_ci95": exact_group_interval([r["paired_delta"] for r in selected]),
                                "candidate_tie_order_bounds": np.mean([r["candidate_tie_order_bounds"] for r in selected], axis=0).tolist()}
        report.update(status="verified", input_hashes=INPUT_HASHES, groups=rows, aggregates=aggregates,
                      original_six_group_mean_with_average_ties=float(np.mean([r["legacy_candidate_sq"] for r in rows])),
                      official_scorer_parity_verified=True, frozen_model_replay_verified=True,
                      out_of_group_preprocessing_verified=True, paper_table3_anchors_verified=True,
                      official_total_score=None, new_independent_scientific_gain_verified=False,
                      compute_budget_parity_verified=False,
                      claim="Retrospective public SQ replay only; not a new independent experiment or official full competition score.",
                      environment={"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform()},
                      verifier_sha256=digest(Path(__file__)))
        with (output / "replayed-predictions.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
            writer.writeheader()
            writer.writerows(predictions)
        save(output / "independent-replay.json", report)
        save(output / "artifact-manifest.json", {"schema": "evomind.e2lmc_replay_manifest.v1",
             "files": [{"name": p.name, "bytes": p.stat().st_size, "sha256": digest(p)} for p in sorted(output.iterdir()) if p.is_file()]})
        print(json.dumps({"status": report["status"], "aggregates": aggregates, "output": str(output),
                          "new_independent_scientific_gain_verified": False, "official_total_score": None}))
        return 0
    except Exception as error:
        report.update(error_class=type(error).__name__, error=str(error))
        save(output / "replay-failure.json", report)
        print(json.dumps(report))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
