#!/usr/bin/env python3
"""Run the Russian text-normalization public OOF audit on HPC without remote writes."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import mlebench_remote_ops as remote_ops
from scripts import mlebench_wave2_adapters as wave2


REMOTE_ROOT = "/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra"
REMOTE_DATASET = (
    f"{REMOTE_ROOT}/mlebench_official_data/"
    "text-normalization-challenge-russian-language/prepared/public/ru_train.csv.zip"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "workspace"
    / "mlebench_plans"
    / "text_normalization_russian_public_oof_validation_current.json"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def source_sha256() -> str:
    return hashlib.sha256(Path(wave2.__file__).read_bytes()).hexdigest()


def _remote_rule_source() -> str:
    constants = [
        "_DIGIT_WORDS_RUSSIAN",
        "_RUSSIAN_SMALL",
        "_RUSSIAN_TENS",
        "_RUSSIAN_HUNDREDS",
        "_RUSSIAN_GENITIVE_SMALL",
        "_RUSSIAN_GENITIVE_TENS",
        "_RUSSIAN_GENITIVE_HUNDREDS",
        "_RUSSIAN_ORDINAL_BASE",
        "_RUSSIAN_ORDINAL_HUNDREDS",
        "_RUSSIAN_MONTHS_GENITIVE",
    ]
    functions = [
        wave2._roman_integer,
        wave2._russian_under_thousand,
        wave2.russian_integer_words,
        wave2.russian_cardinal_genitive_words,
        wave2._inflect_russian_ordinal_word,
        wave2._russian_ordinal_base_phrase,
        wave2.russian_ordinal_words,
        wave2.russian_date_words,
        wave2.russian_letters_words,
        wave2.russian_digit_words,
        wave2.russian_telephone_words,
        wave2._russian_feminine_cardinal,
        wave2.russian_decimal_words,
        wave2.russian_fraction_words,
        wave2.russian_time_words,
        wave2._russian_plural_unit,
        wave2.russian_measure_words,
        wave2.russian_ordinal_token_words,
        wave2.russian_contextual_token_words,
        wave2.infer_normalization_class,
        wave2.normalize_token,
    ]
    chunks = ["from __future__ import annotations", "import re"]
    chunks.extend(f"{name} = {getattr(wave2, name)!r}" for name in constants)
    chunks.extend(inspect.getsource(function).strip() for function in functions)
    return "\n\n".join(chunks)


def _remote_evaluator_source(*, row_limit: int, folds: int) -> str:
    rule_source = _remote_rule_source()
    return rf'''{rule_source}

import csv
import io
import json
import zipfile
from collections import Counter, defaultdict

DATASET = {REMOTE_DATASET!r}
ROW_LIMIT = {row_limit}
FOLDS = {folds}


def read_rows():
    rows = []
    with zipfile.ZipFile(DATASET) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not members:
            raise RuntimeError("Russian training CSV is absent from the ZIP archive")
        with archive.open(members[0]) as raw:
            reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8", newline=""))
            required = {{"sentence_id", "class", "before", "after"}}
            missing = required.difference(reader.fieldnames or ())
            if missing:
                raise RuntimeError(f"Russian training CSV missing columns: {{sorted(missing)}}")
            for index, row in enumerate(reader):
                if ROW_LIMIT and index >= ROW_LIMIT:
                    break
                rows.append((row["sentence_id"], row["class"], row["before"], row["after"]))
    return rows


def greedy_group_folds(rows):
    sizes = Counter(row[0] for row in rows)
    loads = [0] * FOLDS
    assignment = {{}}
    for sentence_id, size in sorted(sizes.items(), key=lambda item: (-item[1], item[0])):
        fold = min(range(FOLDS), key=lambda index: (loads[index], index))
        assignment[sentence_id] = fold
        loads[fold] += size
    return assignment, loads


def build_map(rows, include, keep_identity=False):
    pairs = Counter((before, after) for sentence_id, token_class, before, after in rows if include(sentence_id))
    best = {{}}
    distinct = Counter()
    for (before, after), count in pairs.items():
        distinct[before] += 1
        current = best.get(before)
        candidate = (count, after)
        if current is None or candidate[0] > current[0] or (candidate[0] == current[0] and candidate[1] < current[1]):
            best[before] = candidate
    mapping = {{
        before: after for before, (count, after) in best.items()
        if keep_identity or before != after
    }}
    ambiguous = {{before for before, count in distinct.items() if count > 1}}
    return mapping, set(best), ambiguous


def neighbor(rows, index, offset):
    target = index + offset
    if target < 0 or target >= len(rows) or rows[target][0] != rows[index][0]:
        return "<BOS>" if offset < 0 else "<EOS>"
    return rows[target][2]


def token_shape(token):
    if token in {{"<BOS>", "<EOS>"}}:
        return token
    if re.fullmatch(r"[-+]?\d[\d.,:/-]*", token):
        return "<NUM>"
    if re.fullmatch(r"[A-Za-z]+", token):
        return "<LATIN>"
    if re.fullmatch(r"[А-Яа-яЁё]+", token):
        return "<CYRILLIC>"
    return token.lower()


def finalize_context(counter):
    grouped = {{}}
    for (key, after), count in counter.items():
        record = grouped.setdefault(key, {{"total": 0, "best_count": 0, "after": after}})
        record["total"] += count
        if count > record["best_count"] or (count == record["best_count"] and after < record["after"]):
            record["best_count"] = count
            record["after"] = after
    return grouped


def build_context_maps(rows, assignment, held_out_fold, ambiguous_before):
    counters = {{name: Counter() for name in ("exact", "prev", "next", "shape")}}
    for index, (sentence_id, token_class, before, after) in enumerate(rows):
        if assignment[sentence_id] == held_out_fold or before not in ambiguous_before:
            continue
        previous = neighbor(rows, index, -1)
        following = neighbor(rows, index, 1)
        counters["exact"][((before, previous, following), after)] += 1
        counters["prev"][((before, previous), after)] += 1
        counters["next"][((before, following), after)] += 1
        counters["shape"][((before, token_shape(previous), token_shape(following)), after)] += 1
    return {{name: finalize_context(counter) for name, counter in counters.items()}}


CONTEXT_CONFIGS = {{
    "token_identity": None,
    "context_conservative": {{"exact": (2, 0.80), "prev": (4, 0.90), "next": (4, 0.90), "shape": (5, 0.92)}},
    "context_balanced": {{"exact": (2, 0.67), "prev": (3, 0.78), "next": (3, 0.78), "shape": (4, 0.82)}},
    "context_aggressive": {{"exact": (1, 0.50), "prev": (2, 0.60), "next": (2, 0.60), "shape": (3, 0.67)}},
}}


def contextual_prediction(rows, index, maps, config):
    sentence_id, token_class, before, after = rows[index]
    previous = neighbor(rows, index, -1)
    following = neighbor(rows, index, 1)
    keys = {{
        "exact": (before, previous, following),
        "prev": (before, previous),
        "next": (before, following),
        "shape": (before, token_shape(previous), token_shape(following)),
    }}
    candidates = []
    specificity = {{"exact": 4, "prev": 3, "next": 3, "shape": 2}}
    for name, key in keys.items():
        record = maps[name].get(key)
        if record is None:
            continue
        minimum, purity_minimum = config[name]
        purity = record["best_count"] / record["total"]
        if record["best_count"] >= minimum and purity >= purity_minimum:
            candidates.append((purity, min(record["best_count"], 20), specificity[name], record["after"]))
    return max(candidates)[3] if candidates else None


def empty_stats():
    return {{"correct": 0, "total": 0}}


def update(stats, correct):
    stats["total"] += 1
    stats["correct"] += int(correct)


def accuracy(stats):
    return stats["correct"] / stats["total"] if stats["total"] else None


def run_grouped_oof(rows):
    assignment, fold_sizes = greedy_group_folds(rows)
    inferred = empty_stats()
    oracle = empty_stats()
    unchanged = empty_stats()
    changed = empty_stats()
    unseen = empty_stats()
    ambiguous = empty_stats()
    class_stats = defaultdict(empty_stats)
    candidate_stats = {{name: empty_stats() for name in CONTEXT_CONFIGS}}
    errors = Counter()
    class_errors = Counter()
    unseen_errors = Counter()
    inference_confusion = Counter()
    rule_impact = Counter()
    mapping_entries = []
    mapping_hits = 0
    for fold in range(FOLDS):
        mapping, known_before, ambiguous_before = build_map(
            rows, lambda sentence_id: assignment[sentence_id] != fold, keep_identity=True
        )
        identity_mapping = mapping
        context_maps = build_context_maps(rows, assignment, fold, ambiguous_before)
        mapping_entries.append(len(mapping))
        for index, (sentence_id, token_class, before, after) in enumerate(rows):
            if assignment[sentence_id] != fold:
                continue
            contextual_value = russian_contextual_token_words(
                before, neighbor(rows, index, -1), neighbor(rows, index, 1)
            )
            inferred_prediction = contextual_value or normalize_token(before, mapping, "russian")
            oracle_prediction = normalize_token(before, mapping, "russian", token_class)
            inferred_class = infer_normalization_class(before, "russian")
            base_prediction = mapping.get(before, before)
            inferred_correct = inferred_prediction == after
            update(inferred, inferred_correct)
            update(oracle, oracle_prediction == after)
            update(class_stats[token_class], inferred_correct)
            update(changed if before != after else unchanged, inferred_correct)
            if before not in known_before:
                update(unseen, inferred_correct)
                if not inferred_correct:
                    unseen_errors[(token_class, inferred_class, before, after, inferred_prediction)] += 1
            if before in ambiguous_before:
                update(ambiguous, inferred_correct)
            inference_confusion[(token_class, inferred_class)] += 1
            if inferred_prediction != base_prediction:
                if inferred_correct and base_prediction != after:
                    rule_impact[(inferred_class, "gain")] += 1
                elif not inferred_correct and base_prediction == after:
                    rule_impact[(inferred_class, "loss")] += 1
                else:
                    rule_impact[(inferred_class, "neutral")] += 1
            if not inferred_correct:
                errors[(token_class, inferred_class, before, after, inferred_prediction)] += 1
                class_errors[(
                    token_class,
                    before,
                    after,
                    inferred_prediction,
                    neighbor(rows, index, -1),
                    neighbor(rows, index, 1),
                )] += 1
            token_identity_prediction = inferred_prediction
            update(candidate_stats["token_identity"], token_identity_prediction == after)
            for name, config in CONTEXT_CONFIGS.items():
                if config is None:
                    continue
                context_value = contextual_prediction(rows, index, context_maps, config)
                candidate = context_value if context_value is not None else token_identity_prediction
                update(candidate_stats[name], candidate == after)
            mapping_hits += int(before in mapping)
    return {{
        "fold_sizes": fold_sizes,
        "mapping_entries_by_fold": mapping_entries,
        "inferred_class_overall_accuracy": accuracy(inferred),
        "oracle_class_overall_accuracy": accuracy(oracle),
        "identity_accuracy": accuracy(unchanged),
        "changed_token_accuracy": accuracy(changed),
        "changed_rows": changed["total"],
        "unseen_before_accuracy": accuracy(unseen),
        "unseen_before_rows": unseen["total"],
        "ambiguous_before_accuracy": accuracy(ambiguous),
        "ambiguous_before_rows": ambiguous["total"],
        "mapping_hit_fraction": mapping_hits / len(rows),
        "candidate_context_metrics": {{
            name: {{"accuracy": accuracy(stats), "correct": stats["correct"], "rows": stats["total"]}}
            for name, stats in candidate_stats.items()
        }},
        "class_metrics": {{
            name: {{"rows": stats["total"], "accuracy": accuracy(stats)}}
            for name, stats in sorted(class_stats.items())
        }},
        "rule_impact": {{
            inferred_class: {{
                label: rule_impact.get((inferred_class, label), 0)
                for label in ("gain", "loss", "neutral")
            }}
            for inferred_class in sorted({{key[0] for key in rule_impact}})
        }},
        "inference_confusion": [
            {{"true_class": true_class, "inferred_class": inferred_class, "rows": count}}
            for (true_class, inferred_class), count in inference_confusion.most_common()
        ],
        "top_errors": [
            {{"true_class": item[0], "inferred_class": item[1], "before": item[2],
              "after": item[3], "prediction": item[4], "count": count}}
            for item, count in errors.most_common(160)
        ],
        "top_errors_by_class": {{
            token_class: [
                {{"before": item[1], "after": item[2], "prediction": item[3],
                  "previous": item[4], "following": item[5], "count": count}}
                for item, count in class_errors.most_common()
                if item[0] == token_class
            ][:80]
            for token_class in sorted({{item[0] for item in class_errors}})
        }},
        "top_unseen_errors": [
            {{"true_class": item[0], "inferred_class": item[1], "before": item[2],
              "after": item[3], "prediction": item[4], "count": count}}
            for item, count in unseen_errors.most_common(160)
        ],
    }}


def run_source_stress(rows):
    sentence_ids = list(dict.fromkeys(row[0] for row in rows))
    split = max(1, int(len(sentence_ids) * 0.8))
    training_ids = set(sentence_ids[:split])
    validation_ids = set(sentence_ids[split:])
    mapping, _, _ = build_map(rows, lambda sentence_id: sentence_id in training_ids, keep_identity=True)
    stats = empty_stats()
    for index, (sentence_id, token_class, before, after) in enumerate(rows):
        if sentence_id in validation_ids:
            contextual_value = russian_contextual_token_words(
                before, neighbor(rows, index, -1), neighbor(rows, index, 1)
            )
            prediction = contextual_value or normalize_token(before, mapping, "russian")
            update(stats, prediction == after)
    return {{"rows": stats["total"], "accuracy": accuracy(stats), "mapping_entries": len(mapping)}}


rows = read_rows()
grouped = run_grouped_oof(rows)
stress = run_source_stress(rows)
payload = {{
    "rows": len(rows),
    "sentences": len({{row[0] for row in rows}}),
    "changed_fraction": grouped.pop("changed_rows") / len(rows),
    **grouped,
    "source_stress_rows": stress["rows"],
    "source_stress_accuracy": stress["accuracy"],
    "source_stress_mapping_entries": stress["mapping_entries"],
}}
print("EVOMIND_RESULT=" + json.dumps(payload, ensure_ascii=False, sort_keys=True))
'''


def _run_remote_source(client: Any, source: str, *, timeout: int) -> tuple[int, str, str]:
    stdin, stdout, stderr = client.exec_command("nice -n 10 python3 -", timeout=timeout)
    stdout.channel.settimeout(timeout)
    stderr.channel.settimeout(timeout)
    stdin.write(source)
    stdin.flush()
    stdin.channel.shutdown_write()
    output = stdout.read().decode("utf-8", errors="replace")
    error = stderr.read().decode("utf-8", errors="replace")
    return stdout.channel.recv_exit_status(), output, error


def run_audit(*, row_limit: int, folds: int, timeout: int) -> dict[str, Any]:
    if not REMOTE_DATASET.startswith(REMOTE_ROOT + "/"):
        raise RuntimeError("Remote dataset escaped the dedicated HPC root")
    client = remote_ops._connect()
    try:
        preflight = (
            f"test -d {REMOTE_ROOT} && test -w {REMOTE_ROOT} && "
            f"test -r {REMOTE_DATASET} && printf 'PREFLIGHT_OK'"
        )
        code, output, error = remote_ops._run_remote(client, preflight, timeout=30)
        if code != 0 or output.strip() != "PREFLIGHT_OK":
            raise RuntimeError(f"Remote read-only preflight failed (exit={code}, stderr_type={{'nonempty' if error else 'empty'}})")
        source = _remote_evaluator_source(row_limit=row_limit, folds=folds)
        code, output, error = _run_remote_source(client, source, timeout=timeout)
    finally:
        client.close()
    if code != 0:
        tail = "\n".join(error.splitlines()[-12:])
        raise RuntimeError(f"Remote Russian OOF audit failed (exit={code}):\n{tail}")
    marker = "EVOMIND_RESULT="
    lines = [line for line in output.splitlines() if line.startswith(marker)]
    if len(lines) != 1:
        raise RuntimeError("Remote Russian OOF audit did not return exactly one result marker")
    metrics = json.loads(lines[0][len(marker) :])
    bronze = 0.97592
    grouped_target = 0.9800
    stress_target = 0.9760
    return {
        "schema": "evomind.mlebench.text_normalization_public_oof_validation.v1",
        "dataset": "text-normalization-challenge-russian-language",
        "path_scope_verified": True,
        "access_mode": "remote_read_only_cpu_probe",
        "split_strategy": "dependency_free_groupkfold_compatible_greedy_balance",
        "stress_strategy": "source_ordered_last_20pct_complete_sentences",
        "folds": folds,
        **metrics,
        "bronze_threshold": bronze,
        "grouped_target_accuracy": grouped_target,
        "source_stress_target_accuracy": stress_target,
        "bronze_passed": metrics["inferred_class_overall_accuracy"] >= bronze,
        "grouped_target_passed": metrics["inferred_class_overall_accuracy"] >= grouped_target,
        "source_stress_target_passed": metrics["source_stress_accuracy"] >= stress_target,
        "created_at": utc_now(),
        "local_source_sha256": source_sha256(),
        "remote_stderr_type": "nonempty" if error else "empty",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=500_000)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=3_600)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.rows <= 0 or args.folds < 2 or args.timeout <= 0:
        raise SystemExit("--rows/--timeout must be positive and --folds must be at least 2")
    result = run_audit(row_limit=args.rows, folds=args.folds, timeout=args.timeout)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = output.with_name(f"{output.stem}_before_{{stamp}}{{output.suffix}}")
        shutil.copy2(output, backup)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "output": str(output),
        "rows": result["rows"],
        "inferred_class_overall_accuracy": result["inferred_class_overall_accuracy"],
        "source_stress_accuracy": result["source_stress_accuracy"],
        "candidate_context_metrics": result.get("candidate_context_metrics", {}),
        "grouped_target_passed": result["grouped_target_passed"],
        "source_stress_target_passed": result["source_stress_target_passed"],
    }
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
