from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "e2lmc_candidate2_goal_v2.py"
SPEC = importlib.util.spec_from_file_location("e2lmc_candidate2_goal_v2", SCRIPT)
assert SPEC and SPEC.loader
e2lmc = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = e2lmc
SPEC.loader.exec_module(e2lmc)


def _payload() -> dict:
    iterations = list(range(0, 16000, 1000))
    experiments: dict[str, dict[str, list[float]]] = {}
    for index, name in enumerate(
        ("dense-500m-arch2", "dense-1b-arch2", "dense-2b-arch1", "dense-2b-arch2")
    ):
        # Candidate metric-a is a clean learning curve; mmluvar is noisy on
        # the fresh tail.  The fixture is public curve data, not labels.
        base = [float(i + ((i + index) % 3) * 0.2) for i in range(len(iterations))]
        if index >= 2:
            base = [float(i if i < 10 else 14 - i * 0.1) for i in range(len(iterations))]
        experiments[name] = {
            "iteration": [float(value) for value in iterations],
            "gtokens": [float(value) * 1000.0 for value in iterations],
            "mmluvar": base,
            "metric_a": [float(i) for i in range(len(iterations))],
            "metric_b": [float(len(iterations) - i) for i in range(len(iterations))],
            "metric_c": [float((i % 4) * 0.1 + i) for i in range(len(iterations))],
        }
    return {
        "metadata": {"metrics": ["iteration", "gtokens", "mmluvar", "metric_a", "metric_b", "metric_c"]},
        "data": experiments,
    }


def _write_inputs(tmp_path: Path, *, ledger: dict | None = None) -> tuple[Path, Path, str]:
    source = tmp_path / "plot-data-combined-all.json"
    raw = (json.dumps(_payload(), ensure_ascii=False, sort_keys=True) + "\n").encode()
    source.write_bytes(raw)
    source_sha = hashlib.sha256(raw).hexdigest()
    ledger_path = tmp_path / "holdout-ledger.v1.json"
    if ledger is None:
        ledger = {
            "schema": e2lmc.LEDGER_SCHEMA,
            "official_source_sha256": source_sha,
            "consumed_experiments": ["dense-500m-arch2", "dense-1b-arch2"],
            "consumed_iteration_points": [],
            "consumed_point_count": 0,
            "hidden_test_labels_used": False,
        }
    ledger_path.write_text(json.dumps(ledger, sort_keys=True) + "\n", encoding="utf-8")
    return source, ledger_path, source_sha


def test_candidate1_points_and_prior_architectures_are_denied(tmp_path):
    source, ledger_path, source_sha = _write_inputs(tmp_path)
    raw, payload = e2lmc.load_official_source(source, expected_sha256=source_sha)
    ledger = e2lmc.load_holdout_ledger(ledger_path, source_sha256=source_sha)
    result = e2lmc.run_candidate(raw, payload, ledger, source_sha=source_sha)
    assert result["denied_architecture_holdouts"] == ["dense-1b-arch2", "dense-500m-arch2"]
    assert set(result["eligible_experiments"]) == {"dense-2b-arch1", "dense-2b-arch2"}
    assert all(
        not token.endswith(("|0", "|4000", "|8000", "|12000"))
        for row in result["holdout_point_rows"]
        for token in [e2lmc.point_token(row["experiment"], row["iteration"])]
    )
    assert result["independent"]["holdout_point_overlap"] == 0
    assert result["independent"]["hidden_test_labels_used"] is False


def test_explicit_consumed_iteration_point_is_excluded(tmp_path):
    source, ledger_path, source_sha = _write_inputs(tmp_path)
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["consumed_iteration_points"] = [{"experiment": "dense-2b-arch1", "iteration": 9000}]
    ledger["consumed_point_count"] = 1
    ledger_path.write_text(json.dumps(ledger) + "\n", encoding="utf-8")
    raw, payload = e2lmc.load_official_source(source, expected_sha256=source_sha)
    loaded = e2lmc.load_holdout_ledger(ledger_path, source_sha256=source_sha)
    result = e2lmc.run_candidate(raw, payload, loaded, source_sha=source_sha)
    assert e2lmc.point_token("dense-2b-arch1", 9000) not in {
        e2lmc.point_token(row["experiment"], row["iteration"]) for row in result["holdout_point_rows"]
    }


def test_source_hash_mismatch_is_exact_gate(tmp_path):
    source, _, _ = _write_inputs(tmp_path)
    with pytest.raises(e2lmc.ExactGateError) as caught:
        e2lmc.load_official_source(source, expected_sha256="0" * 64)
    assert caught.value.code == "OFFICIAL_SOURCE_SHA_MISMATCH"


def test_ledger_source_binding_and_schema_are_exact_gates(tmp_path):
    source, ledger_path, source_sha = _write_inputs(tmp_path)
    bad = json.loads(ledger_path.read_text(encoding="utf-8"))
    bad["official_source_sha256"] = "1" * 64
    ledger_path.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(e2lmc.ExactGateError) as caught:
        e2lmc.load_holdout_ledger(ledger_path, source_sha256=source_sha)
    assert caught.value.code == "HOLDOUT_LEDGER_SOURCE_MISMATCH"
    bad["official_source_sha256"] = source_sha
    bad["schema"] = "wrong"
    ledger_path.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(e2lmc.ExactGateError) as caught:
        e2lmc.load_holdout_ledger(ledger_path, source_sha256=source_sha)
    assert caught.value.code == "HOLDOUT_LEDGER_SCHEMA_MISMATCH"


def test_hidden_test_labels_are_rejected(tmp_path):
    source, ledger_path, source_sha = _write_inputs(tmp_path)
    payload = _payload()
    payload["test_labels"] = {"hidden": [1, 2]}
    raw = (json.dumps(payload, sort_keys=True) + "\n").encode()
    source.write_bytes(raw)
    source_sha = hashlib.sha256(raw).hexdigest()
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["official_source_sha256"] = source_sha
    ledger_path.write_text(json.dumps(ledger) + "\n", encoding="utf-8")
    raw, loaded_payload = e2lmc.load_official_source(source, expected_sha256=source_sha)
    loaded_ledger = e2lmc.load_holdout_ledger(ledger_path, source_sha256=source_sha)
    with pytest.raises(e2lmc.ExactGateError) as caught:
        e2lmc.run_candidate(raw, loaded_payload, loaded_ledger, source_sha=source_sha)
    assert caught.value.code == "HIDDEN_TEST_LABELS_PRESENT"


def test_ledger_reports_hidden_labels_as_exact_gate(tmp_path):
    source, ledger_path, source_sha = _write_inputs(tmp_path)
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["hidden_test_labels_used"] = True
    ledger_path.write_text(json.dumps(ledger) + "\n", encoding="utf-8")
    with pytest.raises(e2lmc.ExactGateError) as caught:
        e2lmc.load_holdout_ledger(ledger_path, source_sha256=source_sha)
    assert caught.value.code == "HOLDOUT_LEDGER_TEST_LABELS"


def test_ledger_explicitly_mutable_is_rejected(tmp_path):
    source, ledger_path, source_sha = _write_inputs(tmp_path)
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["read_only"] = False
    ledger_path.write_text(json.dumps(ledger) + "\n", encoding="utf-8")
    with pytest.raises(e2lmc.ExactGateError) as caught:
        e2lmc.load_holdout_ledger(ledger_path, source_sha256=source_sha)
    assert caught.value.code == "HOLDOUT_LEDGER_NOT_READ_ONLY"


def test_available_point_whitelist_is_applied(tmp_path):
    source, ledger_path, source_sha = _write_inputs(tmp_path)
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["available_experiments"] = ["dense-2b-arch1", "dense-2b-arch2"]
    ledger["available_points"] = [
        {"experiment": "dense-2b-arch1", "iteration": 1000},
        {"experiment": "dense-2b-arch1", "iteration": 2000},
        {"experiment": "dense-2b-arch1", "iteration": 3000},
        {"experiment": "dense-2b-arch1", "iteration": 5000},
        {"experiment": "dense-2b-arch1", "iteration": 6000},
        {"experiment": "dense-2b-arch1", "iteration": 7000},
        {"experiment": "dense-2b-arch1", "iteration": 9000},
        {"experiment": "dense-2b-arch1", "iteration": 10000},
        {"experiment": "dense-2b-arch2", "iteration": 1000},
        {"experiment": "dense-2b-arch2", "iteration": 2000},
        {"experiment": "dense-2b-arch2", "iteration": 3000},
        {"experiment": "dense-2b-arch2", "iteration": 5000},
        {"experiment": "dense-2b-arch2", "iteration": 6000},
        {"experiment": "dense-2b-arch2", "iteration": 7000},
        {"experiment": "dense-2b-arch2", "iteration": 9000},
        {"experiment": "dense-2b-arch2", "iteration": 10000},
    ]
    ledger_path.write_text(json.dumps(ledger) + "\n", encoding="utf-8")
    raw, payload = e2lmc.load_official_source(source, expected_sha256=source_sha)
    loaded = e2lmc.load_holdout_ledger(ledger_path, source_sha256=source_sha)
    result = e2lmc.run_candidate(raw, payload, loaded, source_sha=source_sha)
    assert set(result["eligible_experiments"]) == {"dense-2b-arch1", "dense-2b-arch2"}
    assert all(row["iteration"] in {5000.0, 6000.0, 7000.0, 9000.0, 10000.0} for row in result["holdout_point_rows"])


def test_paired_bootstrap_is_deterministic_and_paired():
    rows = [
        {"experiment": "a", "baseline": 0.2, "candidate": 0.4},
        {"experiment": "b", "baseline": 0.3, "candidate": 0.5},
        {"experiment": "c", "baseline": 0.4, "candidate": 0.6},
    ]
    first = e2lmc.paired_group_bootstrap(rows, rounds=300, seed=17)
    second = e2lmc.paired_group_bootstrap(rows, rounds=300, seed=17)
    assert first == second
    assert first["resampling_unit"] == "experiment_group_paired_delta"
    assert first["ci95_lower"] > 0


def test_temporal_split_uses_latest_fresh_block():
    rows = {str(index): {"iteration": float(index), "mmluvar": 0.1, "metric_a": 0.2} for index in range(12)}
    split = e2lmc.temporal_split(rows, experiment="x")
    assert split.development == tuple(range(8))
    assert split.holdout == tuple(range(8, 12))


def test_insufficient_fresh_points_is_exact_gate():
    rows = {str(index): {"iteration": float(index), "mmluvar": 0.1, "metric_a": 0.2} for index in range(7)}
    with pytest.raises(e2lmc.ExactGateError) as caught:
        e2lmc.temporal_split(rows, experiment="small")
    assert caught.value.code == "INSUFFICIENT_FRESH_POINTS"


def test_main_writes_manifest_and_no_network(tmp_path, monkeypatch):
    source, ledger, source_sha = _write_inputs(tmp_path)
    monkeypatch.setattr(e2lmc, "SOURCE_SHA256", source_sha)
    output = tmp_path / "out"
    exit_code = e2lmc.main(
        [
            "--source-json",
            str(source),
            "--holdout-ledger",
            str(ledger),
            "--out-dir",
            str(output),
        ]
    )
    assert exit_code == 0
    manifest = json.loads((output / "artifact-manifest.json").read_text(encoding="utf-8"))
    names = {item["name"] for item in manifest["files"]}
    assert "solution.py" in names
    assert manifest["network_access"] is False
    assert manifest["holdout_ledger_sha256"]
    independent = json.loads((output / "independent-verification.json").read_text(encoding="utf-8"))
    assert independent["test_labels_used"] is False


def test_main_missing_inputs_returns_exact_gate(tmp_path):
    output = tmp_path / "out"
    assert e2lmc.main(["--out-dir", str(output)]) == 2
    gate = json.loads((output / "exact-gate.json").read_text(encoding="utf-8"))
    assert gate["status"] == "waiting_exact_gate"
    assert gate["code"] == "OFFICIAL_SOURCE_SNAPSHOT_MISSING"


def test_no_network_import_or_remote_fallback():
    source_text = SCRIPT.read_text(encoding="utf-8")
    assert "urllib" not in source_text
    assert "urlopen" not in source_text


def test_candidate1_script_hash_is_not_changed(tmp_path):
    original = Path(__file__).resolve().parents[1] / "scripts" / "e2lmc_goal_v2.py"
    before = hashlib.sha256(original.read_bytes()).hexdigest()
    _write_inputs(tmp_path)
    after = hashlib.sha256(original.read_bytes()).hexdigest()
    assert after == before


def test_point_token_canonicalizes_integer_float():
    assert e2lmc.point_token("x", 4000) == "x|4000"
    assert e2lmc.point_token("x", 4000.0) == "x|4000"
    assert e2lmc.point_token("x", 0.125) == "x|0.125"
