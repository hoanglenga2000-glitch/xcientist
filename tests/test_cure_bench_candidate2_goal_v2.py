from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "cure_bench_candidate2_goal_v2.py"
SPEC = importlib.util.spec_from_file_location("cure_bench_candidate2_goal_v2", MODULE_PATH)
assert SPEC and SPEC.loader
cure = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cure
SPEC.loader.exec_module(cure)


class FakeFrozenEncoder:
    manifest_info = {
        "schema": "evomind.cure_bench.frozen_encoder_manifest.v1",
        "backend": "test-fixture",
        "model_id": "fixture-only",
        "revision": "fixture",
        "pooling": "fixture",
        "hidden_size": 8,
        "manifest_sha256": "a" * 64,
        "frozen": True,
        "offline_only": True,
    }

    def encode(self, texts, *, batch_size: int = 16):
        values = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            values.append(np.asarray([byte / 255.0 for byte in digest[:8]], dtype=np.float32))
        return np.stack(values)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _choice_row(index: int, *, answer: str | int = "A") -> dict:
    label = index % 3
    return {
        "id": f"q-{index:03d}",
        "question": f"Which marker identifies class {label}?",
        "question_type": "multi_choice",
        "options": {"A": "marker alpha", "B": "marker beta", "C": "marker gamma"},
        "correct_answer": answer if answer != "A" else chr(ord("A") + label),
    }


def _data_root(tmp_path: Path, rows: int = 90) -> Path:
    root = tmp_path / "data-root"
    validation = [_choice_row(index) for index in range(rows)]
    _write_jsonl(root / "data" / "curebench_valset_pharse1.jsonl", validation)
    _write_jsonl(
        root / "data" / "curebench_testset_phase1.jsonl",
        [
            {
                "id": f"test-a-{index:03d}",
                "question": "Which marker is present?",
                "question_type": "multi_choice",
                "options": ["marker alpha", "marker beta", "marker gamma"],
            }
            for index in range(6)
        ],
    )
    _write_jsonl(
        root / "data" / "curebench_testset_phase2.jsonl",
        [
            {
                "id": f"test-b-{index:03d}",
                "question": "Which marker is present?",
                "question_type": "multi_choice",
                "options": ["marker alpha", "marker beta", "marker gamma"],
            }
            for index in range(6)
        ],
    )
    return root


def _primary_baseline_manifest(
    root: Path,
    *,
    baseline_type: str = "publicly_reproducible_strong",
    omit_role: str = "",
    extra_file: bool = False,
) -> Path:
    evidence = root / "managed_runtime" / "primary-baseline-evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    roles = ("source", "protocol", "implementation", "data_manifest", "environment", "reproduction_receipt")
    files = []
    for role in roles:
        if role == omit_role:
            continue
        path = evidence / f"{role}.txt"
        raw = f"fixture {role}\n".encode("utf-8")
        path.write_bytes(raw)
        files.append(
            {
                "role": role,
                "path": path.name,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    if extra_file:
        (evidence / "unlisted-extra.txt").write_text("extra", encoding="utf-8")
    protocol = "The same frozen CURE-Bench scored choice rows and accuracy aggregation are used."
    manifest = root / "managed_runtime" / "cure-bench-primary-baseline.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": cure.PRIMARY_BASELINE_SCHEMA,
                "competition": "cure_bench",
                "status": "VERIFIED",
                "baseline_type": baseline_type,
                "source_url": "https://curebench.example.edu/frozen-baseline-v1",
                "source_authority": "official_dataset",
                "title": "Frozen reproducible CURE-Bench baseline fixture",
                "protocol_id": "cure-bench-frozen-accuracy-v1",
                "protocol": protocol,
                "protocol_sha256": hashlib.sha256(protocol.encode("utf-8")).hexdigest(),
                "protocol_comparable": True,
                "metric": cure.PRIMARY_BASELINE_METRIC,
                "direction": "higher_is_better",
                "value": 0.5,
                "publicly_reproducible": baseline_type == "publicly_reproducible_strong",
                "evidence_dir": "primary-baseline-evidence",
                "files": files,
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_numeric_answer_requires_explicit_base_when_ambiguous():
    keys = ["A", "B", "C"]
    values = ["alpha", "beta", "gamma"]
    with pytest.raises(cure.ExactGateError) as error:
        cure.answer_index({"correct_answer": 1}, keys, values)
    assert error.value.code == "AMBIGUOUS_NUMERIC_ANSWER_ENCODING"
    assert cure.answer_index({"correct_answer": 1, "answer_index_base": 0}, keys, values)[0] == 1
    assert cure.answer_index({"correct_answer": 1, "answer_index_base": 1}, keys, values)[0] == 0


def test_recursive_test_label_detection_is_fail_closed(tmp_path: Path):
    path = tmp_path / "test.jsonl"
    _write_jsonl(
        path,
        [
            {
                "id": "t-1",
                "question": "q",
                "question_type": "multi_choice",
                "options": ["a", "b"],
                "metadata": {"gold": "A"},
            }
        ],
    )
    with pytest.raises(cure.ExactGateError) as error:
        cure.load_jsonl(path, require_answer=False)
    assert error.value.code == "TEST_LABELS_PRESENT"


def test_deep_or_hyphenated_test_label_keys_cannot_bypass_scan(tmp_path: Path):
    path = tmp_path / "deep-test.jsonl"
    nested = {"correct-answer": "A"}
    for _ in range(34):
        nested = {"metadata": nested}
    _write_jsonl(
        path,
        [
            {
                "id": "t-deep",
                "question": "q",
                "question_type": "multi_choice",
                "options": ["a", "b"],
                "metadata": nested,
            }
        ],
    )
    with pytest.raises(cure.ExactGateError) as error:
        cure.load_jsonl(path, require_answer=False)
    assert error.value.code in {"TEST_LABELS_PRESENT", "TEST_SCHEMA_NESTING_TOO_DEEP"}


def test_fresh_holdout_is_disjoint_from_candidate1(tmp_path: Path):
    root = _data_root(tmp_path)
    rows, _, _ = cure.load_jsonl(root / "data" / "curebench_valset_pharse1.jsonl", require_answer=True)
    splits = cure.make_fresh_splits(rows)
    prior = {rows[index].row_id for index in splits.prior_holdout}
    fresh = {rows[index].row_id for index in splits.holdout}
    assert prior.isdisjoint(fresh)
    assert len(fresh) == round(len(rows) * cure.HOLDOUT_FRACTION)
    assert len(splits.inner_train) + len(splits.inner_validation) == len(splits.development)


def test_missing_encoder_manifest_is_an_exact_gate(tmp_path: Path):
    root = _data_root(tmp_path)
    with pytest.raises(cure.ExactGateError) as error:
        cure.discover_encoder_manifest(root, None)
    assert error.value.code == "PRETRAINED_ENCODER_MANIFEST_MISSING"


def test_missing_primary_baseline_manifest_is_an_exact_gate(tmp_path: Path):
    root = _data_root(tmp_path)
    with pytest.raises(cure.ExactGateError) as error:
        cure.discover_primary_baseline_manifest(root, None)
    assert error.value.code == "PRIMARY_BASELINE_MANIFEST_MISSING"
    assert error.value.resume_point == "primary_baseline_discovery"
    assert "internal TF-IDF reference is not a public strong baseline" in error.value.required_action


def test_internal_tfidf_reference_cannot_be_renamed_primary(tmp_path: Path):
    root = _data_root(tmp_path)
    manifest = _primary_baseline_manifest(root, baseline_type="internal_reproducible_reference")
    with pytest.raises(cure.ExactGateError) as error:
        cure.load_primary_baseline_manifest(manifest, root)
    assert error.value.code == "PRIMARY_BASELINE_TYPE_INVALID"


def test_public_strong_primary_baseline_requires_source_roles_and_exact_closure(tmp_path: Path):
    root = _data_root(tmp_path)
    missing = _primary_baseline_manifest(root, omit_role="reproduction_receipt")
    with pytest.raises(cure.ExactGateError) as error:
        cure.load_primary_baseline_manifest(missing, root)
    assert error.value.code == "PRIMARY_BASELINE_FILE_ROLES_MISSING"

    root = _data_root(tmp_path / "extra")
    extra = _primary_baseline_manifest(root, extra_file=True)
    with pytest.raises(cure.ExactGateError) as error:
        cure.load_primary_baseline_manifest(extra, root)
    assert error.value.code == "PRIMARY_BASELINE_FILE_CLOSURE_MISMATCH"


def test_public_strong_primary_baseline_manifest_is_source_and_sha_bound(tmp_path: Path):
    root = _data_root(tmp_path)
    manifest = _primary_baseline_manifest(root)
    value = cure.load_primary_baseline_manifest(manifest, root)
    assert value["baseline_type"] == "publicly_reproducible_strong"
    assert value["protocol_comparable"] is True
    assert value["metric"] == cure.PRIMARY_BASELINE_METRIC
    assert value["manifest_sha256"] == cure.sha256_file(manifest)
    assert {entry["role"] for entry in value["verified_files"]} == cure.PRIMARY_BASELINE_STRONG_ROLES


def test_encoder_manifest_rejects_absolute_model_path(tmp_path: Path):
    root = _data_root(tmp_path)
    manifest = root / "managed_runtime" / "frozen-encoder.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "schema": "evomind.cure_bench.frozen_encoder_manifest.v1",
                "backend": "transformers",
                "model_dir": str((root / "outside").resolve()),
                "frozen": True,
                "offline_only": True,
                "files": [{"path": "config.json", "sha256": "0" * 64}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(cure.ExactGateError) as error:
        cure._validate_encoder_manifest(manifest, root)
    assert error.value.code == "PRETRAINED_ENCODER_MODEL_PATH_INVALID"


def test_encoder_manifest_requires_exact_regular_file_closure(tmp_path: Path):
    root = _data_root(tmp_path)
    model_dir = root / "managed_runtime" / "snapshot"
    model_dir.mkdir(parents=True, exist_ok=True)
    config = model_dir / "config.json"
    extra = model_dir / "extra.bin"
    config.write_text("{}", encoding="utf-8")
    extra.write_bytes(b"extra")
    manifest = root / "managed_runtime" / "frozen-encoder.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "evomind.cure_bench.frozen_encoder_manifest.v1",
                "backend": "transformers",
                "model_id": "fixture",
                "revision": "frozen",
                "model_dir": "snapshot",
                "hidden_size": 8,
                "max_length": 128,
                "frozen": True,
                "offline_only": True,
                "files": [{"path": "config.json", "sha256": hashlib.sha256(b"{}").hexdigest()}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(cure.ExactGateError) as error:
        cure._validate_encoder_manifest(manifest, root)
    assert error.value.code == "PRETRAINED_ENCODER_FILE_CLOSURE_MISMATCH"


def _valid_encoder_fixture(tmp_path: Path, *, device: str = "cpu"):
    root = _data_root(tmp_path)
    model_dir = root / "managed_runtime" / "snapshot"
    model_dir.mkdir(parents=True, exist_ok=True)
    config = model_dir / "config.json"
    config.write_text("{}", encoding="utf-8")
    manifest = root / "managed_runtime" / "frozen-encoder.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "evomind.cure_bench.frozen_encoder_manifest.v1",
                "backend": "transformers",
                "model_id": "fixture",
                "revision": "frozen",
                "model_dir": "snapshot",
                "hidden_size": 8,
                "max_length": 128,
                "device": device,
                "frozen": True,
                "offline_only": True,
                "files": [{"path": "config.json", "sha256": hashlib.sha256(b"{}").hexdigest()}],
            }
        ),
        encoding="utf-8",
    )
    return root, manifest


def _fake_transformer_modules(monkeypatch, *, cuda_available: bool):
    class FakeParameter:
        def __init__(self):
            self.device = "cpu"
            self.frozen = False

        def requires_grad_(self, value):
            self.frozen = not value
            return self

    class FakeModel:
        def __init__(self):
            self.parameter = FakeParameter()
            self.to_device = None
            self.eval_called = False

        def parameters(self):
            return iter([self.parameter])

        def to(self, device):
            self.to_device = device
            self.parameter.device = device
            return self

        def eval(self):
            self.eval_called = True
            return self

    model = FakeModel()
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: cuda_available),
        device=lambda name: name,
    )
    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: object()),
        AutoModel=types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: model),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    return model


def test_encoder_device_policy_is_enforced_and_model_is_frozen(tmp_path: Path, monkeypatch):
    root, manifest = _valid_encoder_fixture(tmp_path, device="cpu")
    model = _fake_transformer_modules(monkeypatch, cuda_available=False)
    encoder = cure.FrozenTransformerEncoder(manifest, root)
    encoder._load()
    assert model.to_device == "cpu"
    assert model.eval_called is True
    assert model.parameter.frozen is True


def test_encoder_cuda_policy_returns_exact_gate_when_cuda_missing(tmp_path: Path, monkeypatch):
    root, manifest = _valid_encoder_fixture(tmp_path, device="cuda")
    _fake_transformer_modules(monkeypatch, cuda_available=False)
    encoder = cure.FrozenTransformerEncoder(manifest, root)
    with pytest.raises(cure.ExactGateError) as error:
        encoder._load()
    assert error.value.code == "PRETRAINED_ENCODER_GPU_UNAVAILABLE"


def test_bootstrap_rounds_must_be_at_least_two():
    y_true = np.asarray([0, 1])
    prediction = np.asarray([0, 1])
    for rounds in (0, 1, True):
        with pytest.raises(cure.ExactGateError) as error:
            cure.paired_bootstrap(y_true, prediction, prediction, rounds=rounds)
        assert error.value.code == "BOOTSTRAP_ROUNDS_INVALID"


def test_missing_holdout_ledger_is_an_exact_gate_for_real_encoder(tmp_path: Path):
    root = _data_root(tmp_path)
    with pytest.raises(cure.ExactGateError) as error:
        cure.run_experiment(root, tmp_path / "output", encoder=FakeFrozenEncoder())
    assert error.value.code == "HOLDOUT_LEDGER_MISSING"


def test_real_candidate_stops_before_encoding_without_qualified_primary_baseline(tmp_path: Path):
    root = _data_root(tmp_path)
    ledger = cure.HoldoutLedger(
        path=str(root / "fixture-ledger.json"),
        sha256="b" * 64,
        schema="evomind.cure_bench.holdout_ledger.v1",
        consumed_hashes=frozenset(),
        available_hashes=frozenset(),
    )
    with pytest.raises(cure.ExactGateError) as error:
        cure.run_experiment(root, tmp_path / "output", encoder=FakeFrozenEncoder(), holdout_ledger=ledger)
    assert error.value.code == "PRIMARY_BASELINE_MANIFEST_MISSING"
    assert not (tmp_path / "output").exists()


def test_end_to_end_fixture_writes_hash_bound_manifest_without_test_labels(tmp_path: Path):
    root = _data_root(tmp_path)
    output = tmp_path / "candidate2-output"
    result = cure.run_experiment(root, output, encoder=FakeFrozenEncoder(), allow_test_encoder=True)
    assert result["status"] == "completed"
    assert result["id_overlap_with_prior_candidate"] == 0
    assert result["production_goal_eligible"] is False
    assert (output / "artifact-manifest.json").is_file()
    assert (output / "artifact-manifest-receipt.json").is_file()
    assert json.loads((output / "dataset-audit.json").read_text(encoding="utf-8"))["test_labels_used"] is False
    internal = json.loads((output / "baseline-evidence.json").read_text(encoding="utf-8"))
    primary = json.loads((output / "primary-baseline-evidence.json").read_text(encoding="utf-8"))
    assert internal["classification"] == "INTERNAL_REPRODUCIBLE_REFERENCE"
    assert internal["eligible_as_primary_baseline"] is False
    assert primary["recovery_gate"] == "PRIMARY_BASELINE_MANIFEST_MISSING"
    assert primary["production_goal_eligible"] is False
    predictions = json.loads((output / "holdout-predictions.json").read_text(encoding="utf-8"))
    assert predictions and all("y_true" in row for row in predictions)
    manifest = json.loads((output / "artifact-manifest.json").read_text(encoding="utf-8"))
    listed = {entry["name"] for entry in manifest["files"]}
    actual = {
        path.name
        for path in output.iterdir()
        if path.is_file() and path.name not in {"artifact-manifest.json", "artifact-manifest-receipt.json"}
    }
    assert listed == actual
    for entry in manifest["files"]:
        path = output / entry["name"]
        assert path.stat().st_size == entry["bytes"]
        assert cure.sha256_file(path) == entry["sha256"]


def test_valid_primary_baseline_is_preserved_separately_from_internal_reference(tmp_path: Path):
    root = _data_root(tmp_path)
    primary_manifest = _primary_baseline_manifest(root)
    primary = cure.load_primary_baseline_manifest(primary_manifest, root)
    output = tmp_path / "candidate2-primary-output"
    result = cure.run_experiment(
        root,
        output,
        encoder=FakeFrozenEncoder(),
        allow_test_encoder=True,
        primary_baseline=primary,
    )
    evidence = json.loads((output / "primary-baseline-evidence.json").read_text(encoding="utf-8"))
    internal = json.loads((output / "baseline-evidence.json").read_text(encoding="utf-8"))
    assert evidence["baseline_type"] == "publicly_reproducible_strong"
    assert evidence["manifest_sha256"] == cure.sha256_file(primary_manifest)
    assert result["primary_baseline_manifest_sha256"] == evidence["manifest_sha256"]
    assert internal["classification"] == "INTERNAL_REPRODUCIBLE_REFERENCE"
    assert internal["eligible_as_primary_baseline"] is False


def test_holdout_ledger_rejects_consumed_identity(tmp_path: Path):
    root = _data_root(tmp_path)
    rows, _, _ = cure.load_jsonl(root / "data" / "curebench_valset_pharse1.jsonl", require_answer=True)
    split = cure.make_fresh_splits(rows)
    ledger_path = root / ".evomind" / "cure-bench" / "holdout-ledger.json"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(
        json.dumps(
            {
                "schema": "evomind.cure_bench.holdout_ledger.v1",
                "consumed_holdout_id_sha256": [split.holdout_id_sha256],
                "available_holdout_id_sha256": [],
            }
        ),
        encoding="utf-8",
    )
    ledger = cure.load_holdout_ledger(ledger_path)
    with pytest.raises(cure.ExactGateError) as error:
        cure.run_experiment(root, tmp_path / "output", encoder=FakeFrozenEncoder(), holdout_ledger=ledger)
    assert error.value.code == "HOLDOUT_ALREADY_CONSUMED"


def test_holdout_ledger_available_identity_binds_artifacts(tmp_path: Path):
    root = _data_root(tmp_path)
    rows, _, _ = cure.load_jsonl(root / "data" / "curebench_valset_pharse1.jsonl", require_answer=True)
    split = cure.make_fresh_splits(rows)
    ledger_path = root / ".evomind" / "cure-bench" / "holdout-ledger.json"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(
        json.dumps(
            {
                "schema": "evomind.cure_bench.holdout_ledger.v1",
                "consumed_holdout_id_sha256": [],
                "available_holdout_id_sha256": [split.holdout_id_sha256],
            }
        ),
        encoding="utf-8",
    )
    ledger = cure.load_holdout_ledger(ledger_path)
    output = tmp_path / "bound-output"
    result = cure.run_experiment(
        root,
        output,
        encoder=FakeFrozenEncoder(),
        allow_test_encoder=True,
        holdout_ledger=ledger,
    )
    split_manifest = json.loads((output / "split-manifest.json").read_text(encoding="utf-8"))
    assert result["holdout_ledger_sha256"] == ledger.sha256
    assert split_manifest["holdout_ledger_sha256"] == ledger.sha256


def test_open_ended_first_pass_omits_options():
    row = cure.CanonicalRow(
        row_id="q",
        question_type="open_ended_multi_choice",
        question="Explain the mechanism",
        option_keys=("A", "B"),
        option_values=("first", "second"),
        label=0,
        scored=True,
    )
    text = cure.row_text(row)
    assert "OPTION_A" not in text
    assert "Explain the mechanism" in text
