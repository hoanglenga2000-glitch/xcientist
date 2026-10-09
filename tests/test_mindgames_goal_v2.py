from __future__ import annotations

import importlib.util
import json
import hashlib
import sys
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "mindgames_goal_v2.py"
SPEC = importlib.util.spec_from_file_location("mindgames_goal_v2_fixture", MODULE_PATH)
assert SPEC and SPEC.loader
mindgames = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mindgames
SPEC.loader.exec_module(mindgames)


def _fixture_runtime(tmp_path: Path) -> tuple[Path, Path, Path]:
    data_root = tmp_path / "competition-data" / "mindgames"
    data_root.mkdir(parents=True)
    runtime_root = data_root.parent / ".runtime" / "mindgames"
    site_packages = runtime_root / "lib" / "python3.12" / "site-packages"
    site_packages.mkdir(parents=True)
    (site_packages / "textarena").mkdir()
    (site_packages / "textarena" / "__init__.py").write_text("__version__ = '0.7.4'\n", encoding="utf-8")
    (site_packages / "trueskill").mkdir()
    (site_packages / "trueskill" / "__init__.py").write_text("class Rating: pass\n", encoding="utf-8")
    return data_root, runtime_root, site_packages


def _formal_protocol(
    data_root: Path,
    runtime_root: Path,
    site_packages: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    evaluator = data_root / "starter-kit" / "src" / "offline_evaluation.py"
    evaluator.parent.mkdir(parents=True)
    evaluator.write_text("# frozen evaluator\n", encoding="utf-8")
    evaluator_sha = mindgames.sha256_file(evaluator)
    monkeypatch.setattr(mindgames, "OFFLINE_EVALUATOR_SHA256", evaluator_sha)
    roots = [
        site_packages / "textarena",
        site_packages / "trueskill",
        runtime_root / "models" / "Qwen3-8B",
        runtime_root / "models" / "STARS",
        runtime_root / "models" / "tungsten",
    ]
    for root in roots[2:]:
        root.mkdir(parents=True)
        (root / "config.json").write_text(json.dumps({"model": root.name}) + "\n", encoding="utf-8")
    files = []
    for root in roots:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                files.append(
                    {
                        "path": path.relative_to(runtime_root).as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": mindgames.sha256_file(path),
                    }
                )
    manifest = {
        "schema": mindgames.FORMAL_PROTOCOL_SCHEMA,
        "frozen": True,
        "offline_only": True,
        "trust_remote_code": False,
        "official_repository_commit": mindgames.OFFICIAL_REPOSITORY_COMMIT,
        "official_evaluator_sha256": evaluator_sha,
        "textarena_version": mindgames.EXPECTED_TEXTARENA_VERSION,
        "trueskill_version": "0.4.5",
        "environments": [value[0] for value in mindgames.ENVIRONMENTS],
        "episodes_per_environment": mindgames.EPISODES_PER_ENVIRONMENT,
        "seed": mindgames.SEED,
        "seat_balanced": True,
        "candidate_model_id": "Qwen/Qwen3-8B",
        "candidate_model_dir": "models/Qwen3-8B",
        "reference_agents": [
            {"name": "STARS", "model_dir": "models/STARS", "agent_type": "HFLocalAgent"},
            {"name": "tungsten", "model_dir": "models/tungsten", "agent_type": "HFLocalAgent"},
        ],
        "closure_roots": [root.relative_to(runtime_root).as_posix() for root in roots],
        "files": files,
    }
    manifest_path = runtime_root / mindgames.FORMAL_PROTOCOL_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return evaluator, manifest_path


@pytest.fixture(autouse=True)
def _clear_dependency_modules(monkeypatch: pytest.MonkeyPatch):
    for name in ("textarena", "trueskill"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    for name in mindgames.MANAGED_RUNTIME_ENVS:
        monkeypatch.delenv(name, raising=False)


def test_existing_adjacent_runtime_is_discovered_and_imported_from_bound_site_packages(tmp_path: Path):
    data_root, runtime_root, site_packages = _fixture_runtime(tmp_path)
    before = list(sys.path)
    before_runtime_files = sorted(path.relative_to(runtime_root).as_posix() for path in runtime_root.rglob("*"))

    runtime = mindgames.discover_managed_runtime(data_root)
    assert runtime.root == runtime_root.resolve()
    assert runtime.site_packages == (site_packages.resolve(),)
    textarena, trueskill, loaded = mindgames.load_managed_dependencies(data_root)

    assert loaded == runtime
    assert textarena.__version__ == "0.7.4"
    assert Path(textarena.__file__).resolve().is_relative_to(site_packages.resolve())
    assert Path(trueskill.__file__).resolve().is_relative_to(site_packages.resolve())
    assert sys.path == before
    after_runtime_files = sorted(path.relative_to(runtime_root).as_posix() for path in runtime_root.rglob("*"))
    assert after_runtime_files == before_runtime_files


def test_missing_existing_runtime_is_an_exact_dependency_gate(tmp_path: Path):
    data_root = tmp_path / "competition-data" / "mindgames"
    data_root.mkdir(parents=True)

    with pytest.raises(mindgames.ManagedRuntimeGate) as error:
        mindgames.load_managed_dependencies(data_root)

    assert error.value.code == "RUNTIME_DEPENDENCY_MISSING"
    assert error.value.dependency == "textarena"
    assert "online" not in error.value.reason


def test_runtime_override_cannot_escape_authorized_parent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_root = tmp_path / "competition-data" / "mindgames"
    data_root.mkdir(parents=True)
    outside = tmp_path / "outside-runtime"
    outside.mkdir()
    monkeypatch.setenv(mindgames.MANAGED_RUNTIME_ENV, str(outside))

    with pytest.raises(mindgames.ManagedRuntimeGate) as error:
        mindgames.discover_managed_runtime(data_root)

    assert error.value.code == "MANAGED_RUNTIME_PATH_ESCAPE"
    assert str(outside) not in error.value.reason


def test_runtime_symlink_escape_is_rejected_when_platform_allows_symlinks(tmp_path: Path):
    data_root = tmp_path / "competition-data" / "mindgames"
    data_root.mkdir(parents=True)
    runtime_parent = data_root.parent / ".runtime"
    runtime_parent.mkdir()
    outside = tmp_path / "outside-runtime"
    outside.mkdir()
    runtime_link = runtime_parent / "mindgames"
    try:
        runtime_link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable on this test host")

    with pytest.raises(mindgames.ManagedRuntimeGate) as error:
        mindgames.discover_managed_runtime(data_root)

    assert error.value.code == "MANAGED_RUNTIME_PATH_ESCAPE"


def test_model_discovery_includes_existing_managed_runtime_model(tmp_path: Path):
    data_root, runtime_root, _site_packages = _fixture_runtime(tmp_path)
    model = runtime_root / "models" / "Qwen3-8B"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}\n", encoding="utf-8")

    candidates = mindgames.model_candidates(data_root)

    assert model.resolve() in candidates


def test_frozen_formal_protocol_closes_dependencies_models_references_and_evaluator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    data_root, runtime_root, site_packages = _fixture_runtime(tmp_path)
    evaluator, manifest_path = _formal_protocol(data_root, runtime_root, site_packages, monkeypatch)
    runtime = mindgames.discover_managed_runtime(data_root)

    protocol = mindgames.load_frozen_formal_protocol(data_root, runtime, evaluator)

    assert protocol.manifest_path == manifest_path
    assert protocol.manifest_sha256 == mindgames.sha256_file(manifest_path)
    assert protocol.candidate_model == (runtime_root / "models/Qwen3-8B").resolve()
    assert [name for name, _path in protocol.reference_models] == ["STARS", "tungsten"]
    assert protocol.closure_file_count == 5


def test_missing_formal_protocol_is_exact_gate_before_any_candidate_execution(tmp_path: Path):
    data_root, _runtime_root, _site_packages = _fixture_runtime(tmp_path)
    runtime = mindgames.discover_managed_runtime(data_root)
    evaluator = data_root / "offline_evaluation.py"
    evaluator.write_text("fixture\n", encoding="utf-8")

    with pytest.raises(mindgames.ManagedRuntimeGate) as error:
        mindgames.load_frozen_formal_protocol(data_root, runtime, evaluator)

    assert error.value.code == "FORMAL_PROTOCOL_MANIFEST_MISSING"


def test_formal_protocol_rejects_extra_file_and_hash_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    data_root, runtime_root, site_packages = _fixture_runtime(tmp_path)
    evaluator, manifest_path = _formal_protocol(data_root, runtime_root, site_packages, monkeypatch)
    runtime = mindgames.discover_managed_runtime(data_root)
    (runtime_root / "models/Qwen3-8B/extra.bin").write_bytes(b"extra")
    with pytest.raises(mindgames.ManagedRuntimeGate) as extra:
        mindgames.load_frozen_formal_protocol(data_root, runtime, evaluator)
    assert extra.value.code == "FORMAL_PROTOCOL_FILE_CLOSURE_MISMATCH"
    (runtime_root / "models/Qwen3-8B/extra.bin").unlink()
    (runtime_root / "models/Qwen3-8B/config.json").write_text("drift\n", encoding="utf-8")
    with pytest.raises(mindgames.ManagedRuntimeGate) as drift:
        mindgames.load_frozen_formal_protocol(data_root, runtime, evaluator)
    assert drift.value.code == "FORMAL_PROTOCOL_FILE_HASH_MISMATCH"


def test_formal_path_has_no_install_download_or_ollama_fallback() -> None:
    text = MODULE_PATH.read_text(encoding="utf-8")
    assert "pip install" not in text
    assert "ollama" not in text
    assert "subprocess.run" not in text


def test_main_emits_exact_gate_when_formal_manifest_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root, _runtime_root, _site_packages = _fixture_runtime(tmp_path)
    evaluator = data_root / "starter-kit/src/offline_evaluation.py"
    evaluator.parent.mkdir(parents=True)
    evaluator.write_text("# evaluator fixture\n", encoding="utf-8")
    monkeypatch.setattr(mindgames, "OFFLINE_EVALUATOR_SHA256", mindgames.sha256_file(evaluator))
    output = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        ["mindgames_goal_v2.py", "--data-dir", str(data_root), "--out-dir", str(output)],
    )

    assert mindgames.main() == 0

    gate = json.loads((output / "exact-gate.json").read_text(encoding="utf-8"))
    assert gate["code"] == "FORMAL_PROTOCOL_MANIFEST_MISSING"
    assert gate["training_performed"] is False
    assert gate["automatic_install_performed"] is False
    assert gate["automatic_download_performed"] is False
