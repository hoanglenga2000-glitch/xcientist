from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from research_agent_workstation import dashboard as legacy_dashboard
from research_agent_workstation.server.adapters.storage_adapter import LocalStorageAdapter
from research_agent_workstation.server.services.code_agent_context_service import CodeAgentContextService
from scripts import run_full_acceptance, verify_code_agent_patch_lifecycle, verify_runtime_completeness
from scripts.validate_tabular_experiment import resolve_path as resolve_tabular_evidence_path
from scripts.validate_titanic_experiment import resolve_evidence_path as resolve_titanic_evidence_path
from scripts.verify_research_integrity import (
    configured_experiment_root,
    display_evidence_path,
    latest_complete_experiment,
)
from scripts.verify_research_sources import REQUIRED_SOURCE_NAMES, validate_source_metadata
from scripts.verify_security_invariants import repository_files as security_repository_files

ROOT = Path(__file__).resolve().parents[1]
EXEMPT = {
    "scripts/build_reproducible_submission_package.py",
    "scripts/verify_external_resources_manifest.py",
    "scripts/verify_security_invariants.py",
}
PATTERNS = {
    "personal_hpc_mount": re.compile(r"/hpc2(?:hdd|ssd)/", re.IGNORECASE),
    "personal_hpc_account": re.compile(r"\b" + "aims" + r"lab\b", re.IGNORECASE),
    "personal_workspace_name": re.compile(r"(?:~[/\\])?" + "jing" + "hw", re.IGNORECASE),
    "windows_user_home": re.compile(r"C:\\Users\\[^\\/\r\n]+", re.IGNORECASE),
    "release_validation_path": re.compile("EvoMind-" + "release-validation", re.IGNORECASE),
}
TEXT_SUFFIXES = {
    ".bat",
    ".cmd",
    ".js",
    ".json",
    ".jsx",
    ".md",
    ".mjs",
    ".ps1",
    ".py",
    ".sh",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}


def test_quarantine_is_recursively_export_ignored() -> None:
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8-sig")
    rules = {line.split("#", 1)[0].strip() for line in attributes.splitlines() if line.split("#", 1)[0].strip()}

    assert {
        "/scripts/_quarantine export-ignore",
        "/scripts/_quarantine/** export-ignore",
    } <= rules


def test_git_inventory_contains_no_machine_specific_release_text() -> None:
    paths, discovery_findings = security_repository_files()
    assert discovery_findings == []

    findings: list[str] = []
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        if (
            relative in EXEMPT
            or relative.startswith(("tests/", "scripts/_quarantine/"))
            or not path.is_file()
            or path.suffix.casefold() not in TEXT_SUFFIXES
        ):
            continue
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        for rule, pattern in PATTERNS.items():
            if pattern.search(text):
                findings.append(f"{relative}:{rule}")

    assert findings == []


def test_security_inventory_uses_filesystem_fallback_without_git(tmp_path: Path) -> None:
    source = tmp_path / "src" / "safe.py"
    source.parent.mkdir()
    source.write_text("value = 1\n", encoding="utf-8")
    vendor = tmp_path / "node_modules" / "ignored.js"
    vendor.parent.mkdir()
    vendor.write_text("ignored = true;\n", encoding="utf-8")

    paths, discovery_findings = security_repository_files(tmp_path)

    assert discovery_findings == []
    assert {path.relative_to(tmp_path).as_posix() for path in paths} == {"src/safe.py"}


def test_verified_launcher_guidance_configures_llm_first() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8-sig")
    quick_start = readme.partition("## Quick Start")[2].partition("## CLI Commands")[0]
    installer = (ROOT / "install.ps1").read_text(encoding="utf-8-sig")
    next_steps = installer.partition('Write-Host "Next steps:"')[2]

    assert quick_start.index("evomind setup") < quick_start.index("start_verified_workstation.ps1")
    assert next_steps.index("evomind setup") < next_steps.index("start_verified_workstation.ps1")
    assert "evomind dashboard start" in quick_start
    assert "evomind dashboard start" in next_steps


def test_research_source_verifier_uses_only_release_tracked_metadata() -> None:
    verifier = (ROOT / "scripts" / "verify_research_sources.py").read_text(encoding="utf-8-sig")
    config = yaml.safe_load((ROOT / "configs" / "research_sources.yaml").read_text(encoding="utf-8-sig"))

    assert "FINAL_AUDIT_PATH" not in verifier
    validate_source_metadata(config["sources"])
    names = {source["id"]: source["name"] for source in config["sources"]}
    assert names.items() >= REQUIRED_SOURCE_NAMES.items()


def test_research_integrity_uses_latest_complete_release_workspace_evidence(tmp_path: Path) -> None:
    task_root = tmp_path / "experiments" / "titanic"
    complete = task_root / "20260701_complete"
    incomplete = task_root / "20260702_incomplete"
    complete.mkdir(parents=True)
    incomplete.mkdir()
    for name in ("validation_gate.json", "experiment_log.json"):
        (complete / name).write_text("{}\n", encoding="utf-8")
    (incomplete / "validation_gate.json").write_text("{}\n", encoding="utf-8")

    selected = latest_complete_experiment(task_root, ("validation_gate.json", "experiment_log.json"))
    integrity_source = (ROOT / "scripts" / "verify_research_integrity.py").read_text(encoding="utf-8-sig")

    assert selected == complete
    assert "科研Agent工作站最终完成审计.md" not in integrity_source
    assert "20260606_192118" not in integrity_source


def test_research_integrity_supports_external_runtime_evidence_without_path_disclosure(
    monkeypatch, tmp_path: Path
) -> None:
    evidence_root = tmp_path / "private-profile" / "experiments"
    artifact = evidence_root / "titanic" / "20260716" / "validation_gate.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"status":"passed"}\n', encoding="utf-8")
    monkeypatch.setenv("RESEARCH_EXPERIMENT_ROOT", str(evidence_root))

    configured = configured_experiment_root()

    assert configured == evidence_root.resolve()
    assert display_evidence_path(artifact, configured) == "runtime/experiments/titanic/20260716/validation_gate.json"
    assert str(tmp_path) not in display_evidence_path(artifact, configured)


def test_research_integrity_fails_closed_without_runtime_evidence(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="experiment root unavailable for task 'titanic'"):
        latest_complete_experiment(tmp_path / "missing" / "titanic", ("validation_gate.json",))


def test_full_acceptance_uses_explicit_external_experiment_root(monkeypatch, tmp_path: Path) -> None:
    evidence_root = tmp_path / "sealed-evidence"
    run_dir = evidence_root / "titanic" / "20260716"
    run_dir.mkdir(parents=True)
    for name in ("validation_gate.json", "experiment_log.json"):
        (run_dir / name).write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("RESEARCH_EXPERIMENT_ROOT", str(evidence_root))

    selected = run_full_acceptance.latest_experiment(
        "titanic",
        ["validation_gate.json", "experiment_log.json"],
    )

    assert Path(selected) == run_dir.resolve()


def test_validators_resolve_task_data_from_external_evidence_root(monkeypatch, tmp_path: Path) -> None:
    evidence_root = tmp_path / "sealed-evidence"
    sample = evidence_root / "tasks" / "titanic" / "data" / "sample_submission.csv"
    sample.parent.mkdir(parents=True)
    sample.write_text("PassengerId,Survived\n1,0\n", encoding="utf-8")
    monkeypatch.setenv("RESEARCH_EVIDENCE_ROOT", str(evidence_root))

    configured_path = "tasks/titanic/data/sample_submission.csv"

    assert resolve_titanic_evidence_path(configured_path) == sample
    assert resolve_tabular_evidence_path(configured_path) == sample


def test_full_acceptance_derives_experiments_from_evidence_root(monkeypatch, tmp_path: Path) -> None:
    evidence_root = tmp_path / "sealed-evidence"
    monkeypatch.setenv("RESEARCH_EVIDENCE_ROOT", str(evidence_root))
    monkeypatch.delenv("RESEARCH_EXPERIMENT_ROOT", raising=False)

    assert run_full_acceptance.configured_experiment_root() == (evidence_root / "experiments").resolve()


def test_full_acceptance_redacts_external_evidence_paths(monkeypatch, tmp_path: Path) -> None:
    evidence_root = tmp_path / "private-profile"
    monkeypatch.setenv("RESEARCH_EVIDENCE_ROOT", str(evidence_root))
    value = f"python validator.py --input {evidence_root / 'experiments' / 'titanic'}"

    redacted = run_full_acceptance.redact_runtime_paths(value)

    assert str(tmp_path) not in redacted
    assert "<runtime-evidence>" in redacted


def test_patch_lifecycle_selects_external_comparable_runs(monkeypatch, tmp_path: Path) -> None:
    evidence_root = tmp_path / "private-profile"
    task_root = evidence_root / "experiments" / "house_prices"
    before_run = task_root / "20260715"
    after_run = task_root / "20260716"
    incomplete_run = task_root / "20260717_incomplete"
    for run_dir in (before_run, after_run):
        run_dir.mkdir(parents=True)
        (run_dir / "model_results.json").write_text("{}\n", encoding="utf-8")
    incomplete_run.mkdir()
    monkeypatch.setenv("RESEARCH_EVIDENCE_ROOT", str(evidence_root))
    monkeypatch.delenv("RESEARCH_EXPERIMENT_ROOT", raising=False)

    selected_before, selected_after = verify_code_agent_patch_lifecycle.latest_comparable_runs("house_prices")

    assert selected_before == before_run
    assert selected_after == after_run


def test_code_agent_comparison_redacts_external_run_paths(tmp_path: Path) -> None:
    release_root = tmp_path / "release"
    evidence_root = tmp_path / "private-profile" / "experiments" / "house_prices"
    before_run = evidence_root / "20260715"
    after_run = evidence_root / "20260716"
    for run_dir, score in ((before_run, 0.11), (after_run, 0.09)):
        run_dir.mkdir(parents=True)
        (run_dir / "model_results.json").write_text(
            '{"best_model":"ridge","model_results":{"ridge":{"rmse":' + str(score) + "}}}\n",
            encoding="utf-8",
        )
    service = CodeAgentContextService(LocalStorageAdapter(release_root), release_root)

    comparison = service.compare_runs_before_after_patch("house_prices", before_run, after_run)

    assert comparison["before_run"] == "runtime/experiments/house_prices/20260715"
    assert comparison["after_run"] == "runtime/experiments/house_prices/20260716"
    assert comparison["before_metrics"] == {"rmse": 0.11}
    assert comparison["after_metrics"] == {"rmse": 0.09}
    comparison_file = next((release_root / "workspace" / "tasks" / "house_prices" / "code" / "comparisons").glob("*.json"))
    persisted = comparison_file.read_text(encoding="utf-8")
    assert str(tmp_path / "private-profile") not in persisted
    assert "runtime/experiments/house_prices/20260716" in persisted


def test_legacy_dashboard_uses_external_experiments_without_disclosing_root(monkeypatch, tmp_path: Path) -> None:
    evidence_root = tmp_path / "private-profile"
    run_dir = evidence_root / "experiments" / "titanic" / "20260716"
    run_dir.mkdir(parents=True)
    (run_dir / "validation_gate.json").write_text('{"status":"passed"}', encoding="utf-8")
    monkeypatch.setenv("RESEARCH_EVIDENCE_ROOT", str(evidence_root))
    monkeypatch.delenv("RESEARCH_EXPERIMENT_ROOT", raising=False)

    selected = legacy_dashboard.latest_experiment("titanic")

    assert selected == run_dir
    assert legacy_dashboard.rel(selected) == "runtime/experiments/titanic/20260716"
    assert str(tmp_path) not in str(legacy_dashboard.file_info(run_dir / "validation_gate.json"))


def test_runtime_completeness_uses_external_experiments_without_disclosing_root(monkeypatch, tmp_path: Path) -> None:
    evidence_root = tmp_path / "private-profile"
    run_dir = evidence_root / "experiments" / "titanic" / "20260716"
    run_dir.mkdir(parents=True)
    for name in verify_runtime_completeness.REQUIRED_RUNTIME_FILES:
        (run_dir / name).write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("RESEARCH_EVIDENCE_ROOT", str(evidence_root))
    monkeypatch.delenv("RESEARCH_EXPERIMENT_ROOT", raising=False)

    selected = verify_runtime_completeness.latest_run("titanic")

    assert selected == run_dir
    assert verify_runtime_completeness.display_run_path(selected) == "runtime/experiments/titanic/20260716"
    assert str(tmp_path) not in verify_runtime_completeness.display_run_path(selected)


def test_legacy_dashboard_reads_isolated_integrity_gate_without_leaking_path(monkeypatch, tmp_path: Path) -> None:
    gate_path = tmp_path / "private-profile" / "research_integrity_gate.json"
    gate_path.parent.mkdir()
    gate_path.write_text('{"status":"passed","dimensions":[{"dimension":"validity"}]}', encoding="utf-8")
    monkeypatch.setenv("RESEARCH_INTEGRITY_GATE_PATH", str(gate_path))

    summary = legacy_dashboard.summarize_integrity_gate()

    assert summary["status"] == "passed"
    assert summary["path"] == "runtime/research_integrity_gate.json"
    assert str(tmp_path) not in str(summary)


def test_legacy_dashboard_resolves_parent_segments_before_disclosing_gate_path(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "release"
    gate_path = tmp_path / "private-profile" / "research_integrity_gate.json"
    repo_root.mkdir()
    gate_path.parent.mkdir()
    gate_path.write_text('{"status":"passed"}', encoding="utf-8")
    monkeypatch.setattr(legacy_dashboard, "ROOT", repo_root)
    monkeypatch.setenv("RESEARCH_INTEGRITY_GATE_PATH", "../private-profile/research_integrity_gate.json")

    summary = legacy_dashboard.summarize_integrity_gate()

    assert summary["status"] == "passed"
    assert summary["path"] == "runtime/research_integrity_gate.json"
    assert ".." not in summary["path"]
