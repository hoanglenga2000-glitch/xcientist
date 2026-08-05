from __future__ import annotations

import json
import stat
import zipfile
from pathlib import Path

import pytest

from scripts.verify_release_artifacts import (
    VerificationError,
    npm_ls_inventory,
    verify_cyclonedx_sbom,
    verify_npm_audit,
    verify_npm_audits,
    verify_pip_audit,
    verify_workstation_source_bundle,
)

ROOT = Path(__file__).resolve().parents[1]
VALID_SPLIT75_TEXT = (
    ROOT / "benchmark" / "mle_bench_75" / "openai_split75_507f92e.txt"
).read_text(encoding="utf-8")
VALID_SPLIT75 = VALID_SPLIT75_TEXT.splitlines()
VALID_UPSTREAM = json.loads(
    (ROOT / "benchmark" / "mle_bench_75" / "UPSTREAM.json").read_text(encoding="utf-8")
)
VALID_REGISTRY = json.loads(
    (ROOT / "benchmark" / "mle_bench_75" / "tasks_template.json").read_text(
        encoding="utf-8"
    )
)
VALID_REGISTRY_TASK = VALID_REGISTRY["tasks"][0]
VALID_RUNNER_SOURCE = (ROOT / "scripts" / "mlebench_server_runner.py").read_text(
    encoding="utf-8"
)
VALID_RULES_SOURCE = (ROOT / "scripts" / "accept_kaggle_rules_all75.py").read_text(
    encoding="utf-8"
)
CRITICAL_SOURCE_FILES = (
    ".github/workflows/ci.yml",
    "configs/schemas/benchmark_task.schema.json",
    "scripts/extract_capability_evidence_bundle.py",
    "scripts/mle_bench_split75_contract.py",
    "scripts/run_ci_checks.py",
    "scripts/verify_workstation_click_smoke.mjs",
    "scripts/verify_workstation_interactive_controls.mjs",
    "scripts/verify_capability_certification.py",
    "src/research_os/benchmark_manager.py",
    "src/xsci/capability_certification.py",
    "src/xsci/kaggle.py",
    "src/xsci/kaggle_conversation.py",
    "src/xsci/kaggle_intent.py",
    "src/xsci/scientist_adaptive_loop.py",
    "src/xsci/scientist_hypothesis_panel.py",
    "src/xsci/scientist_release_evidence.py",
    "src/xsci/scientist_upgrade_controller.py",
    "src/xsci/scientist_upgrade_gateway.py",
    "src/xsci/terminal_agent.py",
    "src/xsci/terminal_events.py",
    "src/xsci/terminal_tools.py",
    "web/research-agent-workstation/src/app/api/scientist/upgrade-campaign/route.ts",
    "web/research-agent-workstation/scripts/verify-upgrade-campaign-contract.mjs",
    "web/research-agent-workstation/src/components/workstation/AiControlConsole.tsx",
    "web/research-agent-workstation/src/lib/api/client.ts",
    "web/research-agent-workstation/src/lib/api/types.ts",
)

REQUIRED_FILES = {
    ".env.example": "# environment\n",
    "benchmark/mle_bench_75/UPSTREAM.json": json.dumps(VALID_UPSTREAM),
    "benchmark/mle_bench_75/openai_split75_507f92e.txt": VALID_SPLIT75_TEXT,
    "benchmark/mle_bench_75/tasks_template.json": json.dumps(VALID_REGISTRY),
    "configs/schemas/benchmark_task.schema.json": (
        ROOT / "configs" / "schemas" / "benchmark_task.schema.json"
    ).read_text(encoding="utf-8"),
    "Dockerfile": "FROM scratch\n",
    "LICENSE": "MIT license text\n",
    "README.md": "# EvoMind\n",
    "SECURITY.md": "# Security\n",
    "docker-compose.yml": "127.0.0.1:3090:3090\n127.0.0.1:8088:3090\n",
    "docs/NEW_USER_ONBOARDING_GUIDE.md": "# Onboarding\n",
    "docs/CAPABILITY_CERTIFICATION.md": "# Capability certification\n",
    "docs/RELEASE_CHECKLIST.md": "# Release\n",
    "install.ps1": "# installer\n",
    "pyproject.toml": "[project]\nname='xcientist'\n",
    "requirements.txt": "\n",
    "scripts/dpapi_credential_store.ps1": (
        "Export-Clixml\nImport-EvoMindHpcCredential\nResolve-EvoMindHpcGenerationPayload\n"
        "icacls.exe\nSystem.Threading.Mutex\nCommit-EvoMindCredentialFiles\n"
    ),
    "scripts/install_autokaggle_cli.ps1": "# cli\n",
    "scripts/mle_bench_split75_contract.py": (
        ROOT / "scripts" / "mle_bench_split75_contract.py"
    ).read_text(encoding="utf-8"),
    "scripts/mlebench_server_runner.py": VALID_RUNNER_SOURCE,
    "scripts/manage_deepseek_secret.ps1": (
        "dpapi_credential_store.ps1\nImport-Clixml\nEnter-EvoMindCredentialStoreLock\nSecretFromStdin\n"
    ),
    "scripts/manage_hpc_proxy_bridge.ps1": "# proxy manager\n",
    "scripts/manage_hpc_ssh_secret.ps1": (
        "dpapi_credential_store.ps1\nEnter-EvoMindCredentialStoreLock\nSecretFromStdin\n"
        "Get-EvoMindHpcCredentialStorePaths\nResolve-EvoMindHpcCredentialGeneration\n"
        "New-EvoMindHpcCredentialGeneration\nRemove-EvoMindHpcCredentialStore\n"
    ),
    "scripts/manage_kaggle_secret.ps1": (
        "dpapi_credential_store.ps1\nImport-Clixml\nEnter-EvoMindCredentialStoreLock\nSecretFromStdin\n"
    ),
    "scripts/manage_workstation_dashboard.py": "# manager\n",
    "scripts/hpc_socks_bridge.py": "# proxy bridge\n",
    "scripts/quick_setup.ps1": "# setup\n",
    "scripts/accept_kaggle_rules_all75.py": VALID_RULES_SOURCE,
    "scripts/restart_workstation_frontend.ps1": "# restart\n",
    "scripts/run_ci_checks.py": (ROOT / "scripts" / "run_ci_checks.py").read_text(
        encoding="utf-8"
    ),
    "scripts/run_new_user_release_acceptance.ps1": "# acceptance\n",
    "scripts/verify_workstation_click_smoke.mjs": (
        ROOT / "scripts" / "verify_workstation_click_smoke.mjs"
    ).read_text(encoding="utf-8"),
    "scripts/verify_workstation_interactive_controls.mjs": (
        ROOT / "scripts" / "verify_workstation_interactive_controls.mjs"
    ).read_text(encoding="utf-8"),
    "scripts/start_hpc_socks_bridge.py": "# safe proxy launcher\n",
    "scripts/start_verified_workstation.ps1": "# start\n",
    "scripts/verify_backend_resource_status.py": "# backend status\n",
    "scripts/verify_new_user_release_readiness.py": "# readiness\n",
    "scripts/verify_no_plaintext_secrets.py": "# scanner\n",
    "scripts/verify_security_invariants.py": "# security\n",
    "scripts/verify_verified_workstation_launch_audit.py": "# audit\n",
    "scripts/verify_workstation_launch_readiness.py": "# launch\n",
    "scripts/verify_workstation_ui_truthfulness.py": "# ui truth\n",
    "src/research_os/benchmark_manager.py": (
        ROOT / "src" / "research_os" / "benchmark_manager.py"
    ).read_text(encoding="utf-8"),
    "src/xsci/dashboard.py": "# bridge\n",
    "web/research-agent-workstation/package-lock.json": "{}\n",
    "web/research-agent-workstation/package.json": "{}\n",
    "web/research-agent-workstation/scripts/verify-kaggle-status-contract.mjs": "// contract\n",
    "web/research-agent-workstation/scripts/verify-upgrade-campaign-contract.mjs": "// contract\n",
    "web/research-agent-workstation/src/lib/connector-status.ts": "export {};\n",
    "web/research-agent-workstation/src/lib/server/kaggle-status.ts": "export {};\n",
    "web/research-agent-workstation/src/middleware.ts": "export {};\n",
}
REQUIRED_FILES.update({
    relative: (ROOT / relative).read_text(encoding="utf-8")
    for relative in CRITICAL_SOURCE_FILES
})


def _bundle(dist_dir, extra_files=None, omit_files=None):
    archive_path = dist_dir / "xcientist-0.2.2-workstation-source.zip"
    files = {**REQUIRED_FILES, **(extra_files or {})}
    for relative in omit_files or ():
        files.pop(relative)
    with zipfile.ZipFile(archive_path, mode="w") as archive:
        for relative, content in files.items():
            archive.writestr(f"xcientist-0.2.2/{relative}", content)
    return archive_path


def test_workstation_source_bundle_verifier_accepts_complete_safe_archive(tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    source_license = tmp_path / "LICENSE"
    source_license.write_text(REQUIRED_FILES["LICENSE"], encoding="utf-8")
    _bundle(dist)

    result = verify_workstation_source_bundle(dist, source_license)

    assert result["root"] == "xcientist-0.2.2"
    assert result["files"] == len(REQUIRED_FILES)
    assert result["mle_bench_registered_tasks"] == len(VALID_REGISTRY["tasks"])
    assert result["mle_bench_official_tasks"] == 75
    assert result["mle_bench_local_official_overlap"] == VALID_REGISTRY["mle_bench_reference"][
        "locally_registered_official_competitions"
    ]
    assert set(result["critical_source_sha256"]) == set(CRITICAL_SOURCE_FILES)


@pytest.mark.parametrize(
    "critical_path",
    CRITICAL_SOURCE_FILES,
)
def test_workstation_source_bundle_requires_critical_runtime_sources(
    tmp_path,
    critical_path,
):
    dist = tmp_path / "dist"
    dist.mkdir()
    source_license = tmp_path / "LICENSE"
    source_license.write_text(REQUIRED_FILES["LICENSE"], encoding="utf-8")
    _bundle(dist, omit_files={critical_path})

    with pytest.raises(VerificationError, match="missing workstation files"):
        verify_workstation_source_bundle(dist, source_license)


@pytest.mark.parametrize(
    "critical_path",
    CRITICAL_SOURCE_FILES,
)
def test_workstation_source_bundle_rejects_placeholder_or_drifted_critical_source(
    tmp_path,
    critical_path,
):
    dist = tmp_path / "dist"
    dist.mkdir()
    source_license = tmp_path / "LICENSE"
    source_license.write_text(REQUIRED_FILES["LICENSE"], encoding="utf-8")
    _bundle(dist, {critical_path: "# placeholder that cannot enforce release gates\n"})

    with pytest.raises(VerificationError, match="critical source differs"):
        verify_workstation_source_bundle(dist, source_license)


@pytest.mark.parametrize(
    "failure",
    ["total", "planned", "duplicate_task", "missing_field", "range_note"],
)
def test_workstation_source_bundle_verifier_rejects_invalid_mle_registry(tmp_path, failure):
    dist = tmp_path / "dist"
    dist.mkdir()
    source_license = tmp_path / "LICENSE"
    source_license.write_text(REQUIRED_FILES["LICENSE"], encoding="utf-8")
    registry = json.loads(json.dumps(VALID_REGISTRY))
    registered_count = len(registry["tasks"])

    if failure == "total":
        registry["total_tasks"] = registered_count + 1
    elif failure == "planned":
        registry["total_tasks"] = registered_count + 1
        registry["remaining_tasks_planned"] = 1
        registry["planned_tasks_summary"] = [
            {"modality": "tabular", "count": 2, "status": "planned"}
        ]
    elif failure == "duplicate_task":
        registry["tasks"].append(dict(VALID_REGISTRY_TASK))
        registry["total_tasks"] = registered_count + 1
    elif failure == "missing_field":
        del registry["tasks"][0]["metric"]
    elif failure == "range_note":
        registry["total_tasks"] = registered_count + 1
        registry["remaining_tasks_planned"] = 1
        registry["planned_tasks_summary"] = [
            {"modality": "tabular", "count": 1, "status": "planned"}
        ]
        registry["normalization_note"] = "Tasks 9-9 are planned but not registered."

    _bundle(
        dist,
        {"benchmark/mle_bench_75/tasks_template.json": json.dumps(registry)},
    )
    with pytest.raises(VerificationError, match="tasks_template.json is invalid"):
        verify_workstation_source_bundle(dist, source_license)


@pytest.mark.parametrize(
    "failure",
    [
        "manifest",
        "runner",
        "runner_mutation",
        "runner_alias_augassign",
        "runner_helper_mutation",
        "wrong_metadata",
        "slug_mapping",
        "slug_value",
        "slug_alias_augassign",
        "rules_order",
        "rules_mutation",
        "rules_alias_augassign",
        "overlap",
        "upstream",
    ],
)
def test_workstation_source_bundle_verifier_rejects_split75_drift(tmp_path, failure):
    dist = tmp_path / "dist"
    dist.mkdir()
    source_license = tmp_path / "LICENSE"
    source_license.write_text(REQUIRED_FILES["LICENSE"], encoding="utf-8")
    files = {}
    if failure == "manifest":
        files["benchmark/mle_bench_75/openai_split75_507f92e.txt"] = (
            VALID_SPLIT75_TEXT.replace(VALID_SPLIT75[0], "drifted-task", 1)
        )
    elif failure == "runner":
        files["scripts/mlebench_server_runner.py"] = "COMPETITIONS = " + repr(
            [(task_id, "fixture", "fixture") for task_id in [*VALID_SPLIT75, "extra-task"]]
        ) + "\nKAGGLE_SLUGS = " + repr({task_id: task_id for task_id in VALID_SPLIT75})
    elif failure == "runner_mutation":
        files["scripts/mlebench_server_runner.py"] = (
            VALID_RUNNER_SOURCE
            + '\nCOMPETITIONS.append(("evil-task", "lite", "tabular"))\n'
        )
    elif failure == "runner_alias_augassign":
        files["scripts/mlebench_server_runner.py"] = (
            VALID_RUNNER_SOURCE
            + '\nalias = COMPETITIONS\nalias += [("evil-task", "lite", "tabular")]\n'
        )
    elif failure == "runner_helper_mutation":
        files["scripts/mlebench_server_runner.py"] = (
            VALID_RUNNER_SOURCE
            + '\ndef mutate(value):\n    value.append(("evil-task", "lite", "tabular"))\n'
            + "mutate(COMPETITIONS)\n"
        )
    elif failure == "wrong_metadata":
        files["scripts/mlebench_server_runner.py"] = VALID_RUNNER_SOURCE.replace(
            '("aerial-cactus-identification", "lite", "image_classification")',
            '("aerial-cactus-identification", "lite", "tabular")',
            1,
        )
    elif failure == "slug_mapping":
        files["scripts/mlebench_server_runner.py"] = "COMPETITIONS = " + repr(
            [(task_id, "fixture", "fixture") for task_id in VALID_SPLIT75]
        ) + "\nKAGGLE_SLUGS = " + repr({task_id: task_id for task_id in VALID_SPLIT75[:-1]})
    elif failure == "slug_value":
        files["scripts/mlebench_server_runner.py"] = VALID_RUNNER_SOURCE.replace(
            '"chaii-hindi-tamil-question-answering"',
            '"wrong-download-target"',
            1,
        )
    elif failure == "slug_alias_augassign":
        files["scripts/mlebench_server_runner.py"] = (
            VALID_RUNNER_SOURCE
            + '\nalias = KAGGLE_SLUGS\nalias |= {"evil-task": "evil-task"}\n'
        )
    elif failure == "rules_order":
        files["scripts/accept_kaggle_rules_all75.py"] = "SPLIT75 = " + repr(
            list(reversed(VALID_SPLIT75))
        )
    elif failure == "rules_mutation":
        files["scripts/accept_kaggle_rules_all75.py"] = VALID_RULES_SOURCE + "\nSPLIT75.reverse()\n"
    elif failure == "rules_alias_augassign":
        files["scripts/accept_kaggle_rules_all75.py"] = (
            VALID_RULES_SOURCE + '\nalias = SPLIT75\nalias += ["evil-task"]\n'
        )
    elif failure == "overlap":
        registry = json.loads(json.dumps(VALID_REGISTRY))
        registry["mle_bench_reference"]["locally_registered_official_competitions"] = []
        files["benchmark/mle_bench_75/tasks_template.json"] = json.dumps(registry)
    elif failure == "upstream":
        upstream = dict(VALID_UPSTREAM)
        upstream["commit"] = "1" * 40
        files["benchmark/mle_bench_75/UPSTREAM.json"] = json.dumps(upstream)

    _bundle(dist, files)
    with pytest.raises(VerificationError, match="split75 contract is invalid"):
        verify_workstation_source_bundle(dist, source_license)


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "../escape.txt",
        "scripts/_quarantine/payload.py",
        ".env",
        "node_modules/payload.js",
        "src/backup/payload.py",
        "src/.bak/payload.py",
    ],
)
def test_workstation_source_bundle_verifier_rejects_unsafe_members(tmp_path, unsafe_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    source_license = tmp_path / "LICENSE"
    source_license.write_text(REQUIRED_FILES["LICENSE"], encoding="utf-8")
    _bundle(dist, {unsafe_path: "unsafe\n"})

    with pytest.raises(VerificationError):
        verify_workstation_source_bundle(dist, source_license)


@pytest.mark.parametrize(
    "unsafe_directory",
    [
        "../escape/",
        "scripts/_quarantine/",
        "node_modules/",
        "src/backup/",
        "src/.bak/",
    ],
)
def test_workstation_source_bundle_verifier_rejects_unsafe_directory_members(
    tmp_path, unsafe_directory
):
    dist = tmp_path / "dist"
    dist.mkdir()
    source_license = tmp_path / "LICENSE"
    source_license.write_text(REQUIRED_FILES["LICENSE"], encoding="utf-8")
    archive_path = _bundle(dist)
    with zipfile.ZipFile(archive_path, mode="a") as archive:
        archive.writestr(f"xcientist-0.2.2/{unsafe_directory}", b"")

    with pytest.raises(VerificationError):
        verify_workstation_source_bundle(dist, source_license)


@pytest.mark.parametrize(
    "backup_path",
    [
        "src/Screens.tsx.bak_ui_redesign_20260625",
        "src/module.py.backup",
        "src/module.py.orig",
        "src/module.py.rej",
        "src/module.py.tmp",
        "src/module.py~",
    ],
)
def test_workstation_source_bundle_verifier_rejects_backup_members(tmp_path, backup_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    source_license = tmp_path / "LICENSE"
    source_license.write_text(REQUIRED_FILES["LICENSE"], encoding="utf-8")
    _bundle(dist, {backup_path: "stale source\n"})

    with pytest.raises(VerificationError, match="backup file"):
        verify_workstation_source_bundle(dist, source_license)


def test_workstation_source_bundle_verifier_rejects_duplicate_members(tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    source_license = tmp_path / "LICENSE"
    source_license.write_text(REQUIRED_FILES["LICENSE"], encoding="utf-8")
    archive_path = _bundle(dist)
    with zipfile.ZipFile(archive_path, mode="a") as archive:
        archive.writestr("xcientist-0.2.2/README.md", "replacement\n")

    with pytest.raises(VerificationError, match="duplicate"):
        verify_workstation_source_bundle(dist, source_license)


def test_workstation_source_bundle_verifier_rejects_symlink_members(tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    source_license = tmp_path / "LICENSE"
    source_license.write_text(REQUIRED_FILES["LICENSE"], encoding="utf-8")
    archive_path = _bundle(dist)
    link = zipfile.ZipInfo("xcientist-0.2.2/link")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive_path, mode="a") as archive:
        archive.writestr(link, "../../outside")

    with pytest.raises(VerificationError, match="symbolic link"):
        verify_workstation_source_bundle(dist, source_license)


@pytest.mark.parametrize(
    "content",
    [
        "remote=/" + "hpc2hdd/home/operator/project\n",
        "workspace=~/" + "jing" + "hw/project\n",
        "root=C:" + "\\Users\\operator\\project\n",
        "build=D:/EvoMind-" + "release-validation/source\n",
    ],
)
def test_workstation_source_bundle_rejects_machine_specific_text(tmp_path, content):
    dist = tmp_path / "dist"
    dist.mkdir()
    source_license = tmp_path / "LICENSE"
    source_license.write_text(REQUIRED_FILES["LICENSE"], encoding="utf-8")
    _bundle(dist, {"configs/private.yaml": content})

    with pytest.raises(VerificationError, match="portability"):
        verify_workstation_source_bundle(dist, source_license)


@pytest.mark.parametrize(
    "required_path",
    [
        "scripts/manage_hpc_proxy_bridge.ps1",
        "scripts/hpc_socks_bridge.py",
        "scripts/start_hpc_socks_bridge.py",
    ],
)
def test_workstation_source_bundle_verifier_requires_hpc_proxy_chain(tmp_path, required_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    source_license = tmp_path / "LICENSE"
    source_license.write_text(REQUIRED_FILES["LICENSE"], encoding="utf-8")
    _bundle(dist, omit_files={required_path})

    with pytest.raises(VerificationError, match="missing workstation files"):
        verify_workstation_source_bundle(dist, source_license)


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _npm_audit_report(scope):
    omit = [] if scope == "full" else ["dev"]
    return {
        "auditReportVersion": 2,
        "vulnerabilities": {},
        "metadata": {
            "vulnerabilities": {
                "info": 0,
                "low": 0,
                "moderate": 0,
                "high": 0,
                "critical": 0,
                "total": 0,
            },
            "dependencies": {
                "prod": 40,
                "dev": 20 if scope == "full" else 0,
                "optional": 5,
                "peer": 2,
                "peerOptional": 0,
                "total": 67 if scope == "full" else 47,
            },
        },
        "_evomindEvidence": {
            "schemaVersion": 1,
            "kind": "npm-audit",
            "scope": scope,
            "omit": omit,
        },
    }


def test_pip_audit_accepts_nonempty_complete_records(tmp_path):
    report = _write_json(
        tmp_path / "pip-audit.json",
        {"dependencies": [{"name": "requests", "version": "2.32.4", "vulns": []}]},
    )

    assert verify_pip_audit(report) == {"dependencies": 1, "vulnerabilities": 0}


def test_pip_audit_rejects_single_fabricated_package_as_complete_inventory(tmp_path):
    report = _write_json(
        tmp_path / "fabricated-pip-audit.json",
        {"dependencies": [{"name": "totally-real-package", "version": "9.9.9", "vulns": []}]},
    )
    installed = {("requests", "2.32.4"), ("urllib3", "2.5.0")}

    with pytest.raises(VerificationError, match="does not cover the installed Python tree"):
        verify_pip_audit(report, installed)


@pytest.mark.parametrize(
    "dependencies",
    [
        [],
        [{}],
        [{"name": "", "version": "1.0", "vulns": []}],
        [{"name": "requests", "version": "", "vulns": []}],
        [{"name": "requests", "version": "2.32.4", "vulns": {}}],
        [{"name": "requests", "version": "2.32.4", "vulns": [{}]}],
    ],
)
def test_pip_audit_rejects_empty_or_malformed_records(tmp_path, dependencies):
    report = _write_json(tmp_path / "pip-audit.json", {"dependencies": dependencies})

    with pytest.raises(VerificationError):
        verify_pip_audit(report)


def test_npm_audits_accept_distinct_explicit_full_and_production_scopes(tmp_path):
    full = _write_json(tmp_path / "full.json", _npm_audit_report("full"))
    production = _write_json(tmp_path / "production.json", _npm_audit_report("production"))

    result = verify_npm_audits(full, production)

    assert result["full"]["scope"] == "full"
    assert result["production"]["scope"] == "production"


def test_npm_audit_rejects_old_zero_count_false_green(tmp_path):
    report = _write_json(
        tmp_path / "old.json",
        {
            "metadata": {
                "vulnerabilities": {
                    "info": 0,
                    "low": 0,
                    "moderate": 0,
                    "high": 0,
                    "critical": 0,
                    "total": 0,
                }
            }
        },
    )

    with pytest.raises(VerificationError):
        verify_npm_audit(report, "production")


def test_npm_audit_rejects_swapped_scope(tmp_path):
    report = _write_json(tmp_path / "production.json", _npm_audit_report("production"))

    with pytest.raises(VerificationError, match="scope 'full'"):
        verify_npm_audit(report, "full")


def test_npm_audits_reject_same_file_for_both_scopes(tmp_path):
    report = _write_json(tmp_path / "audit.json", _npm_audit_report("full"))

    with pytest.raises(VerificationError, match="different files"):
        verify_npm_audits(report, report)


def test_npm_ls_inventory_collects_nested_scoped_packages_and_ignores_uninstalled_optionals(tmp_path):
    inventory_path = _write_json(
        tmp_path / "npm-ls.json",
        {
            "name": "research-agent-workstation",
            "version": "0.1.0",
            "dependencies": {
                "@scope/parent": {
                    "version": "1.2.3",
                    "dependencies": {
                        "nested.package": {"version": "4.5.6"},
                        "optional-not-installed": {},
                    },
                }
            },
        },
    )

    assert npm_ls_inventory(inventory_path, "research-agent-workstation", "0.1.0") == {
        ("@scope/parent", "1.2.3"),
        ("nested.package", "4.5.6"),
    }


def test_npm_ls_inventory_rejects_broken_dependency_tree(tmp_path):
    inventory_path = _write_json(
        tmp_path / "broken-npm-ls.json",
        {
            "name": "research-agent-workstation",
            "version": "0.1.0",
            "problems": ["missing: required-package@1.0.0"],
            "dependencies": {"required-package": {}},
        },
    )

    with pytest.raises(VerificationError, match="invalid npm dependency tree"):
        npm_ls_inventory(inventory_path, "research-agent-workstation", "0.1.0")


def test_npm_ls_inventory_rejects_optional_extraneous_even_with_lockfile(tmp_path):
    inventory_path = _write_json(
        tmp_path / "npm-ls.json",
        {
            "name": "research-agent-workstation",
            "version": "0.1.0",
            "problems": [
                "extraneous: @emnapi/wasi-threads@1.2.2 C:\\repo\\node_modules\\@emnapi\\wasi-threads"
            ],
            "dependencies": {
                "@emnapi/wasi-threads": {
                    "version": "1.2.2",
                    "extraneous": True,
                }
            },
        },
    )
    lock_path = _write_json(
        tmp_path / "package-lock.json",
        {
            "lockfileVersion": 3,
            "packages": {
                "": {"name": "research-agent-workstation", "version": "0.1.0"},
                "node_modules/@emnapi/wasi-threads": {
                    "version": "1.2.2",
                    "optional": True,
                    "dev": True,
                },
            },
        },
    )

    assert lock_path.is_file()
    with pytest.raises(VerificationError, match="invalid npm dependency tree"):
        npm_ls_inventory(inventory_path, "research-agent-workstation", "0.1.0")


def test_npm_ls_inventory_rejects_unlisted_extraneous(tmp_path):
    inventory_path = _write_json(
        tmp_path / "npm-ls.json",
        {
            "name": "research-agent-workstation",
            "version": "0.1.0",
            "problems": ["extraneous: injected-package@9.9.9 C:\\repo\\node_modules\\injected-package"],
            "dependencies": {"injected-package": {"version": "9.9.9", "extraneous": True}},
        },
    )
    with pytest.raises(VerificationError, match="invalid npm dependency tree"):
        npm_ls_inventory(inventory_path, "research-agent-workstation", "0.1.0")


def _cyclonedx_payload(name, version, purl_type, root_dependencies=None):
    root_ref = f"pkg:{purl_type}/{name}@{version}"
    dependency_ref = f"pkg:{purl_type}/dependency@1.0.0"
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "metadata": {
            "component": {
                "type": "application",
                "name": name,
                "version": version,
                "bom-ref": root_ref,
                "purl": root_ref,
            }
        },
        "components": [
            {
                "type": "library",
                "name": "dependency",
                "version": "1.0.0",
                "bom-ref": dependency_ref,
                "purl": dependency_ref,
            }
        ],
        "dependencies": [
            {
                "ref": root_ref,
                "dependsOn": [dependency_ref] if root_dependencies is None else root_dependencies,
            },
            {"ref": dependency_ref},
        ],
    }


@pytest.mark.parametrize(
    ("name", "version", "purl_type"),
    [
        ("xcientist", "0.2.2", "pypi"),
        ("research-agent-workstation", "0.1.0", "npm"),
    ],
)
def test_cyclonedx_verifier_accepts_ecosystem_root_and_nonempty_graph(tmp_path, name, version, purl_type):
    sbom = _write_json(
        tmp_path / f"{name}.cdx.json",
        _cyclonedx_payload(name, version, purl_type),
    )

    result = verify_cyclonedx_sbom(sbom, name, version, purl_type)

    assert result["components"] == 1
    assert result["dependencies"] == 2
    assert result["root_dependencies"] == 1


def test_cyclonedx_verifier_accepts_exact_nested_scoped_npm_inventory(tmp_path):
    payload = _cyclonedx_payload("research-agent-workstation", "0.1.0", "npm")
    parent_ref = "pkg:npm/%40scope/dependency@1.0.0"
    child_ref = "pkg:npm/nested.package@2.0.0"
    payload["components"] = [
        {
            "type": "library",
            "group": "@scope",
            "name": "dependency",
            "version": "1.0.0",
            "bom-ref": parent_ref,
            "purl": parent_ref,
            "components": [
                {
                    "type": "library",
                    "name": "nested.package",
                    "version": "2.0.0",
                    "bom-ref": child_ref,
                    "purl": child_ref,
                }
            ],
        }
    ]
    root_ref = payload["metadata"]["component"]["bom-ref"]
    payload["dependencies"] = [
        {"ref": root_ref, "dependsOn": [parent_ref]},
        {"ref": parent_ref, "dependsOn": [child_ref]},
        {"ref": child_ref},
    ]
    sbom = _write_json(tmp_path / "nested-npm.cdx.json", payload)

    result = verify_cyclonedx_sbom(
        sbom,
        "research-agent-workstation",
        "0.1.0",
        "npm",
        {("@scope/dependency", "1.0.0"), ("nested.package", "2.0.0")},
    )

    assert result["components"] == 2


def test_cyclonedx_verifier_rejects_single_component_as_complete_inventory(tmp_path):
    sbom = _write_json(
        tmp_path / "incomplete.cdx.json",
        _cyclonedx_payload("xcientist", "0.2.2", "pypi"),
    )

    with pytest.raises(VerificationError, match="does not cover the installed dependency tree"):
        verify_cyclonedx_sbom(
            sbom,
            "xcientist",
            "0.2.2",
            "pypi",
            {("dependency", "1.0.0"), ("missing-package", "2.0.0")},
        )


@pytest.mark.parametrize(
    ("name", "version"),
    [("other-project", "0.2.2"), ("xcientist", "9.9.9")],
)
def test_cyclonedx_verifier_rejects_wrong_project_identity(tmp_path, name, version):
    sbom = _write_json(tmp_path / "bad.cdx.json", _cyclonedx_payload(name, version, "pypi"))

    with pytest.raises(VerificationError):
        verify_cyclonedx_sbom(sbom, "xcientist", "0.2.2", "pypi")


def test_cyclonedx_verifier_rejects_project_only_in_dependency_components(tmp_path):
    payload = _cyclonedx_payload("other-project", "0.2.2", "pypi")
    payload["components"].append(
        {
            "type": "application",
            "name": "xcientist",
            "version": "0.2.2",
            "bom-ref": "pkg:pypi/xcientist@0.2.2",
            "purl": "pkg:pypi/xcientist@0.2.2",
        }
    )
    sbom = _write_json(tmp_path / "false-green.cdx.json", payload)

    with pytest.raises(VerificationError, match="metadata root"):
        verify_cyclonedx_sbom(sbom, "xcientist", "0.2.2", "pypi")


def test_cyclonedx_verifier_rejects_empty_root_dependencies(tmp_path):
    sbom = _write_json(
        tmp_path / "empty-root.cdx.json",
        _cyclonedx_payload("xcientist", "0.2.2", "pypi", root_dependencies=[]),
    )

    with pytest.raises(VerificationError, match="root component has no dependency edges"):
        verify_cyclonedx_sbom(sbom, "xcientist", "0.2.2", "pypi")


def test_cyclonedx_verifier_rejects_wrong_purl_ecosystem(tmp_path):
    sbom = _write_json(
        tmp_path / "wrong-ecosystem.cdx.json",
        _cyclonedx_payload("xcientist", "0.2.2", "npm"),
    )

    with pytest.raises(VerificationError, match="'pypi' ecosystem"):
        verify_cyclonedx_sbom(sbom, "xcientist", "0.2.2", "pypi")


def test_cyclonedx_verifier_accepts_python_root_without_purl(tmp_path):
    payload = _cyclonedx_payload("xcientist", "0.2.2", "pypi")
    payload["metadata"]["component"].pop("purl")
    payload["metadata"]["component"]["bom-ref"] = "root-component"
    payload["dependencies"][0]["ref"] = "root-component"
    sbom = _write_json(tmp_path / "python-tool-format.cdx.json", payload)

    result = verify_cyclonedx_sbom(sbom, "xcientist", "0.2.2", "pypi")

    assert result["root_dependencies"] == 1


def test_cyclonedx_verifier_rejects_unresolved_root_dependency(tmp_path):
    sbom = _write_json(
        tmp_path / "unresolved.cdx.json",
        _cyclonedx_payload(
            "xcientist",
            "0.2.2",
            "pypi",
            root_dependencies=["pkg:pypi/missing@1.0.0"],
        ),
    )

    with pytest.raises(VerificationError, match="missing components"):
        verify_cyclonedx_sbom(sbom, "xcientist", "0.2.2", "pypi")
