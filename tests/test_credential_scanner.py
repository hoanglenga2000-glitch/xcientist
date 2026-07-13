from __future__ import annotations

import json
import subprocess

import pytest

from scripts.verify_no_plaintext_secrets import (
    _is_secret_text,
    discover_candidate_files,
    discover_filesystem_candidates,
    main,
    python_literal_secret_assignments,
    scan_file,
    scan_history_repository,
    scan_repository,
    text_literal_secret_assignments,
)


def test_python_credential_scanner_detects_environment_assignment(tmp_path):
    source = tmp_path / "leak.py"
    source.write_text(
        "import os\nos.environ['GPU_SSH_PASSWORD'] = 'real-looking-value-123'\n",
        encoding="utf-8",
    )

    assert python_literal_secret_assignments(source) == [2]


def test_python_credential_scanner_detects_setdefault_but_allows_placeholders(tmp_path):
    leaked = tmp_path / "leaked.py"
    leaked.write_text(
        "import os\nos.environ.setdefault('API_KEY', 'live-value-123456')\n",
        encoding="utf-8",
    )
    placeholder = tmp_path / "placeholder.py"
    placeholder.write_text("PASSWORD = '<required>'\n", encoding="utf-8")

    assert python_literal_secret_assignments(leaked) == [2]
    assert python_literal_secret_assignments(placeholder) == []


def test_python_credential_scanner_detects_password_keyword_and_sendline(tmp_path):
    source = tmp_path / "transport.py"
    source.write_text(
        "client.connect(password='live-value-123456')\n"
        "child.sendline('another-live-value-123456')\n",
        encoding="utf-8",
    )

    assert python_literal_secret_assignments(source) == [1, 2]


def test_python_credential_scanner_detects_positional_auth_password(tmp_path):
    leaked = tmp_path / "transport.py"
    leaked.write_text(
        "transport.auth_password('worker', 'live-value-123456')\n",
        encoding="utf-8",
    )
    safe = tmp_path / "safe_transport.py"
    safe.write_text(
        "import os\ntransport.auth_password('worker', os.environ['HPC_PASSWORD'])\n",
        encoding="utf-8",
    )

    assert python_literal_secret_assignments(leaked) == [1]
    assert python_literal_secret_assignments(safe) == []


def test_python_credential_scanner_detects_dict_literal(tmp_path):
    source = tmp_path / "settings.py"
    source.write_text(
        "SERVER = {'username': 'worker', 'password': 'live-value-123456'}\n",
        encoding="utf-8",
    )

    assert python_literal_secret_assignments(source) == [1]

    mapping = tmp_path / "mapping.py"
    mapping.write_text(
        "FIELDS = {'GPU_SSH_PASSWORD': 'credentials.gpu_ssh_password'}\n",
        encoding="utf-8",
    )
    assert python_literal_secret_assignments(mapping) == []


def test_text_credential_scanner_detects_quoted_literals(tmp_path):
    source = tmp_path / "handoff.md"
    source.write_text(
        "password='live-value-123456'\n"
        '\"access_token\": \"another-live-value-123456\"\n',
        encoding="utf-8",
    )

    assert text_literal_secret_assignments(source) == [1, 2]


def test_text_credential_scanner_allows_safe_examples_and_regex_fixtures(tmp_path):
    source = tmp_path / "safe.md"
    source.write_text(
        "password='${HPC_TTA2_PASSWORD}'\n"
        "api_key='fixture-value-123456'\n"
        "const pattern = \"password='live-value-123456'\";\n"
        'api_key_status: "not_configured"\n'
        'return ready ? "password_env_dpapi" : "not_configured";\n',
        encoding="utf-8",
    )

    assert text_literal_secret_assignments(source) == []


def test_python_credential_scanner_accepts_utf8_bom(tmp_path):
    source = tmp_path / "bom.py"
    source.write_text("PASSWORD = 'synthetic-live-value-123456'\n", encoding="utf-8-sig")

    assert python_literal_secret_assignments(source) == [1]
    findings, scanned = scan_file(source, tmp_path)
    assert scanned is True
    assert not any(finding["pattern"] == "python_parse_error" for finding in findings)
    assert any(finding["pattern"] == "python_literal_secret_assignment" for finding in findings)


def test_scan_file_fails_closed_on_python_parse_error(tmp_path):
    source = tmp_path / "broken.py"
    source.write_text("def broken(:\n    pass\n", encoding="utf-8")

    findings, scanned = scan_file(source, tmp_path)

    assert scanned is True
    assert any(finding["pattern"] == "python_parse_error" for finding in findings)


def test_scan_file_fails_closed_on_nul_in_python_source(tmp_path):
    source = tmp_path / "hidden.py"
    source.write_bytes(b"PASSWORD = 'synthetic-live-value-123456'\0\n")

    findings, scanned = scan_file(source, tmp_path)

    assert scanned is True
    assert any(finding["pattern"] == "python_parse_error" for finding in findings)


def test_scan_file_fails_closed_on_unicode_and_read_errors(tmp_path):
    undecodable = tmp_path / "undecodable.py"
    undecodable.write_bytes(b"PASSWORD = '\xff'\n")

    unicode_findings, unicode_scanned = scan_file(undecodable, tmp_path)
    missing_findings, missing_scanned = scan_file(tmp_path / "missing.py", tmp_path)

    assert unicode_scanned is True
    assert [finding["pattern"] for finding in unicode_findings] == ["unicode_decode_error"]
    assert missing_scanned is False
    assert [finding["pattern"] for finding in missing_findings] == ["file_read_error"]


def test_placeholder_exemption_requires_a_whole_value_match():
    assert _is_secret_text("<redacted-password>") is False
    assert _is_secret_text("prefix-<redacted-password>-suffix") is True


def test_scanner_checks_each_credential_match_on_the_same_line(tmp_path):
    source = tmp_path / "settings.md"
    source.write_text(
        "password='<redacted-password>'; "
        "password='synthetic-live-value-123456'; "
        "access_token='synthetic-second-value-123456'\n",
        encoding="utf-8",
    )

    assert text_literal_secret_assignments(source) == [1, 1]
    findings, scanned = scan_file(source, tmp_path)
    literal_findings = [
        finding for finding in findings if finding["pattern"] == "text_literal_secret_assignment"
    ]

    assert scanned is True
    assert len(literal_findings) == 2
    assert len({finding["column"] for finding in literal_findings}) == 2


def _git(repo, *args):
    subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def _commit(repo, message):
    _git(
        repo,
        "-c",
        "user.name=EvoMind Test",
        "-c",
        "user.email=evomind-test@example.invalid",
        "commit",
        "--quiet",
        "-m",
        message,
    )


def test_repository_scan_covers_tracked_and_untracked_sources_outside_legacy_roots(tmp_path):
    _git(tmp_path, "init", "--quiet")
    workflow = tmp_path / ".github" / "workflows" / "release.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text("password: 'synthetic-workflow-value-123456'\n", encoding="utf-8")
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("ENV PASSWORD='synthetic-container-value-123456'\n", encoding="utf-8")
    documentation = tmp_path / "docs" / "deployment.md"
    documentation.parent.mkdir()
    documentation.write_text("api_key='synthetic-doc-value-123456'\n", encoding="utf-8")
    web_source = tmp_path / "web" / "app.ts"
    web_source.parent.mkdir()
    web_source.write_text("const access_token = 'synthetic-web-value-123456';\n", encoding="utf-8")
    script = tmp_path / "scripts" / "deploy.sh"
    script.parent.mkdir()
    script.write_text("password='synthetic-script-value-123456'\n", encoding="utf-8")
    quarantined = tmp_path / "scripts" / "_quarantine" / "old_remote.py"
    quarantined.parent.mkdir()
    quarantined.write_text(
        "transport.auth_password('worker', 'synthetic-retired-value-123456')\n",
        encoding="utf-8",
    )
    config = tmp_path / "configs" / "production.yaml"
    config.parent.mkdir()
    config.write_text("api_key: 'synthetic-config-value-123456'\n", encoding="utf-8")
    fixture = tmp_path / "tests" / "fixtures" / "credential-corpus.txt"
    fixture.parent.mkdir(parents=True)
    fixture.write_text("password='synthetic-fixture-secret-123456'\n", encoding="utf-8")
    binary = tmp_path / "docs" / "diagram.png"
    binary.write_bytes(b"\x89PNG\r\n\x1a\n\x00binary-fixture")
    _git(
        tmp_path,
        "add",
        ".github/workflows/release.yml",
        "web/app.ts",
        "scripts/deploy.sh",
        "scripts/_quarantine/old_remote.py",
        "configs/production.yaml",
        "tests/fixtures",
        "docs/diagram.png",
    )

    candidates, discovery_findings = discover_candidate_files(tmp_path)
    relative_candidates = {
        candidate.relative_to(tmp_path).as_posix() for candidate in candidates
    }
    findings, scanned_files = scan_repository(tmp_path)
    finding_files = {finding["file"] for finding in findings}

    assert [finding["pattern"] for finding in discovery_findings] == [
        "tracked_quarantine_path"
    ]
    assert {
        ".github/workflows/release.yml",
        "Dockerfile",
        "configs/production.yaml",
        "docs/deployment.md",
        "docs/diagram.png",
        "scripts/deploy.sh",
        "scripts/_quarantine/old_remote.py",
        "tests/fixtures/credential-corpus.txt",
        "web/app.ts",
    } <= relative_candidates
    assert {
        ".github/workflows/release.yml",
        "Dockerfile",
        "configs/production.yaml",
        "docs/deployment.md",
        "scripts/deploy.sh",
        "web/app.ts",
    } <= finding_files
    assert "tests/fixtures/credential-corpus.txt" not in finding_files
    assert "docs/diagram.png" not in finding_files
    assert scanned_files == 7


def test_source_bundle_without_git_uses_strict_filesystem_inventory(tmp_path):
    safe = tmp_path / "README.md"
    safe.write_text("source bundle\n", encoding="utf-8")
    leaked = tmp_path / "src" / "settings.py"
    leaked.parent.mkdir()
    leaked.write_text("password='synthetic-bundle-value-123456'\n", encoding="utf-8")
    ignored_vendor = tmp_path / "node_modules" / "package.js"
    ignored_vendor.parent.mkdir()
    ignored_vendor.write_text("api_key='synthetic-vendor-value-123456'\n", encoding="utf-8")
    ignored_fixture = tmp_path / "tests" / "fixtures" / "credential.txt"
    ignored_fixture.parent.mkdir(parents=True)
    ignored_fixture.write_text("password='synthetic-fixture-value-123456'\n", encoding="utf-8")

    candidates, discovery_findings = discover_candidate_files(tmp_path)
    findings, scanned_files = scan_repository(tmp_path)

    assert discovery_findings == []
    assert {path.relative_to(tmp_path).as_posix() for path in candidates} == {
        "README.md",
        "src/settings.py",
    }
    assert {finding["file"] for finding in findings} == {"src/settings.py"}
    assert scanned_files == 2


def test_source_bundle_quarantine_directory_is_fail_closed(tmp_path):
    quarantined = tmp_path / "scripts" / "_quarantine" / "retired.py"
    quarantined.parent.mkdir(parents=True)
    quarantined.write_text("print('synthetic safe file')\n", encoding="utf-8")

    candidates, findings = discover_filesystem_candidates(tmp_path)

    assert quarantined in candidates
    assert [finding["pattern"] for finding in findings] == [
        "source_bundle_quarantine_path"
    ]


def test_history_scan_detects_secret_removed_from_current_tree(tmp_path):
    _git(tmp_path, "init", "--quiet")
    source = tmp_path / "docs" / "old-deployment.md"
    source.parent.mkdir(parents=True)
    source.write_text("password='synthetic-history-value-123456'\n", encoding="utf-8")
    _git(tmp_path, "add", "docs/old-deployment.md")
    _commit(tmp_path, "add historical deployment note")
    leaked_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        encoding="ascii",
    ).stdout.strip()

    source.write_text("Credential values are stored outside Git.\n", encoding="utf-8")
    _git(tmp_path, "add", "docs/old-deployment.md")
    _commit(tmp_path, "remove historical credential")

    current_findings, _ = scan_repository(tmp_path)
    history_findings, scanned_files, commit_count, blob_count = scan_history_repository(tmp_path)

    assert current_findings == []
    assert scanned_files >= 2
    assert commit_count == 2
    assert blob_count >= 2
    leaked = [item for item in history_findings if item["file"] == "docs/old-deployment.md"]
    assert leaked
    assert any(item["commit"] == leaked_commit for item in leaked)
    assert all(len(str(item["object_id"])) == 40 for item in leaked)


def test_repository_main_history_mode_fails_on_removed_secret(tmp_path):
    _git(tmp_path, "init", "--quiet")
    source = tmp_path / "release.env.example"
    source.write_text("API_KEY='synthetic-history-token-123456'\n", encoding="utf-8")
    _git(tmp_path, "add", "release.env.example")
    _commit(tmp_path, "add old release settings")
    source.write_text("API_KEY='<your-api-key>'\n", encoding="utf-8")
    _git(tmp_path, "add", "release.env.example")
    _commit(tmp_path, "replace old release credential")

    with pytest.raises(SystemExit) as caught:
        main(tmp_path, history=True)

    payload = json.loads(str(caught.value))
    assert payload["status"] == "failed"
    assert payload["history_included"] is True
    assert payload["history_commit_count"] == 2
    assert payload["history_scanned_files"] >= 2
    assert any(finding["file"] == "release.env.example" for finding in payload["findings"])


def test_repository_main_exits_nonzero_for_tracked_parse_failure(tmp_path):
    _git(tmp_path, "init", "--quiet")
    source = tmp_path / "automation" / "broken.py"
    source.parent.mkdir()
    source.write_text("if True print('broken')\n", encoding="utf-8")
    _git(tmp_path, "add", "automation/broken.py")

    with pytest.raises(SystemExit) as caught:
        main(tmp_path)

    payload = json.loads(str(caught.value))
    assert payload["status"] == "failed"
    assert payload["finding_count"] >= 1
    assert any(finding["pattern"] == "python_parse_error" for finding in payload["findings"])


def test_repository_main_exits_nonzero_for_unicode_and_read_failures(tmp_path):
    _git(tmp_path, "init", "--quiet")
    undecodable = tmp_path / "automation" / "undecodable.py"
    undecodable.parent.mkdir()
    undecodable.write_bytes(b"PASSWORD = '\xff'\n")
    missing = tmp_path / "automation" / "missing.py"
    missing.write_text("print('staged then removed')\n", encoding="utf-8")
    _git(tmp_path, "add", "automation/undecodable.py", "automation/missing.py")
    missing.unlink()

    with pytest.raises(SystemExit) as caught:
        main(tmp_path)

    payload = json.loads(str(caught.value))
    patterns = {finding["pattern"] for finding in payload["findings"]}
    assert payload["status"] == "failed"
    assert {"unicode_decode_error", "file_read_error"} <= patterns
