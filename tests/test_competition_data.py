from __future__ import annotations

import json
import re
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from evomind_runtime.competition_data import (
    COMPETITION_CATALOG,
    _parse_weather4cast_sftp_long_listing,
    _weather4cast_local_file_set_matches_remote,
    _weather4cast_listing_profile,
    _weather4cast_nonroot_listing_timeout_is_skippable,
    _weather4cast_remote_file_set_sha256,
    _weather4cast_sftp_supports_short_option,
    build_adapter_script,
    catalog_projection,
    kaggle_catalog_projection,
    normalize_competition,
    persistent_root,
    source_catalog_sha256,
    validate_receipt,
)
from evomind_runtime.models import ToolResult
from evomind_runtime.policy import PolicyEngine
from evomind_runtime.tools import (
    _competition_data_hpc_identity,
    _competition_data_status,
    _managed_kaggle_credential_payload,
    build_default_registry,
)


def test_six_competition_catalog_has_fixed_hpc_roots_and_stable_identity() -> None:
    assert list(COMPETITION_CATALOG) == [
        "cure_bench",
        "e2lmc",
        "mindgames",
        "ariel_2025",
        "weather4cast",
        "open_polymer",
    ]
    assert len(catalog_projection()) == 6
    assert len(kaggle_catalog_projection()) == 3
    assert source_catalog_sha256() == "5540fffdea8741766783b18b2915f3c6feac7d5174c1368c4ab7e8902b919a8b"
    for item in catalog_projection():
        assert item["persistent_root"].startswith(
            "/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/competition_data/"
        )


@pytest.mark.parametrize(
    ("supplied", "expected"),
    [
        ("CURE-Bench", "cure_bench"),
        ("E2LM", "e2lmc"),
        ("Mind Games", "mindgames"),
        ("Ariel R2", "ariel_2025"),
        ("Weather4cast", "weather4cast"),
        ("Open Polymer", "open_polymer"),
    ],
)
def test_competition_aliases_are_normalized_without_model_written_paths(supplied: str, expected: str) -> None:
    assert normalize_competition(supplied) == expected
    assert persistent_root(supplied).endswith("/" + expected)


def test_standard_competition_tools_are_selected_by_ordinary_user_language_and_never_accept_secrets() -> None:
    registry = build_default_registry()
    selected = {item.name for item in registry.specs_for_prompt("把 Weather4cast 比赛数据下载到 HPC 并验收")}
    assert {"competition_data_prepare", "competition_data_status", "hpc_verify"} <= selected
    ordinary_six = {
        item.name
        for item in registry.specs_for_prompt("把六个比赛都下载并训练完成")
    }
    assert {
        "competition_data_prepare",
        "competition_data_status",
        "hpc_verify",
        "hpc_execute_solution",
        "training_route",
    } <= ordinary_six
    for name in ("competition_data_prepare", "competition_data_status"):
        spec = registry.get(name)
        assert spec is not None
        properties = spec.input_schema["properties"]
        assert set(properties) == {"competition", "timeout_seconds"}
        assert not any(term in json.dumps(properties).casefold() for term in ("password", "token", "credential", "secret_ref"))


def test_managed_kaggle_connector_projects_only_an_already_loaded_environment_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(FileNotFoundError, match="not loaded"):
        _managed_kaggle_credential_payload()

    monkeypatch.setenv("KAGGLE_API_TOKEN", "managed-token-fixture")
    assert _managed_kaggle_credential_payload() == {
        "schema": "evomind.run_secret_payload.v1",
        "purpose": "kaggle_api",
        "token": "managed-token-fixture",
    }

    monkeypatch.delenv("KAGGLE_API_TOKEN")
    monkeypatch.setenv("KAGGLE_USERNAME", "managed.user")
    monkeypatch.setenv("KAGGLE_KEY", "legacy-key-fixture")
    assert _managed_kaggle_credential_payload() == {
        "schema": "evomind.run_secret_payload.v1",
        "purpose": "kaggle_api",
        "username": "managed.user",
        "token": "legacy-key-fixture",
    }


def test_competition_status_is_read_only_and_prepare_is_workspace_managed(tmp_path) -> None:
    policy = PolicyEngine()

    status = policy.evaluate(
        tool_name="competition_data_status",
        arguments={"competition": "open_polymer"},
        permission_level="observe",
        workspace_root=tmp_path,
    )
    prepare = policy.evaluate(
        tool_name="competition_data_prepare",
        arguments={"competition": "open_polymer"},
        permission_level="workspace-write",
        workspace_root=tmp_path,
    )

    assert status.allowed is True
    assert status.requires_approval is False
    assert status.reason == "read-only capability"
    assert prepare.allowed is True
    assert prepare.requires_approval is False


def test_competition_data_hpc_identity_projects_complete_non_secret_generation_receipt() -> None:
    context = SimpleNamespace(metadata={
        "managed_hpc_identity": {
            "allocation_generation": 19,
            "owner_principal_id": "admin",
            "allocation_binding_id": "fixture-binding-do-not-project",
        },
    })
    evidence = {
        "status": "job_container_verified",
        "designated_proxy_path_verified": True,
        "pinned_gateway_host_key_verified": True,
        "allocation_role_authenticated": True,
        "expected_host_uuid_match": True,
        "expected_gpu_uuid_match": True,
        "expected_gpu_model_and_memory_match": True,
        "allowed_remote_root_match": True,
        "job_container_verified": True,
        "identity_consistent": True,
        "samples_requested": 5,
        "samples_passed": 5,
        "host_uuid": "fixture-host-uuid",
        "gpu_uuids": ["fixture-gpu-uuid"],
        "gpu_name": "NVIDIA A800 80GB PCIe",
        "gpu_memory_total_mib": 81920,
        "remote_root": "/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra",
        "read_only": True,
        "signals_sent": 0,
        "other_processes_modified": False,
    }

    projected = _competition_data_hpc_identity(
        evidence,
        job_id=91805,
        credential_profile="tenant_fixture_job91805_g19",
        context=context,
    )

    assert projected["job_id"] == 91805
    assert projected["profile_alias"] == "G19"
    assert projected["allocation_generation"] == 19
    assert projected["designated_proxy_path_verified"] is True
    assert projected["job_container_verified"] is True
    assert projected["samples_passed"] == projected["samples_requested"] == 5
    assert projected["remote_root"] == "/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra"
    assert projected["secrets_returned"] is False
    serialized = json.dumps(projected, sort_keys=True)
    assert "owner_principal_id" not in serialized
    assert "allocation_binding_id" not in serialized
    assert "fixture-host-uuid" not in serialized
    assert "fixture-gpu-uuid" not in serialized


def test_competition_data_status_returns_exact_verified_job_and_hpc_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    tenant_id = "tenant_aaaaaaaaaaaaaaaaaaaaaaaa"
    profile = f"{tenant_id}_job91805_g19"
    managed_identity = {
        "tenant_id": tenant_id,
        "owner_principal_id": "admin",
        "job_id": 91805,
        "credential_profile": profile,
        "allocation_generation": 19,
        "profile_instance_id": "35279c5f-6a99-4dd0-b53a-789b7f03b376",
        "allocation_binding_id": "aimslab-job91805-fixture",
    }
    evidence = {
        "status": "job_container_verified",
        "designated_proxy_path_verified": True,
        "pinned_gateway_host_key_verified": True,
        "allocation_role_authenticated": True,
        "expected_host_uuid_match": True,
        "expected_gpu_uuid_match": True,
        "expected_gpu_model_and_memory_match": True,
        "allowed_remote_root_match": True,
        "job_container_verified": True,
        "identity_consistent": True,
        "samples_requested": 5,
        "samples_passed": 5,
        "host_uuid": "fixture-host-uuid",
        "gpu_uuids": ["fixture-gpu-uuid"],
        "gpu_name": "NVIDIA A800 80GB PCIe",
        "gpu_memory_total_mib": 81920,
        "remote_root": "/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra",
        "read_only": True,
        "signals_sent": 0,
        "other_processes_modified": False,
    }

    class FakeHpcRuntime:
        def __init__(self, **_kwargs) -> None:
            pass

        def execute_managed_data_task(self, **_kwargs):
            return {
                "schema": "evomind.managed_competition_data_execution.v1",
                "status": "completed",
                "receipt": {
                    "schema": "evomind.competition_data_receipt.v1",
                    "competition": "open_polymer",
                    "status": "NOT_STARTED",
                    "persistent_root": persistent_root("open_polymer"),
                    "secret_values_logged": False,
                },
                "stdout_tail": "",
                "stderr_tail": "",
                "local_log": "",
            }

    monkeypatch.setattr(
        "evomind_runtime.tools._hpc_verify",
        lambda _args, _context: ToolResult("", True, evidence, "verified"),
    )
    monkeypatch.setattr("evomind_runtime.tools._load_bound_hpc_config", lambda *_args: object())
    monkeypatch.setattr("evomind_runtime.hpc_runtime_overlay.HpcRuntime", FakeHpcRuntime)
    monkeypatch.setattr(
        "evomind_runtime.tools._artifact_publish",
        lambda _args, _context: ToolResult(
            "", True, {"artifact": {"id": "artifact_fixture", "sha256": "a" * 64}}, "published",
        ),
    )
    context = SimpleNamespace(
        session_id="run_fixture",
        workspace_root=tmp_path,
        runtime_root=tmp_path / "runtime",
        metadata={"managed_hpc_identity": managed_identity},
    )

    result = _competition_data_status({"competition": "open_polymer"}, context)

    assert result.ok is True
    assert result.content["status"] == "NOT_STARTED"
    assert result.content["hpc_identity_gate"] == "passed_5_of_5"
    assert result.content["hpc_identity"]["job_id"] == 91805
    assert result.content["hpc_identity"]["profile_alias"] == "G19"
    assert result.content["hpc_identity"]["designated_proxy_path_verified"] is True
    assert result.content["hpc_identity"]["job_container_verified"] is True
    assert result.content["hpc_identity"]["samples_passed"] == 5


def test_mindgames_adapter_runs_four_distinct_real_environment_loaders() -> None:
    script = build_adapter_script("mindgames", "prepare")
    for environment in ("SecretMafia-v0", "Codenames-v0", "ColonelBlotto-v0", "ThreePlayerIPD-v0"):
        assert environment in script
    assert "ta.make(env_id=env_id)" in script
    assert "env.reset(num_players=players)" in script
    assert '[\"SecretMafia-v0\", 6]' in script
    assert "textarena==0.7.4" in script
    assert "54ed02917d6771dcc3e8141218960d020947f7f2ccfd9ac9b320979349746015" in script
    assert "6025f530624335c67d6547d44757b357b4e79bae030a0383e9887a92c1718f0b" in script
    assert "write_receipt VALIDATION_FAILED" in script
    assert '"textarena_version"' in script
    assert '"runtime_dependencies"' in script
    assert 'expected_environments=set(["SecretMafia-v0", "Codenames-v0", "ColonelBlotto-v0", "ThreePlayerIPD-v0"])' in build_adapter_script("mindgames", "status")
    assert "find_spec('textarena')" not in script and 'find_spec("textarena")' not in script
    loader = script.split(
        '"$RUNTIME/bin/python" - "$ROOT/.evomind/loader-smoke.json" <<\'PY\'\n',
        1,
    )[1].split("\nPY\n", 1)[0]
    compile(loader, "mindgames-loader-smoke.py", "exec")


def test_e2lmc_adapter_requires_eight_unique_valid_notebooks() -> None:
    script = build_adapter_script("e2lmc", "prepare")
    assert script.count("download '") == 8
    assert "valid_unique_notebooks" in script
    assert 'data.get("cells")' in script
    assert 'data.get("nbformat")' in script


def test_weather_adapter_uses_ephemeral_secret_file_and_resumable_background_mirror() -> None:
    script = build_adapter_script("weather4cast", "prepare")
    assert "ACCELERATE=0" in script
    assert "EVOMIND_SECRET_WEATHER4CAST_FILE" in script
    assert "exec 9<\"$SECRET\"" in script
    assert "rm -f \"$SECRET\"" in script
    assert "mirror --continue" not in script
    assert "OpenSSH sftp is required for exact remote file-set verification" in script
    assert "command -v sftp" in script
    assert "reget -R ." not in script
    assert "reget -a -R ." not in script
    assert "SSH_ASKPASS_REQUIRE" in script
    assert "SFTP_AUTH_FAILED" in script
    assert "SFTP_CONNECTION_FAILED" in script
    assert "SFTP_BATCH_FAILED" in script
    assert "sshpass" in script
    assert '["-B", "262144"]' in script
    assert '["-R", "256"]' in script
    assert '"-oIPQoS=throughput"' in script
    assert '"-oCompression=no"' in script
    assert "0o600" in script
    assert "#!/bin/sh" in script
    assert "for temporary in (askpass, password_file, batch, *parallel_batches)" in script
    assert '#!/bin/sh' in script
    assert 'EVOMIND_PASSWORD_FILE' in script
    assert 'open("/dev/fd/9"' in script
    assert "secret-consumed" in script
    assert "echo $password" not in script.casefold()
    assert "--password" not in script.casefold()
    assert "\nWORKER\n" in script
    assert "\nPY\n" in script
    accelerated = build_adapter_script("weather4cast", "accelerate")
    assert "ACCELERATE=1" in accelerated
    assert "controlled Weather4cast acceleration credential failed preflight" in accelerated
    assert "existing Weather4cast worker identity could not be verified" in accelerated
    assert 'kill -TERM -- "-$EXISTING_PID"' in accelerated
    assert "existing Weather4cast worker did not stop before acceleration" in accelerated
    assert 'exec 6>"$JOB/accelerate.lock"' in accelerated
    assert "existing Weather4cast worker did not release the inherited lock" in accelerated
    assert '6>&- 8>&- > /dev/null' in accelerated
    assert "ThreadPoolExecutor(" in accelerated
    assert "max_workers=max(1, min(requested_parallelism, len(pending_batches)))" in accelerated
    assert "parallelism = min(16, len(remote_files))" in accelerated
    assert 'sftp-list-{listing_index:04d}.batch' in accelerated
    assert 'sftp-transfer-{index:02d}.batch' in accelerated
    assert "ls -ln ./*" not in accelerated
    assert 'f"reget {lftp_quote(name)} {lftp_quote(name)}"' in accelerated
    assert "tuned=False" in accelerated
    assert "except subprocess.TimeoutExpired:" in accelerated
    assert "for listing_retry in range(4):" in accelerated
    assert '"SFTP_CONNECTION_FAILED", "SFTP_TIMEOUT", "SFTP_FAILED"' in accelerated
    assert "time.sleep(min(4, 2 ** listing_retry))" in accelerated
    assert "listing_failure = classify(listing.stderr)" in accelerated
    assert "pending_directories.pop(0)" in accelerated
    assert "def reject_listing_structure(reason):" in accelerated
    assert 'f"listing_structure_{reason}"' in accelerated
    assert 'reject_listing_structure("duplicate_directory")' in accelerated
    assert 'reject_listing_structure("depth")' in accelerated
    assert 'reject_listing_structure("invalid_relative")' in accelerated
    assert 'reject_listing_structure("duplicate_file")' in accelerated
    assert 'reject_listing_structure("entry_cap")' in accelerated
    assert '"adapter_listing_archive_files": 0' in accelerated
    assert '"adapter_listing_scientific_files": 0' in accelerated
    assert '"adapter_listing_metadata_files": 0' in accelerated
    assert '"adapter_listing_other_files": 0' in accelerated
    assert '"adapter_listing_max_depth": 0' in accelerated
    assert '"adapter_listing_skipped_directories": 0' in accelerated
    assert '"adapter_listing_timeout_exhaustions": 0' in accelerated
    skip_guard = accelerated.index(
        "if _weather4cast_nonroot_listing_timeout_is_skippable("
    )
    skip_count = accelerated.index(
        'adapter_telemetry["adapter_listing_skipped_directories"] += 1',
        skip_guard,
    )
    timeout_count = accelerated.index(
        'adapter_telemetry["adapter_listing_timeout_exhaustions"] += 1',
        skip_count,
    )
    skip_state = accelerated.index('state("RUNNING", "downloading")', timeout_count)
    skip_continue = accelerated.index("continue", skip_state)
    terminal_failure = accelerated.index(
        'state("DOWNLOAD_FAILED", "failed", listing_failure)', skip_continue
    )
    assert skip_guard < skip_count < timeout_count < skip_state < skip_continue < terminal_failure
    completed_listing = accelerated.index('"ok_with_skips"')
    incomplete_gate = accelerated.index(
        'state("DOWNLOAD_FAILED", "failed", "WEATHER_LISTING_INCOMPLETE")',
        completed_listing,
    )
    remote_set_gate = accelerated.index(
        "remote_file_set_sha256 = _weather4cast_remote_file_set_sha256(remote_files)",
        incomplete_gate,
    )
    transfer_start = accelerated.index("parallelism = min(16, len(remote_files))")
    assert completed_listing < incomplete_gate < remote_set_gate < transfer_start
    assert '"lftp_parallel"' not in accelerated
    assert "_weather4cast_local_file_set_matches_remote(local_remote_files, remote_files)" in accelerated
    assert '"skipped_directory_count": adapter_telemetry["adapter_listing_skipped_directories"]' in accelerated
    assert '"remote_file_set_sha256": remote_file_set_sha256' in accelerated
    assert "WEATHER_LISTING_INCOMPLETE" in accelerated
    symlink_gate = accelerated.index("if root.is_symlink() or data_root.is_symlink():")
    first_remote_command = accelerated.index("run_batch(list_batch", symlink_gate)
    assert symlink_gate < first_remote_command
    entry_cap = accelerated.index(
        "if len(remote_files) + len(local_directories) > 1000:"
    )
    profile_refresh = accelerated.rfind("refresh_listing_profile()", 0, entry_cap)
    assert profile_refresh >= 0
    assert profile_refresh < entry_cap
    reject_start = accelerated.index("def reject_listing_structure(reason):")
    reject_state = accelerated.index('state("DOWNLOAD_FAILED", "failed"', reject_start)
    assert accelerated.index("refresh_listing_profile()", reject_start) < reject_state
    assert "> 16" in accelerated
    assert "_parse_weather4cast_sftp_long_listing" in accelerated
    assert "_weather4cast_sftp_supports_short_option" in accelerated
    assert "__EVOMIND_WEATHER_SFTP_HELPERS__" not in accelerated
    assert '"adapter_transfer_mode": "starting"' in accelerated
    assert '"openssh_parallel_files"' in accelerated
    assert 'pending_directories = [""]' in accelerated
    assert 'list_commands = [f"cd {lftp_quote(remote)}"]' in accelerated
    assert 'list_commands.append(f"cd {lftp_quote(current_directory)}")' in accelerated
    assert 'list_commands.extend(["ls -ln .", "quit"])' in accelerated
    assert 'ls -ln {lftp_quote(current_directory)}/*' not in accelerated
    assert '"parallel_retry_exhausted"' in accelerated
    assert "SFTP_PARALLEL_RETRY_EXHAUSTED" in accelerated
    assert "pending_batches = failed_batches" in accelerated
    assert accelerated.count("reget -R .") == 0
    assert '"adapter_tuning_requested": False' in accelerated
    assert "mirror --continue" not in accelerated
    with pytest.raises(ValueError, match="only for Weather4cast"):
        build_adapter_script("ariel_2025", "accelerate")
    status_script = build_adapter_script("weather4cast", "status")
    assert '"DOWNLOAD_FAILED"' in status_script
    assert 'attempt_root / "exit-code"' in status_script
    # Every retry owns an immutable attempt directory selected atomically by a
    # single current-attempt record.  This prevents an old worker or failure
    # state from overwriting the receipt for a newer retry.
    assert "current-attempt.json" in script
    assert 'attempts" / attempt_id' in script
    assert "flock -n 8" in script
    assert 'open("/dev/fd/9"' in script
    assert "umask 077" in script
    assert "os.open(password_file" in script
    assert "SFTP_BINARY_MISSING" in script
    assert "SFTP_TIMEOUT" in script
    assert "timeout=86400" in script
    assert 'state("FULL_DATA_READY", "complete"' in script
    assert "password_file.write_text" not in script
    assert 'python3 - "$SECRET"' not in script
    assert "timeout=120" not in script


def test_weather_sftp_usage_detection_supports_standard_bracketed_openssh_usage() -> None:
    usage = (
        "usage: sftp [-46AaCfNpqrv] [-B buffer_size] [-b batchfile] "
        "[-R num_requests] [-S program] destination"
    )
    assert _weather4cast_sftp_supports_short_option(usage, "-B") is True
    assert _weather4cast_sftp_supports_short_option(usage, "-R") is True
    assert _weather4cast_sftp_supports_short_option("usage: sftp -B value -R value", "-B") is True
    assert _weather4cast_sftp_supports_short_option("usage: sftp -B value -R value", "-R") is True
    with pytest.raises(ValueError, match="unsupported"):
        _weather4cast_sftp_supports_short_option(usage, "-Z")


def test_weather_nonroot_listing_timeout_is_the_only_skippable_listing_failure() -> None:
    assert _weather4cast_nonroot_listing_timeout_is_skippable(
        "SFTP_TIMEOUT", "safe-child-directory"
    ) is True
    assert _weather4cast_nonroot_listing_timeout_is_skippable("SFTP_TIMEOUT", "") is False
    for failure in (
        "SFTP_AUTH_FAILED",
        "SFTP_BATCH_FAILED",
        "SFTP_CONNECTION_FAILED",
        "SFTP_FAILED",
        "SFTP_LISTING_FAILED",
    ):
        assert _weather4cast_nonroot_listing_timeout_is_skippable(
            failure, "safe-child-directory"
        ) is False


def test_weather_skipped_directory_cannot_substitute_a_stale_local_file() -> None:
    remote_files = [f"season/chunk-{index:03d}.bin" for index in range(108)]
    expected_hash = _weather4cast_remote_file_set_sha256(remote_files)
    assert re.fullmatch(r"[a-f0-9]{64}", expected_hash)
    assert _weather4cast_local_file_set_matches_remote(remote_files, remote_files) is True

    stale_substitution = [*remote_files[:-1], "season/stale-extra.bin"]
    assert len(stale_substitution) == 108
    assert _weather4cast_local_file_set_matches_remote(
        stale_substitution, remote_files
    ) is False


def test_weather_missing_remote_member_fails_before_transfer() -> None:
    incomplete_remote_files = [f"season/chunk-{index:03d}.bin" for index in range(107)]
    with pytest.raises(ValueError, match="exactly 108"):
        _weather4cast_remote_file_set_sha256(incomplete_remote_files)

    extra_remote_files = [f"season/chunk-{index:03d}.bin" for index in range(109)]
    with pytest.raises(ValueError, match="exactly 108"):
        _weather4cast_remote_file_set_sha256(extra_remote_files)

    casefold_duplicate = [f"season/chunk-{index:03d}.bin" for index in range(106)]
    casefold_duplicate.extend(["season/DUPLICATE.bin", "season/duplicate.bin"])
    assert len(casefold_duplicate) == 108
    with pytest.raises(ValueError, match="exactly 108"):
        _weather4cast_remote_file_set_sha256(casefold_duplicate)


def test_weather_listing_profile_is_suffix_only_and_aggregate_only() -> None:
    archive_suffixes = [".tar", ".tar.gz", ".tgz", ".zip", ".gz", ".bz2", ".xz", ".7z"]
    scientific_suffixes = [".nc", ".npy", ".npz", ".h5", ".hdf5", ".grib", ".grb"]
    metadata_suffixes = [
        ".txt", ".md", ".json", ".csv", ".xml", ".yaml", ".yml", ".sha256", ".html",
    ]
    remote_files = (
        [f"archive-{index}{suffix}" for index, suffix in enumerate(archive_suffixes)]
        + [f"science-{index}{suffix.upper()}" for index, suffix in enumerate(scientific_suffixes)]
        + [f"metadata-{index}{suffix}" for index, suffix in enumerate(metadata_suffixes)]
        + ["other.parquet", "nested/deeper/no-extension"]
    )

    profile = _weather4cast_listing_profile(
        remote_files,
        {"nested", "nested/deeper"},
    )

    assert profile == {
        "adapter_listing_entries": len(remote_files) + 2,
        "adapter_listing_files": len(remote_files),
        "adapter_listing_directories": 2,
        "adapter_listing_archive_files": len(archive_suffixes),
        "adapter_listing_scientific_files": len(scientific_suffixes),
        "adapter_listing_metadata_files": len(metadata_suffixes),
        "adapter_listing_other_files": 2,
        "adapter_listing_max_depth": 3,
    }
    assert all(isinstance(value, int) for value in profile.values())
    assert sum(
        profile[key]
        for key in (
            "adapter_listing_archive_files",
            "adapter_listing_scientific_files",
            "adapter_listing_metadata_files",
            "adapter_listing_other_files",
        )
    ) == profile["adapter_listing_files"]


def test_weather_sftp_long_listing_distinguishes_files_directories_and_control_lines() -> None:
    listing = """\
Connected to fixture.example.
sftp> cd "/official"
sftp> ls -ln ./*
-rw-r--r--    1 1000 1000 123 Jan 01 12:00 ./plain-file.tar
-rw-r--r--    1 1000 1000 456 Jan 01 12:01 ./file with spaces.zip
drwxr-xr-x    2 1000 1000 512 Jan 01 12:02 ./nested-data
-rw-r--r--    1 1000 1000 789 Jan 01 12:03 /official/prefixed.nc
"""
    entries, diagnostics = _parse_weather4cast_sftp_long_listing(listing, "/official")
    assert entries == [
        ("file", "file with spaces.zip"),
        ("directory", "nested-data"),
        ("file", "plain-file.tar"),
        ("file", "prefixed.nc"),
    ]
    assert diagnostics == {
        "raw_lines": 7,
        "control_lines": 3,
        "rejected_lines": 0,
        "files": 3,
        "directories": 1,
    }


def test_weather_sftp_long_listing_strips_exact_recursive_listing_prefix() -> None:
    listing = """\
sftp> ls -ln dynamic
dynamic:
-rw-r--r-- 1 1000 1000 123 Jan 01 12:00 dynamic/chunk-001.tar
drwxr-xr-x 2 1000 1000 512 Jan 01 12:01 dynamic/nested
"""
    entries, diagnostics = _parse_weather4cast_sftp_long_listing(
        listing, "/", "dynamic",
    )
    assert entries == [("file", "chunk-001.tar"), ("directory", "nested")]
    assert diagnostics["rejected_lines"] == 0
    assert diagnostics["control_lines"] == 2
    assert diagnostics["files"] == 1
    assert diagnostics["directories"] == 1


def test_weather_sftp_long_listing_accepts_canonical_absolute_rows() -> None:
    listing = "\n".join(
        [
            "Connected to fixture.example.",
            'sftp> cd "/submitted-root"',
            'sftp> ls -ln "training-data"/*',
            *[
                (
                    f"-rw-r--r-- 1 1000 1000 {1000 + index} "
                    f"Aug 27 09:2{index} /canonical/official/training-data/chunk-{index:03d}.tar"
                )
                for index in range(8)
            ],
        ]
    )

    entries, diagnostics = _parse_weather4cast_sftp_long_listing(
        listing, "/submitted-root", "training-data",
    )

    assert entries == [("file", f"chunk-{index:03d}.tar") for index in range(8)]
    assert diagnostics == {
        "raw_lines": 11,
        "control_lines": 3,
        "rejected_lines": 0,
        "files": 8,
        "directories": 0,
    }


def test_weather_sftp_long_listing_rejects_non_child_canonical_row() -> None:
    listing = (
        "-rw-r--r-- 1 1000 1000 1000 Aug 27 09:20 "
        "/canonical/official/training-data/nested/chunk-000.tar"
    )

    entries, diagnostics = _parse_weather4cast_sftp_long_listing(
        listing, "/submitted-root", "training-data",
    )

    assert entries == []
    assert diagnostics["rejected_lines"] == 1


def test_weather_sftp_root_remote_preserves_canonical_absolute_subdirectory_rows() -> None:
    listing = "\n".join(
        [
            "Connected to fixture.example.",
            'sftp> cd "/"',
            'sftp> cd "training-data"',
            "sftp> ls -ln .",
            *[
                (
                    f"-rw-r--r-- 1 1000 1000 {1000 + index} "
                    f"Aug 27 09:2{index} /opaque/canonical/training-data/chunk-{index:03d}.tar"
                )
                for index in range(8)
            ],
        ]
    )

    entries, diagnostics = _parse_weather4cast_sftp_long_listing(
        listing, "/", "training-data",
    )

    assert entries == [("file", f"chunk-{index:03d}.tar") for index in range(8)]
    assert diagnostics == {
        "raw_lines": 12,
        "control_lines": 4,
        "rejected_lines": 0,
        "files": 8,
        "directories": 0,
    }


def test_weather_sftp_long_listing_rejects_mixed_canonical_parents() -> None:
    listing = """\
-rw-r--r-- 1 1000 1000 1000 Aug 27 09:20 /canonical/alpha/training-data/chunk-000.tar
-rw-r--r-- 1 1000 1000 1001 Aug 27 09:21 /canonical/bravo/training-data/chunk-001.tar
"""

    entries, diagnostics = _parse_weather4cast_sftp_long_listing(
        listing, "/submitted-root", "training-data",
    )

    assert entries == [("file", "chunk-000.tar")]
    assert diagnostics["rejected_lines"] == 1
    assert diagnostics["files"] == 1


def test_weather_sftp_long_listing_accepts_nested_listing_root_canonical_child() -> None:
    listing = (
        "-rw-r--r-- 1 1000 1000 1000 Aug 27 09:20 "
        "/canonical/official/season-2024/training-data/chunk-000.tar"
    )

    entries, diagnostics = _parse_weather4cast_sftp_long_listing(
        listing, "/submitted-root", "season-2024/training-data",
    )

    assert entries == [("file", "chunk-000.tar")]
    assert diagnostics["rejected_lines"] == 0


def test_weather_accelerate_listing_enters_nested_directory_before_listing_dot() -> None:
    accelerated = build_adapter_script("weather4cast", "accelerate")

    assert 'if current_directory:' in accelerated
    assert 'list_commands.append(f"cd {lftp_quote(current_directory)}")' in accelerated
    assert 'list_commands.extend(["ls -ln .", "quit"])' in accelerated
    assert 'ls -ln {lftp_quote(current_directory)}/*' not in accelerated


@pytest.mark.parametrize(
    "mode",
    [
        "crw-rw----",
        "brw-rw----",
        "prw-r-----",
        "srw-rw----",
    ],
)
def test_weather_sftp_long_listing_rejects_canonical_special_files(mode: str) -> None:
    listing = (
        f"{mode} 1 1000 1000 0 Aug 27 09:20 "
        "/canonical/official/training-data/special-entry"
    )

    entries, diagnostics = _parse_weather4cast_sftp_long_listing(
        listing, "/submitted-root", "training-data",
    )

    assert entries == []
    assert diagnostics["rejected_lines"] == 1


def test_weather_sftp_long_listing_skips_symlink_without_following_target() -> None:
    listing = (
        "lrwxrwxrwx 1 owner group 4 Jan 01 12:00 "
        "/canonical/official/training-data/link -> ../../outside"
    )

    entries, diagnostics = _parse_weather4cast_sftp_long_listing(
        listing,
        "/submitted-root",
        "training-data",
    )

    assert entries == []
    assert diagnostics["control_lines"] == 1
    assert diagnostics["rejected_lines"] == 0


@pytest.mark.parametrize(
    "opaque_metadata",
    [
        "? owner group ?",
        r"links domain\owner group/name size",
        "1 owner:role forecast-team+etl 4096",
        "not-a-count 用户 owner;metadata unknown-size",
    ],
)
def test_weather_sftp_long_listing_treats_unused_server_metadata_as_opaque(
    opaque_metadata: str,
) -> None:
    listing = f"-server-mode {opaque_metadata} Aug 27 09:20 ./official-file.tar"

    entries, diagnostics = _parse_weather4cast_sftp_long_listing(listing, "/official")

    assert entries == [("file", "official-file.tar")]
    assert diagnostics["rejected_lines"] == 0


def test_weather_sftp_long_listing_still_rejects_malformed_field_count() -> None:
    entries, diagnostics = _parse_weather4cast_sftp_long_listing(
        "-server-mode opaque metadata without a complete longname row",
        "/official",
    )

    assert entries == []
    assert diagnostics["rejected_lines"] == 1


def test_weather_sftp_long_listing_accepts_quoted_safe_tilde_leaf() -> None:
    listing = "-server-mode opaque owner group size Aug 27 09:20 ./forecast~archive.tar"

    entries, diagnostics = _parse_weather4cast_sftp_long_listing(listing, "/official")

    assert entries == [("file", "forecast~archive.tar")]
    assert diagnostics["rejected_lines"] == 0


def test_weather_sftp_long_listing_accepts_exact_numeric_total_control_line() -> None:
    listing = """\
sftp> cd "/official"
sftp> cd "training-data"
sftp> ls -ln .
total 123456
-server-mode opaque owner group size Aug 27 09:20 ./chunk-000.tar
"""

    entries, diagnostics = _parse_weather4cast_sftp_long_listing(
        listing,
        "/official",
        "training-data",
    )

    assert entries == [("file", "chunk-000.tar")]
    assert diagnostics == {
        "raw_lines": 5,
        "control_lines": 4,
        "rejected_lines": 0,
        "files": 1,
        "directories": 0,
    }


@pytest.mark.parametrize("unsafe_total", ["total", "total -1", "total 1.0", "total all"])
def test_weather_sftp_long_listing_rejects_non_numeric_total_lines(unsafe_total: str) -> None:
    entries, diagnostics = _parse_weather4cast_sftp_long_listing(unsafe_total, "/official")

    assert entries == []
    assert diagnostics["rejected_lines"] == 1


@pytest.mark.parametrize(
    "unsafe_line",
    [
        "-rw-r--r-- 1 owner group 1 Jan 01 12:00 ../escape",
        "-rw-r--r-- 1 owner group 1 Jan 01 12:00 ./nested/escape",
        "-rw-r--r-- 1 owner group 1 Jan 01 12:00 /official-prefix/not-exact",
        "unknown control output",
    ],
)
def test_weather_sftp_long_listing_rejects_unsafe_or_unknown_rows(unsafe_line: str) -> None:
    entries, diagnostics = _parse_weather4cast_sftp_long_listing(unsafe_line, "/official")
    assert entries == []
    assert diagnostics["rejected_lines"] == 1


@pytest.mark.parametrize(
    ("expected_reason", "listing"),
    [
        ("total_like_format", "total /private/reject-marker"),
        ("field_count", "-rw-r--r-- private-reject-marker"),
        (
            "special_type",
            "crw-rw---- 1 owner group 0 Jan 01 12:00 ./private-reject-marker",
        ),
        (
            "unknown_type",
            "zrw-r--r-- 1 owner group 1 Jan 01 12:00 ./private-reject-marker",
        ),
        (
            "path",
            "-rw-r--r-- 1 owner group 1 Jan 01 12:00 ../private-reject-marker",
        ),
        (
            "leading_dash",
            "-rw-r--r-- 1 owner group 1 Jan 01 12:00 -private-reject-marker",
        ),
        (
            "colon",
            "-rw-r--r-- 1 owner group 1 Jan 01 12:00 private:reject-marker",
        ),
        (
            "bracket",
            "-rw-r--r-- 1 owner group 1 Jan 01 12:00 private[reject-marker]",
        ),
        (
            "percent",
            "-rw-r--r-- 1 owner group 1 Jan 01 12:00 private%reject-marker",
        ),
        (
            "ampersand",
            "-rw-r--r-- 1 owner group 1 Jan 01 12:00 private&reject-marker",
        ),
        (
            "hash",
            "-rw-r--r-- 1 owner group 1 Jan 01 12:00 private#reject-marker",
        ),
        (
            "apostrophe",
            "-rw-r--r-- 1 owner group 1 Jan 01 12:00 private'reject-marker",
        ),
        (
            "exclamation",
            "-rw-r--r-- 1 owner group 1 Jan 01 12:00 private!reject-marker",
        ),
        (
            "unicode_character",
            "-rw-r--r-- 1 owner group 1 Jan 01 12:00 private-数据",
        ),
        (
            "duplicate",
            "\n".join(
                [
                    "-rw-r--r-- 1 owner group 1 Jan 01 12:00 ./private-reject-marker",
                    "-rw-r--r-- 1 owner group 1 Jan 01 12:01 ./PRIVATE-REJECT-MARKER",
                ]
            ),
        ),
    ],
)
def test_weather_sftp_long_listing_reports_only_safe_reject_reason(
    expected_reason: str,
    listing: str,
) -> None:
    entries, diagnostics = _parse_weather4cast_sftp_long_listing(listing, "/official")

    assert diagnostics["rejected_lines"] == 1
    assert diagnostics["reject_reason"] == expected_reason
    serialized = json.dumps(diagnostics, sort_keys=True)
    assert "private-reject-marker" not in serialized.casefold()
    assert "/private/" not in serialized


def test_weather_accelerate_uses_safe_parser_reason_as_fallback_enum() -> None:
    accelerated = build_adapter_script("weather4cast", "accelerate")

    assert 'diagnostics.get("reject_reason", "unknown_type")' in accelerated
    assert "listing_parse_rejected_{reject_reason}" in accelerated


def test_weather_accelerate_has_recursive_resume_worker_contract() -> None:
    accelerated = build_adapter_script("weather4cast", "accelerate")

    assert 'if test "$ACCELERATE" = "1"; then' in accelerated
    assert '"adapter_transfer_mode": "openssh_recursive_get"' in accelerated
    assert '"get -aR ."' in accelerated
    assert 'state("RUNNING", "transferring")' in accelerated
    assert 'state("RUNNING", "validating")' in accelerated
    assert '"expected_file_count": 108' in accelerated
    assert 'len(rows) != 108' in accelerated
    assert 'kill -TERM -- "-$EXISTING_PID"' in accelerated
    assert 'existing Weather4cast worker did not stop before acceleration' in accelerated
    assert 'secret_values_logged": False' in accelerated
    assert 'echo $password' not in accelerated.casefold()
    assert "--password" not in accelerated.casefold()

    embedded_python = re.findall(r"<<'PY'\n(.*?)\nPY\n", accelerated, flags=re.DOTALL)
    assert embedded_python
    for index, body in enumerate(embedded_python):
        compile(body, f"weather4cast-accelerate-embedded-{index}.py", "exec")


def test_weather_status_prefers_a_current_valid_manifest_over_stale_failure_state() -> None:
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("bash unavailable on this Windows test host")

    status_script = build_adapter_script("weather4cast", "status")
    harness = r'''\
set -euo pipefail
TMP_ROOT=$(mktemp -d)
trap 'rm -rf "$TMP_ROOT"' EXIT
export EVOMIND_COMPETITION_DATA_ROOT="$TMP_ROOT/weather4cast"
export EVOMIND_COMPETITION_RECEIPT="$TMP_ROOT/receipt.json"
mkdir -p "$EVOMIND_COMPETITION_DATA_ROOT/.evomind/download-job"
python3 - "$EVOMIND_COMPETITION_DATA_ROOT" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
attempt_id = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
rows = []
for index in range(108):
    path = root / "official" / f"file-{index:03d}.bin"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = bytes([(index % 251) + 1])
    path.write_bytes(payload)
    rows.append({"path": path.relative_to(root).as_posix(), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
smoke = {item["path"]: {"ok": True, "loader": "nonempty_binary"} for item in rows}
remote_paths = [item["path"][len("official/"):] for item in rows]
remote_file_set_sha256 = hashlib.sha256(
    ("\n".join(sorted(remote_paths, key=str.casefold)) + "\n").encode("utf-8")
).hexdigest()
manifest = {
    "schema": "evomind.competition_data_manifest.v1",
    "competition": "weather4cast",
    "attempt_id": attempt_id,
    "file_count": 108,
    "total_bytes": 108,
    "files": rows,
    "loader_smoke": smoke,
    "duplicate_files": 0,
    "expected_file_count": 108,
    "skipped_directory_count": 0,
    "remote_file_set_sha256": remote_file_set_sha256,
    "rules_accepted": False,
    "secret_values_logged": False,
}
manifest_path = root / ".evomind" / "data-manifest.json"
manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
job = root / ".evomind" / "download-job"
attempt = job / "attempts" / attempt_id
attempt.mkdir(parents=True, exist_ok=True)
(job / "current-attempt.json").write_text(json.dumps({
    "schema": "evomind.weather4cast_current_attempt.v1",
    "attempt_id": attempt_id,
    "pid": 1,
}) + "\n", encoding="utf-8")
(attempt / "state.json").write_text(json.dumps({
    "schema": "evomind.weather4cast_download_state.v2",
    "attempt_id": attempt_id,
    "status": "FULL_DATA_READY",
    "phase": "complete",
    "failure_code": "",
    "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    "adapter_listing_status": "ok",
    "adapter_listing_skipped_directories": 0,
    "adapter_listing_timeout_exhaustions": 0,
    "rules_accepted": False,
    "secret_values_logged": False,
}) + "\n", encoding="utf-8")
(attempt / "exit-code").write_text("0\n", encoding="utf-8")
(job / "state.json").write_text(
    json.dumps({
        "schema": "evomind.weather4cast_download_state.v1",
        "status": "DOWNLOAD_FAILED",
        "phase": "failed",
        "failure_code": "SFTP_AUTH_FAILED",
        "adapter_listing_status": "ok",
        "adapter_listing_skipped_directories": 0,
        "adapter_listing_timeout_exhaustions": 0,
        "rules_accepted": False,
    }) + "\n",
    encoding="utf-8",
)
(job / "exit-code").write_text("1\n", encoding="utf-8")
PY
''' + status_script + r'''
python3 - "$EVOMIND_COMPETITION_RECEIPT" <<'PY'
import json, sys
receipt = json.load(open(sys.argv[1], encoding="utf-8"))
print(json.dumps({"status": receipt.get("status"), "failure_code": receipt.get("failure_code", "")}))
PY
'''
    completed = subprocess.run([bash, "-s"], input=harness.encode("utf-8"), capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    observed = json.loads(completed.stdout.decode("utf-8", "replace").strip().splitlines()[-1])
    assert observed == {"status": "FULL_DATA_READY", "failure_code": ""}


@pytest.mark.parametrize(
    "evidence_tamper",
    ["missing", "stale_extra", "state_skipped", "state_timeout_bool", "listing_status"],
)
def test_weather_status_rejects_false_full_when_evidence_binding_differs(
    evidence_tamper: str,
) -> None:
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("bash unavailable on this Windows test host")

    status_script = build_adapter_script("weather4cast", "status")
    harness = r'''\
set -euo pipefail
TMP_ROOT=$(mktemp -d)
trap 'rm -rf "$TMP_ROOT"' EXIT
export EVOMIND_COMPETITION_DATA_ROOT="$TMP_ROOT/weather4cast"
export EVOMIND_COMPETITION_RECEIPT="$TMP_ROOT/receipt.json"
JOB="$EVOMIND_COMPETITION_DATA_ROOT/.evomind/download-job"
ATTEMPT="cccccccccccccccccccccccccccccccc"
mkdir -p "$JOB/attempts/$ATTEMPT"
python3 - "$EVOMIND_COMPETITION_DATA_ROOT" "$ATTEMPT" ''' + json.dumps(evidence_tamper) + r''' <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1]); attempt_id = sys.argv[2]; tamper = sys.argv[3]
rows = []
for index in range(108):
    path = root / "official" / f"file-{index:03d}.bin"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = bytes([(index % 251) + 1])
    path.write_bytes(payload)
    rows.append({
        "path": path.relative_to(root).as_posix(),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    })
remote_paths = [item["path"][len("official/"):] for item in rows]
if tamper == "missing":
    bound_remote_paths = remote_paths[:-1]
elif tamper == "stale_extra":
    bound_remote_paths = [*remote_paths[:-1], "stale-extra.bin"]
else:
    bound_remote_paths = remote_paths
remote_hash = hashlib.sha256(
    ("\n".join(sorted(bound_remote_paths, key=str.casefold)) + "\n").encode("utf-8")
).hexdigest()
smoke = {item["path"]: {"ok": True, "loader": "nonempty_binary"} for item in rows}
manifest = {
    "schema": "evomind.competition_data_manifest.v1",
    "competition": "weather4cast",
    "attempt_id": attempt_id,
    "file_count": 108,
    "total_bytes": 108,
    "files": rows,
    "loader_smoke": smoke,
    "duplicate_files": 0,
    "expected_file_count": 108,
    "skipped_directory_count": 0,
    "remote_file_set_sha256": remote_hash,
    "rules_accepted": False,
    "secret_values_logged": False,
}
manifest_path = root / ".evomind" / "data-manifest.json"
manifest_path.parent.mkdir(parents=True, exist_ok=True)
manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
job = root / ".evomind" / "download-job"
(job / "current-attempt.json").write_text(json.dumps({
    "schema": "evomind.weather4cast_current_attempt.v1",
    "attempt_id": attempt_id,
    "pid": 1,
}) + "\n", encoding="utf-8")
attempt = job / "attempts" / attempt_id
(attempt / "state.json").write_text(json.dumps({
    "schema": "evomind.weather4cast_download_state.v2",
    "attempt_id": attempt_id,
    "status": "FULL_DATA_READY",
    "phase": "complete",
    "failure_code": "",
    "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    "adapter_listing_status": "ok_with_skips" if tamper == "listing_status" else "ok",
    "adapter_listing_skipped_directories": 1 if tamper == "state_skipped" else 0,
    "adapter_listing_timeout_exhaustions": False if tamper == "state_timeout_bool" else 0,
    "rules_accepted": False,
    "secret_values_logged": False,
}) + "\n", encoding="utf-8")
(attempt / "exit-code").write_text("0\n", encoding="utf-8")
PY
''' + status_script + r'''
python3 - "$EVOMIND_COMPETITION_RECEIPT" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8")).get("status"))
PY
'''
    completed = subprocess.run(
        [bash, "-s"], input=harness.encode("utf-8"), capture_output=True, check=False
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    assert completed.stdout.decode("utf-8", "replace").strip().splitlines()[-1] != (
        "FULL_DATA_READY"
    )


def test_weather_status_never_falls_back_when_current_attempt_selector_is_invalid() -> None:
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("bash unavailable on this Windows test host")

    status_script = build_adapter_script("weather4cast", "status")
    harness = r'''\
set -euo pipefail
TMP_ROOT=$(mktemp -d)
trap 'rm -rf "$TMP_ROOT"' EXIT
export EVOMIND_COMPETITION_DATA_ROOT="$TMP_ROOT/weather4cast"
export EVOMIND_COMPETITION_RECEIPT="$TMP_ROOT/receipt.json"
mkdir -p "$EVOMIND_COMPETITION_DATA_ROOT/.evomind/download-job"
printf '{"schema":"evomind.weather4cast_current_attempt.v1","attempt_id":"invalid"}\n' > "$EVOMIND_COMPETITION_DATA_ROOT/.evomind/download-job/current-attempt.json"
''' + status_script + r'''
python3 - "$EVOMIND_COMPETITION_RECEIPT" <<'PY'
import json, sys
receipt = json.load(open(sys.argv[1], encoding="utf-8"))
print(receipt.get("status"))
PY
'''
    completed = subprocess.run([bash, "-s"], input=harness.encode("utf-8"), capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    assert completed.stdout.decode("utf-8", "replace").strip().splitlines()[-1] == "VALIDATION_FAILED"


def test_weather_running_receipt_does_not_inherit_a_previous_attempt_manifest() -> None:
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("bash unavailable on this Windows test host")

    status_script = build_adapter_script("weather4cast", "status")
    harness = r'''\
set -euo pipefail
TMP_ROOT=$(mktemp -d)
trap 'rm -rf "$TMP_ROOT"' EXIT
export EVOMIND_COMPETITION_DATA_ROOT="$TMP_ROOT/weather4cast"
export EVOMIND_COMPETITION_RECEIPT="$TMP_ROOT/receipt.json"
JOB="$EVOMIND_COMPETITION_DATA_ROOT/.evomind/download-job"
CURRENT="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
mkdir -p "$JOB/attempts/$CURRENT"
printf '{"schema":"evomind.weather4cast_current_attempt.v1","attempt_id":"%s","pid":%s}\n' "$CURRENT" "$$" > "$JOB/current-attempt.json"
printf '{"schema":"evomind.weather4cast_download_state.v2","attempt_id":"%s","status":"RUNNING","phase":"downloading","failure_code":"","rules_accepted":false,"secret_values_logged":false}\n' "$CURRENT" > "$JOB/attempts/$CURRENT/state.json"
printf '%s\n' "$$" > "$JOB/attempts/$CURRENT/pid"
mkdir -p "$EVOMIND_COMPETITION_DATA_ROOT/.evomind"
printf '{"schema":"evomind.competition_data_manifest.v1","competition":"weather4cast","attempt_id":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","file_count":108,"total_bytes":108,"files":[],"loader_smoke":{},"expected_file_count":108}\n' > "$EVOMIND_COMPETITION_DATA_ROOT/.evomind/data-manifest.json"
''' + status_script + r'''
python3 - "$EVOMIND_COMPETITION_RECEIPT" <<'PY'
import json, sys
receipt = json.load(open(sys.argv[1], encoding="utf-8"))
print(json.dumps({
    "status": receipt.get("status"),
    "files": receipt.get("files"),
    "bytes": receipt.get("bytes"),
    "manifest_sha256": receipt.get("manifest_sha256"),
    "loader_smoke": receipt.get("loader_smoke"),
}))
PY
'''
    completed = subprocess.run([bash, "-s"], input=harness.encode("utf-8"), capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    observed = json.loads(completed.stdout.decode("utf-8", "replace").strip().splitlines()[-1])
    assert observed == {
        "status": "RUNNING",
        "files": 0,
        "bytes": 0,
        "manifest_sha256": "",
        "loader_smoke": {},
    }


def test_kaggle_adapters_use_official_slugs_resumable_download_and_real_loaders() -> None:
    expected = {
        "cure_bench": "cure-bench",
        "ariel_2025": "ariel-data-challenge-2025",
        "open_polymer": "neurips-open-polymer-prediction-2025",
    }
    for competition, slug in expected.items():
        script = build_adapter_script(competition, "prepare")
        assert "EVOMIND_SECRET_KAGGLE_FILE" in script
        assert slug in script
        assert "Range" in script
        assert "Content-Range" in script
        assert "kaggle_range_not_honored" in script
        assert "kaggle_range_contract_invalid" in script
        assert "official_archive_invalid_after_resume" in script
        assert "zipfile.is_zipfile" in script
        assert "archive_sha256" in script
        assert "unsafe_archive_member" in script
        assert "rules_accepted" in script
        assert "secret_values_logged" in script
        status_script = build_adapter_script(competition, "status")
        assert 'payload["partial_archive_bytes"]' in status_script
        assert 'payload["worker_alive"]' in status_script
        assert 'payload["worker_exit_code"]' in status_script
        assert 'value="DOWNLOAD_FAILED"' in status_script
        assert "persistent Kaggle worker is no longer alive" in status_script


def test_kaggle_worker_resumes_after_clean_eof_leaves_a_truncated_zip(tmp_path) -> None:
    import http.server
    import io
    import sys
    import threading
    import types
    import zipfile

    script = build_adapter_script("cure_bench", "prepare")
    bodies = re.findall(r"<<'PY'[^\n]*\n(.*?)\nPY\n", script, flags=re.DOTALL)
    worker = next(body for body in bodies if "def download(headers):" in body)

    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("train.csv", "id,target\n1,0\n2,1\n")
        archive.writestr("test.csv", "id\n3\n")
        archive.writestr("sample_submission.csv", "id,target\n3,0\n")
    archive_bytes = archive_buffer.getvalue()
    split = len(archive_bytes) // 2
    requests: list[str] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib callback name
            supplied = self.headers.get("Range") or ""
            requests.append(supplied)
            if not supplied:
                payload = archive_bytes[:split]
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
            else:
                assert supplied == f"bytes={split}-"
                payload = archive_bytes[split:]
                self.send_response(206)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Content-Range", f"bytes {split}-{len(archive_bytes) - 1}/{len(archive_bytes)}")
            self.send_header("Content-Type", "application/zip")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}/download"
    worker = re.sub(
        r'endpoint = f"https://www\.kaggle\.com/api/v1/competitions/data/download-all/\{slug\}"',
        f'endpoint = {endpoint!r}',
        worker,
        count=1,
    )

    root = tmp_path / "cure_bench"
    root.mkdir()
    (root / ".evomind" / "download-job").mkdir(parents=True)
    secret = tmp_path / "secret.json"
    secret.write_text(
        json.dumps({"schema": "evomind.run_secret_payload.v1", "purpose": "kaggle_api", "token": "fixture"}),
        encoding="utf-8",
    )
    fcntl = types.ModuleType("fcntl")
    fcntl.LOCK_EX = 1
    fcntl.LOCK_UN = 2
    fcntl.flock = lambda *_args: None
    previous_fcntl = sys.modules.get("fcntl")
    previous_argv = sys.argv
    sys.modules["fcntl"] = fcntl
    sys.argv = ["worker.py", str(secret), str(root), "cure_bench", "cure-bench", "https://example.invalid/"]
    try:
        exec(compile(worker, "kaggle-worker.py", "exec"), {"__name__": "__main__"})
    finally:
        sys.argv = previous_argv
        if previous_fcntl is None:
            sys.modules.pop("fcntl", None)
        else:
            sys.modules["fcntl"] = previous_fcntl
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert requests == ["", f"bytes={split}-"]
    archive_path = root / "official" / "cure-bench.zip"
    assert zipfile.is_zipfile(archive_path)
    state = json.loads((root / ".evomind" / "download-job" / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "FULL_DATA_READY"
    manifest = json.loads((root / ".evomind" / "data-manifest.json").read_text(encoding="utf-8"))
    assert manifest["file_count"] == 3
    assert all(item["ok"] is True for item in manifest["loader_smoke"].values())


def test_generated_bash_adapters_are_syntax_valid_when_bash_is_available() -> None:
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("bash unavailable on this Windows test host")
    for competition in COMPETITION_CATALOG:
        for action in ("prepare", "status"):
            completed = subprocess.run(
                [bash, "-n"],
                input=build_adapter_script(competition, action).encode("utf-8"),
                capture_output=True,
                check=False,
            )
            assert completed.returncode == 0, f"{competition}/{action}: {completed.stderr.decode('utf-8', 'replace')}"

    for action in ("prepare", "status"):
        script = build_adapter_script("weather4cast", action)
        embedded_python = re.findall(r"<<'PY'\n(.*?)\nPY\n", script, flags=re.DOTALL)
        assert embedded_python, f"weather4cast/{action}: no embedded Python bodies found"
        for index, body in enumerate(embedded_python):
            compile(body, f"weather4cast-{action}-embedded-{index}.py", "exec")


def test_receipt_validation_rejects_root_drift_and_secret_output() -> None:
    loader_smoke = {
        "SecretMafia-v0": {"ok": True},
        "Codenames-v0": {"ok": True},
        "ColonelBlotto-v0": {"ok": True},
        "ThreePlayerIPD-v0": {"ok": True},
    }
    runtime_dependencies = {
        "corpora/words.zip": {
            "source": "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/words.zip",
            "sha256": "54ed02917d6771dcc3e8141218960d020947f7f2ccfd9ac9b320979349746015",
        },
        "taggers/averaged_perceptron_tagger_eng.zip": {
            "source": "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/taggers/averaged_perceptron_tagger_eng.zip",
            "sha256": "6025f530624335c67d6547d44757b357b4e79bae030a0383e9887a92c1718f0b",
        },
    }
    base = {
        "schema": "evomind.competition_data_receipt.v1",
        "competition": "mindgames",
        "status": "FULL_DATA_READY",
        "persistent_root": persistent_root("mindgames"),
        "secret_values_logged": False,
        "expected_environment_count": 4,
        "textarena_version": "0.7.4",
        "runtime_dependencies": runtime_dependencies,
        "loader_smoke": loader_smoke,
    }
    assert validate_receipt("mindgames", base)["status"] == "FULL_DATA_READY"
    with pytest.raises(ValueError, match="root mismatch"):
        validate_receipt("mindgames", {**base, "persistent_root": persistent_root("e2lmc")})
    with pytest.raises(ValueError, match="no-secret-output"):
        validate_receipt("mindgames", {**base, "secret_values_logged": True})
    with pytest.raises(ValueError, match="dependency or loader evidence"):
        validate_receipt("mindgames", {**base, "runtime_dependencies": {}})
    with pytest.raises(ValueError, match="dependency or loader evidence"):
        validate_receipt("mindgames", {**base, "loader_smoke": {**loader_smoke, "SecretMafia-v0": {"ok": False}}})


def test_weather_full_receipt_requires_attempt_bound_manifest_and_all_loaders() -> None:
    loader_smoke = {
        f"official/file-{index:03d}.bin": {"ok": True, "loader": "nonempty_binary"}
        for index in range(108)
    }
    receipt = {
        "schema": "evomind.competition_data_receipt.v1",
        "competition": "weather4cast",
        "status": "FULL_DATA_READY",
        "persistent_root": persistent_root("weather4cast"),
        "secret_values_logged": False,
        "attempt_id": "a" * 32,
        "phase": "complete",
        "failure_code": "",
        "files": 108,
        "expected_file_count": 108,
        "manifest_sha256": "b" * 64,
        "remote_file_set_sha256": "c" * 64,
        "skipped_directory_count": 0,
        "adapter_listing_status": "ok",
        "adapter_listing_skipped_directories": 0,
        "adapter_listing_timeout_exhaustions": 0,
        "loader_smoke": loader_smoke,
        "worker_exit_code": 0,
    }
    assert validate_receipt("weather4cast", receipt)["status"] == "FULL_DATA_READY"
    for field, invalid in (
        ("attempt_id", ""),
        ("files", 107),
        ("manifest_sha256", ""),
        ("remote_file_set_sha256", ""),
        ("skipped_directory_count", 1),
        ("skipped_directory_count", False),
        ("adapter_listing_status", "ok_with_skips"),
        ("adapter_listing_skipped_directories", 1),
        ("adapter_listing_skipped_directories", False),
        ("adapter_listing_timeout_exhaustions", 1),
        ("adapter_listing_timeout_exhaustions", False),
        ("worker_exit_code", 1),
        ("failure_code", "SFTP_AUTH_FAILED"),
        ("loader_smoke", {**loader_smoke, "official/file-000.bin": {"ok": False}}),
    ):
        with pytest.raises(ValueError, match="Weather4cast FULL_DATA_READY"):
            validate_receipt("weather4cast", {**receipt, field: invalid})
