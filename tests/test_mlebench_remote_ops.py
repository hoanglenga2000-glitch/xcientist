"""Fast fail-closed tests for the MLE-Bench remote controller."""
from __future__ import annotations

import hashlib
import inspect
import io
import json
import tarfile
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath

import pytest

from scripts import mlebench_remote_ops as ops


def _write_bundle(path: Path) -> tuple[str, str]:
    payload = b"print('ok')\n"
    payload_sha = hashlib.sha256(payload).hexdigest()
    manifest = {
        "schema": ops.EXPECTED_MANIFEST_SCHEMA,
        "kaggle_submission_enabled": False,
        "human_gate_preserved": True,
        "files": {"scripts/a.py": payload_sha},
    }
    with tarfile.open(path, "w:gz") as archive:
        for name, data in (
            ("bundle_manifest.json", json.dumps(manifest).encode()),
            ("scripts/a.py", payload),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return hashlib.sha256(path.read_bytes()).hexdigest(), payload_sha


def _write_gate(path: Path, *, created_at: datetime, passed: bool = True) -> None:
    samples = [
        {
            "index": index,
            "eligible": passed,
            "hold_reasons": [] if passed else ["fixture_hold"],
            "probe_errors": [],
        }
        for index in range(1, 6)
    ]
    path.write_text(json.dumps({
        "schema": ops.EXPECTED_GATE_SCHEMA,
        "created_at": created_at.isoformat(),
        "policy": {"samples_required": 5, "sample_interval_seconds": 15},
        "dedicated_root_writable": True,
        "passed": passed,
        "samples": samples,
        "identity": {
            "host_uuid": "host-uuid",
            "gpu_uuids": ["gpu-uuid"],
            "stable": True,
        },
        "hold_reasons": [] if passed else ["fixture_hold"],
        "other_processes_modified": False,
        "signals_sent": 0,
    }), encoding="utf-8")


def test_local_bundle_verifies_hashes_and_human_gate(tmp_path):
    bundle = tmp_path / "bundle.tar.gz"
    archive_sha, _ = _write_bundle(bundle)
    result = ops.verify_local_bundle(bundle, expected_sha256=archive_sha)
    assert result["passed"] is True
    assert result["manifest_hash_count"] == 1
    assert result["human_gate_preserved"] is True


def test_local_bundle_rejects_manifest_hash_drift(tmp_path):
    bundle = tmp_path / "bundle.tar.gz"
    archive_sha, _ = _write_bundle(bundle)
    with tarfile.open(bundle, "r:gz") as archive:
        members = {item.name: archive.extractfile(item).read() for item in archive if item.isfile()}
    manifest = json.loads(members["bundle_manifest.json"])
    manifest["files"]["scripts/a.py"] = "0" * 64
    with tarfile.open(bundle, "w:gz") as archive:
        for name, data in (
            ("bundle_manifest.json", json.dumps(manifest).encode()),
            ("scripts/a.py", members["scripts/a.py"]),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    archive_sha = hashlib.sha256(bundle.read_bytes()).hexdigest()
    with pytest.raises(ops.RemoteOpsError, match="Manifest hash mismatch"):
        ops.verify_local_bundle(bundle, expected_sha256=archive_sha)


def test_local_bundle_rejects_unmanifested_member(tmp_path):
    bundle = tmp_path / "bundle.tar.gz"
    _write_bundle(bundle)
    with tarfile.open(bundle, "r:gz") as archive:
        members = {item.name: archive.extractfile(item).read() for item in archive if item.isfile()}
    members["scripts/untracked.py"] = b"print('untracked')\n"
    with tarfile.open(bundle, "w:gz") as archive:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    archive_sha = hashlib.sha256(bundle.read_bytes()).hexdigest()
    with pytest.raises(ops.RemoteOpsError, match="differ from the manifest"):
        ops.verify_local_bundle(bundle, expected_sha256=archive_sha)


def test_gate_requires_five_fresh_clean_identity_bound_samples(tmp_path):
    now = datetime.now(timezone.utc)
    gate = tmp_path / "gate.json"
    _write_gate(gate, created_at=now)
    result = ops.validate_gate_report(gate, max_age_seconds=600, now=now)
    assert result["sample_count"] == 5

    _write_gate(gate, created_at=now - timedelta(hours=1))
    with pytest.raises(ops.RemoteOpsError, match="stale"):
        ops.validate_gate_report(gate, max_age_seconds=600, now=now)

    _write_gate(gate, created_at=now, passed=False)
    with pytest.raises(ops.RemoteOpsError, match="did not pass"):
        ops.validate_gate_report(gate, max_age_seconds=600, now=now)


def _healthy_gate_sample(*, used_mib: int = 4, utilization: int = 0) -> dict:
    return {
        "host_uuid": "host-job89508",
        "gpus": [{
            "index": 0,
            "name": "A800",
            "uuid": "gpu-job89508",
            "memory_total_mib": 81920,
            "memory_used_mib": used_mib,
            "memory_free_mib": 81920 - used_mib,
            "utilization_percent": utilization,
        }],
        "compute_apps": [],
        "processes": [],
        "nvidia_fd_processes": [],
        "state_records": [],
        "probe_errors": [],
        "read_only": True,
        "signals_sent": 0,
        "other_processes_modified": False,
    }


def _evaluate_gate(samples: list[dict]) -> dict:
    return ops.evaluate_gpu_gate_samples(
        samples,
        expected_host_uuid="host-job89508",
        expected_gpu_uuid="gpu-job89508",
        require_expected_identity=True,
        min_free_memory_mib=60 * 1024,
        max_other_process_memory_mib=8 * 1024,
        max_memory_growth_mib=256,
        max_utilization_percent=5,
    )


def test_composite_gate_passes_five_stable_identity_bound_samples():
    samples = [_healthy_gate_sample() for _ in range(5)]

    result = _evaluate_gate(samples)

    assert result["passed"] is True
    assert result["hold_reasons"] == []
    assert result["identity"]["stable"] is True
    assert all(sample["eligible"] is True for sample in samples)


@pytest.mark.parametrize("state_code", ["T", "t", "D", "Z"])
def test_composite_gate_holds_blocked_process_and_nvidia_fd_states(state_code):
    samples = [_healthy_gate_sample() for _ in range(5)]
    samples[2]["processes"] = [{"pid": 123, "state_code": state_code}]
    samples[2]["nvidia_fd_processes"] = [{
        "pid": 123,
        "state_code": state_code,
        "nvidia_fds": ["/dev/nvidia0"],
    }]

    result = _evaluate_gate(samples)

    assert result["passed"] is False
    assert "blocked_process_state" in result["hold_reasons"]
    assert "blocked_nvidia_fd_process" in result["hold_reasons"]


def test_composite_gate_holds_stale_state_log_mismatch_and_identity_drift():
    samples = [_healthy_gate_sample() for _ in range(5)]
    samples[1]["state_records"] = [{
        "pid": 123,
        "process": {"exists": True, "state_code": "S"},
        "mismatch_reasons": ["declared_running_activity_stale"],
    }]
    samples[4]["host_uuid"] = "unexpected-host"

    result = _evaluate_gate(samples)

    assert result["passed"] is False
    assert "state_log_freshness_mismatch" in result["hold_reasons"]
    assert "host_uuid_mismatch" in result["hold_reasons"]
    assert "host_identity_unstable" in result["hold_reasons"]


def test_composite_gate_preserves_but_does_not_block_on_historical_missing_process_state():
    samples = [_healthy_gate_sample() for _ in range(5)]
    for sample in samples:
        sample["state_records"] = [{
            "pid": 123,
            "process": {"exists": False, "state_code": ""},
            "state_age_seconds": 300_000,
            "mismatch_reasons": [
                "declared_running_process_missing",
                "declared_running_activity_stale",
            ],
        }]

    result = _evaluate_gate(samples)

    assert result["passed"] is True
    assert result["hold_reasons"] == []


def test_linux_process_state_classification_never_treats_blocked_states_as_running():
    assert ops.classify_linux_process_state("R", exists=True) == "running"
    assert ops.classify_linux_process_state("S", exists=True) == "running"
    for state_code in ("T", "t", "D", "Z", "X"):
        assert ops.classify_linux_process_state(state_code, exists=True) == "blocked"
    assert ops.classify_linux_process_state(None, exists=False) == "stopped"


def test_remote_gate_probe_is_read_only_and_covers_proc_fd_freshness_and_identity():
    source = ops._gpu_gate_probe_source(300)

    compile(source, "<gpu-gate-probe>", "exec")
    for token in (
        'pathlib.Path("/proc")',
        'fields.get("State", "")',
        'target.startswith("/dev/nvidia")',
        '"/sys/class/dmi/id/product_uuid"',
        '"--query-gpu=index,name,uuid',
        '"state_age_seconds"',
        '"log_age_seconds"',
        '"activity_age_seconds"',
        '"signals_sent": 0',
        '"other_processes_modified": False',
    ):
        assert token in source
    assert "os.kill" not in source


def test_gpu_gate_collects_five_configurable_samples_without_remote_signals(monkeypatch):
    class Client:
        closed = False

        def close(self):
            self.closed = True

    client = Client()
    config = ops.GpuSshConfig(
        host="fixture-host",
        port=22,
        username="fixture-user",
        password="fixture-secret",
        known_hosts_path="fixture-known-hosts",
        credential_profile="job89508",
        expected_host_uuid="host-job89508",
        expected_gpu_uuid="gpu-job89508",
    )
    calls = []

    monkeypatch.setattr(ops, "load_gpu_ssh_config", lambda: config)
    monkeypatch.setattr(ops, "_connect", lambda supplied: client)

    def fake_run(_client, command, *, timeout):
        calls.append((command, timeout))
        if command.startswith("test -d"):
            return 0, "ROOT_EXISTS=1\nROOT_WRITABLE=1\n", ""
        return 0, json.dumps(_healthy_gate_sample()), ""

    monkeypatch.setattr(ops, "_run_remote", fake_run)

    report = ops.sample_gpu_idle_gate(samples_required=5, interval_seconds=0)

    assert report["passed"] is True
    assert report["policy"]["samples_required"] == 5
    assert len(report["samples"]) == 5
    assert len(calls) == 6
    assert report["signals_sent"] == 0
    assert report["other_processes_modified"] is False
    assert client.closed is True


def test_remote_paths_and_identifiers_fail_closed():
    assert ops.ensure_remote_path(ops.REMOTE_OUTPUT_ROOT).startswith(ops.ALLOWED_GPU_REMOTE_ROOT)
    assert ops.ensure_remote_path(ops.REMOTE_UNIFIED_SITE_PACKAGES).startswith(ops.ALLOWED_GPU_REMOTE_ROOT)
    assert ops.ensure_remote_path(ops.REMOTE_GRADER_SITE_PACKAGES).startswith(ops.ALLOWED_GPU_REMOTE_ROOT)
    assert ops.ensure_remote_path(ops.REMOTE_SCIKIT_LEARN_WHEEL).startswith(ops.ALLOWED_GPU_REMOTE_ROOT)
    assert ops.ensure_remote_path(ops.REMOTE_SHARED_TORCH_HOME).startswith(
        ops.ALLOWED_GPU_REMOTE_ROOT
    )
    assert ops.ensure_remote_path(ops.REMOTE_SHARED_XDG_CACHE_HOME).startswith(
        ops.ALLOWED_GPU_REMOTE_ROOT
    )
    assert ops.ensure_remote_path(ops.REMOTE_SHARED_HF_HOME).startswith(
        ops.ALLOWED_GPU_REMOTE_ROOT
    )
    assert ops.ensure_remote_path(ops.REMOTE_RUNTIME_TMP_ROOT).startswith(
        ops.ALLOWED_GPU_REMOTE_ROOT
    )
    with pytest.raises(ops.RemoteOpsError, match="outside"):
        ops.ensure_remote_path("/tmp/out")
    with pytest.raises(ops.RemoteOpsError, match="characters"):
        ops.validate_run_id("../bad")
    with pytest.raises(ops.RemoteOpsError, match="characters"):
        ops.parse_competitions("good,bad/id")
    with pytest.raises(ops.RemoteOpsError, match="Unsupported"):
        ops.parse_waves("Wave9")


def test_runner_argv_preserves_scope_seed_and_human_gate():
    release = ops.release_dir("a" * 64)
    argv = ops.build_runner_argv(
        release=release,
        run_id="may2022_s42_test",
        waves=["Wave0"],
        competitions=["tabular-playground-series-may-2022"],
        seed=42,
        optimization_plan=f"{release}/plans/may2022_gpt56_review_current.json",
        resume=False,
    )
    joined = " ".join(argv)
    assert "--waves Wave0" in joined
    assert "--wave2-fast-kernels" in argv
    assert "--competitions tabular-playground-series-may-2022" in joined
    assert "--seed 42" in joined
    assert argv.count("--candidate-only") == 1
    assert "kaggle" not in joined.lower()
    assert all(not value.startswith("/tmp") for value in argv)

    optimized = ops.build_runner_argv(
        release=release,
        run_id="leaf_s42_a800",
        waves=["Wave1"],
        competitions=["leaf-classification"],
        seed=42,
        optimization_plan=None,
        resume=False,
        runner_performance_overrides=[
            "--siim-workers",
            "8",
            "--wave2-workers",
            "32",
            "--leaf-embedding-batch-size",
            "512",
        ],
    )
    assert optimized[-6:] == [
        "--siim-workers",
        "8",
        "--wave2-workers",
        "32",
        "--leaf-embedding-batch-size",
        "512",
    ]

    with pytest.raises(ops.RemoteOpsError, match="Unsupported"):
        ops.normalize_runner_performance_overrides(["--seed", "99"])
    with pytest.raises(ops.RemoteOpsError, match="outside"):
        ops.normalize_runner_performance_overrides(["--wave2-workers", "128"])
    with pytest.raises(ops.RemoteOpsError, match="outside"):
        ops.normalize_runner_performance_overrides(["--siim-workers", "17"])
    with pytest.raises(ops.RemoteOpsError, match="pairs"):
        ops.normalize_runner_performance_overrides(["--wave2-workers"])

    cache_path = (
        f"{ops.REMOTE_MAY2022_CACHE_ROOT}/"
        "job89941_may2022_public_cache_s42_v1_20260728/cache"
    )
    cache_argv = ops.normalize_runner_contract_args([
        "--may-precomputed-cache-dir",
        cache_path,
        "--may-require-precomputed-cache",
    ])
    assert cache_argv == [
        "--may-precomputed-cache-dir",
        cache_path,
        "--may-require-precomputed-cache",
    ]
    cache_run = ops.build_runner_argv(
        release=release,
        run_id="may2022_cached_s42",
        waves=["Wave0"],
        competitions=["tabular-playground-series-may-2022"],
        seed=42,
        optimization_plan=None,
        resume=False,
        runner_contract_args=cache_argv,
    )
    assert cache_run[-3:] == cache_argv
    with pytest.raises(ops.RemoteOpsError, match="supplied together"):
        ops.normalize_runner_contract_args(["--may-precomputed-cache-dir", cache_path])
    with pytest.raises(ops.RemoteOpsError, match="supplied together"):
        ops.normalize_runner_contract_args(["--may-require-precomputed-cache"])
    with pytest.raises(ops.RemoteOpsError, match="outside"):
        ops.normalize_runner_contract_args([
            "--may-precomputed-cache-dir",
            f"{ops.ALLOWED_GPU_REMOTE_ROOT}/other/cache",
            "--may-require-precomputed-cache",
        ])

    dog_args = [
        "--wave2-dog-breed-backbone",
        "convnext_small",
        "--wave2-dog-breed-training-mode",
        "frozen_backbone_head",
        "--wave2-dog-breed-head-learning-rate",
        "0.001",
        "--wave2-dog-breed-diagnostic-fold-limit",
        "1",
        "--wave2-dog-breed-diagnostic-parent-fold0-epoch1-log-loss",
        "0.156225621700287",
        "--wave2-dog-breed-diagnostic-parent-fold0-top1-accuracy",
        "0.946195652173913",
        "--wave2-dog-breed-epochs",
        "4",
        "--wave2-dog-breed-batch-size",
        "32",
        "--wave2-vision-folds",
        "5",
    ]
    assert ops.normalize_runner_contract_args(dog_args) == dog_args
    with pytest.raises(ops.RemoteOpsError, match="invalid"):
        ops.normalize_runner_contract_args(
            ["--wave2-dog-breed-training-mode", "stability_finetune"]
        )

    precompute = ops.build_birds_precompute_argv(
        release=release,
        run_id="birds_precompute_s42",
        seed=42,
        optimization_plan=f"{release}/plans/wave2_gpt56_current.json",
        audio_workers=8,
        nice_level=15,
    )
    precompute_joined = " ".join(precompute)
    assert precompute[:4] == ["nice", "-n", "15", "python3"]
    assert "--precompute-only birds" in precompute_joined
    assert "--competitions mlsp-2013-birds" in precompute_joined
    assert "--wave2-audio-workers 8" in precompute_joined
    assert "kaggle" not in precompute_joined.lower()


def test_runner_environment_pins_performance_paths_inside_dedicated_root():
    release = ops.release_dir("a" * 64)
    environment = ops.build_runner_environment(release, "may2022_s42_test")

    assert environment["PYTHONPATH"] == ops.build_release_pythonpath(
        release, include_runtime=True
    )
    for key in ("TORCH_HOME", "XDG_CACHE_HOME", "HF_HOME", "TMPDIR"):
        assert environment[key].startswith(ops.ALLOWED_GPU_REMOTE_ROOT + "/")
    tmpdir = PurePosixPath(environment["TMPDIR"])
    assert tmpdir.parent == PurePosixPath(ops.REMOTE_RUNTIME_TMP_ROOT)
    assert len(tmpdir.name) == 12
    assert all(character in "0123456789abcdef" for character in tmpdir.name)
    assert len(environment["TMPDIR"].encode("utf-8")) + 40 <= 107
    assert environment["CUDA_MODULE_LOADING"] == "LAZY"
    assert environment["PYTORCH_CUDA_ALLOC_CONF"] == ops.CUDA_ALLOCATOR_CONF
    assert "expandable_segments:True" in environment["PYTORCH_CUDA_ALLOC_CONF"]


def test_safe_concurrency_allows_one_owned_cpu_light_runner_with_one_gpu_task():
    release = ops.release_dir(ops.EXPECTED_BUNDLE_SHA256)
    russian = {
        "competition_ids": ["text-normalization-challenge-russian-language"],
        "release": release,
        "owned": True,
    }
    assert ops.safe_cpu_light_concurrency(
        ["nomad2018-predict-transparent-conductors"], [russian], release, True
    )
    assert not ops.safe_cpu_light_concurrency(
        ["nomad2018-predict-transparent-conductors"], [russian], release, False
    )
    assert not ops.safe_cpu_light_concurrency(
        ["text-normalization-challenge-english-language"], [russian], release, True
    )
    assert not ops.safe_cpu_light_concurrency(
        ["nomad2018-predict-transparent-conductors"],
        [{**russian, "owned": False}],
        release,
        True,
    )


def test_safe_concurrency_allows_owned_jigsaw_beside_one_gpu_task():
    release = ops.release_dir(ops.EXPECTED_BUNDLE_SHA256)
    jigsaw = {
        "competition_ids": ["jigsaw-toxic-comment-classification-challenge"],
        "release": release,
        "owned": True,
    }

    assert ops.safe_cpu_light_concurrency(
        ["aptos2019-blindness-detection"], [jigsaw], release, True
    )
    assert not ops.safe_cpu_light_concurrency(
        ["aptos2019-blindness-detection"], [jigsaw], release, False
    )


def test_safe_concurrency_allows_new_jigsaw_beside_one_owned_gpu_task():
    release = ops.release_dir(ops.EXPECTED_BUNDLE_SHA256)
    aptos = {
        "competition_ids": ["aptos2019-blindness-detection"],
        "release": release,
        "owned": True,
    }

    assert ops.safe_cpu_light_concurrency(
        ["jigsaw-toxic-comment-classification-challenge"], [aptos], release, True
    )
    assert not ops.safe_cpu_light_concurrency(
        ["jigsaw-toxic-comment-classification-challenge"], [aptos], release, False
    )
    assert not ops.safe_cpu_light_concurrency(
        ["jigsaw-toxic-comment-classification-challenge"],
        [{**aptos, "owned": False}],
        release,
        True,
    )


def test_root_runner_processes_collapses_forked_dataloader_workers():
    candidates = [
        {"pid": 100, "ppid": 1, "competition_ids": ["aptos"]},
        {"pid": 101, "ppid": 100, "competition_ids": ["aptos"]},
        {"pid": 102, "ppid": 101, "competition_ids": ["aptos"]},
    ]

    assert ops.root_runner_processes(candidates) == [candidates[0]]


def test_safe_concurrency_allows_only_explicitly_trusted_cross_release_rollout():
    current = ops.release_dir(ops.EXPECTED_BUNDLE_SHA256)
    previous_sha = next(
        value for value in ops.TRUSTED_CONCURRENT_RELEASE_SHA256S
        if value != ops.EXPECTED_BUNDLE_SHA256
    )
    previous = ops.release_dir(previous_sha)
    jigsaw = {
        "competition_ids": ["jigsaw-toxic-comment-classification-challenge"],
        "release": previous,
        "owned": True,
    }

    assert ops.safe_cpu_light_concurrency(
        ["ranzcr-clip-catheter-line-classification"], [jigsaw], current, True
    )
    assert not ops.safe_cpu_light_concurrency(
        ["ranzcr-clip-catheter-line-classification"],
        [{**jigsaw, "release": ops.release_dir("f" * 64)}],
        current,
        True,
    )


def test_existing_run_state_is_reused_only_for_an_identical_contract():
    release = ops.release_dir("a" * 64)
    argv = ["python3", f"{release}/scripts/run_mlebench_lite_full.py", "--seed", "42"]
    run_dir = f"{ops.REMOTE_OUTPUT_ROOT}/stable_s42"
    state = {
        "schema": "evomind.mlebench_remote_ops.run_state.v1",
        "run_id": "stable_s42",
        "release": release,
        "argv": argv,
        "run_dir": run_dir,
        "human_gate_preserved": True,
        "pid": 123,
        "status": "running",
    }

    reused = ops.idempotent_remote_run_state(
        state,
        run_id="stable_s42",
        release=release,
        argv=argv,
        run_dir=run_dir,
    )
    assert reused["idempotent_reuse"] is True
    assert "idempotent_reuse" not in state

    with pytest.raises(ValueError, match="CONTRACT_MISMATCH"):
        ops.idempotent_remote_run_state(
            state,
            run_id="stable_s42",
            release=release,
            argv=[*argv, "--resume"],
            run_dir=run_dir,
        )


def test_start_source_checks_existing_state_before_spawning():
    source = inspect.getsource(ops.start_run)
    assert "idempotent_remote_run_state" in source
    assert source.index("if state_path.is_file()") < source.index("subprocess.Popen")


def test_status_and_cpu_precompute_liveness_checks_are_proc_based_and_signal_free():
    status_source = inspect.getsource(ops.read_remote_status)
    precompute_source = inspect.getsource(ops.start_birds_cpu_precompute)

    assert 'pathlib.Path("/proc")' in status_source
    assert 'process = "blocked"' in status_source
    assert "activity_age_seconds" in status_source
    assert '"signals_sent":0' in status_source
    assert "os.kill" not in status_source
    assert 'pathlib.Path("/proc")' in precompute_source
    assert "EXISTING_REMOTE_PROCESS_BLOCKED" in precompute_source
    assert "os.kill" not in precompute_source


def test_start_cli_requires_explicit_cpu_light_concurrency_opt_in():
    base = [
        "start", "--run-id", "nomad_s42", "--waves", "Wave2",
        "--competitions", "nomad2018-predict-transparent-conductors",
    ]
    assert ops.parse_args(base).allow_concurrent_with_cpu_light is False
    assert ops.parse_args([*base, "--allow-concurrent-with-cpu-light"]).allow_concurrent_with_cpu_light is True


def test_remote_start_source_embeds_all_concurrency_policy_constants():
    source = inspect.getsource(ops.start_run)

    assert "CPU_LIGHT_CONCURRENT_COMPETITIONS = frozenset" in source
    assert "TRUSTED_CONCURRENT_RELEASE_SHA256S = frozenset" in source
    assert "REMOTE_RELEASE_ROOT =" in source
    assert "from pathlib import PurePosixPath" in source
    assert "runner_release_path.name in TRUSTED_CONCURRENT_RELEASE_SHA256S" in source


def test_release_import_paths_include_package_root_before_script_directories():
    release = ops.release_dir("a" * 64)
    entries = ops.build_release_pythonpath(release, include_runtime=True).split(":")
    assert entries[:4] == [
        release,
        f"{release}/scripts",
        f"{release}/src",
        f"{release}/upstream_mlebench",
    ]
    assert entries[-2:] == [ops.REMOTE_UNIFIED_SITE_PACKAGES, ops.REMOTE_GRADER_SITE_PACKAGES]

    verifier = ops._remote_release_verifier_source(release, "a" * 64)
    compile(verifier, "<remote-release-verifier>", "exec")
    assert "str(release)," in verifier
    assert repr(ops.REMOTE_UNIFIED_SITE_PACKAGES) in verifier
    assert repr(ops.REMOTE_GRADER_SITE_PACKAGES) in verifier


def test_cli_defaults_to_pinned_bundle_and_gate():
    args = ops.parse_args(["verify-bundle"])
    assert args.bundle == ops.DEFAULT_BUNDLE
    assert args.gate_report == ops.DEFAULT_GATE_REPORT
    gate = ops.parse_args(["gpu-gate", "--interval-seconds", "0"])
    assert gate.interval_seconds == 0
    assert gate.samples_required == 5
    assert gate.max_activity_age_seconds == ops.DEFAULT_STATE_LOG_FRESHNESS_SECONDS
    assert gate.min_free_memory_mib == 60 * 1024
    assert ops.parse_args(["prepare-runtime"]).command == "prepare-runtime"
    assert ops.parse_args(["stage-vision-weights"]).command == "stage-vision-weights"
    birds = ops.parse_args(["precompute-birds", "--run-id", "birds_precompute_s42"])
    assert birds.audio_workers == 8
    assert birds.nice_level == 15
    assert birds.optimization_plan_name == "wave2_gpt56_current.json"


def test_birds_cpu_precompute_start_is_gpu_hidden_and_gate_free():
    source = inspect.getsource(ops.start_birds_cpu_precompute)
    assert '"CUDA_VISIBLE_DEVICES": ""' in source
    assert "stage_bundle_cpu_only" in source
    assert "build_birds_precompute_argv" in source
    assert '"gpu_idle_gate_required":False' in source


def test_unified_runtime_probe_is_cpu_only_and_covers_model_families():
    source = ops._runtime_probe_source()
    compile(source, "<runtime-probe>", "exec")
    for module in ("sklearn", "torch", "torchvision", "catboost", "lightgbm", "xgboost", "diskcache"):
        assert module in source
    assert '"cuda_allocated": False' in source
    assert "torch.cuda" not in source


def test_cuda_smoke_covers_every_pinned_vision_backbone():
    source = inspect.getsource(ops.cuda_smoke)
    assert "VISION_BACKBONE_SPECS" in source
    assert "_vision_model(" in source
    assert 'model.to(device="cuda", memory_format=torch.channels_last)' in source
    assert ".cuda(memory_format=" not in source
    assert 'torch.randn(2, 3, 224, 224, device="cuda").to(' in source
    assert "loss.backward()" in source
    assert "optimizer.step()" in source
    assert '"vision_backbone_count"' in source


def test_vision_weight_stage_is_cpu_only_boundary_checked_and_full_hashed():
    source = ops._convnext_weight_stage_source()
    compile(source, "<convnext-weight-stage>", "exec")
    assert repr(ops.REMOTE_SHARED_TORCH_HOME) in source
    assert "ConvNeXt_Tiny_Weights.DEFAULT" in source
    assert "ConvNeXt_Small_Weights.DEFAULT" in source
    assert "EfficientNet_V2_S_Weights.DEFAULT" in source
    assert "check_hash=True" in source
    assert "hashlib.sha256" in source
    for spec in ops.VISION_WEIGHT_SPECS.values():
        assert spec["filename"] in source
        assert spec["sha256"] in source
    assert '"weight_count": len(staged)' in source
    assert '"cuda_allocated": False' in source
    assert "torch.cuda" not in source


def test_pinned_bundle_is_the_verified_current_release():
    assert ops.DEFAULT_BUNDLE.parent.name == "mlebench_unified_20260728_134508"
    assert ops.EXPECTED_BUNDLE_SHA256 == "6ba428151245c9c4ccee1add67e50548193ad281a79482c4086abe1c52bd5721"
    assert "5ab7014aa642a2b88c7b30942b65306c5295f85445b2a8dbba108ae60e1e813b" in ops.TRUSTED_CONCURRENT_RELEASE_SHA256S
    assert "da685ff0166eb3a9bdfb9e61901b2c565f7033356a8ab5540f0859fda3237a55" in ops.TRUSTED_CONCURRENT_RELEASE_SHA256S
    assert "62d03fbe5809d03aa71bb9ae0a6f21a87114f482094a9bc08a6f308bbaef378a" in ops.TRUSTED_CONCURRENT_RELEASE_SHA256S
    assert "421ed5a355984a58f8e93fd3787af8db10dd97d2d0ace1eb45e34ab5dd86a897" in ops.TRUSTED_CONCURRENT_RELEASE_SHA256S
    assert "27eed5ee64e540b42d19aea74f1619939ed1487e17724410292759896f6445fd" in ops.TRUSTED_CONCURRENT_RELEASE_SHA256S
    assert "5681138f2af17e8c500c7086bc2be50240f4fe363c79d0e71fb7b5db377f748d" in ops.TRUSTED_CONCURRENT_RELEASE_SHA256S
    assert "4deb259f9b1e732c57ba6e91086bd7d635fcefccf0d5440bcfafa265042e085e" in ops.TRUSTED_CONCURRENT_RELEASE_SHA256S
    assert "0af1da0e7d5cfa47f00cf0d5451702a7323712a365e413385c2fb5dc954e0b6a" in ops.TRUSTED_CONCURRENT_RELEASE_SHA256S
    assert "10faec425efe0f09cb7dcc5bc8784029885016c907e6cd1fcb16752d439ab92c" in ops.TRUSTED_CONCURRENT_RELEASE_SHA256S
    assert "f31d007b6d6de91073fb8fd8d49e0d2ccf2e21021ab8ff683734fd91d46bc688" in ops.TRUSTED_CONCURRENT_RELEASE_SHA256S
    assert "9ddaf8d478fa99ca13abed805ad92bbf63d56983a868685df26a258f7eb9b9ac" in ops.TRUSTED_CONCURRENT_RELEASE_SHA256S
    assert "ae04b0739d7dcc49cdc7029ab76c04c1777133a968cb806bf3be4f4e70b5ca07" in ops.TRUSTED_CONCURRENT_RELEASE_SHA256S
    assert "02aa2eae888eb58efb960780ff5238c4a8f0437c7e6a40e4a75b9324fd304a69" in ops.TRUSTED_CONCURRENT_RELEASE_SHA256S
    assert "85420302ae73398158ac8acf92ca19b1e855a8946a8b0a4ba09cad8fbd74e10a" in ops.TRUSTED_CONCURRENT_RELEASE_SHA256S
    assert "8f8111aa776ec2f8018ed8f054cd6b8e627160a40715c6a6a24780f1f5c1d4b0" in ops.TRUSTED_CONCURRENT_RELEASE_SHA256S
    assert "a4c32c43082e104f6d37e8d656f906219b72bfa0ec1fd634c3d093d655f13ac3" in ops.TRUSTED_CONCURRENT_RELEASE_SHA256S
    assert "ac25ac8550d517cdb9294b248dbdb85587d2ca10ad4d00343d07debcbb954518" in ops.TRUSTED_CONCURRENT_RELEASE_SHA256S
    assert "b2d8c2d766a3ad5227955ba614d01363f355fc0ee40db5feef8838af60e58e2b" in ops.TRUSTED_CONCURRENT_RELEASE_SHA256S
    assert "02dac0573ef1455550e0782264be40e3cb6eb1071b4dbe7567475e8c433c0145" in ops.TRUSTED_CONCURRENT_RELEASE_SHA256S
    assert "d5e9667597a9305e5d0390747b55b17f9a8a60d071c6c3b7f2ae7742845ce8c8" in ops.TRUSTED_CONCURRENT_RELEASE_SHA256S
    assert "7bffec76e97e3a9105859c8467c49a47b42f87b965d2ab8d5b0cb920ddbd9229" in ops.TRUSTED_CONCURRENT_RELEASE_SHA256S
    assert "fd179a6cf602fdd15284192b060a33a6040bbd5de0c60b89f8457a93d5c12f50" in ops.TRUSTED_CONCURRENT_RELEASE_SHA256S
    assert "fc18233fb55316c4f96ca31b687eac0c38086db36702a9cfa51253e196d57e45" in ops.TRUSTED_CONCURRENT_RELEASE_SHA256S
    assert "459236802ab56c9e7f17c5ce6ccc00f8672b88dd9341695ab9049fe64c12a8d4" in ops.TRUSTED_CONCURRENT_RELEASE_SHA256S
    assert "693346ea65c5828943380e4ddc4bc77e2b440312443b9394659d20852aa2e38f" in ops.TRUSTED_CONCURRENT_RELEASE_SHA256S
    assert "0352816d7243858f3ce9c752450c5b530e5aba2c9048f7214d37675a524f678c" in ops.TRUSTED_CONCURRENT_RELEASE_SHA256S
    assert "d167fb2cccf4f7df29d4b66e3722e02b37a6808e0bc6e3f0847afd2c4e1cb09a" in ops.TRUSTED_CONCURRENT_RELEASE_SHA256S
    assert "2fe51f4197c5136a701d4219ac057ea208dcac17bbb0d02823f2b4af1bb7670a" in ops.TRUSTED_CONCURRENT_RELEASE_SHA256S
    assert ops.DEFAULT_GATE_REPORT == ops.LOCAL_CONTROL_ROOT / "gpu_gate_current.json"


def test_collection_whitelist_contains_siim_multiseed_evidence():
    source = inspect.getsource(ops.collect_run)
    for name in (
        "siim_fold_ensemble.npz",
        "siim_oof_predictions.csv",
        "siim_training_history.json",
        "siim_nested_patient_folds.json",
        "siim_artifact_manifest.json",
    ):
        assert name in source
