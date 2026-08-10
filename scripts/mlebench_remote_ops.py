#!/usr/bin/env python3
"""Gate-controlled deployment and execution for the MLE-Bench Lite bundle.

Every remote path is confined to the dedicated AIMSLAB workspace.  Deployment,
CUDA smoke, and training start fail closed unless they receive a fresh five-
sample identity-bound resource-gate report. Credentials are resolved only
through the shared DPAPI/environment loader and pinned SSH known-hosts file.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import posixpath
import re
import shlex
import stat
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from research_agent_workstation.server.core.gpu_credentials import (  # noqa: E402, I001
    ALLOWED_GPU_REMOTE_ROOT,
    DEFAULT_CREDENTIAL_PROFILE,
    GpuSshConfig,
    connect_ssh,
    load_gpu_ssh_config,
)


DEFAULT_BUNDLE = (
    PROJECT_ROOT
    / "workspace"
    / "deploy"
    / "mlebench_unified_20260806_220719"
    / "mlebench_unified_bundle.tar.gz"
)
LOCAL_CONTROL_ROOT = PROJECT_ROOT / "workspace" / "hpc" / "mlebench_remote_ops"
DEFAULT_GATE_REPORT = LOCAL_CONTROL_ROOT / "gpu_gate_current.json"
REMOTE_CONTROL_ROOT = f"{ALLOWED_GPU_REMOTE_ROOT}/mlebench_remote_ops"
REMOTE_RELEASE_ROOT = f"{REMOTE_CONTROL_ROOT}/releases"
REMOTE_INCOMING_ROOT = f"{REMOTE_CONTROL_ROOT}/incoming"
REMOTE_STATE_ROOT = f"{REMOTE_CONTROL_ROOT}/state"
REMOTE_LOG_ROOT = f"{REMOTE_CONTROL_ROOT}/logs"
REMOTE_DATA_ROOT = f"{ALLOWED_GPU_REMOTE_ROOT}/mlebench_official_data"
REMOTE_OUTPUT_ROOT = f"{ALLOWED_GPU_REMOTE_ROOT}/mlebench_lite_runs"
REMOTE_SHARED_CACHE_ROOT = f"{ALLOWED_GPU_REMOTE_ROOT}/mlebench_model_cache"
REMOTE_SHARED_TORCH_HOME = f"{ALLOWED_GPU_REMOTE_ROOT}/mlebench_model_cache/torch"
REMOTE_SHARED_XDG_CACHE_HOME = f"{REMOTE_SHARED_CACHE_ROOT}/xdg"
REMOTE_SHARED_HF_HOME = f"{REMOTE_SHARED_CACHE_ROOT}/huggingface"
REMOTE_RUNTIME_TMP_ROOT = f"{ALLOWED_GPU_REMOTE_ROOT}/.t"
REMOTE_UNIFIED_RUNTIME_ROOT = (
    f"{ALLOWED_GPU_REMOTE_ROOT}/mlebench_lite_runtime/unified-py310-sklearn1.7.2"
)
REMOTE_UNIFIED_SITE_PACKAGES = f"{REMOTE_UNIFIED_RUNTIME_ROOT}/site-packages"
REMOTE_GRADER_SITE_PACKAGES = (
    f"{ALLOWED_GPU_REMOTE_ROOT}/runtime/python310/mlebench_grader_bundle_v1"
)
REMOTE_PYTHON = "/usr/bin/python3.10"
REMOTE_RUNTIME_WHEEL_ROOT = (
    f"{ALLOWED_GPU_REMOTE_ROOT}/deployments/"
    "mlebench_phase_a_20260725_021916/wheels"
)
RUNNER_PERFORMANCE_OVERRIDE_LIMITS = {
    "--leaf-embedding-batch-size": (1, 1024),
    "--may-mlp-batch-size": (1, 65536),
    "--siim-workers": (0, 16),
    "--wave2-workers": (1, 64),
}
MAY2022_CACHE_ARG = "--may-precomputed-cache-dir"
MAY2022_CACHE_REQUIRED_ARG = "--may-require-precomputed-cache"
DOG_BREED_RUNNER_VALUE_ARGS = frozenset(
    {
        "--wave2-dog-breed-backbone",
        "--wave2-dog-breed-training-mode",
        "--wave2-dog-breed-head-learning-rate",
        "--wave2-dog-breed-diagnostic-fold-limit",
        "--wave2-dog-breed-diagnostic-parent-fold0-epoch1-log-loss",
        "--wave2-dog-breed-diagnostic-parent-fold0-top1-accuracy",
        "--wave2-dog-breed-epochs",
        "--wave2-dog-breed-batch-size",
        "--wave2-vision-folds",
    }
)
REMOTE_MAY2022_CACHE_ROOT = (
    f"{ALLOWED_GPU_REMOTE_ROOT}/evomind_mle22/job89941_may2022_public_cache"
)
REMOTE_SCIKIT_LEARN_WHEEL = (
    f"{REMOTE_RUNTIME_WHEEL_ROOT}/"
    "scikit_learn-1.7.2-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl"
)
REMOTE_JOBLIB_WHEEL = f"{REMOTE_RUNTIME_WHEEL_ROOT}/joblib-1.5.2-py3-none-any.whl"
REMOTE_NUMPY_WHEEL = (
    f"{REMOTE_RUNTIME_WHEEL_ROOT}/"
    "numpy-2.2.6-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
)

EXPECTED_BUNDLE_SHA256 = "5c09e4a9875654e3401d2a205dee43349a71cacca9e6c396766ba7b9bd08e9ec"
TRUSTED_CONCURRENT_RELEASE_SHA256S = frozenset({
    EXPECTED_BUNDLE_SHA256,
    "5ab7014aa642a2b88c7b30942b65306c5295f85445b2a8dbba108ae60e1e813b",
    "da685ff0166eb3a9bdfb9e61901b2c565f7033356a8ab5540f0859fda3237a55",
    "62d03fbe5809d03aa71bb9ae0a6f21a87114f482094a9bc08a6f308bbaef378a",
    "421ed5a355984a58f8e93fd3787af8db10dd97d2d0ace1eb45e34ab5dd86a897",
    "058a2bfbb0672ca3f3d982c79348cedaf8df379596c93edb7ca1f525faa416b5",
    "1e90a0836848c8eb835fc6f24ed9345b070a9dcaa8b49ab28d067cd402d80ee8",
    "b9929e98d9115b56290dd8a1126f63d749c5dd03cf74a2a34bcfd719b42aecc0",
    "27eed5ee64e540b42d19aea74f1619939ed1487e17724410292759896f6445fd",
    "5681138f2af17e8c500c7086bc2be50240f4fe363c79d0e71fb7b5db377f748d",
    "63aaaac2a44ae8de406bb04a0eef3ad2b663f7d782227ab24fb52ceb6f5d621f",
    "4deb259f9b1e732c57ba6e91086bd7d635fcefccf0d5440bcfafa265042e085e",
    "0af1da0e7d5cfa47f00cf0d5451702a7323712a365e413385c2fb5dc954e0b6a",
    "10faec425efe0f09cb7dcc5bc8784029885016c907e6cd1fcb16752d439ab92c",
    "f31d007b6d6de91073fb8fd8d49e0d2ccf2e21021ab8ff683734fd91d46bc688",
    "9ddaf8d478fa99ca13abed805ad92bbf63d56983a868685df26a258f7eb9b9ac",
    "ae04b0739d7dcc49cdc7029ab76c04c1777133a968cb806bf3be4f4e70b5ca07",
    "02aa2eae888eb58efb960780ff5238c4a8f0437c7e6a40e4a75b9324fd304a69",
    "85420302ae73398158ac8acf92ca19b1e855a8946a8b0a4ba09cad8fbd74e10a",
    "8f8111aa776ec2f8018ed8f054cd6b8e627160a40715c6a6a24780f1f5c1d4b0",
    "a4c32c43082e104f6d37e8d656f906219b72bfa0ec1fd634c3d093d655f13ac3",
    "ac25ac8550d517cdb9294b248dbdb85587d2ca10ad4d00343d07debcbb954518",
    "b2d8c2d766a3ad5227955ba614d01363f355fc0ee40db5feef8838af60e58e2b",
    "02dac0573ef1455550e0782264be40e3cb6eb1071b4dbe7567475e8c433c0145",
    "d5e9667597a9305e5d0390747b55b17f9a8a60d071c6c3b7f2ae7742845ce8c8",
    "7bffec76e97e3a9105859c8467c49a47b42f87b965d2ab8d5b0cb920ddbd9229",
    "fd179a6cf602fdd15284192b060a33a6040bbd5de0c60b89f8457a93d5c12f50",
    "fc18233fb55316c4f96ca31b687eac0c38086db36702a9cfa51253e196d57e45",
    "459236802ab56c9e7f17c5ce6ccc00f8672b88dd9341695ab9049fe64c12a8d4",
    "693346ea65c5828943380e4ddc4bc77e2b440312443b9394659d20852aa2e38f",
    "0352816d7243858f3ce9c752450c5b530e5aba2c9048f7214d37675a524f678c",
    "d167fb2cccf4f7df29d4b66e3722e02b37a6808e0bc6e3f0847afd2c4e1cb09a",
    "2fe51f4197c5136a701d4219ac057ea208dcac17bbb0d02823f2b4af1bb7670a",
    "756db62e29bf711203694478ec291777e89c0d82510031208033eb5f3f53b152",
    "ff746136ffe3ea2c2cf175b64d02b7eddde86413e4aa38113fef16ff578e86b2",
    "58378e1d3bd25eff7b56b1f761618a96230db9078821482523bdafc8c3e3fd96",
    "4b61c21d1e25266917895893e14a3fb3cbcf61cc026f3e898feaab0a3f4752c3",
    "8266b359c36f1463e8023f59c6424dae6909f2d789dda453188d98ab1a2bdf91",
    "cbb7fcd12369a61c7cdeb006ebeffd6a2e90821b0a572fb23b5c2aed712b1f9d",
    "cd6f4712e45b453c31dec935949109e09f4f400b6054a01b7d287650a2852c83",
    "6a2ff824a86e460b3e5c78fa633f32e8a14a610343e80b29ac9feebf657d2f79",
    "17b1531104183974e6911714d343b488a87086f5e54883c3b830e32a28befddf",
    "9a80fc8aa4445e1cebadc050421759c498286c30e73fbeda94773e83d02927e6",
    "f8e98eef3567ec35e8a67dab021b0a90e95d8c378f4c3d409f509af1e530d436",
    "77376bff9e793b792437ce7fce7bcbd6f0b687ed6961770cfc1f3f1cbbb4058e",
})
EXPECTED_MANIFEST_SCHEMA = "evomind.mlebench_lite.unified_bundle.v1"
EXPECTED_GATE_SCHEMA = "evomind.hpc_gpu_resource_gate.v2"
DEFAULT_GATE_SAMPLES = 5
DEFAULT_GATE_INTERVAL_SECONDS = 15
DEFAULT_STATE_LOG_FRESHNESS_SECONDS = 300
DEFAULT_MIN_FREE_MEMORY_MIB = 60 * 1024
DEFAULT_MAX_OTHER_PROCESS_MEMORY_MIB = 8 * 1024
DEFAULT_MAX_MEMORY_GROWTH_MIB = 256
BLOCKED_PROCESS_STATES = frozenset({"T", "t", "D", "Z", "X", "x"})
CONVNEXT_TINY_WEIGHT_FILENAME = "convnext_tiny-983f1562.pth"
CONVNEXT_TINY_WEIGHT_SHA256 = (
    "983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d"
)
CONVNEXT_SMALL_WEIGHT_FILENAME = "convnext_small-0c510722.pth"
CONVNEXT_SMALL_WEIGHT_SHA256 = (
    "0c510722adfd92966a2bd72b92f785ca05966bbac03cafe2f7a90b1f54bfab9a"
)
EFFICIENTNET_V2_S_WEIGHT_FILENAME = "efficientnet_v2_s-dd5fe13b.pth"
EFFICIENTNET_V2_S_WEIGHT_SHA256 = (
    "dd5fe13b1d60ec15317ccc8ca158186e134d3366c3dde9cb9a4e301f2dc66c74"
)
VISION_WEIGHT_SPECS = {
    "convnext_tiny": {
        "filename": CONVNEXT_TINY_WEIGHT_FILENAME,
        "sha256": CONVNEXT_TINY_WEIGHT_SHA256,
    },
    "convnext_small": {
        "filename": CONVNEXT_SMALL_WEIGHT_FILENAME,
        "sha256": CONVNEXT_SMALL_WEIGHT_SHA256,
    },
    "efficientnet_v2_s": {
        "filename": EFFICIENTNET_V2_S_WEIGHT_FILENAME,
        "sha256": EFFICIENTNET_V2_S_WEIGHT_SHA256,
    },
}
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SAFE_COMPETITION = re.compile(r"^[a-z0-9][a-z0-9-]{0,127}$")
SUPPORTED_WAVES = {"Wave0", "Wave1", "Wave2"}
CPU_LIGHT_CONCURRENT_COMPETITIONS = frozenset({
    "jigsaw-toxic-comment-classification-challenge",
    "text-normalization-challenge-english-language",
    "text-normalization-challenge-russian-language",
})
CUDA_ALLOCATOR_CONF = "backend:native,expandable_segments:True,garbage_collection_threshold:0.9"


class RemoteOpsError(RuntimeError):
    """Fail-closed operational error without secret-bearing context."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_tar_name(name: str) -> bool:
    candidate = PurePosixPath(name)
    return bool(name) and not candidate.is_absolute() and ".." not in candidate.parts


def verify_local_bundle(bundle: Path, *, expected_sha256: str = EXPECTED_BUNDLE_SHA256) -> dict[str, Any]:
    """Verify archive SHA, traversal safety, manifest hashes, and control flags."""

    bundle = bundle.expanduser().resolve()
    if not bundle.is_file():
        raise RemoteOpsError(f"Bundle is missing: {bundle}")
    archive_sha256 = sha256_file(bundle)
    if expected_sha256 and archive_sha256 != expected_sha256:
        raise RemoteOpsError("Bundle SHA256 differs from the pinned deployment artifact")

    with tarfile.open(bundle, "r:gz") as archive:
        members = archive.getmembers()
        if not members or any(
            not _safe_tar_name(member.name)
            or not member.isfile()
            or member.issym()
            or member.islnk()
            or member.isdev()
            for member in members
        ):
            raise RemoteOpsError("Bundle contains an unsafe archive member")
        by_name = {member.name: member for member in members if member.isfile()}
        manifest_member = by_name.get("bundle_manifest.json")
        if manifest_member is None:
            raise RemoteOpsError("Bundle manifest is missing")
        handle = archive.extractfile(manifest_member)
        if handle is None:
            raise RemoteOpsError("Bundle manifest is unreadable")
        manifest = json.loads(handle.read().decode("utf-8"))
        if manifest.get("schema") != EXPECTED_MANIFEST_SCHEMA:
            raise RemoteOpsError("Bundle manifest schema is unsupported")
        if manifest.get("kaggle_submission_enabled") is not False:
            raise RemoteOpsError("Bundle does not preserve the Kaggle Human Gate")
        if manifest.get("human_gate_preserved") is not True:
            raise RemoteOpsError("Bundle Human Gate declaration is missing")
        files = manifest.get("files")
        if not isinstance(files, dict) or not files:
            raise RemoteOpsError("Bundle manifest has no file hashes")
        expected_member_names = {str(name) for name in files} | {"bundle_manifest.json"}
        if set(by_name) != expected_member_names:
            raise RemoteOpsError("Bundle archive members differ from the manifest")
        verified = 0
        for name, expected in files.items():
            if not _safe_tar_name(str(name)) or name not in by_name:
                raise RemoteOpsError(f"Manifest member is missing or unsafe: {name}")
            member_handle = archive.extractfile(by_name[name])
            if member_handle is None:
                raise RemoteOpsError(f"Manifest member is unreadable: {name}")
            actual = hashlib.sha256(member_handle.read()).hexdigest()
            if actual != expected:
                raise RemoteOpsError(f"Manifest hash mismatch: {name}")
            verified += 1

    return {
        "schema": "evomind.mlebench_remote_ops.bundle_verification.v1",
        "bundle": str(bundle),
        "sha256": archive_sha256,
        "archive_member_count": len(members),
        "manifest_hash_count": verified,
        "manifest_schema": manifest["schema"],
        "human_gate_preserved": True,
        "passed": True,
    }


def validate_gate_report(
    path: Path,
    *,
    max_age_seconds: int = 600,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Require a fresh, identity-bound, five-or-more-sample resource report."""

    path = path.expanduser().resolve()
    if not path.is_file():
        raise RemoteOpsError(f"GPU idle-gate report is missing: {path}")
    report = json.loads(path.read_text(encoding="utf-8-sig"))
    if report.get("schema") != EXPECTED_GATE_SCHEMA:
        raise RemoteOpsError("GPU idle-gate schema is unsupported")
    samples = report.get("samples")
    policy = report.get("policy")
    if not isinstance(policy, dict):
        raise RemoteOpsError("GPU resource gate policy is missing")
    declared_samples = int(policy.get("samples_required") or 0)
    if (
        not isinstance(samples, list)
        or declared_samples < DEFAULT_GATE_SAMPLES
        or len(samples) != declared_samples
    ):
        raise RemoteOpsError("GPU resource gate requires at least five declared samples")
    if report.get("passed") is not True or report.get("dedicated_root_writable") is not True:
        raise RemoteOpsError("GPU idle gate did not pass")
    if report.get("other_processes_modified") is not False or report.get("signals_sent") != 0:
        raise RemoteOpsError("GPU resource gate audit contract is invalid")
    if report.get("hold_reasons"):
        raise RemoteOpsError("GPU resource gate contains HOLD reasons")
    if not all(
        sample.get("eligible") is True
        and sample.get("probe_errors") == []
        and sample.get("hold_reasons") == []
        for sample in samples
    ):
        raise RemoteOpsError("GPU resource samples are not clean")
    identity = report.get("identity")
    if (
        not isinstance(identity, dict)
        or not identity.get("host_uuid")
        or not identity.get("gpu_uuids")
        or identity.get("stable") is not True
    ):
        raise RemoteOpsError("GPU resource gate identity is missing or unstable")
    created_at = datetime.fromisoformat(str(report["created_at"]).replace("Z", "+00:00"))
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    age_seconds = ((now or datetime.now(timezone.utc)) - created_at.astimezone(timezone.utc)).total_seconds()
    if age_seconds < -60 or age_seconds > max_age_seconds:
        raise RemoteOpsError("GPU idle-gate report is stale")
    return {
        "path": str(path),
        "created_at": created_at.isoformat(),
        "age_seconds": age_seconds,
        "sample_count": len(samples),
        "host_uuid": identity["host_uuid"],
        "gpu_uuids": list(identity["gpu_uuids"]),
        "passed": True,
    }


def ensure_remote_path(path: str) -> str:
    """Normalize and prove that a remote path stays under the dedicated root."""

    normalized = posixpath.normpath(path)
    root = posixpath.normpath(ALLOWED_GPU_REMOTE_ROOT)
    if normalized != root and not normalized.startswith(root + "/"):
        raise RemoteOpsError("Remote path is outside the dedicated workspace")
    return normalized


def validate_run_id(run_id: str) -> str:
    if not SAFE_ID.fullmatch(run_id):
        raise RemoteOpsError("Run id contains unsupported characters")
    return run_id


def parse_competitions(value: str) -> list[str]:
    competitions = [item.strip() for item in value.split(",") if item.strip()]
    if any(not SAFE_COMPETITION.fullmatch(item) for item in competitions):
        raise RemoteOpsError("Competition id contains unsupported characters")
    return competitions


def parse_waves(value: str) -> list[str]:
    waves = [item.strip() for item in value.split(",") if item.strip()]
    if not waves or any(item not in SUPPORTED_WAVES for item in waves):
        raise RemoteOpsError("Unsupported wave selection")
    return list(dict.fromkeys(waves))


def normalize_runner_performance_overrides(values: Iterable[str] | None) -> list[str]:
    """Validate the small set of resource-only runner overrides used by GPU lanes."""

    raw = list(values or [])
    if len(raw) % 2:
        raise RemoteOpsError("Runner performance overrides must be option/value pairs")
    normalized: list[str] = []
    seen: set[str] = set()
    for index in range(0, len(raw), 2):
        option, value_text = raw[index], raw[index + 1]
        if option not in RUNNER_PERFORMANCE_OVERRIDE_LIMITS:
            raise RemoteOpsError(f"Unsupported runner performance override: {option}")
        if option in seen:
            raise RemoteOpsError(f"Duplicate runner performance override: {option}")
        try:
            value = int(value_text)
        except ValueError as exc:
            raise RemoteOpsError(f"Runner performance override is not an integer: {option}") from exc
        minimum, maximum = RUNNER_PERFORMANCE_OVERRIDE_LIMITS[option]
        if value < minimum or value > maximum:
            raise RemoteOpsError(
                f"Runner performance override is outside {minimum}..{maximum}: {option}"
            )
        seen.add(option)
        normalized.extend((option, str(value)))
    return normalized


def normalize_runner_contract_args(values: Iterable[str] | None) -> list[str]:
    """Validate fail-closed, non-tunable runner contracts carried into a release."""

    raw = list(values or [])
    normalized: list[str] = []
    cache_path: str | None = None
    require_cache = False
    index = 0
    while index < len(raw):
        option = raw[index]
        if option == MAY2022_CACHE_ARG:
            if cache_path is not None or index + 1 >= len(raw):
                raise RemoteOpsError("May cache argument is duplicated or missing its path")
            candidate = ensure_remote_path(raw[index + 1])
            if not candidate.startswith(REMOTE_MAY2022_CACHE_ROOT + "/"):
                raise RemoteOpsError("May cache path is outside the frozen job89941 cache root")
            if PurePosixPath(candidate).name != "cache" or "private" in {
                part.lower() for part in PurePosixPath(candidate).parts
            }:
                raise RemoteOpsError("May cache path is not a public cache directory")
            cache_path = candidate
            normalized.extend((option, candidate))
            index += 2
            continue
        if option == MAY2022_CACHE_REQUIRED_ARG:
            if require_cache:
                raise RemoteOpsError("May required-cache argument is duplicated")
            require_cache = True
            normalized.append(option)
            index += 1
            continue
        if option in DOG_BREED_RUNNER_VALUE_ARGS:
            if index + 1 >= len(raw):
                raise RemoteOpsError(f"Dog Breed runner contract value is missing: {option}")
            value = str(raw[index + 1])
            if option == "--wave2-dog-breed-backbone":
                valid = value in {"convnext_small", "efficientnet_v2_s"}
            elif option == "--wave2-dog-breed-training-mode":
                valid = value in {"stability_finetune", "frozen_backbone_head"}
            elif option == "--wave2-dog-breed-head-learning-rate":
                try:
                    numeric = float(value)
                except ValueError:
                    numeric = -1.0
                valid = 1e-5 <= numeric <= 5e-3
            elif option == "--wave2-dog-breed-diagnostic-fold-limit":
                valid = value == "1"
            elif option == "--wave2-dog-breed-diagnostic-parent-fold0-epoch1-log-loss":
                try:
                    numeric = float(value)
                except ValueError:
                    numeric = math.nan
                valid = math.isclose(
                    numeric,
                    0.156225621700287,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            elif option == "--wave2-dog-breed-diagnostic-parent-fold0-top1-accuracy":
                try:
                    numeric = float(value)
                except ValueError:
                    numeric = math.nan
                valid = math.isclose(
                    numeric,
                    0.946195652173913,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            elif option == "--wave2-dog-breed-epochs":
                valid = value.isdigit() and 1 <= int(value) <= 10
            elif option == "--wave2-dog-breed-batch-size":
                valid = value.isdigit() and 1 <= int(value) <= 64
            else:
                valid = value == "5"
            if not valid:
                raise RemoteOpsError(f"Dog Breed runner contract value is invalid: {option}")
            normalized.extend((option, value))
            index += 2
            continue
        raise RemoteOpsError(f"Unsupported runner contract argument: {option}")
    if (cache_path is None) != (not require_cache):
        raise RemoteOpsError("May cache path and required-cache flag must be supplied together")
    return normalized


def safe_cpu_light_concurrency(
    new_competitions,
    running,
    release,
    allow,
):
    """Permit one trusted CPU-bounded runner beside one GPU-heavy runner only."""

    new_ids = set(new_competitions)
    if not allow or len(new_ids) != 1:
        return False
    if len(running) != 1:
        return False
    existing = running[0]
    existing_ids = set(existing.get("competition_ids") or [])
    if len(existing_ids) != 1:
        return False
    new_is_cpu_light = new_ids.issubset(CPU_LIGHT_CONCURRENT_COMPETITIONS)
    existing_is_cpu_light = existing_ids.issubset(CPU_LIGHT_CONCURRENT_COMPETITIONS)
    if new_is_cpu_light == existing_is_cpu_light:
        return False
    new_release = PurePosixPath(str(release))
    existing_release = PurePosixPath(str(existing.get("release") or ""))
    release_root = PurePosixPath(REMOTE_RELEASE_ROOT)
    releases_are_trusted = bool(
        new_release.parent == release_root
        and existing_release.parent == release_root
        and new_release.name in TRUSTED_CONCURRENT_RELEASE_SHA256S
        and existing_release.name in TRUSTED_CONCURRENT_RELEASE_SHA256S
    )
    return bool(
        existing.get("owned")
        and releases_are_trusted
        and existing_ids
    )


def root_runner_processes(candidates):
    """Collapse forked DataLoader workers that inherit the runner command line."""

    candidate_pids = {int(item.get("pid") or 0) for item in candidates}
    return [
        item
        for item in candidates
        if int(item.get("ppid") or 0) not in candidate_pids
    ]


def idempotent_remote_run_state(
    existing: Any,
    *,
    run_id: str,
    release: str,
    argv: list[str],
    run_dir: str,
) -> dict[str, Any]:
    """Validate and reuse a previously-created state for the same run id."""

    if not isinstance(existing, dict):
        raise ValueError("EXISTING_RUN_STATE_NOT_AN_OBJECT")
    expected = {
        "schema": "evomind.mlebench_remote_ops.run_state.v1",
        "run_id": run_id,
        "release": release,
        "argv": argv,
        "run_dir": run_dir,
        "human_gate_preserved": True,
    }
    if any(existing.get(key) != value for key, value in expected.items()):
        raise ValueError("EXISTING_RUN_STATE_CONTRACT_MISMATCH")
    response = dict(existing)
    response["idempotent_reuse"] = True
    return response


def _connect(config: GpuSshConfig | None = None):
    """Connect through the shared strict host-key-pinned credential loader."""

    return connect_ssh(config or load_gpu_ssh_config(), timeout=25)


def _run_remote(client: Any, command: str, *, timeout: int = 120) -> tuple[int, str, str]:
    _stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
    output = stdout.read().decode("utf-8", "replace")
    error = stderr.read().decode("utf-8", "replace")
    return stdout.channel.recv_exit_status(), output, error


def _runtime_probe_source() -> str:
    """Return a no-CUDA-allocation import and sparse-model runtime probe."""

    return f'''from __future__ import annotations
import json, pathlib, sys
site = pathlib.Path({REMOTE_UNIFIED_SITE_PACKAGES!r}).resolve()
grader_site = pathlib.Path({REMOTE_GRADER_SITE_PACKAGES!r}).resolve()
if sys.version_info[:2] != (3, 10):
    raise SystemExit("UNIFIED_RUNTIME_REQUIRES_PY310")
import numpy, pandas, scipy, sklearn, joblib, torch, torchvision, catboost, lightgbm, xgboost, diskcache
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
matrix = TfidfVectorizer(ngram_range=(1, 2)).fit_transform(["alpha beta", "beta gamma", "alpha gamma", "delta"])
model = LogisticRegression(max_iter=100, solver="liblinear", random_state=42).fit(matrix, [0, 1, 0, 1])
probability = model.predict_proba(matrix)[:, 1]
sklearn_path = pathlib.Path(sklearn.__file__).resolve()
joblib_path = pathlib.Path(joblib.__file__).resolve()
diskcache_path = pathlib.Path(diskcache.__file__).resolve()
if site not in sklearn_path.parents or site not in joblib_path.parents:
    raise SystemExit("UNIFIED_RUNTIME_TARGET_PRECEDENCE_FAILED")
if grader_site not in diskcache_path.parents:
    raise SystemExit("GRADER_RUNTIME_TARGET_PRECEDENCE_FAILED")
if probability.shape != (4,) or not numpy.isfinite(probability).all():
    raise SystemExit("UNIFIED_RUNTIME_SPARSE_SMOKE_FAILED")
payload = {{
    "status": "UNIFIED_RUNTIME_OK",
    "python": sys.version.split()[0],
    "site_packages": str(site),
    "versions": {{
        "numpy": numpy.__version__, "pandas": pandas.__version__, "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__, "joblib": joblib.__version__,
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "catboost": catboost.__version__, "lightgbm": lightgbm.__version__,
        "xgboost": xgboost.__version__, "diskcache": diskcache.__version__,
    }},
    "paths": {{"sklearn": str(sklearn_path), "joblib": str(joblib_path),
              "diskcache": str(diskcache_path)}},
    "sparse_smoke_rows": int(probability.shape[0]),
    "cuda_allocated": False,
}}
print(json.dumps(payload, sort_keys=True))
'''


def verify_remote_runtime(client: Any) -> dict[str, Any]:
    """Fail closed unless one Python can import every required model family."""

    site = ensure_remote_path(REMOTE_UNIFIED_SITE_PACKAGES)
    grader_site = ensure_remote_path(REMOTE_GRADER_SITE_PACKAGES)
    source = _runtime_probe_source()
    pythonpath = f"{site}:{grader_site}"
    command = (
        f"PYTHONPATH={shlex.quote(pythonpath)} "
        f"{shlex.quote(REMOTE_PYTHON)} - <<'PY'\n{source}\nPY"
    )
    code, output, error = _run_remote(client, command, timeout=180)
    if code:
        raise RemoteOpsError(f"Unified runtime verification failed: {error[-500:] or output[-500:]}")
    try:
        payload = json.loads(output.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise RemoteOpsError("Unified runtime did not emit valid JSON") from exc
    if payload.get("status") != "UNIFIED_RUNTIME_OK" or payload.get("cuda_allocated") is not False:
        raise RemoteOpsError("Unified runtime verification contract failed")
    return payload


def _convnext_weight_stage_source() -> str:
    """Return a CPU-only multi-backbone weight staging and full-hash probe."""

    return f'''from __future__ import annotations
import hashlib, json, os, pathlib, sys, urllib.parse
allowed = pathlib.Path({ALLOWED_GPU_REMOTE_ROOT!r}).resolve()
torch_home = pathlib.Path({REMOTE_SHARED_TORCH_HOME!r}).resolve()
if allowed not in torch_home.parents:
    raise SystemExit("TORCH_HOME_OUTSIDE_ALLOWED_ROOT")
torch_home.mkdir(parents=True, exist_ok=True)
os.environ["TORCH_HOME"] = str(torch_home)
sys.path[:0] = [{REMOTE_UNIFIED_SITE_PACKAGES!r}, {REMOTE_GRADER_SITE_PACKAGES!r}]
import torch, torchvision
from torchvision.models import (
    ConvNeXt_Small_Weights,
    ConvNeXt_Tiny_Weights,
    EfficientNet_V2_S_Weights,
)
registry = {{
    "convnext_tiny": ConvNeXt_Tiny_Weights.DEFAULT,
    "convnext_small": ConvNeXt_Small_Weights.DEFAULT,
    "efficientnet_v2_s": EfficientNet_V2_S_Weights.DEFAULT,
}}
specs = {VISION_WEIGHT_SPECS!r}
staged = []
for name, weights in registry.items():
    spec = specs[name]
    filename = pathlib.Path(urllib.parse.urlparse(weights.url).path).name
    if filename != spec["filename"]:
        raise SystemExit("UNEXPECTED_VISION_WEIGHT_FILENAME:" + name)
    weights.get_state_dict(progress=False, check_hash=True)
    checkpoint = (torch_home / "hub" / "checkpoints" / filename).resolve()
    if allowed not in checkpoint.parents or not checkpoint.is_file():
        raise SystemExit("VISION_WEIGHT_NOT_STAGED:" + name)
    digest = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != spec["sha256"] or checkpoint.stat().st_size <= 0:
        raise SystemExit("VISION_WEIGHT_HASH_INVALID:" + name)
    staged.append({{
        "backbone": name,
        "checkpoint": str(checkpoint),
        "filename": filename,
        "url": weights.url,
        "sha256": actual,
        "expected_sha256": spec["sha256"],
        "size_bytes": checkpoint.stat().st_size,
    }})
payload = {{
    "status": "VISION_BACKBONE_WEIGHTS_READY",
    "torch_home": str(torch_home),
    "weights": staged,
    "weight_count": len(staged),
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "cuda_allocated": False,
}}
if payload["weight_count"] != len(specs):
    raise SystemExit("VISION_WEIGHT_COUNT_INVALID")
print(json.dumps(payload, sort_keys=True))
'''


def stage_vision_weights() -> dict[str, Any]:
    """Stage every pinned vision backbone without allocating or gating the GPU."""

    ensure_remote_path(REMOTE_SHARED_TORCH_HOME)
    client = _connect()
    try:
        runtime = verify_remote_runtime(client)
        source = _convnext_weight_stage_source()
        pythonpath = f"{REMOTE_UNIFIED_SITE_PACKAGES}:{REMOTE_GRADER_SITE_PACKAGES}"
        command = (
            f"PYTHONPATH={shlex.quote(pythonpath)} "
            f"{shlex.quote(REMOTE_PYTHON)} - <<'PY'\n{source}\nPY"
        )
        code, output, error = _run_remote(client, command, timeout=1200)
    finally:
        client.close()
    if code:
        raise RemoteOpsError(
            f"Vision weight staging failed: {error[-500:] or output[-500:]}"
        )
    weight_cache = json.loads(output.strip().splitlines()[-1])
    if (
        weight_cache.get("status") != "VISION_BACKBONE_WEIGHTS_READY"
        or weight_cache.get("weight_count") != len(VISION_WEIGHT_SPECS)
    ):
        raise RemoteOpsError("Vision weight staging returned an invalid status")
    return {
        "schema": "evomind.mlebench_remote_ops.vision_weight_cache.v2",
        "created_at": utc_now(),
        "remote_root": ALLOWED_GPU_REMOTE_ROOT,
        "runtime": runtime,
        "weight_cache": weight_cache,
        "gpu_allocated": False,
        "passed": True,
    }


def prepare_remote_runtime() -> dict[str, Any]:
    """Install only pinned local CPU wheels into a dedicated Python 3.10 overlay."""

    site = ensure_remote_path(REMOTE_UNIFIED_SITE_PACKAGES)
    wheel_paths = [
        ensure_remote_path(REMOTE_NUMPY_WHEEL),
        ensure_remote_path(REMOTE_SCIKIT_LEARN_WHEEL),
        ensure_remote_path(REMOTE_JOBLIB_WHEEL),
    ]
    client = _connect()
    installed = False
    try:
        try:
            runtime = verify_remote_runtime(client)
        except RemoteOpsError:
            command = " && ".join((
                f"mkdir -p {shlex.quote(site)}",
                f"{shlex.quote(REMOTE_PYTHON)} -m pip install "
                "--disable-pip-version-check --no-index --no-deps "
                f"--upgrade --target {shlex.quote(site)} "
                + " ".join(shlex.quote(path) for path in wheel_paths),
            ))
            code, output, error = _run_remote(client, command, timeout=600)
            if code:
                raise RemoteOpsError(f"Unified runtime installation failed: {error[-500:] or output[-500:]}")
            installed = True
            runtime = verify_remote_runtime(client)
    finally:
        client.close()
    return {
        "schema": "evomind.mlebench_remote_ops.unified_runtime.v1",
        "created_at": utc_now(),
        "remote_root": REMOTE_UNIFIED_RUNTIME_ROOT,
        "site_packages": site,
        "wheel_paths": wheel_paths,
        "network_install": False,
        "installed": installed,
        "runtime": runtime,
        "passed": True,
    }


def classify_linux_process_state(state_code: str | None, *, exists: bool) -> str:
    """Classify Linux process state without sending signal 0 or any other signal."""

    if not exists:
        return "stopped"
    code = str(state_code or "")[:1]
    if code in BLOCKED_PROCESS_STATES or not code:
        return "blocked"
    if code in {"R", "S", "I", "W"}:
        return "running"
    return "blocked"


def _gpu_gate_probe_source(max_activity_age_seconds: int) -> str:
    """Build the read-only remote /proc, NVIDIA-FD, identity and freshness probe."""

    if max_activity_age_seconds <= 0:
        raise RemoteOpsError("GPU gate freshness threshold must be positive")
    source = r'''from __future__ import annotations
import csv, io, json, os, pathlib, socket, subprocess, time

ALLOWED_ROOT = pathlib.Path(__ALLOWED_ROOT__).resolve()
STATE_ROOT = pathlib.Path(__STATE_ROOT__).resolve()
BLOCKED_STATES = frozenset(__BLOCKED_STATES__)
MAX_ACTIVITY_AGE_SECONDS = __MAX_ACTIVITY_AGE_SECONDS__
now = time.time()
errors = []

def read_first(paths):
    for path in paths:
        try:
            value = pathlib.Path(path).read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if value:
            return value
    return ""

def inside_allowed(path):
    try:
        resolved = pathlib.Path(path).resolve(strict=False)
        return resolved == ALLOWED_ROOT or ALLOWED_ROOT in resolved.parents
    except OSError:
        return False

def age_seconds(path):
    try:
        return max(0.0, now - pathlib.Path(path).stat().st_mtime)
    except OSError:
        return None

def process_info(pid):
    result = {"pid": int(pid), "exists": False, "state_code": "", "state": "",
              "name": "", "ppid": 0, "nvidia_fds": []}
    status_path = pathlib.Path("/proc") / str(pid) / "status"
    try:
        status_lines = status_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return result
    fields = {}
    for line in status_lines:
        key, separator, value = line.partition(":")
        if separator:
            fields[key] = value.strip()
    state_text = fields.get("State", "")
    result.update({
        "exists": True,
        "state_code": state_text[:1],
        "state": state_text,
        "name": fields.get("Name", ""),
    })
    try:
        result["ppid"] = int(fields.get("PPid", "0").split()[0])
    except (ValueError, IndexError):
        result["ppid"] = 0
    fd_root = status_path.parent / "fd"
    try:
        entries = list(fd_root.iterdir())
    except OSError:
        entries = []
    for entry in entries:
        try:
            target = os.readlink(entry)
        except OSError:
            continue
        if target.startswith("/dev/nvidia"):
            result["nvidia_fds"].append(target)
    result["nvidia_fds"] = sorted(set(result["nvidia_fds"]))
    return result

def run_nvidia_smi(query):
    command = ["nvidia-smi", query, "--format=csv,noheader,nounits"]
    try:
        completed = subprocess.run(command, check=False, capture_output=True,
                                   text=True, encoding="utf-8", timeout=20)
    except (OSError, subprocess.SubprocessError) as exc:
        errors.append("nvidia_smi_execution_failed:" + type(exc).__name__)
        return []
    if completed.returncode != 0:
        errors.append("nvidia_smi_query_failed")
        return []
    return [row for row in csv.reader(io.StringIO(completed.stdout)) if row]

host_uuid = read_first(("/sys/class/dmi/id/product_uuid", "/etc/machine-id"))
if not host_uuid:
    errors.append("host_uuid_missing")

gpus = []
for row in run_nvidia_smi(
    "--query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu"
):
    try:
        total = int(float(row[3].strip()))
        used = int(float(row[4].strip()))
        gpus.append({
            "index": int(row[0].strip()),
            "name": row[1].strip(),
            "uuid": row[2].strip(),
            "memory_total_mib": total,
            "memory_used_mib": used,
            "memory_free_mib": max(0, total - used),
            "utilization_percent": int(float(row[5].strip())),
        })
    except (IndexError, ValueError):
        errors.append("gpu_row_parse_failed")

compute_apps = []
for row in run_nvidia_smi(
    "--query-compute-apps=pid,process_name,used_gpu_memory,gpu_uuid"
):
    try:
        compute_apps.append({
            "pid": int(row[0].strip()),
            "process_name": row[1].strip(),
            "used_memory_mib": int(float(row[2].strip())),
            "gpu_uuid": row[3].strip(),
        })
    except (IndexError, ValueError):
        errors.append("compute_app_row_parse_failed")

state_records = []
state_pids = set()
if STATE_ROOT.is_dir():
    for state_path in sorted(STATE_ROOT.glob("*.json")):
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            pid = int(state.get("pid") or 0)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            errors.append("state_file_unreadable:" + state_path.name)
            continue
        process = process_info(pid) if pid > 0 else process_info(-1)
        if pid > 0:
            state_pids.add(pid)
        declared_status = str(state.get("status") or "").lower()
        log_path = pathlib.Path(str(state.get("log_path") or ""))
        run_dir = pathlib.Path(str(state.get("run_dir") or ""))
        mismatch_reasons = []
        state_age = age_seconds(state_path)
        log_age = age_seconds(log_path) if inside_allowed(log_path) else None
        if str(log_path) and not inside_allowed(log_path):
            mismatch_reasons.append("log_path_outside_allowed_root")
        activity_ages = [value for value in (state_age, log_age) if value is not None]
        if inside_allowed(run_dir):
            for name in ("checkpoint.json", "summary.json", "manifest.json"):
                candidate_age = age_seconds(run_dir / name)
                if candidate_age is not None:
                    activity_ages.append(candidate_age)
        elif str(run_dir):
            mismatch_reasons.append("run_dir_outside_allowed_root")
        activity_age = min(activity_ages) if activity_ages else None
        active_declared = declared_status in {"running", "started", "resuming"}
        terminal_declared = declared_status in {"passed", "failed", "stopped", "completed"}
        if active_declared:
            if not process["exists"]:
                mismatch_reasons.append("declared_running_process_missing")
            if process["state_code"] in BLOCKED_STATES:
                mismatch_reasons.append("declared_running_process_blocked")
            if activity_age is None or activity_age > MAX_ACTIVITY_AGE_SECONDS:
                mismatch_reasons.append("declared_running_activity_stale")
        if terminal_declared and process["exists"] and process["nvidia_fds"]:
            mismatch_reasons.append("terminal_state_process_holds_nvidia_fd")
        state_records.append({
            "state_file": state_path.name,
            "declared_status": declared_status,
            "pid": pid,
            "process": process,
            "state_age_seconds": state_age,
            "log_age_seconds": log_age,
            "activity_age_seconds": activity_age,
            "mismatch_reasons": mismatch_reasons,
        })

candidate_pids = state_pids | {item["pid"] for item in compute_apps}
own_nvidia_fd_processes = []
for proc_dir in pathlib.Path("/proc").glob("[0-9]*"):
    try:
        if proc_dir.stat().st_uid != os.getuid():
            continue
        pid = int(proc_dir.name)
    except (OSError, ValueError):
        continue
    info = process_info(pid)
    if info["nvidia_fds"]:
        own_nvidia_fd_processes.append(info)
        candidate_pids.add(pid)

processes = [process_info(pid) for pid in sorted(candidate_pids)]
payload = {
    "captured_at_epoch": now,
    "hostname": socket.gethostname(),
    "host_uuid": host_uuid,
    "gpus": gpus,
    "compute_apps": compute_apps,
    "processes": processes,
    "nvidia_fd_processes": own_nvidia_fd_processes,
    "state_records": state_records,
    "probe_errors": sorted(set(errors)),
    "read_only": True,
    "signals_sent": 0,
    "other_processes_modified": False,
}
print(json.dumps(payload, sort_keys=True))
'''
    return (
        source.replace("__ALLOWED_ROOT__", repr(ALLOWED_GPU_REMOTE_ROOT))
        .replace("__STATE_ROOT__", repr(REMOTE_STATE_ROOT))
        .replace("__BLOCKED_STATES__", repr(sorted(BLOCKED_PROCESS_STATES)))
        .replace("__MAX_ACTIVITY_AGE_SECONDS__", str(max_activity_age_seconds))
    )


def _expected_uuid_set(value: str | None) -> set[str]:
    return {item.strip().lower() for item in str(value or "").split(",") if item.strip()}


def evaluate_gpu_gate_samples(
    samples: list[dict[str, Any]],
    *,
    expected_host_uuid: str | None,
    expected_gpu_uuid: str | None,
    require_expected_identity: bool,
    min_free_memory_mib: int,
    max_other_process_memory_mib: int,
    max_memory_growth_mib: int,
    max_utilization_percent: int,
) -> dict[str, Any]:
    """Evaluate read-only probe samples and return deterministic HOLD reasons."""

    all_reasons: list[str] = []
    expected_gpu_uuids = _expected_uuid_set(expected_gpu_uuid)
    expected_host = str(expected_host_uuid or "").strip().lower()
    observed_hosts: list[str] = []
    observed_gpu_sets: list[tuple[str, ...]] = []
    total_memory_used: list[int] = []
    maximum_utilizations: list[int] = []

    if require_expected_identity and not expected_host:
        all_reasons.append("expected_host_uuid_missing")
    if require_expected_identity and not expected_gpu_uuids:
        all_reasons.append("expected_gpu_uuid_missing")

    for sample in samples:
        reasons: list[str] = []
        if sample.get("probe_errors"):
            reasons.append("probe_error")
        if sample.get("read_only") is not True:
            reasons.append("probe_not_read_only")
        if sample.get("signals_sent") != 0:
            reasons.append("probe_signal_contract_invalid")
        if sample.get("other_processes_modified") is not False:
            reasons.append("probe_process_mutation_contract_invalid")

        host_uuid = str(sample.get("host_uuid") or "").strip().lower()
        observed_hosts.append(host_uuid)
        if not host_uuid:
            reasons.append("host_uuid_missing")
        elif expected_host and host_uuid != expected_host:
            reasons.append("host_uuid_mismatch")

        gpus = sample.get("gpus") if isinstance(sample.get("gpus"), list) else []
        gpu_uuids = tuple(sorted(str(gpu.get("uuid") or "").strip().lower() for gpu in gpus))
        observed_gpu_sets.append(gpu_uuids)
        if not gpus or any(not value for value in gpu_uuids):
            reasons.append("gpu_identity_missing")
        if expected_gpu_uuids and set(gpu_uuids) != expected_gpu_uuids:
            reasons.append("gpu_uuid_mismatch")

        used_total = 0
        utilization_max = 0
        for gpu in gpus:
            try:
                free_mib = int(gpu["memory_free_mib"])
                used_mib = int(gpu["memory_used_mib"])
                utilization = int(gpu["utilization_percent"])
            except (KeyError, TypeError, ValueError):
                reasons.append("gpu_metrics_invalid")
                continue
            used_total += used_mib
            utilization_max = max(utilization_max, utilization)
            if free_mib < min_free_memory_mib:
                reasons.append("gpu_free_memory_below_floor")
            if utilization > max_utilization_percent:
                reasons.append("gpu_utilization_above_limit")
        total_memory_used.append(used_total)
        maximum_utilizations.append(utilization_max)

        compute_apps = (
            sample.get("compute_apps")
            if isinstance(sample.get("compute_apps"), list)
            else []
        )
        try:
            other_memory = sum(int(item.get("used_memory_mib") or 0) for item in compute_apps)
        except (AttributeError, TypeError, ValueError):
            other_memory = max_other_process_memory_mib + 1
            reasons.append("compute_app_metrics_invalid")
        if other_memory > max_other_process_memory_mib:
            reasons.append("other_process_memory_above_limit")
        app_pids = {int(item.get("pid") or -1) for item in compute_apps if isinstance(item, dict)}

        for process in sample.get("processes") or []:
            if str(process.get("state_code") or "")[:1] in BLOCKED_PROCESS_STATES:
                reasons.append("blocked_process_state")
        for record in sample.get("state_records") or []:
            mismatches = set(record.get("mismatch_reasons") or [])
            process = record.get("process") if isinstance(record.get("process"), dict) else {}
            path_contract_broken = bool(
                mismatches
                & {"log_path_outside_allowed_root", "run_dir_outside_allowed_root"}
            )
            # Old state files are append-only evidence. A missing process with a
            # days-old "running" marker is audit drift, not current GPU usage;
            # blocking on every historical marker would make the gate impossible
            # to clear without deleting evidence. Freshness remains fail-closed
            # for live processes and for any path-containment violation.
            if mismatches and (process.get("exists") is True or path_contract_broken):
                reasons.append("state_log_freshness_mismatch")
        for process in sample.get("nvidia_fd_processes") or []:
            state_code = str(process.get("state_code") or "")[:1]
            if state_code in BLOCKED_PROCESS_STATES:
                reasons.append("blocked_nvidia_fd_process")
            try:
                process_pid = int(process.get("pid") or -1)
            except (TypeError, ValueError):
                process_pid = -1
            if process_pid not in app_pids:
                reasons.append("unaccounted_nvidia_fd_process")

        sample["hold_reasons"] = sorted(set(reasons))
        sample["eligible"] = not sample["hold_reasons"]
        sample["idle"] = sample["eligible"]
        all_reasons.extend(sample["hold_reasons"])

    if len(set(observed_hosts)) != 1 or not observed_hosts or not observed_hosts[0]:
        all_reasons.append("host_identity_unstable")
    if len(set(observed_gpu_sets)) != 1 or not observed_gpu_sets or not observed_gpu_sets[0]:
        all_reasons.append("gpu_identity_unstable")
    if total_memory_used and max(total_memory_used) - min(total_memory_used) > max_memory_growth_mib:
        all_reasons.append("gpu_memory_growth_unstable")
    if (
        len(maximum_utilizations) >= 3
        and all(
            later > earlier
            for earlier, later in zip(maximum_utilizations[-3:], maximum_utilizations[-2:])
        )
    ):
        all_reasons.append("gpu_utilization_sustained_rise")

    unique_reasons = sorted(set(all_reasons))
    return {
        "passed": not unique_reasons and bool(samples),
        "hold_reasons": unique_reasons,
        "identity": {
            "host_uuid": observed_hosts[0] if observed_hosts else "",
            "gpu_uuids": list(observed_gpu_sets[0]) if observed_gpu_sets else [],
            "expected_host_uuid_bound": bool(expected_host),
            "expected_gpu_uuid_bound": bool(expected_gpu_uuids),
            "stable": bool(
                observed_hosts
                and observed_hosts[0]
                and len(set(observed_hosts)) == 1
                and observed_gpu_sets
                and observed_gpu_sets[0]
                and len(set(observed_gpu_sets)) == 1
            ),
        },
    }


def sample_gpu_idle_gate(
    *,
    samples_required: int = DEFAULT_GATE_SAMPLES,
    interval_seconds: int = DEFAULT_GATE_INTERVAL_SECONDS,
    max_activity_age_seconds: int = DEFAULT_STATE_LOG_FRESHNESS_SECONDS,
    min_free_memory_mib: int = DEFAULT_MIN_FREE_MEMORY_MIB,
    max_other_process_memory_mib: int = DEFAULT_MAX_OTHER_PROCESS_MEMORY_MIB,
    max_memory_growth_mib: int = DEFAULT_MAX_MEMORY_GROWTH_MIB,
    max_utilization_percent: int = 5,
    expected_host_uuid: str | None = None,
    expected_gpu_uuid: str | None = None,
    config: GpuSshConfig | None = None,
    client: Any | None = None,
) -> dict[str, Any]:
    """Run the authoritative identity-bound, read-only GPU prelaunch gate."""

    if samples_required < DEFAULT_GATE_SAMPLES or interval_seconds < 0:
        raise RemoteOpsError("GPU gate requires at least five non-negative-interval samples")
    if (
        max_activity_age_seconds <= 0
        or min_free_memory_mib <= 0
        or max_other_process_memory_mib < 0
        or max_memory_growth_mib < 0
        or max_utilization_percent < 0
    ):
        raise RemoteOpsError("GPU gate resource limits are invalid")

    config = config or load_gpu_ssh_config()
    expected_host_uuid = expected_host_uuid or config.expected_host_uuid
    expected_gpu_uuid = expected_gpu_uuid or config.expected_gpu_uuid
    require_expected_identity = config.credential_profile != DEFAULT_CREDENTIAL_PROFILE
    owns_client = client is None
    client = client or _connect(config)
    samples: list[dict[str, Any]] = []
    pre_code = -1
    pre_output = ""
    pre_error = ""
    root_exists = False
    root_writable = False
    try:
        preflight_command = (
            f"test -d {shlex.quote(ALLOWED_GPU_REMOTE_ROOT)} && echo ROOT_EXISTS=1 || echo ROOT_EXISTS=0; "
            f"test -w {shlex.quote(ALLOWED_GPU_REMOTE_ROOT)} && echo ROOT_WRITABLE=1 || echo ROOT_WRITABLE=0"
        )
        pre_code, pre_output, pre_error = _run_remote(client, preflight_command, timeout=30)
        root_exists = "ROOT_EXISTS=1" in pre_output
        root_writable = "ROOT_WRITABLE=1" in pre_output
        probe_source = _gpu_gate_probe_source(max_activity_age_seconds)
        for index in range(1, samples_required + 1):
            code, output, error = _run_remote(
                client,
                _python_command(probe_source),
                timeout=max(45, interval_seconds + 30),
            )
            try:
                sample = json.loads(output.strip().splitlines()[-1]) if code == 0 else {}
            except (IndexError, json.JSONDecodeError):
                sample = {}
            if not isinstance(sample, dict):
                sample = {}
            sample["index"] = index
            sample["captured_at"] = utc_now()
            sample["exit_code"] = code
            sample["stderr_type"] = type(error).__name__ if error else ""
            sample.setdefault("compute_apps", [])
            sample.setdefault("gpus", [])
            sample.setdefault("processes", [])
            sample.setdefault("nvidia_fd_processes", [])
            sample.setdefault("state_records", [])
            sample.setdefault("read_only", True)
            sample.setdefault("signals_sent", 0)
            sample.setdefault("other_processes_modified", False)
            sample.setdefault("probe_errors", [])
            if not isinstance(sample["probe_errors"], list):
                sample["probe_errors"] = ["probe_errors_invalid"]
            if code != 0 or not output.strip():
                sample["probe_errors"] = sorted(
                    set([*sample["probe_errors"], "remote_probe_failed"])
                )
            samples.append(sample)
            if index < samples_required:
                time.sleep(interval_seconds)
    finally:
        if owns_client:
            client.close()

    evaluation = evaluate_gpu_gate_samples(
        samples,
        expected_host_uuid=expected_host_uuid,
        expected_gpu_uuid=expected_gpu_uuid,
        require_expected_identity=require_expected_identity,
        min_free_memory_mib=min_free_memory_mib,
        max_other_process_memory_mib=max_other_process_memory_mib,
        max_memory_growth_mib=max_memory_growth_mib,
        max_utilization_percent=max_utilization_percent,
    )
    hold_reasons = list(evaluation["hold_reasons"])
    if pre_code != 0 or not root_exists:
        hold_reasons.append("dedicated_root_missing")
    if not root_writable:
        hold_reasons.append("dedicated_root_not_writable")
    hold_reasons = sorted(set(hold_reasons))
    passed = bool(
        pre_code == 0
        and root_exists
        and root_writable
        and evaluation["passed"]
        and not hold_reasons
    )
    return {
        "schema": EXPECTED_GATE_SCHEMA,
        "created_at": utc_now(),
        "remote_root": ALLOWED_GPU_REMOTE_ROOT,
        "credential_profile": config.credential_profile,
        "policy": {
            "samples_required": samples_required,
            "sample_interval_seconds": interval_seconds,
            "max_activity_age_seconds": max_activity_age_seconds,
            "min_free_memory_mib": min_free_memory_mib,
            "max_other_process_memory_mib": max_other_process_memory_mib,
            "max_memory_growth_mib": max_memory_growth_mib,
            "max_utilization_percent": max_utilization_percent,
            "blocked_process_states": sorted(BLOCKED_PROCESS_STATES),
            "expected_identity_required": require_expected_identity,
        },
        "preflight": {
            "exit_code": pre_code,
            "root_exists": root_exists,
            "root_writable": root_writable,
            "stderr_type": type(pre_error).__name__ if pre_error else "",
        },
        "samples": samples,
        "identity": evaluation["identity"],
        "hold_reasons": hold_reasons,
        "read_only_gate_passed": evaluation["passed"],
        "dedicated_root_writable": root_writable,
        "other_processes_modified": False,
        "signals_sent": 0,
        "passed": passed,
    }


def _python_command(source: str) -> str:
    import base64

    encoded = base64.b64encode(source.encode("utf-8")).decode("ascii")
    return (
        f"{shlex.quote(REMOTE_PYTHON)} -c "
        f"{shlex.quote(f'import base64;exec(base64.b64decode({encoded!r}))')}"
    )


def release_dir(bundle_sha256: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", bundle_sha256):
        raise RemoteOpsError("Release SHA256 is malformed")
    return ensure_remote_path(f"{REMOTE_RELEASE_ROOT}/{bundle_sha256}")


def _remote_release_verifier_source(release: str, expected_sha256: str) -> str:
    return f'''from __future__ import annotations
import hashlib, json, pathlib, py_compile, sys
release = pathlib.Path({release!r})
manifest_path = release / "bundle_manifest.json"
if not release.is_dir() or not manifest_path.is_file():
    raise SystemExit("REMOTE_RELEASE_MISSING")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("schema") != {EXPECTED_MANIFEST_SCHEMA!r}:
    raise SystemExit("REMOTE_MANIFEST_SCHEMA_INVALID")
verified = 0
for name, expected in manifest["files"].items():
    path = release / name
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit("REMOTE_MANIFEST_HASH_INVALID:" + name)
    verified += 1
for name in (
    "scripts/run_mlebench_lite_full.py",
    "scripts/run_mlebench_lite_wave0.py",
    "scripts/mlebench_wave2_adapters.py",
    "scripts/russian_transliteration.py",
    "scripts/mlebench_medal_recovery_adapters.py",
    "scripts/regrade_mlebench_lite_run.py",
    "src/research_os/mlebench_phase_a.py",
):
    py_compile.compile(str(release / name), doraise=True)
sys.path[:0] = [
    str(release),
    str(release / "scripts"),
    str(release / "src"),
    str(release / "upstream_mlebench"),
    {REMOTE_UNIFIED_SITE_PACKAGES!r},
    {REMOTE_GRADER_SITE_PACKAGES!r},
]
import run_mlebench_lite_full as runner
print(json.dumps({{"status":"REMOTE_RELEASE_OK","manifest_hash_count":verified,
                  "runner":str(pathlib.Path(runner.__file__).resolve()),
                  "bundle_sha256":{expected_sha256!r}}}, sort_keys=True))
'''


def verify_remote_release(client: Any, bundle_sha256: str) -> dict[str, Any]:
    release = release_dir(bundle_sha256)
    code, output, error = _run_remote(
        client,
        _python_command(_remote_release_verifier_source(release, bundle_sha256)),
        timeout=240,
    )
    if code:
        raise RemoteOpsError(f"Remote release verification failed: {error[-500:] or output[-500:]}")
    return json.loads(output.strip().splitlines()[-1])


def deploy_bundle(bundle: Path, gate_report: Path, *, max_gate_age: int) -> dict[str, Any]:
    gate = validate_gate_report(gate_report, max_age_seconds=max_gate_age)
    verification = verify_local_bundle(bundle)
    bundle_sha256 = verification["sha256"]
    release = release_dir(bundle_sha256)
    transfer_stamp = int(time.time() * 1000)
    incoming = ensure_remote_path(
        f"{REMOTE_INCOMING_ROOT}/{bundle_sha256}-{transfer_stamp}.tar.gz"
    )
    incoming_part = incoming + ".part"
    client = _connect()
    try:
        mkdir = "mkdir -p " + " ".join(
            shlex.quote(ensure_remote_path(path))
            for path in (REMOTE_RELEASE_ROOT, REMOTE_INCOMING_ROOT, REMOTE_STATE_ROOT, REMOTE_LOG_ROOT)
        )
        code, output, error = _run_remote(client, mkdir, timeout=60)
        if code:
            raise RemoteOpsError(f"Remote deployment roots could not be created: {error[-500:]}")

        code, output, error = _run_remote(client, f"test -d {shlex.quote(release)}", timeout=30)
        uploaded = code != 0
        if uploaded:
            with client.open_sftp() as sftp:
                sftp.put(str(bundle), incoming_part)
                sftp.rename(incoming_part, incoming)
            stamp = int(time.time())
            stage = ensure_remote_path(f"{REMOTE_RELEASE_ROOT}/.stage-{bundle_sha256}-{stamp}")
            source = f'''from __future__ import annotations
import hashlib, json, os, pathlib, tarfile
archive_path = pathlib.Path({incoming!r})
expected_archive = {bundle_sha256!r}
stage = pathlib.Path({stage!r})
release = pathlib.Path({release!r})
if hashlib.sha256(archive_path.read_bytes()).hexdigest() != expected_archive:
    raise SystemExit("REMOTE_ARCHIVE_SHA_INVALID")
stage.mkdir(parents=False, exist_ok=False)
with tarfile.open(archive_path, "r:gz") as archive:
    members = archive.getmembers()
    for member in members:
        path = pathlib.PurePosixPath(member.name)
        if (path.is_absolute() or ".." in path.parts or member.issym()
                or member.islnk() or member.isdev()):
            raise SystemExit("REMOTE_ARCHIVE_MEMBER_UNSAFE")
    archive.extractall(stage)
manifest = json.loads((stage / "bundle_manifest.json").read_text(encoding="utf-8"))
for name, expected in manifest["files"].items():
    actual = hashlib.sha256((stage / name).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit("REMOTE_MANIFEST_HASH_INVALID:" + name)
os.replace(stage, release)
print(json.dumps({{"status":"REMOTE_EXTRACT_OK","release":str(release),"members":len(members)}}))
'''
            code, output, error = _run_remote(client, _python_command(source), timeout=600)
            if code:
                raise RemoteOpsError(f"Remote bundle extraction failed: {error[-500:] or output[-500:]}")
        remote = verify_remote_release(client, bundle_sha256)
    finally:
        client.close()
    return {
        "schema": "evomind.mlebench_remote_ops.deployment.v1",
        "created_at": utc_now(),
        "gate": gate,
        "local_bundle": verification,
        "remote_release": release,
        "uploaded": uploaded,
        "remote_verification": remote,
        "passed": True,
    }


def stage_bundle_cpu_only(bundle: Path) -> dict[str, Any]:
    """Upload and verify a release for CPU-only work without consuming the GPU idle gate."""

    verification = verify_local_bundle(bundle)
    bundle_sha256 = verification["sha256"]
    release = release_dir(bundle_sha256)
    transfer_stamp = int(time.time() * 1000)
    incoming = ensure_remote_path(
        f"{REMOTE_INCOMING_ROOT}/{bundle_sha256}-{transfer_stamp}.tar.gz"
    )
    incoming_part = incoming + ".part"
    client = _connect()
    try:
        mkdir = "mkdir -p " + " ".join(
            shlex.quote(ensure_remote_path(path))
            for path in (REMOTE_RELEASE_ROOT, REMOTE_INCOMING_ROOT, REMOTE_STATE_ROOT, REMOTE_LOG_ROOT)
        )
        code, output, error = _run_remote(client, mkdir, timeout=60)
        if code:
            raise RemoteOpsError(f"Remote CPU staging roots could not be created: {error[-500:]}")

        code, output, error = _run_remote(client, f"test -d {shlex.quote(release)}", timeout=30)
        uploaded = code != 0
        if uploaded:
            with client.open_sftp() as sftp:
                sftp.put(str(bundle), incoming_part)
                sftp.rename(incoming_part, incoming)
            stamp = int(time.time())
            stage = ensure_remote_path(f"{REMOTE_RELEASE_ROOT}/.stage-{bundle_sha256}-{stamp}")
            source = f'''from __future__ import annotations
import hashlib, json, os, pathlib, tarfile
archive_path = pathlib.Path({incoming!r})
expected_archive = {bundle_sha256!r}
stage = pathlib.Path({stage!r})
release = pathlib.Path({release!r})
if hashlib.sha256(archive_path.read_bytes()).hexdigest() != expected_archive:
    raise SystemExit("REMOTE_ARCHIVE_SHA_INVALID")
stage.mkdir(parents=False, exist_ok=False)
with tarfile.open(archive_path, "r:gz") as archive:
    members = archive.getmembers()
    for member in members:
        path = pathlib.PurePosixPath(member.name)
        if (path.is_absolute() or ".." in path.parts or member.issym()
                or member.islnk() or member.isdev()):
            raise SystemExit("REMOTE_ARCHIVE_MEMBER_UNSAFE")
    archive.extractall(stage)
manifest = json.loads((stage / "bundle_manifest.json").read_text(encoding="utf-8"))
for name, expected in manifest["files"].items():
    actual = hashlib.sha256((stage / name).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit("REMOTE_MANIFEST_HASH_INVALID:" + name)
os.replace(stage, release)
print(json.dumps({{"status":"REMOTE_CPU_STAGE_EXTRACT_OK","release":str(release),"members":len(members)}}))
'''
            code, output, error = _run_remote(client, _python_command(source), timeout=600)
            if code:
                raise RemoteOpsError(f"Remote CPU bundle extraction failed: {error[-500:] or output[-500:]}")
        remote = verify_remote_release(client, bundle_sha256)
    finally:
        client.close()
    return {
        "schema": "evomind.mlebench_remote_ops.cpu_release_stage.v1",
        "created_at": utc_now(),
        "local_bundle": verification,
        "remote_release": release,
        "uploaded": uploaded,
        "remote_verification": remote,
        "gpu_idle_gate_required": False,
        "passed": True,
    }


def cuda_smoke(bundle: Path, gate_report: Path, *, max_gate_age: int) -> dict[str, Any]:
    gate = validate_gate_report(gate_report, max_age_seconds=max_gate_age)
    verification = verify_local_bundle(bundle)
    release = release_dir(verification["sha256"])
    source = f'''from __future__ import annotations
import json, pathlib, sys
release = pathlib.Path({release!r})
sys.path[:0] = [
    str(release),
    str(release / "scripts"),
    str(release / "src"),
    str(release / "upstream_mlebench"),
    {REMOTE_UNIFIED_SITE_PACKAGES!r},
    {REMOTE_GRADER_SITE_PACKAGES!r},
]
import torch
from mlebench_medal_recovery_adapters import build_may2022_residual_mlp
from mlebench_wave2_adapters import VISION_BACKBONE_SPECS, _vision_model
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
x = torch.randn(256, 276, device="cuda")
y = torch.randint(0, 2, (256,), device="cuda", dtype=torch.float32)
model = build_may2022_residual_mlp(276, width=768, block_count=5).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)
optimizer.zero_grad(set_to_none=True)
with torch.autocast(device_type="cuda", dtype=torch.float16):
    logits = model(x)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
loss.backward()
optimizer.step()
torch.cuda.synchronize()
payload = {{"status":"CUDA_MLP_SMOKE_OK", "shape":list(logits.shape),
           "loss":float(loss.detach()), "finite":bool(torch.isfinite(logits).all()),
           "device":torch.cuda.get_device_name(0),
           "max_memory_allocated":int(torch.cuda.max_memory_allocated())}}
if not payload["finite"] or payload["shape"] != [256]:
    raise SystemExit("CUDA_MLP_SMOKE_INVALID")
del model, optimizer, x, y, logits, loss
torch.cuda.empty_cache()
amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
vision = []
for backbone in VISION_BACKBONE_SPECS:
    torch.cuda.reset_peak_memory_stats()
    model, pretrained, identity = _vision_model(
        2,
        backbone=backbone,
        require_pretrained=True,
    )
    model = model.to(device="cuda", memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    images = torch.randn(2, 3, 224, 224, device="cuda").to(
        memory_format=torch.channels_last
    )
    labels = torch.tensor([0, 1], device="cuda")
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        logits = model(images)
        loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    optimizer.step()
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=amp_dtype):
        inference = model(images)
    torch.cuda.synchronize()
    record = {{
        "backbone": backbone,
        "pretrained": bool(pretrained),
        "weight_identity": identity,
        "shape": list(inference.shape),
        "finite": bool(torch.isfinite(inference).all()),
        "loss": float(loss.detach()),
        "peak_memory_allocated_mib": int(torch.cuda.max_memory_allocated() / 2**20),
    }}
    if not record["pretrained"] or not record["finite"] or record["shape"] != [2, 2]:
        raise SystemExit("CUDA_VISION_SMOKE_INVALID:" + backbone)
    vision.append(record)
    del model, optimizer, images, labels, logits, loss, inference
    torch.cuda.empty_cache()
payload["vision"] = vision
payload["vision_backbone_count"] = len(vision)
payload["amp_dtype"] = str(amp_dtype)
if payload["vision_backbone_count"] != len(VISION_BACKBONE_SPECS):
    raise SystemExit("CUDA_VISION_BACKBONE_COUNT_INVALID")
print(json.dumps(payload, sort_keys=True))
'''
    client = _connect()
    try:
        verify_remote_release(client, verification["sha256"])
        command = (
            f"TORCH_HOME={shlex.quote(ensure_remote_path(REMOTE_SHARED_TORCH_HOME))} "
            + _python_command(source)
        )
        code, output, error = _run_remote(client, command, timeout=600)
    finally:
        client.close()
    if code:
        raise RemoteOpsError(f"CUDA MLP smoke failed: {error[-500:] or output[-500:]}")
    return {
        "schema": "evomind.mlebench_remote_ops.cuda_smoke.v1",
        "created_at": utc_now(),
        "gate": gate,
        "release": release,
        "smoke": json.loads(output.strip().splitlines()[-1]),
        "passed": True,
    }


def build_runner_argv(
    *,
    release: str,
    run_id: str,
    waves: list[str],
    competitions: list[str],
    seed: int,
    optimization_plan: str | None,
    resume: bool,
    runner_performance_overrides: Iterable[str] | None = None,
    runner_contract_args: Iterable[str] | None = None,
) -> list[str]:
    validate_run_id(run_id)
    if seed < 0:
        raise RemoteOpsError("Seed must be non-negative")
    argv = [
        REMOTE_PYTHON,
        f"{release}/scripts/run_mlebench_lite_full.py",
        "--data-root", REMOTE_DATA_ROOT,
        "--output-root", REMOTE_OUTPUT_ROOT,
        "--allowed-root", ALLOWED_GPU_REMOTE_ROOT,
        "--official-source-root", f"{release}/upstream_mlebench",
        "--waves", ",".join(waves),
        "--run-id", run_id,
        "--seed", str(seed),
        "--candidate-only",
        "--wave2-fast-kernels",
    ]
    if competitions:
        argv.extend(["--competitions", ",".join(competitions)])
    if optimization_plan:
        argv.extend(["--optimization-plan", optimization_plan])
    if resume:
        argv.append("--resume")
    argv.extend(normalize_runner_performance_overrides(runner_performance_overrides))
    argv.extend(normalize_runner_contract_args(runner_contract_args))
    return argv


def build_birds_precompute_argv(
    *,
    release: str,
    run_id: str,
    seed: int,
    optimization_plan: str | None,
    audio_workers: int,
    nice_level: int,
) -> list[str]:
    validate_run_id(run_id)
    if seed < 0:
        raise RemoteOpsError("Seed must be non-negative")
    if audio_workers < 1 or audio_workers > 32:
        raise RemoteOpsError("Birds audio workers must be between 1 and 32")
    nice_level = max(0, min(19, int(nice_level)))
    argv = [
        "nice",
        "-n",
        str(nice_level),
        REMOTE_PYTHON,
        f"{release}/scripts/run_mlebench_lite_full.py",
        "--data-root", REMOTE_DATA_ROOT,
        "--output-root", REMOTE_OUTPUT_ROOT,
        "--allowed-root", ALLOWED_GPU_REMOTE_ROOT,
        "--official-source-root", f"{release}/upstream_mlebench",
        "--waves", "Wave2",
        "--competitions", "mlsp-2013-birds",
        "--run-id", run_id,
        "--seed", str(seed),
        "--precompute-only", "birds",
        "--wave2-audio-workers", str(audio_workers),
    ]
    if optimization_plan:
        argv.extend(["--optimization-plan", optimization_plan])
    return argv


def build_release_pythonpath(release: str, *, include_runtime: bool) -> str:
    """Return an import path that supports both package and script-style imports."""

    entries = [
        release,
        f"{release}/scripts",
        f"{release}/src",
        f"{release}/upstream_mlebench",
    ]
    if include_runtime:
        entries.extend((REMOTE_UNIFIED_SITE_PACKAGES, REMOTE_GRADER_SITE_PACKAGES))
    return ":".join(entries)


def build_runner_environment(release: str, run_id: str) -> dict[str, str]:
    """Pin performance-sensitive caches and CUDA settings inside the dedicated root."""

    validate_run_id(run_id)
    short_run_token = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:12]
    return {
        "PYTHONPATH": build_release_pythonpath(release, include_runtime=True),
        "TORCH_HOME": ensure_remote_path(REMOTE_SHARED_TORCH_HOME),
        "XDG_CACHE_HOME": ensure_remote_path(REMOTE_SHARED_XDG_CACHE_HOME),
        "HF_HOME": ensure_remote_path(REMOTE_SHARED_HF_HOME),
        "TMPDIR": ensure_remote_path(f"{REMOTE_RUNTIME_TMP_ROOT}/{short_run_token}"),
        "CUDA_MODULE_LOADING": "LAZY",
        "PYTORCH_CUDA_ALLOC_CONF": CUDA_ALLOCATOR_CONF,
    }


def start_run(
    bundle: Path,
    gate_report: Path,
    *,
    run_id: str,
    waves: list[str],
    competitions: list[str],
    seed: int,
    optimization_plan_name: str | None,
    resume: bool,
    allow_concurrent_with_cpu_light: bool,
    max_gate_age: int,
    runner_performance_overrides: Iterable[str] | None = None,
    runner_contract_args: Iterable[str] | None = None,
) -> dict[str, Any]:
    competition_ids = set(competitions)
    cpu_light_request = bool(
        allow_concurrent_with_cpu_light
        and len(competition_ids) == 1
        and competition_ids.issubset(CPU_LIGHT_CONCURRENT_COMPETITIONS)
    )
    if cpu_light_request:
        gate = {
            "schema": "evomind.mlebench_remote_ops.cpu_light_start_gate.v1",
            "created_at": utc_now(),
            "competition_ids": sorted(competition_ids),
            "gpu_idle_gate_required": False,
            "concurrency_policy_enforced_remotely": True,
            "passed": True,
        }
    else:
        gate = validate_gate_report(gate_report, max_age_seconds=max_gate_age)
    verification = verify_local_bundle(bundle)
    release = release_dir(verification["sha256"])
    run_id = validate_run_id(run_id)
    plan = f"{release}/plans/{optimization_plan_name}" if optimization_plan_name else None
    argv = build_runner_argv(
        release=release,
        run_id=run_id,
        waves=waves,
        competitions=competitions,
        seed=seed,
        optimization_plan=plan,
        resume=resume,
        runner_performance_overrides=runner_performance_overrides,
        runner_contract_args=runner_contract_args,
    )
    state_path = ensure_remote_path(f"{REMOTE_STATE_ROOT}/{run_id}.json")
    log_path = ensure_remote_path(f"{REMOTE_LOG_ROOT}/{run_id}.log")
    run_dir = ensure_remote_path(f"{REMOTE_OUTPUT_ROOT}/{run_id}")
    runtime_environment = build_runner_environment(release, run_id)
    concurrency_policy_source = inspect.getsource(safe_cpu_light_concurrency)
    runner_process_policy_source = inspect.getsource(root_runner_processes)
    idempotency_policy_source = inspect.getsource(idempotent_remote_run_state)
    source = f'''from __future__ import annotations
import json, os, pathlib, subprocess, time
from pathlib import PurePosixPath
CPU_LIGHT_CONCURRENT_COMPETITIONS = frozenset({sorted(CPU_LIGHT_CONCURRENT_COMPETITIONS)!r})
TRUSTED_CONCURRENT_RELEASE_SHA256S = frozenset({sorted(TRUSTED_CONCURRENT_RELEASE_SHA256S)!r})
REMOTE_RELEASE_ROOT = {REMOTE_RELEASE_ROOT!r}
{concurrency_policy_source}
{runner_process_policy_source}
{idempotency_policy_source}
argv = {argv!r}
release = pathlib.Path({release!r})
state_path = pathlib.Path({state_path!r})
log_path = pathlib.Path({log_path!r})
state_path.parent.mkdir(parents=True, exist_ok=True)
log_path.parent.mkdir(parents=True, exist_ok=True)
if state_path.is_file():
    existing = json.loads(state_path.read_text(encoding="utf-8"))
    response = idempotent_remote_run_state(
        existing,
        run_id={run_id!r},
        release=str(release),
        argv=argv,
        run_dir={run_dir!r},
    )
    print(json.dumps(response, sort_keys=True))
    raise SystemExit(0)
for key in ("TORCH_HOME", "XDG_CACHE_HOME", "HF_HOME", "TMPDIR"):
    pathlib.Path({runtime_environment!r}[key]).mkdir(parents=True, exist_ok=True)
runner_candidates = []
for path in pathlib.Path("/proc").glob("[0-9]*/cmdline"):
    try:
        raw = path.read_bytes()
        parts = [item.decode("utf-8", "replace") for item in raw.split(b"\\0") if item]
    except OSError:
        continue
    if any(pathlib.PurePosixPath(item).name == "run_mlebench_lite_full.py" for item in parts):
        def option(flag):
            try:
                return parts[parts.index(flag) + 1]
            except (ValueError, IndexError):
                return ""
        runner = next((item for item in parts if pathlib.PurePosixPath(item).name == "run_mlebench_lite_full.py"), "")
        runner_path = pathlib.Path(runner)
        runner_release = str(runner_path.parents[1]) if len(runner_path.parents) >= 2 else ""
        runner_release_path = pathlib.PurePosixPath(runner_release)
        existing_ids = [item for item in option("--competitions").split(",") if item]
        owned = bool(
            runner_release_path.parent == pathlib.PurePosixPath(REMOTE_RELEASE_ROOT)
            and runner_release_path.name in TRUSTED_CONCURRENT_RELEASE_SHA256S
            and option("--allowed-root") == {ALLOWED_GPU_REMOTE_ROOT!r}
        )
        try:
            status_lines = (path.parent / "status").read_text(encoding="utf-8").splitlines()
            ppid = int(next(line.split()[1] for line in status_lines if line.startswith("PPid:")))
        except (OSError, StopIteration, ValueError, IndexError):
            ppid = 0
        runner_candidates.append({{"pid":int(path.parent.name),"ppid":ppid,
                        "command":" ".join(parts)[:500],"competition_ids":existing_ids,
                        "release":runner_release,"owned":owned}})
running = root_runner_processes(runner_candidates)
concurrency_allowed = safe_cpu_light_concurrency(
    {competitions!r}, running, str(release), {allow_concurrent_with_cpu_light!r}
)
if running and not concurrency_allowed:
    print(json.dumps({{"status":"EXISTING_RUNNER","running":running}}, sort_keys=True))
    raise SystemExit(19)
environment = os.environ.copy()
environment.update({runtime_environment!r})
with log_path.open("ab", buffering=0) as log:
    process = subprocess.Popen(argv, cwd=release, env=environment,
        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True, close_fds=True)
state = {{"schema":"evomind.mlebench_remote_ops.run_state.v1",
         "run_id":{run_id!r},"pid":process.pid,"status":"running",
         "created_at":time.time(),"argv":argv,"release":str(release),
         "run_dir":{run_dir!r},"log_path":str(log_path),"human_gate_preserved":True}}
state["concurrent_with"] = running if concurrency_allowed else []
state["performance_contract"] = {{
    "wave2_fast_kernels": "--wave2-fast-kernels" in argv,
    "torch_home": environment["TORCH_HOME"],
    "xdg_cache_home": environment["XDG_CACHE_HOME"],
    "hf_home": environment["HF_HOME"],
    "tmpdir": environment["TMPDIR"],
    "cuda_module_loading": environment["CUDA_MODULE_LOADING"],
    "pytorch_cuda_alloc_conf": environment["PYTORCH_CUDA_ALLOC_CONF"],
}}
state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
print(json.dumps(state, sort_keys=True))
'''
    client = _connect()
    try:
        verify_remote_release(client, verification["sha256"])
        runtime = verify_remote_runtime(client)
        code, output, error = _run_remote(client, _python_command(source), timeout=120)
    finally:
        client.close()
    if code:
        raise RemoteOpsError(f"Remote runner did not start: {error[-500:] or output[-500:]}")
    state = json.loads(output.strip().splitlines()[-1])
    payload = {
        "schema": "evomind.mlebench_remote_ops.start.v1",
        "created_at": utc_now(),
        "gate": gate,
        "bundle_sha256": verification["sha256"],
        "remote": state,
        "unified_runtime": runtime,
        "kaggle_submission_enabled": False,
        "human_gate_preserved": True,
        "passed": True,
    }
    LOCAL_CONTROL_ROOT.mkdir(parents=True, exist_ok=True)
    (LOCAL_CONTROL_ROOT / f"{run_id}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (LOCAL_CONTROL_ROOT / "current.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def start_birds_cpu_precompute(
    bundle: Path,
    *,
    run_id: str,
    seed: int,
    optimization_plan_name: str | None,
    audio_workers: int,
    nice_level: int,
) -> dict[str, Any]:
    stage = stage_bundle_cpu_only(bundle)
    verification = stage["local_bundle"]
    release = release_dir(verification["sha256"])
    run_id = validate_run_id(run_id)
    plan = f"{release}/plans/{optimization_plan_name}" if optimization_plan_name else None
    argv = build_birds_precompute_argv(
        release=release,
        run_id=run_id,
        seed=seed,
        optimization_plan=plan,
        audio_workers=audio_workers,
        nice_level=nice_level,
    )
    state_path = ensure_remote_path(f"{REMOTE_STATE_ROOT}/{run_id}.json")
    log_path = ensure_remote_path(f"{REMOTE_LOG_ROOT}/{run_id}.log")
    run_dir = ensure_remote_path(f"{REMOTE_OUTPUT_ROOT}/{run_id}")
    runtime_environment = build_runner_environment(release, run_id)
    runtime_environment.update({
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": str(max(1, min(8, int(audio_workers)))),
        "OPENBLAS_NUM_THREADS": str(max(1, min(8, int(audio_workers)))),
        "MKL_NUM_THREADS": str(max(1, min(8, int(audio_workers)))),
    })
    source = f'''from __future__ import annotations
import json, os, pathlib, subprocess, time
argv = {argv!r}
release = pathlib.Path({release!r})
state_path = pathlib.Path({state_path!r})
log_path = pathlib.Path({log_path!r})
run_dir = pathlib.Path({run_dir!r})
state_path.parent.mkdir(parents=True, exist_ok=True)
log_path.parent.mkdir(parents=True, exist_ok=True)
if state_path.is_file():
    existing = json.loads(state_path.read_text(encoding="utf-8"))
    summary_path = pathlib.Path(existing.get("run_dir", "")) / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") == "passed":
            existing["idempotent_reuse"] = True
            existing["summary_status"] = summary.get("status")
            print(json.dumps(existing, sort_keys=True))
            raise SystemExit(0)
    pid = int(existing.get("pid") or -1)
    if pid > 0:
        proc_status = pathlib.Path("/proc") / str(pid) / "status"
        try:
            status_lines = proc_status.read_text(encoding="utf-8", errors="replace").splitlines()
            state_code = next(
                line.partition(":")[2].strip()[:1]
                for line in status_lines
                if line.startswith("State:")
            )
        except (OSError, StopIteration):
            state_code = ""
        if state_code in {sorted(BLOCKED_PROCESS_STATES)!r}:
            raise SystemExit("EXISTING_REMOTE_PROCESS_BLOCKED")
        if state_code:
            existing["idempotent_reuse"] = True
            existing["status"] = "running"
            existing["process_state_code"] = state_code
            print(json.dumps(existing, sort_keys=True))
            raise SystemExit(0)
for key in ("TORCH_HOME", "XDG_CACHE_HOME", "HF_HOME", "TMPDIR"):
    pathlib.Path({runtime_environment!r}[key]).mkdir(parents=True, exist_ok=True)
environment = os.environ.copy()
environment.update({runtime_environment!r})
with log_path.open("ab", buffering=0) as log:
    process = subprocess.Popen(argv, cwd=release, env=environment,
        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True, close_fds=True)
state = {{"schema":"evomind.mlebench_remote_ops.run_state.v1",
         "run_id":{run_id!r},"pid":process.pid,"status":"running",
         "created_at":time.time(),"argv":argv,"release":str(release),
         "run_dir":str(run_dir),"log_path":str(log_path),"human_gate_preserved":True,
         "cpu_precompute_only":True,"competition_ids":["mlsp-2013-birds"],
         "gpu_idle_gate_required":False}}
state["performance_contract"] = {{
    "cuda_visible_devices": environment.get("CUDA_VISIBLE_DEVICES"),
    "nice_level": {int(max(0, min(19, int(nice_level))))!r},
    "audio_workers": {int(audio_workers)!r},
    "torch_home": environment["TORCH_HOME"],
    "xdg_cache_home": environment["XDG_CACHE_HOME"],
    "hf_home": environment["HF_HOME"],
    "tmpdir": environment["TMPDIR"],
}}
state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
print(json.dumps(state, sort_keys=True))
'''
    client = _connect()
    try:
        runtime = verify_remote_runtime(client)
        code, output, error = _run_remote(client, _python_command(source), timeout=120)
    finally:
        client.close()
    if code:
        raise RemoteOpsError(f"Remote Birds CPU precompute did not start: {error[-500:] or output[-500:]}")
    state = json.loads(output.strip().splitlines()[-1])
    payload = {
        "schema": "evomind.mlebench_remote_ops.birds_cpu_precompute_start.v1",
        "created_at": utc_now(),
        "bundle_sha256": verification["sha256"],
        "stage": stage,
        "remote": state,
        "unified_runtime": runtime,
        "kaggle_submission_enabled": False,
        "human_gate_preserved": True,
        "gpu_idle_gate_required": False,
        "passed": True,
    }
    LOCAL_CONTROL_ROOT.mkdir(parents=True, exist_ok=True)
    (LOCAL_CONTROL_ROOT / f"{run_id}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def read_remote_status(run_id: str) -> dict[str, Any]:
    run_id = validate_run_id(run_id)
    state_path = ensure_remote_path(f"{REMOTE_STATE_ROOT}/{run_id}.json")
    source = f'''from __future__ import annotations
import json, pathlib, time
state_path = pathlib.Path({state_path!r})
if not state_path.is_file():
    raise SystemExit("REMOTE_STATE_MISSING")
state = json.loads(state_path.read_text(encoding="utf-8"))
pid = int(state["pid"])
proc_status = pathlib.Path("/proc") / str(pid) / "status"
try:
    status_lines = proc_status.read_text(encoding="utf-8", errors="replace").splitlines()
    process_state_text = next(
        line.partition(":")[2].strip()
        for line in status_lines
        if line.startswith("State:")
    )
    process_state_code = process_state_text[:1]
except (OSError, StopIteration):
    process_state_text = ""
    process_state_code = ""
if not process_state_code:
    process = "stopped"
elif process_state_code in {sorted(BLOCKED_PROCESS_STATES)!r}:
    process = "blocked"
elif process_state_code in ["R", "S", "I", "W"]:
    process = "running"
else:
    process = "blocked"
run_dir = pathlib.Path(state["run_dir"])
def read_json(name):
    path = run_dir / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
tail = ""
log_path = pathlib.Path(state["log_path"])
if log_path.is_file():
    with log_path.open("rb") as handle:
        handle.seek(0, 2); size = handle.tell(); handle.seek(max(0, size - 12000))
        tail = handle.read().decode("utf-8", "replace")[-12000:]
now = time.time()
def age(path):
    try:
        return max(0.0, now - pathlib.Path(path).stat().st_mtime)
    except OSError:
        return None
state_age = age(state_path)
log_age = age(log_path)
activity_ages = [value for value in (state_age, log_age) if value is not None]
for name in ("checkpoint.json", "summary.json", "manifest.json"):
    value = age(run_dir / name)
    if value is not None:
        activity_ages.append(value)
activity_age = min(activity_ages) if activity_ages else None
hold_reasons = []
if str(state.get("status") or "").lower() in ["running", "started", "resuming"]:
    if process != "running":
        hold_reasons.append("declared_running_process_not_runnable")
    if activity_age is None or activity_age > {DEFAULT_STATE_LOG_FRESHNESS_SECONDS!r}:
        hold_reasons.append("declared_running_activity_stale")
payload = {{"state":state,"process":process,"process_state_code":process_state_code,
           "process_state":process_state_text,"state_age_seconds":state_age,
           "log_age_seconds":log_age,"activity_age_seconds":activity_age,
           "hold_reasons":hold_reasons,"signals_sent":0,"other_processes_modified":False,
           "manifest":read_json("manifest.json"),
           "checkpoint":read_json("checkpoint.json"),"summary":read_json("summary.json"),
           "log_tail":tail}}
print(json.dumps(payload, sort_keys=True))
'''
    client = _connect()
    try:
        code, output, error = _run_remote(client, _python_command(source), timeout=120)
    finally:
        client.close()
    if code:
        raise RemoteOpsError(f"Remote status failed: {error[-500:] or output[-500:]}")
    payload = json.loads(output.strip().splitlines()[-1])
    target = LOCAL_CONTROL_ROOT / f"{run_id}_status.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    payload["local_status_path"] = str(target.resolve())
    return payload


def collect_run(run_id: str, *, include_checkpoints: bool = False) -> dict[str, Any]:
    status = read_remote_status(run_id)
    state = status["state"]
    remote_run = ensure_remote_path(str(state["run_dir"]))
    local_root = LOCAL_CONTROL_ROOT / "collected" / run_id
    local_root.mkdir(parents=True, exist_ok=True)
    allow_names = {
        "manifest.json", "phase_a_audit.json", "environment.json", "checkpoint.json",
        "results_current.json", "summary.json", "result.json", "submission.csv",
        "submission_validation.json", "private_grader.json", "run.log",
        "may2022_ensemble_diagnostics.json", "may2022_fold_assignments.npz",
        "may2022_oof_ensemble.npz",
        "vision_oof_and_test.npz",
        "taxi_oof_manifest.csv", "taxi_fold_test_predictions.csv",
        "taxi_oof_fold_records.json", "taxi_duplicate_audit.json",
        "siim_fold_ensemble.npz", "siim_oof_predictions.csv",
        "siim_test_components.csv", "siim_training_history.json",
        "siim_nested_patient_folds.json", "siim_duplicate_connected_groups.json",
        "siim_image_content_manifest.csv", "siim_artifact_manifest.json",
        "siim_preprocessing_ablation_gate.json", "promotion_gate.json",
    }
    downloaded: list[dict[str, Any]] = []
    client = _connect()
    try:
        with client.open_sftp() as sftp:
            stack = [(remote_run, local_root)]
            while stack:
                remote_dir, local_dir = stack.pop()
                local_dir.mkdir(parents=True, exist_ok=True)
                for item in sftp.listdir_attr(remote_dir):
                    remote_path = ensure_remote_path(posixpath.join(remote_dir, item.filename))
                    local_path = local_dir / item.filename
                    is_dir = stat.S_ISDIR(item.st_mode)
                    if is_dir:
                        stack.append((remote_path, local_path))
                        continue
                    checkpoint = item.filename.endswith((".pt", ".pth", ".cbm", ".ubj"))
                    if item.filename not in allow_names and not (include_checkpoints and checkpoint):
                        continue
                    sftp.get(remote_path, str(local_path))
                    downloaded.append({
                        "remote": remote_path,
                        "local": str(local_path.resolve()),
                        "bytes": local_path.stat().st_size,
                        "sha256": sha256_file(local_path),
                    })
    finally:
        client.close()
    report = {
        "schema": "evomind.mlebench_remote_ops.collection.v1",
        "created_at": utc_now(),
        "run_id": run_id,
        "remote_run": remote_run,
        "local_root": str(local_root.resolve()),
        "include_checkpoints": include_checkpoints,
        "file_count": len(downloaded),
        "files": downloaded,
        "passed": bool(downloaded),
    }
    (local_root / "collection_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def regrade_run(bundle: Path, run_id: str, *, force: bool) -> dict[str, Any]:
    verification = verify_local_bundle(bundle)
    release = release_dir(verification["sha256"])
    run_id = validate_run_id(run_id)
    status = read_remote_status(run_id)
    if status["process"] == "running":
        raise RemoteOpsError("Regrade requires the training process to be stopped")
    argv = [
        REMOTE_PYTHON, f"{release}/scripts/regrade_mlebench_lite_run.py",
        "--run-id", run_id,
        "--output-root", REMOTE_OUTPUT_ROOT,
        "--data-root", REMOTE_DATA_ROOT,
        "--official-source-root", f"{release}/upstream_mlebench",
        "--allowed-root", ALLOWED_GPU_REMOTE_ROOT,
    ]
    if force:
        argv.append("--force")
    env = f"PYTHONPATH={shlex.quote(build_release_pythonpath(release, include_runtime=True))}"
    command = env + " " + " ".join(shlex.quote(value) for value in argv)
    client = _connect()
    try:
        verify_remote_release(client, verification["sha256"])
        code, output, error = _run_remote(client, command, timeout=3600)
    finally:
        client.close()
    payload = {
        "schema": "evomind.mlebench_remote_ops.regrade.v1",
        "created_at": utc_now(),
        "run_id": run_id,
        "exit_code": code,
        "stdout_tail": output[-4000:],
        "stderr_tail": error[-4000:],
        "training_executed": False,
        "kaggle_submission_executed": False,
        "passed": code == 0,
    }
    if code:
        raise RemoteOpsError(f"Remote regrade failed: {error[-500:] or output[-500:]}")
    return payload


def _write_evidence(name: str, payload: dict[str, Any]) -> Path:
    LOCAL_CONTROL_ROOT.mkdir(parents=True, exist_ok=True)
    target = LOCAL_CONTROL_ROOT / name
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--gate-report", type=Path, default=DEFAULT_GATE_REPORT)
    parser.add_argument("--max-gate-age", type=int, default=600)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("verify-bundle")
    gate = subparsers.add_parser("gpu-gate")
    gate.add_argument("--samples-required", type=int, default=DEFAULT_GATE_SAMPLES)
    gate.add_argument("--interval-seconds", type=int, default=DEFAULT_GATE_INTERVAL_SECONDS)
    gate.add_argument(
        "--max-activity-age-seconds",
        type=int,
        default=DEFAULT_STATE_LOG_FRESHNESS_SECONDS,
    )
    gate.add_argument("--min-free-memory-mib", type=int, default=DEFAULT_MIN_FREE_MEMORY_MIB)
    gate.add_argument(
        "--max-other-process-memory-mib",
        type=int,
        default=DEFAULT_MAX_OTHER_PROCESS_MEMORY_MIB,
    )
    gate.add_argument(
        "--max-memory-growth-mib",
        type=int,
        default=DEFAULT_MAX_MEMORY_GROWTH_MIB,
    )
    gate.add_argument("--max-utilization-percent", type=int, default=5)
    subparsers.add_parser("deploy")
    subparsers.add_parser("cuda-smoke")
    subparsers.add_parser("prepare-runtime")
    subparsers.add_parser("stage-vision-weights")

    start = subparsers.add_parser("start")
    start.add_argument("--run-id", required=True)
    start.add_argument("--waves", required=True)
    start.add_argument("--competitions", default="")
    start.add_argument("--seed", type=int, default=42)
    start.add_argument("--optimization-plan-name")
    start.add_argument("--resume", action="store_true")
    start.add_argument("--allow-concurrent-with-cpu-light", action="store_true")
    start.add_argument("--runner-performance-override", action="append", default=[])
    start.add_argument("--runner-contract-arg", action="append", default=[])

    birds = subparsers.add_parser("precompute-birds")
    birds.add_argument("--run-id", required=True)
    birds.add_argument("--seed", type=int, default=42)
    birds.add_argument("--optimization-plan-name", default="wave2_gpt56_current.json")
    birds.add_argument("--audio-workers", type=int, default=8)
    birds.add_argument("--nice-level", type=int, default=15)

    status = subparsers.add_parser("status")
    status.add_argument("--run-id", required=True)
    collect = subparsers.add_parser("collect")
    collect.add_argument("--run-id", required=True)
    collect.add_argument("--include-checkpoints", action="store_true")
    regrade = subparsers.add_parser("regrade")
    regrade.add_argument("--run-id", required=True)
    regrade.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "verify-bundle":
            payload = verify_local_bundle(args.bundle)
            name = "bundle_verification_current.json"
        elif args.command == "gpu-gate":
            payload = sample_gpu_idle_gate(
                samples_required=args.samples_required,
                interval_seconds=args.interval_seconds,
                max_activity_age_seconds=args.max_activity_age_seconds,
                min_free_memory_mib=args.min_free_memory_mib,
                max_other_process_memory_mib=args.max_other_process_memory_mib,
                max_memory_growth_mib=args.max_memory_growth_mib,
                max_utilization_percent=args.max_utilization_percent,
            )
            name = "gpu_gate_current.json"
        elif args.command == "deploy":
            payload = deploy_bundle(args.bundle, args.gate_report, max_gate_age=args.max_gate_age)
            name = "deployment_current.json"
        elif args.command == "cuda-smoke":
            payload = cuda_smoke(args.bundle, args.gate_report, max_gate_age=args.max_gate_age)
            name = "cuda_smoke_current.json"
        elif args.command == "prepare-runtime":
            payload = prepare_remote_runtime()
            name = "unified_runtime_current.json"
        elif args.command == "stage-vision-weights":
            payload = stage_vision_weights()
            name = "vision_weight_cache_current.json"
        elif args.command == "start":
            payload = start_run(
                args.bundle,
                args.gate_report,
                run_id=args.run_id,
                waves=parse_waves(args.waves),
                competitions=parse_competitions(args.competitions),
                seed=args.seed,
                optimization_plan_name=args.optimization_plan_name,
                resume=args.resume,
                allow_concurrent_with_cpu_light=args.allow_concurrent_with_cpu_light,
                max_gate_age=args.max_gate_age,
                runner_performance_overrides=args.runner_performance_override,
                runner_contract_args=args.runner_contract_arg,
            )
            name = f"{args.run_id}_start.json"
        elif args.command == "precompute-birds":
            payload = start_birds_cpu_precompute(
                args.bundle,
                run_id=args.run_id,
                seed=args.seed,
                optimization_plan_name=args.optimization_plan_name,
                audio_workers=args.audio_workers,
                nice_level=args.nice_level,
            )
            name = f"{args.run_id}_birds_cpu_precompute_start.json"
        elif args.command == "status":
            payload = read_remote_status(args.run_id)
            name = f"{args.run_id}_status_current.json"
        elif args.command == "collect":
            payload = collect_run(args.run_id, include_checkpoints=args.include_checkpoints)
            name = f"{args.run_id}_collection_current.json"
        else:
            payload = regrade_run(args.bundle, args.run_id, force=args.force)
            name = f"{args.run_id}_regrade_current.json"
        evidence = _write_evidence(name, payload)
        print(json.dumps({"evidence": str(evidence.resolve()), **payload}, ensure_ascii=False, indent=2))
        return 0
    except (RemoteOpsError, OSError, ValueError, json.JSONDecodeError) as exc:
        payload = {
            "schema": "evomind.mlebench_remote_ops.failure.v1",
            "created_at": utc_now(),
            "command": args.command,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "passed": False,
        }
        evidence = _write_evidence(f"{args.command}_failure_current.json", payload)
        print(json.dumps({"evidence": str(evidence.resolve()), **payload}, ensure_ascii=False, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
