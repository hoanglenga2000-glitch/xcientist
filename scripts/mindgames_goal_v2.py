from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.machinery
import json
import os
import platform
import random
import shutil
import stat
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, NamedTuple

import numpy as np


SEED = 20260830
EPISODES_PER_ENVIRONMENT = 8
ENVIRONMENTS = (
    ("Codenames-v0", 4, "generalization"),
    ("ColonelBlotto-v0", 2, "generalization"),
    ("ThreePlayerIPD-v0", 3, "generalization"),
    ("SecretMafia-v0", 7, "social_detection"),
)
OFFICIAL_REPOSITORY_COMMIT = "b7ff02ab4a6b01398d186290e0981c37e31ba3ab"
OFFLINE_EVALUATOR_SHA256 = "4c4af5d2ff2e52599f82468b77a9aadcd0b46996d0101b21e3a21c28ac58e693"
EXPECTED_TEXTARENA_VERSION = "0.7.4"
MANAGED_RUNTIME_ENVS = (
    "EVOMIND_MINDGAMES_RUNTIME_ROOT",
    "EVOMIND_MINDGAMES_RUNTIME",
    "MINDGAMES_RUNTIME_ROOT",
)
# Backwards-compatible name retained for callers/tests that inspect the
# canonical selector directly.
MANAGED_RUNTIME_ENV = MANAGED_RUNTIME_ENVS[0]
MANAGED_RUNTIME_RELATIVE = Path(".runtime") / "mindgames"
MANAGED_PACKAGE_NAMES = ("textarena", "trueskill")
FORMAL_PROTOCOL_SCHEMA = "evomind.mindgames.frozen_official_protocol.v1"
FORMAL_PROTOCOL_NAME = "formal-protocol.json"
FORMAL_REFERENCE_NAMES = ("STARS", "tungsten")


class ManagedRuntimeGate(ValueError):
    """A precise gate raised when the existing managed runtime is unusable.

    The runner must never create a venv or install packages.  Keeping this
    error structured lets the CLI emit a redacted receipt without exposing
    absolute paths or import tracebacks.
    """

    def __init__(self, code: str, dependency: str, reason: str) -> None:
        super().__init__(reason)
        self.code = code
        self.dependency = dependency
        self.reason = reason


class FormalProtocol(NamedTuple):
    manifest_path: Path
    manifest_sha256: str
    candidate_model: Path
    reference_models: tuple[tuple[str, Path], ...]
    closure_file_count: int


class ManagedRuntime(NamedTuple):
    """An already-existing, path-bound runtime used for this run only."""

    root: Path
    site_packages: tuple[Path, ...]
    source: str = "adjacent-managed-runtime"

    @property
    def site_package_count(self) -> int:
        return len(self.site_packages)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    data = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _is_link_or_reparse(path: Path) -> bool:
    """Return whether *path* is a symlink or Windows reparse point."""

    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    reparse = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
    return path.is_symlink() or bool(reparse and attributes & reparse)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _safe_relative(value: Any, *, dependency: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ManagedRuntimeGate("FORMAL_PROTOCOL_PATH_INVALID", dependency, "formal protocol path is missing")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or any(part in {"", "."} for part in path.parts):
        raise ManagedRuntimeGate("FORMAL_PROTOCOL_PATH_INVALID", dependency, "formal protocol path is unsafe")
    return path


def _validate_formal_closure(runtime: ManagedRuntime, manifest: dict[str, Any]) -> tuple[dict[str, Path], int]:
    roots = manifest.get("closure_roots")
    files = manifest.get("files")
    if not isinstance(roots, list) or not roots or not isinstance(files, list) or not files:
        raise ManagedRuntimeGate("FORMAL_PROTOCOL_FILE_MANIFEST_MISSING", "formal_protocol", "formal protocol file closure is missing")
    runtime_root = runtime.root.resolve(strict=True)
    resolved_roots: list[Path] = []
    for value in roots:
        relative = _safe_relative(value, dependency="formal_protocol")
        candidate = runtime.root / relative
        if _has_link_or_reparse_between(candidate, runtime.root):
            raise ManagedRuntimeGate("FORMAL_PROTOCOL_PATH_ESCAPE", "formal_protocol", "formal protocol closure contains a link")
        try:
            resolved = candidate.resolve(strict=True)
        except (FileNotFoundError, OSError) as exc:
            raise ManagedRuntimeGate("FORMAL_PROTOCOL_ASSET_MISSING", "formal_protocol", "formal protocol closure root is missing") from exc
        if not resolved.is_dir() or not _is_within(resolved, runtime_root):
            raise ManagedRuntimeGate("FORMAL_PROTOCOL_PATH_ESCAPE", "formal_protocol", "formal protocol closure root escapes runtime")
        resolved_roots.append(resolved)
    declared: dict[str, dict[str, Any]] = {}
    for row in files:
        if not isinstance(row, dict):
            raise ManagedRuntimeGate("FORMAL_PROTOCOL_FILE_ENTRY_INVALID", "formal_protocol", "formal protocol file entry is invalid")
        relative = _safe_relative(row.get("path"), dependency="formal_protocol").as_posix()
        if relative in declared or not isinstance(row.get("bytes"), int) or isinstance(row.get("bytes"), bool):
            raise ManagedRuntimeGate("FORMAL_PROTOCOL_FILE_ENTRY_INVALID", "formal_protocol", "formal protocol file entry is invalid")
        digest = str(row.get("sha256") or "").lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ManagedRuntimeGate("FORMAL_PROTOCOL_FILE_ENTRY_INVALID", "formal_protocol", "formal protocol file SHA is invalid")
        declared[relative] = row
    actual: dict[str, Path] = {}
    for closure_root in resolved_roots:
        for path in closure_root.rglob("*"):
            if not path.is_file():
                continue
            if _is_link_or_reparse(path) or _has_link_or_reparse_between(path, runtime.root):
                raise ManagedRuntimeGate("FORMAL_PROTOCOL_PATH_ESCAPE", "formal_protocol", "formal protocol asset is linked")
            relative = path.resolve(strict=True).relative_to(runtime_root).as_posix()
            actual[relative] = path
    if set(actual) != set(declared):
        raise ManagedRuntimeGate("FORMAL_PROTOCOL_FILE_CLOSURE_MISMATCH", "formal_protocol", "formal protocol file closure has missing or extra files")
    for relative, path in actual.items():
        row = declared[relative]
        if path.stat().st_size != int(row["bytes"]) or sha256_file(path) != str(row["sha256"]).lower():
            raise ManagedRuntimeGate("FORMAL_PROTOCOL_FILE_HASH_MISMATCH", "formal_protocol", "formal protocol asset bytes or SHA drifted")
    return actual, len(actual)


def load_frozen_formal_protocol(data_root: Path, runtime: ManagedRuntime, official_evaluator: Path) -> FormalProtocol:
    manifest_path = runtime.root / FORMAL_PROTOCOL_NAME
    if not manifest_path.is_file() or _is_link_or_reparse(manifest_path):
        raise ManagedRuntimeGate("FORMAL_PROTOCOL_MANIFEST_MISSING", "formal_protocol", "frozen formal protocol manifest is unavailable")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManagedRuntimeGate("FORMAL_PROTOCOL_MANIFEST_INVALID", "formal_protocol", "frozen formal protocol manifest is invalid") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != FORMAL_PROTOCOL_SCHEMA
        or manifest.get("frozen") is not True
        or manifest.get("offline_only") is not True
        or manifest.get("trust_remote_code") is not False
        or manifest.get("official_repository_commit") != OFFICIAL_REPOSITORY_COMMIT
        or manifest.get("official_evaluator_sha256") != OFFLINE_EVALUATOR_SHA256
        or manifest.get("textarena_version") != EXPECTED_TEXTARENA_VERSION
        or not isinstance(manifest.get("trueskill_version"), str)
        or not manifest.get("trueskill_version")
        or manifest.get("environments") != [value[0] for value in ENVIRONMENTS]
        or manifest.get("episodes_per_environment") != EPISODES_PER_ENVIRONMENT
        or manifest.get("seed") != SEED
        or manifest.get("seat_balanced") is not True
        or manifest.get("candidate_model_id") != "Qwen/Qwen3-8B"
    ):
        raise ManagedRuntimeGate("FORMAL_PROTOCOL_CONTRACT_MISMATCH", "formal_protocol", "frozen formal protocol identity drifted")
    if sha256_file(official_evaluator) != OFFLINE_EVALUATOR_SHA256:
        raise ManagedRuntimeGate("OFFICIAL_EVALUATOR_MISSING_OR_DRIFTED", "official_evaluator", "official evaluator SHA drifted")
    actual, count = _validate_formal_closure(runtime, manifest)
    candidate_relative = _safe_relative(manifest.get("candidate_model_dir"), dependency="qwen3_8b")
    candidate = (runtime.root / candidate_relative).resolve(strict=True)
    if not candidate.is_dir() or not _is_within(candidate, runtime.root.resolve(strict=True)):
        raise ManagedRuntimeGate("FROZEN_QWEN3_MODEL_MISSING", "qwen3_8b", "frozen Qwen3-8B model is missing")
    references = manifest.get("reference_agents")
    if not isinstance(references, list) or len(references) != len(FORMAL_REFERENCE_NAMES):
        raise ManagedRuntimeGate("REFERENCE_AGENT_MANIFEST_MISSING", "reference_agents", "frozen reference agents are missing")
    by_name = {str(row.get("name")): row for row in references if isinstance(row, dict)}
    if set(by_name) != set(FORMAL_REFERENCE_NAMES):
        raise ManagedRuntimeGate("REFERENCE_AGENT_MANIFEST_MISMATCH", "reference_agents", "reference agent allowlist drifted")
    loaded: list[tuple[str, Path]] = []
    for name in FORMAL_REFERENCE_NAMES:
        if by_name[name].get("agent_type") != "HFLocalAgent":
            raise ManagedRuntimeGate("REFERENCE_AGENT_MANIFEST_MISMATCH", name, "reference agent type drifted")
        relative = _safe_relative(by_name[name].get("model_dir"), dependency=name)
        model = (runtime.root / relative).resolve(strict=True)
        if not model.is_dir() or not _is_within(model, runtime.root.resolve(strict=True)):
            raise ManagedRuntimeGate("REFERENCE_AGENT_ASSET_MISSING", name, "frozen reference agent model is missing")
        loaded.append((name, model))
    required_prefixes = [candidate_relative.as_posix(), *[str(row[1].relative_to(runtime.root.resolve(strict=True))).replace("\\", "/") for row in loaded]]
    if any(not any(path == prefix or path.startswith(prefix + "/") for path in actual) for prefix in required_prefixes):
        raise ManagedRuntimeGate("FORMAL_PROTOCOL_FILE_CLOSURE_MISMATCH", "formal_protocol", "model roots are not SHA-closed")
    return FormalProtocol(manifest_path, sha256_file(manifest_path), candidate, tuple(loaded), count)


def _has_link_or_reparse_between(path: Path, root: Path) -> bool:
    """Check existing path components from *root* through *path*.

    Missing leaves are allowed during discovery.  Every existing component is
    checked before ``resolve`` so a symlink cannot redirect the runtime outside
    the authorized data-root sibling.
    """

    absolute = path.expanduser().absolute()
    anchor = root.expanduser().absolute()
    if not _is_within(absolute, anchor):
        return True
    relative_parts = absolute.relative_to(anchor).parts
    current = anchor
    if _is_link_or_reparse(current):
        return True
    for part in relative_parts:
        current = current / part
        if _is_link_or_reparse(current):
            return True
    return False


def _safe_data_root(data_root: Path) -> Path:
    """Resolve the supplied data root while rejecting link/reparse escapes."""

    candidate = Path(data_root).expanduser().absolute()
    if _is_link_or_reparse(candidate):
        raise ManagedRuntimeGate("MANAGED_RUNTIME_PATH_ESCAPE", "textarena", "data root is a link or reparse point")
    try:
        resolved = candidate.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise ManagedRuntimeGate("RUNTIME_DATA_ROOT_MISSING", "textarena", "data root is unavailable") from exc
    if not resolved.is_dir() or _is_link_or_reparse(resolved):
        raise ManagedRuntimeGate("RUNTIME_DATA_ROOT_INVALID", "textarena", "data root is not a regular directory")
    return resolved


def _runtime_root_candidates(data_root: Path) -> tuple[Path, ...]:
    """Return only runtime roots authorized by the competition data layout.

    The adapter creates ``<competition-data-parent>/.runtime/mindgames``.  An
    optional environment selector is accepted only when it remains below that
    same parent ``.runtime`` directory; it never broadens the authorized scope.
    """

    root = _safe_data_root(data_root)
    parent = root.parent
    allowed_runtime_parent = parent / ".runtime"
    raw_override = next((os.getenv(name, "").strip() for name in MANAGED_RUNTIME_ENVS if os.getenv(name, "").strip()), "")
    raw_candidates: list[Path] = []
    if raw_override:
        raw_candidates.append(Path(raw_override).expanduser())
    raw_candidates.append(parent / MANAGED_RUNTIME_RELATIVE)
    result: list[Path] = []
    seen: set[str] = set()
    for raw in raw_candidates:
        # Relative selectors are interpreted from the authorized data parent,
        # never from the caller's working directory.
        absolute = (parent / raw if not raw.is_absolute() else raw).absolute()
        # The override is a selector, never permission to broaden the root.
        if not _is_within(absolute, allowed_runtime_parent.absolute()):
            raise ManagedRuntimeGate("MANAGED_RUNTIME_PATH_ESCAPE", "textarena", "managed runtime is outside the authorized data scope")
        if absolute.name != "mindgames":
            raise ManagedRuntimeGate("MANAGED_RUNTIME_PATH_INVALID", "textarena", "managed runtime name is not mindgames")
        if _has_link_or_reparse_between(absolute, allowed_runtime_parent):
            raise ManagedRuntimeGate("MANAGED_RUNTIME_PATH_ESCAPE", "textarena", "managed runtime contains a link or reparse point")
        try:
            resolved = absolute.resolve(strict=False)
        except OSError as exc:
            raise ManagedRuntimeGate("MANAGED_RUNTIME_PATH_INVALID", "textarena", "managed runtime path cannot be resolved") from exc
        if not _is_within(resolved, allowed_runtime_parent.resolve()) or resolved.name != "mindgames":
            raise ManagedRuntimeGate("MANAGED_RUNTIME_PATH_ESCAPE", "textarena", "managed runtime resolves outside the authorized data scope")
        key = os.path.normcase(str(resolved))
        if key not in seen:
            seen.add(key)
            result.append(resolved)
    return tuple(result)


def _site_package_candidates(runtime_root: Path) -> tuple[Path, ...]:
    """Find conventional venv site-package directories without recursion."""

    roots: list[Path] = [runtime_root / "site-packages", runtime_root / "Lib" / "site-packages"]
    for lib_name in ("lib", "lib64"):
        lib_root = runtime_root / lib_name
        if not lib_root.is_dir() or _is_link_or_reparse(lib_root):
            continue
        try:
            children = sorted(lib_root.iterdir(), key=lambda item: item.name.casefold())
        except OSError:
            continue
        for child in children:
            if child.is_dir() and child.name.casefold().startswith("python"):
                roots.append(child / "site-packages")
    found: list[Path] = []
    seen: set[str] = set()
    for candidate in roots:
        if not candidate.is_dir():
            continue
        if _is_link_or_reparse(candidate) or _has_link_or_reparse_between(candidate, runtime_root):
            raise ManagedRuntimeGate("MANAGED_RUNTIME_PATH_ESCAPE", "textarena", "site-packages contains a link or reparse point")
        resolved = candidate.resolve(strict=True)
        if not _is_within(resolved, runtime_root.resolve(strict=True)):
            raise ManagedRuntimeGate("MANAGED_RUNTIME_PATH_ESCAPE", "textarena", "site-packages resolves outside managed runtime")
        key = os.path.normcase(str(resolved))
        if key not in seen:
            seen.add(key)
            found.append(resolved)
    return tuple(found)


def discover_managed_runtime(data_root: Path) -> ManagedRuntime:
    """Discover the existing MindGames runtime, never provision one."""

    roots = _runtime_root_candidates(data_root)
    existing = [root for root in roots if root.is_dir()]
    if not existing:
        raise ManagedRuntimeGate(
            "RUNTIME_DEPENDENCY_MISSING",
            "textarena",
            "the existing adjacent MindGames runtime is unavailable",
        )
    for root in existing:
        if _is_link_or_reparse(root) or _has_link_or_reparse_between(root, root.parent.parent):
            raise ManagedRuntimeGate("MANAGED_RUNTIME_PATH_ESCAPE", "textarena", "managed runtime contains a link or reparse point")
        sites = _site_package_candidates(root)
        if sites:
            return ManagedRuntime(root=root, site_packages=sites)
    raise ManagedRuntimeGate(
        "RUNTIME_SITE_PACKAGES_MISSING",
        "textarena",
        "the existing MindGames runtime has no conventional site-packages directory",
    )


def discover_managed_site_packages(data_root: Path) -> tuple[Path, ...]:
    """Return the validated site-package roots for callers that need only paths."""

    return discover_managed_runtime(data_root).site_packages


def _module_is_within(module: Any, site_packages: tuple[Path, ...]) -> bool:
    """Verify an imported module came from one of the managed site roots."""

    origins: list[Path] = []
    origin = getattr(module, "__file__", None)
    if origin:
        origins.append(Path(origin))
    locations = getattr(module, "__path__", None)
    if locations:
        origins.extend(Path(item) for item in locations)
    if not origins:
        return False
    for item in origins:
        try:
            resolved = item.expanduser().resolve(strict=True)
        except (FileNotFoundError, OSError):
            return False
        if _is_link_or_reparse(item) or not any(
            _is_within(resolved, site) and not _has_link_or_reparse_between(item, site)
            for site in site_packages
        ):
            return False
        if not any(_is_within(resolved, site) for site in site_packages):
            return False
    return True


def _spec_is_within(spec: Any, site_packages: tuple[Path, ...]) -> bool:
    """Validate an import spec before executing package code."""

    origins: list[Path] = []
    origin = getattr(spec, "origin", None)
    if origin and origin not in {"built-in", "frozen"}:
        origins.append(Path(origin))
    locations = getattr(spec, "submodule_search_locations", None)
    if locations:
        origins.extend(Path(item) for item in locations)
    if not origins:
        return False
    for item in origins:
        try:
            resolved = item.expanduser().resolve(strict=True)
        except (FileNotFoundError, OSError):
            return False
        if _is_link_or_reparse(item):
            return False
        if not any(
            _is_within(resolved, site) and not _has_link_or_reparse_between(item, site)
            for site in site_packages
        ):
            return False
    return True


def load_managed_dependencies(data_root: Path) -> tuple[Any, Any, ManagedRuntime]:
    """Import TextArena and TrueSkill exclusively from the existing runtime."""

    runtime = discover_managed_runtime(data_root)
    previous_path = list(sys.path)
    previous_dont_write_bytecode = sys.dont_write_bytecode
    # Importing a source package can otherwise create __pycache__ files in the
    # supervisor-owned runtime; dependency discovery must remain read-only.
    sys.dont_write_bytecode = True
    for site in reversed(runtime.site_packages):
        sys.path.insert(0, str(site))
    try:
        modules: list[Any] = []
        for dependency in MANAGED_PACKAGE_NAMES:
            existing = sys.modules.get(dependency)
            if existing is not None and not _module_is_within(existing, runtime.site_packages):
                raise ManagedRuntimeGate("RUNTIME_MODULE_ORIGIN_INVALID", dependency, "dependency is loaded outside managed runtime")
            spec = importlib.machinery.PathFinder.find_spec(
                dependency,
                [str(site) for site in runtime.site_packages],
            )
            if spec is None:
                raise ManagedRuntimeGate("RUNTIME_DEPENDENCY_MISSING", dependency, "managed runtime dependency is unavailable")
            if not _spec_is_within(spec, runtime.site_packages):
                raise ManagedRuntimeGate("RUNTIME_MODULE_ORIGIN_INVALID", dependency, "dependency spec resolved outside managed runtime")
            try:
                module = importlib.import_module(dependency)
            except (ImportError, ModuleNotFoundError) as exc:
                raise ManagedRuntimeGate("RUNTIME_DEPENDENCY_MISSING", dependency, "managed runtime dependency is unavailable") from exc
            except Exception as exc:
                raise ManagedRuntimeGate("RUNTIME_DEPENDENCY_IMPORT_FAILED", dependency, "managed runtime dependency failed to import") from exc
            if not _module_is_within(module, runtime.site_packages):
                raise ManagedRuntimeGate("RUNTIME_MODULE_ORIGIN_INVALID", dependency, "dependency resolved outside managed runtime")
            modules.append(module)
        textarena = modules[0]
        declared_version = getattr(textarena, "__version__", None)
        if declared_version is not None and str(declared_version) != EXPECTED_TEXTARENA_VERSION:
            raise ManagedRuntimeGate("RUNTIME_DEPENDENCY_VERSION_MISMATCH", "textarena", "managed TextArena version is not the pinned protocol version")
        return textarena, modules[1], runtime
    finally:
        sys.path[:] = previous_path
        sys.dont_write_bytecode = previous_dont_write_bytecode


# Keep a descriptive alias for integrations that call this operation an
# import/discovery step rather than a loader.
import_managed_dependencies = load_managed_dependencies


def strategy_prompt(observation: Any) -> str:
    return (
        "You are being evaluated in a fixed MindGames environment. Follow the environment's exact action format. "
        "Reason privately, then return one concise legal action only. Preserve role and teammate constraints. "
        "For repeated games, use reciprocity and avoid exploitable fixed patterns. For allocation games, distribute "
        "resources according to opponent uncertainty rather than uniformly. For word games, maximize clue coverage "
        "while minimizing forbidden-word risk. For social deduction, separate public evidence from role-conditioned "
        "private information and never invent observations.\n\nOBSERVATION:\n"
        + str(observation)
    )


class StrategyWrapper:
    def __init__(self, base_agent: Callable[[Any], str]) -> None:
        self.base_agent = base_agent

    def __call__(self, observation: Any) -> str:
        return self.base_agent(strategy_prompt(observation))


def run_episode(
    make_environment: Callable[[str], Any],
    environment_id: str,
    player_count: int,
    candidate: Callable[[Any], str],
    baseline: Callable[[Any], str],
    *,
    candidate_seat: int,
    seed: int,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    environment = make_environment(environment_id)
    environment.reset(num_players=player_count)
    done = False
    while not done:
        player_id, observation = environment.get_observation()
        action = candidate(observation) if player_id == candidate_seat else baseline(observation)
        done, _ = environment.step(action=action)
    rewards, game_info = environment.close()
    candidate_reward = float(rewards[candidate_seat])
    opponent_reward = float(np.mean([rewards[index] for index in range(player_count) if index != candidate_seat]))
    info = game_info[candidate_seat] if isinstance(game_info, (list, tuple, dict)) else {}
    if isinstance(game_info, dict):
        info = game_info.get(candidate_seat, game_info.get(str(candidate_seat), {}))
    return {
        "candidate_reward": candidate_reward,
        "opponent_reward": opponent_reward,
        "win": candidate_reward > opponent_reward,
        "loss": candidate_reward < opponent_reward,
        "draw": candidate_reward == opponent_reward,
        "invalid_move": bool((info or {}).get("invalid_move", False)),
        "turn_count": int((info or {}).get("turn_count", 0)),
    }


def paired_bootstrap(differences: np.ndarray, rounds: int, seed: int) -> dict[str, float]:
    if differences.size < 4:
        raise ValueError("insufficient paired episodes")
    generator = np.random.default_rng(seed)
    values = []
    for _ in range(rounds):
        indices = generator.integers(0, len(differences), len(differences))
        values.append(float(np.mean(differences[indices])))
    array = np.asarray(values)
    return {
        "rounds": rounds,
        "mean": float(array.mean()),
        "standard_error": float(array.std(ddof=1)),
        "ci95_lower": float(np.quantile(array, 0.025)),
        "ci95_upper": float(np.quantile(array, 0.975)),
    }


def model_candidates(data_root: Path | None = None) -> list[Path]:
    """Find an existing local model, preferring the bound managed runtime.

    No network lookup or model provisioning occurs here.  Runtime-local model
    locations are limited to fixed descendants of the already validated
    ``.runtime/mindgames`` root; explicit model environment variables retain
    compatibility with previously provisioned, regular directories.
    """

    candidates: list[Path] = []
    for name in ("MINDGAMES_MODEL_PATH", "QWEN3_MODEL_PATH"):
        value = os.getenv(name, "").strip()
        if value:
            candidates.append(Path(value).expanduser())
    if data_root is not None:
        try:
            runtime = discover_managed_runtime(data_root)
        except ManagedRuntimeGate as exc:
            if exc.code.startswith("MANAGED_RUNTIME_PATH"):
                raise
            runtime = None
        if runtime is not None:
            model_roots = (
                runtime.root / "models" / "Qwen3-8B",
                runtime.root / "models" / "Qwen" / "Qwen3-8B",
                runtime.root / "models--Qwen--Qwen3-8B" / "snapshots",
                runtime.root / "hub" / "models--Qwen--Qwen3-8B" / "snapshots",
            )
            for model_root in model_roots:
                if model_root.name == "snapshots" and model_root.is_dir():
                    candidates.extend(sorted(item for item in model_root.iterdir() if item.is_dir()))
                else:
                    candidates.append(model_root)
    home = Path.home()
    roots = [
        Path(os.getenv("HF_HOME", "")).expanduser() if os.getenv("HF_HOME") else None,
        home / ".cache" / "huggingface",
    ]
    for root in roots:
        if root is None:
            continue
        snapshots = root / "hub" / "models--Qwen--Qwen3-8B" / "snapshots"
        if snapshots.is_dir():
            candidates.extend(sorted(path for path in snapshots.iterdir() if path.is_dir()))
    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        try:
            if not path.is_dir() or _is_link_or_reparse(path):
                continue
            resolved = path.resolve(strict=True)
        except (FileNotFoundError, OSError):
            continue
        key = os.path.normcase(str(resolved))
        if key not in seen:
            seen.add(key)
            unique.append(resolved)
    return unique


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    started = utc_now()
    data_root = Path(args.data_dir).resolve(strict=True)
    output_root = Path(args.out_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    official_evaluator = data_root / "starter-kit" / "src" / "offline_evaluation.py"
    if not official_evaluator.is_file():
        official_evaluator = data_root / "src" / "offline_evaluation.py"
    if not official_evaluator.is_file() or sha256_file(official_evaluator) != OFFLINE_EVALUATOR_SHA256:
        write_json(
            output_root / "exact-gate.json",
            {
                "schema": "evomind.mindgames.exact_gate.v2",
                "code": "OFFICIAL_EVALUATOR_MISSING_OR_DRIFTED",
                "required_sha256": OFFLINE_EVALUATOR_SHA256,
                "training_performed": False,
            },
        )
        return 0

    try:
        ta, trueskill, managed_runtime = load_managed_dependencies(data_root)
    except ManagedRuntimeGate as exc:
        write_json(
            output_root / "exact-gate.json",
            {
                "schema": "evomind.mindgames.exact_gate.v2",
                "code": exc.code,
                "dependency": exc.dependency,
                "runtime_source": "adjacent-managed-runtime",
                "reason": exc.reason,
                "required_action": "make the existing SHA-bound managed runtime available; do not install online",
                "resume_point": "managed_runtime_dependency_discovery",
                "automatic_install_performed": False,
                "terms_accepted": False,
                "training_performed": False,
            },
        )
        return 0

    try:
        formal_protocol = load_frozen_formal_protocol(data_root, managed_runtime, official_evaluator)
    except ManagedRuntimeGate as exc:
        write_json(
            output_root / "exact-gate.json",
            {
                "schema": "evomind.mindgames.exact_gate.v2",
                "code": exc.code,
                "dependency": exc.dependency,
                "reason": exc.reason,
                "resume_point": "frozen_formal_protocol_validation",
                "automatic_download_performed": False,
                "automatic_install_performed": False,
                "terms_accepted": False,
                "training_performed": False,
            },
        )
        return 0

    model_path = formal_protocol.candidate_model
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    base_agent = ta.agents.HFLocalAgent(model_name=str(model_path), max_new_tokens=512)
    candidate_agent = StrategyWrapper(base_agent)
    reference_agents = [
        (name, ta.agents.HFLocalAgent(model_name=str(model), max_new_tokens=512))
        for name, model in formal_protocol.reference_models
    ]
    episode_rows: list[dict[str, Any]] = []
    candidate_rating = trueskill.Rating()
    baseline_rating = trueskill.Rating()
    for environment_id, player_count, track in ENVIRONMENTS:
        for reference_name, reference_agent in reference_agents:
            for episode in range(EPISODES_PER_ENVIRONMENT):
                seat = episode % player_count
                seed = SEED + 1000 * list(value[0] for value in ENVIRONMENTS).index(environment_id) + episode
                outcome = run_episode(
                    ta.make,
                    environment_id,
                    player_count,
                    candidate_agent,
                    reference_agent,
                    candidate_seat=seat,
                    seed=seed,
                )
                if outcome["win"]:
                    candidate_rating, baseline_rating = trueskill.rate_1vs1(candidate_rating, baseline_rating)
                elif outcome["loss"]:
                    baseline_rating, candidate_rating = trueskill.rate_1vs1(baseline_rating, candidate_rating)
                else:
                    candidate_rating, baseline_rating = trueskill.rate_1vs1(candidate_rating, baseline_rating, drawn=True)
                episode_rows.append(
                    {
                        "environment": environment_id,
                        "track": track,
                        "reference_agent": reference_name,
                        "episode": episode,
                        "seed": seed,
                        "candidate_seat": seat,
                        **outcome,
                    }
                )

    differences = np.asarray(
        [row["candidate_reward"] - row["opponent_reward"] for row in episode_rows], dtype=float
    )
    bootstrap = paired_bootstrap(differences, 2000, SEED + 99)
    candidate_reward = float(np.mean([row["candidate_reward"] for row in episode_rows]))
    baseline_reward = float(np.mean([row["opponent_reward"] for row in episode_rows]))
    minimum_margin = max(0.01 * max(abs(baseline_reward), 1.0), bootstrap["standard_error"])
    internal_gate = bool(bootstrap["ci95_lower"] > minimum_margin and candidate_rating.mu > baseline_rating.mu)
    protocol_gate = bool(candidate_rating.mu >= 26.8 and bootstrap["ci95_lower"] > minimum_margin)
    official_gate = {
        "generalization_required_trueskill": 26.8,
        "social_detection_required_trueskill_mean": 23.8,
        "candidate_local_trueskill_mu": float(candidate_rating.mu),
        "candidate_local_trueskill_sigma": float(candidate_rating.sigma),
        "protocol_comparable": True,
        "passed": protocol_gate,
        "reason": "frozen official evaluator, dependency closure, Qwen3-8B candidate, and STARS/tungsten reference pool verified",
        "formal_protocol_manifest_sha256": formal_protocol.manifest_sha256,
    }

    source_path = Path(__file__).resolve(strict=True)
    source_sha = sha256_file(source_path)
    task_contract = {
        "schema": "evomind.mindgames.task_contract.v2",
        "environments": [value[0] for value in ENVIRONMENTS],
        "episodes_per_environment": EPISODES_PER_ENVIRONMENT,
        "role_balance": True,
        "fixed_seeds": True,
        "metrics": ["reward difference", "win/draw/loss rate", "invalid move rate", "TrueSkill"],
        "external_submission": False,
        "runtime_dependency_policy": "existing_managed_runtime_only",
        "formal_protocol_manifest_sha256": formal_protocol.manifest_sha256,
        "reference_agents": [name for name, _model in formal_protocol.reference_models],
    }
    baseline_evidence = {
        "schema": "evomind.mindgames.baseline_evidence.v2",
        "official_repository_commit": OFFICIAL_REPOSITORY_COMMIT,
        "official_evaluator_sha256": OFFLINE_EVALUATOR_SHA256,
        "frozen_reference_agents": [name for name, _model in formal_protocol.reference_models],
        "official_generalization_reference_trueskill": 26.8,
        "official_social_reference_trueskill_mean": 23.8,
        "runtime_dependency_source": managed_runtime.source,
        "runtime_site_package_count": managed_runtime.site_package_count,
        "classification": "OFFICIAL_ORGANIZER_BASELINE",
        "formal_protocol_manifest_sha256": formal_protocol.manifest_sha256,
    }
    metrics = {
        "schema": "evomind.mindgames.metrics.v2",
        "candidate_mean_reward": candidate_reward,
        "baseline_mean_reward": baseline_reward,
        "candidate_trueskill_mu": float(candidate_rating.mu),
        "candidate_trueskill_sigma": float(candidate_rating.sigma),
        "baseline_trueskill_mu": float(baseline_rating.mu),
        "baseline_trueskill_sigma": float(baseline_rating.sigma),
        "bootstrap": bootstrap,
        "internal_canary_gate": internal_gate,
        "official_gate": official_gate,
    }
    independent = {
        "schema": "evomind.mindgames.independent_verification.v2",
        "episode_count": len(episode_rows),
        "seat_balance": {
            environment_id: sorted(
                Counter(row["candidate_seat"] for row in episode_rows if row["environment"] == environment_id).items()
            )
            for environment_id, _count, _track in ENVIRONMENTS
        },
        "seed_count": len({row["seed"] for row in episode_rows}),
        "external_submission": False,
        "official_gate_passed": protocol_gate,
        "internal_canary_gate_passed": internal_gate,
    }

    shutil.copyfile(source_path, output_root / "solution.py")
    write_json(output_root / "task-contract-v2.json", task_contract)
    write_json(output_root / "baseline-evidence.json", baseline_evidence)
    write_json(output_root / "episode-results.json", episode_rows)
    write_json(output_root / "metrics.json", metrics)
    write_json(output_root / "candidate-vs-baseline.json", {"internal_gate": internal_gate, "official_gate": official_gate})
    write_json(output_root / "independent-verification.json", independent)
    write_json(
        output_root / "environment-lock.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "textarena": getattr(ta, "__version__", None),
            "trueskill": getattr(trueskill, "__version__", None),
            "runtime_source": managed_runtime.source,
            "runtime_site_package_count": managed_runtime.site_package_count,
            "model_path_sha256": hashlib.sha256(str(model_path).encode()).hexdigest(),
            "formal_protocol_manifest_sha256": formal_protocol.manifest_sha256,
            "formal_protocol_closure_file_count": formal_protocol.closure_file_count,
        },
    )
    with (output_root / "training.log").open("w", encoding="utf-8", newline="\n") as handle:
        for row in (
            {"at_utc": started, "event": "start", "source_sha256": source_sha},
            {"at_utc": utc_now(), "event": "episodes_complete", "episodes": len(episode_rows)},
            {"at_utc": utc_now(), "event": "independent_verification", "internal_gate": internal_gate},
        ):
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    write_json(
        output_root / "retrospective-memory.json",
        {
            "schema": "evomind.retrospective_memory.v2",
            "candidate": "Qwen3-8B strategy prompt wrapper",
            "internal_canary_gate": internal_gate,
            "official_gate": protocol_gate,
            "next_step": "independent review" if protocol_gate else "revise only on development data and require a fresh frozen protocol",
            "memory_writeback_allowed": protocol_gate,
        },
    )
    files: list[dict[str, Any]] = []
    for path in sorted(output_root.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name not in {"artifact-manifest.json", "artifact-manifest-receipt.json"}:
            files.append({"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = {
        "schema": "evomind.mindgames.artifact_manifest.v2",
        "started_at_utc": started,
        "completed_at_utc": utc_now(),
        "source_sha256": source_sha,
        "exit_code": 0,
        "internal_canary_gate": internal_gate,
        "official_gate": protocol_gate,
        "files": files,
    }
    write_json(output_root / "artifact-manifest.json", manifest)
    write_json(
        output_root / "artifact-manifest-receipt.json",
        {"artifact": "artifact-manifest.json", "bytes": (output_root / "artifact-manifest.json").stat().st_size, "sha256": sha256_file(output_root / "artifact-manifest.json")},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
