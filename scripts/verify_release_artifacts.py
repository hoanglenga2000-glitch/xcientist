from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import stat
import sys
import tarfile
import zipfile
from email import policy
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote

_SOURCE_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_PACKAGE_DIR = _SOURCE_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))
if str(_SOURCE_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(_SOURCE_PACKAGE_DIR))

from research_os.benchmark_manager import (  # noqa: E402
    BenchmarkRegistryError,
    validate_task_registry_payload,
)
from scripts.mle_bench_split75_contract import (  # noqa: E402
    Split75ContractError,
    verify_split75_contract,
)

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


class VerificationError(RuntimeError):
    pass


_PORTABILITY_EXEMPT_PATHS = {
    "scripts/build_reproducible_submission_package.py",
    "scripts/verify_external_resources_manifest.py",
    "scripts/verify_security_invariants.py",
}
_PORTABILITY_PATTERNS = {
    "personal_hpc_mount": re.compile(r"/hpc2(?:hdd|ssd)/", re.IGNORECASE),
    "personal_hpc_account": re.compile(r"\b" + "aims" + r"lab\b", re.IGNORECASE),
    "personal_workspace_name": re.compile(r"(?:~[/\\])?" + "jing" + "hw", re.IGNORECASE),
    "windows_user_home": re.compile(r"C:\\Users\\[^\\/\r\n]+", re.IGNORECASE),
    "release_validation_path": re.compile("EvoMind-" + "release-validation", re.IGNORECASE),
}
_PORTABLE_TEXT_SUFFIXES = {
    ".bat", ".cmd", ".js", ".json", ".jsx", ".md", ".mjs", ".ps1", ".py",
    ".sh", ".toml", ".ts", ".tsx", ".txt", ".yaml", ".yml",
}
_BACKUP_MEMBER_PATTERN = re.compile(r"(?:^|[._-])(?:bak|backup)(?:$|[._-])", re.IGNORECASE)
_CRITICAL_ARCHIVE_SOURCE_PATHS = (
    "configs/schemas/benchmark_task.schema.json",
    "scripts/mle_bench_split75_contract.py",
    "scripts/run_ci_checks.py",
    "src/research_os/benchmark_manager.py",
)


def _is_backup_member(path: PurePosixPath) -> bool:
    for part in path.parts:
        part_path = PurePosixPath(part)
        if (
            _BACKUP_MEMBER_PATTERN.search(part) is not None
            or part_path.suffix.casefold() in {".orig", ".rej", ".tmp"}
            or part.endswith("~")
        ):
            return True
    return False


def _normalized_text_sha256(data: bytes, *, context: str) -> str:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise VerificationError(f"{context} is not UTF-8 text: {exc}") from exc
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def _single(paths: list[Path], label: str) -> Path:
    if len(paths) != 1:
        names = ", ".join(path.name for path in paths) or "none"
        raise VerificationError(f"expected exactly one {label}, found {names}")
    return paths[0]


def _license_member(names: list[str], archive_name: str) -> str:
    candidates = [name for name in names if PurePosixPath(name).name.casefold() == "license"]
    if len(candidates) != 1:
        found = ", ".join(candidates) or "none"
        raise VerificationError(f"{archive_name} must contain exactly one LICENSE file, found {found}")
    return candidates[0]


def _canonical_project_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).casefold()


def _project_identity(pyproject: Path) -> tuple[str, str]:
    try:
        payload = tomllib.loads(pyproject.read_text(encoding="utf-8-sig"))
        project = payload["project"]
        name = str(project["name"])
        version = str(project["version"])
    except (OSError, UnicodeError, KeyError, TypeError, tomllib.TOMLDecodeError) as exc:
        raise VerificationError(f"cannot read project identity from {pyproject}: {exc}") from exc
    if not name or not version:
        raise VerificationError(f"project name/version is empty in {pyproject}")
    return name, version


def _npm_project_identity(package_json: Path) -> tuple[str, str]:
    payload = _read_json(package_json)
    if not isinstance(payload, dict):
        raise VerificationError(f"{package_json} is not an npm package JSON object")
    name = payload.get("name")
    version = payload.get("version")
    if not isinstance(name, str) or not name.strip():
        raise VerificationError(f"npm package name is empty in {package_json}")
    if not isinstance(version, str) or not version.strip():
        raise VerificationError(f"npm package version is empty in {package_json}")
    return name, version


def _verify_metadata(
    data: bytes,
    archive_name: str,
    expected_name: str,
    expected_version: str,
) -> dict[str, Any]:
    metadata = BytesParser(policy=policy.default).parsebytes(data)
    name = str(metadata.get("Name") or "")
    version = str(metadata.get("Version") or "")
    license_expression = metadata.get("License-Expression")
    license_files = metadata.get_all("License-File", [])
    if _canonical_project_name(name) != _canonical_project_name(expected_name):
        raise VerificationError(f"{archive_name} has Name={name!r}, expected {expected_name!r}")
    if version != expected_version:
        raise VerificationError(f"{archive_name} has Version={version!r}, expected {expected_version!r}")
    if license_expression != "MIT":
        raise VerificationError(f"{archive_name} has License-Expression={license_expression!r}, expected 'MIT'")
    if not any(PurePosixPath(value).name.casefold() == "license" for value in license_files):
        raise VerificationError(f"{archive_name} metadata does not declare License-File: LICENSE")
    return {
        "name": name,
        "version": version,
        "license_expression": license_expression,
        "license_files": license_files,
    }


def verify_distribution_licenses(
    dist_dir: Path,
    source_license: Path,
    expected_name: str,
    expected_version: str,
) -> dict[str, Any]:
    expected = source_license.read_bytes()
    wheel = _single(sorted(dist_dir.glob("*.whl")), "wheel")
    sdist = _single(sorted(dist_dir.glob("*.tar.gz")), "source distribution")

    with zipfile.ZipFile(wheel) as archive:
        wheel_member = _license_member(archive.namelist(), wheel.name)
        wheel_license = archive.read(wheel_member)
        metadata_member = _single(
            [Path(name) for name in archive.namelist() if name.endswith(".dist-info/METADATA")],
            "wheel METADATA",
        ).as_posix()
        wheel_metadata = _verify_metadata(
            archive.read(metadata_member),
            wheel.name,
            expected_name,
            expected_version,
        )
    if wheel_license != expected:
        raise VerificationError(f"{wheel.name}:{wheel_member} does not match {source_license}")

    with tarfile.open(sdist, mode="r:gz") as archive:
        sdist_member_name = _license_member(archive.getnames(), sdist.name)
        member = archive.getmember(sdist_member_name)
        extracted = archive.extractfile(member)
        if extracted is None:
            raise VerificationError(f"cannot read {sdist.name}:{sdist_member_name}")
        sdist_license = extracted.read()
        pkg_info_name = _single(
            [
                Path(name)
                for name in archive.getnames()
                if PurePosixPath(name).name == "PKG-INFO" and len(PurePosixPath(name).parts) == 2
            ],
            "source distribution PKG-INFO",
        ).as_posix()
        pkg_info = archive.extractfile(archive.getmember(pkg_info_name))
        if pkg_info is None:
            raise VerificationError(f"cannot read {sdist.name}:{pkg_info_name}")
        sdist_metadata = _verify_metadata(
            pkg_info.read(),
            sdist.name,
            expected_name,
            expected_version,
        )
    if sdist_license != expected:
        raise VerificationError(f"{sdist.name}:{sdist_member_name} does not match {source_license}")

    return {
        "wheel_license": wheel_member,
        "sdist_license": sdist_member_name,
        "wheel_metadata": wheel_metadata,
        "sdist_metadata": sdist_metadata,
    }


def verify_workstation_source_bundle(dist_dir: Path, source_license: Path) -> dict[str, Any]:
    bundle = _single(sorted(dist_dir.glob("*-workstation-source.zip")), "workstation source bundle")
    expected_license = source_license.read_text(encoding="utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    required_paths = {
        ".env.example",
        "benchmark/mle_bench_75/UPSTREAM.json",
        "benchmark/mle_bench_75/openai_split75_507f92e.txt",
        "benchmark/mle_bench_75/tasks_template.json",
        "configs/schemas/benchmark_task.schema.json",
        "Dockerfile",
        "LICENSE",
        "README.md",
        "SECURITY.md",
        "docker-compose.yml",
        "docs/NEW_USER_ONBOARDING_GUIDE.md",
        "docs/RELEASE_CHECKLIST.md",
        "install.ps1",
        "pyproject.toml",
        "requirements.txt",
        "scripts/dpapi_credential_store.ps1",
        "scripts/install_autokaggle_cli.ps1",
        "scripts/mle_bench_split75_contract.py",
        "scripts/mlebench_server_runner.py",
        "scripts/manage_deepseek_secret.ps1",
        "scripts/manage_hpc_proxy_bridge.ps1",
        "scripts/manage_hpc_ssh_secret.ps1",
        "scripts/manage_kaggle_secret.ps1",
        "scripts/manage_workstation_dashboard.py",
        "scripts/hpc_socks_bridge.py",
        "scripts/quick_setup.ps1",
        "scripts/accept_kaggle_rules_all75.py",
        "scripts/restart_workstation_frontend.ps1",
        "scripts/run_ci_checks.py",
        "scripts/run_new_user_release_acceptance.ps1",
        "scripts/start_hpc_socks_bridge.py",
        "scripts/start_verified_workstation.ps1",
        "scripts/verify_backend_resource_status.py",
        "scripts/verify_new_user_release_readiness.py",
        "scripts/verify_no_plaintext_secrets.py",
        "scripts/verify_security_invariants.py",
        "scripts/verify_verified_workstation_launch_audit.py",
        "scripts/verify_workstation_launch_readiness.py",
        "scripts/verify_workstation_ui_truthfulness.py",
        "src/xsci/dashboard.py",
        "src/research_os/benchmark_manager.py",
        "web/research-agent-workstation/package-lock.json",
        "web/research-agent-workstation/package.json",
        "web/research-agent-workstation/scripts/verify-kaggle-status-contract.mjs",
        "web/research-agent-workstation/src/lib/connector-status.ts",
        "web/research-agent-workstation/src/lib/server/kaggle-status.ts",
        "web/research-agent-workstation/src/middleware.ts",
    }
    forbidden_parts = {".git", ".next", "__pycache__", "_quarantine", "node_modules"}

    with zipfile.ZipFile(bundle) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise VerificationError(f"{bundle.name} contains duplicate member names")
        for info in infos:
            unix_mode = info.external_attr >> 16
            if stat.S_IFMT(unix_mode) == stat.S_IFLNK:
                raise VerificationError(f"{bundle.name} contains a symbolic link: {info.filename!r}")
        member_paths = [PurePosixPath(name) for name in names]
        for name, member_path in zip(names, member_paths):
            if "\\" in name or member_path.is_absolute() or ".." in member_path.parts:
                raise VerificationError(f"{bundle.name} contains an unsafe member path: {name!r}")
            if forbidden_parts.intersection(member_path.parts):
                raise VerificationError(f"{bundle.name} contains a forbidden generated directory: {name!r}")
            if _is_backup_member(member_path):
                raise VerificationError(f"{bundle.name} contains a forbidden backup file: {name!r}")

        file_names = [name for name in names if not name.endswith("/")]
        paths = [PurePosixPath(name) for name in file_names]
        for name, member_path in zip(file_names, paths):
            if member_path.name == ".env" or member_path.suffix.casefold() in {".key", ".pem", ".p12", ".pfx"}:
                raise VerificationError(f"{bundle.name} contains a forbidden credential file: {name!r}")

        roots = {member.parts[0] for member in paths if member.parts}
        if len(roots) != 1:
            raise VerificationError(f"{bundle.name} must contain exactly one root directory, found {sorted(roots)}")
        root = next(iter(roots))
        expected_root = bundle.name.removesuffix("-workstation-source.zip")
        if root != expected_root:
            raise VerificationError(f"{bundle.name} root is {root!r}, expected {expected_root!r}")
        relative_names = {PurePosixPath(*member.parts[1:]).as_posix() for member in paths}
        missing = sorted(required_paths - relative_names)
        if missing:
            raise VerificationError(f"{bundle.name} is missing workstation files: {missing}")

        critical_source_hashes: dict[str, str] = {}
        for relative in _CRITICAL_ARCHIVE_SOURCE_PATHS:
            workspace_path = _SOURCE_ROOT / relative
            try:
                workspace_bytes = workspace_path.read_bytes()
                archive_bytes = archive.read(f"{root}/{relative}")
            except (OSError, KeyError) as exc:
                raise VerificationError(
                    f"cannot compare critical archived source {relative}: {exc}"
                ) from exc
            workspace_digest = _normalized_text_sha256(
                workspace_bytes,
                context=f"workspace:{relative}",
            )
            archive_digest = _normalized_text_sha256(
                archive_bytes,
                context=f"{bundle.name}:{relative}",
            )
            if archive_digest != workspace_digest:
                raise VerificationError(
                    f"{bundle.name}:{relative} critical source differs from the verifier workspace: "
                    f"archive={archive_digest}, workspace={workspace_digest}"
                )
            critical_source_hashes[relative] = archive_digest

        for name, member_path in zip(file_names, paths):
            relative = PurePosixPath(*member_path.parts[1:]).as_posix()
            if relative in _PORTABILITY_EXEMPT_PATHS or member_path.suffix.casefold() not in _PORTABLE_TEXT_SUFFIXES:
                continue
            try:
                text = archive.read(name).decode("utf-8-sig")
            except UnicodeDecodeError:
                continue
            for rule, pattern in _PORTABILITY_PATTERNS.items():
                if pattern.search(text):
                    raise VerificationError(f"{bundle.name}:{relative} violates release portability rule {rule}")

        license_text = archive.read(f"{root}/LICENSE").decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
        if license_text != expected_license:
            raise VerificationError(f"{bundle.name} LICENSE does not match {source_license}")

        registry_member = f"{root}/benchmark/mle_bench_75/tasks_template.json"
        try:
            registry_payload = json.loads(archive.read(registry_member).decode("utf-8-sig"))
            upstream_payload = json.loads(
                archive.read(f"{root}/benchmark/mle_bench_75/UPSTREAM.json").decode("utf-8-sig")
            )
            manifest_bytes = archive.read(
                f"{root}/benchmark/mle_bench_75/openai_split75_507f92e.txt"
            )
        except (KeyError, UnicodeError, json.JSONDecodeError) as exc:
            raise VerificationError(
                f"{bundle.name} MLE-Bench metadata is unreadable: {exc}"
            ) from exc

        try:
            split75_contract = verify_split75_contract(
                manifest_bytes=manifest_bytes,
                upstream=upstream_payload,
                registry=registry_payload,
                runner_source=archive.read(f"{root}/scripts/mlebench_server_runner.py").decode("utf-8-sig"),
                rules_source=archive.read(
                    f"{root}/scripts/accept_kaggle_rules_all75.py"
                ).decode("utf-8-sig"),
            )
        except (KeyError, UnicodeError, json.JSONDecodeError, Split75ContractError) as exc:
            raise VerificationError(f"{bundle.name} MLE-Bench split75 contract is invalid: {exc}") from exc

        try:
            registry_tasks = validate_task_registry_payload(
                registry_payload,
                official_competition_ids=split75_contract["official_competition_ids"],
            )
        except (KeyError, BenchmarkRegistryError) as exc:
            raise VerificationError(
                f"{bundle.name}:benchmark/mle_bench_75/tasks_template.json is invalid: {exc}"
            ) from exc

        compose = archive.read(f"{root}/docker-compose.yml").decode("utf-8")
        for mapping in ("127.0.0.1:3090:3090", "127.0.0.1:8088:3090"):
            if mapping not in compose:
                raise VerificationError(f"{bundle.name} does not enforce loopback mapping {mapping}")
        helper_source = archive.read(f"{root}/scripts/dpapi_credential_store.ps1").decode("utf-8-sig")
        for required_token in (
            "Export-Clixml",
            "Import-EvoMindHpcCredential",
            "Resolve-EvoMindHpcGenerationPayload",
            "icacls.exe",
            "System.Threading.Mutex",
            "Commit-EvoMindCredentialFiles",
        ):
            if required_token not in helper_source:
                raise VerificationError(f"{bundle.name}:scripts/dpapi_credential_store.ps1 lacks {required_token}")
        manager_markers = {
            "scripts/manage_deepseek_secret.ps1": ("Import-Clixml",),
            "scripts/manage_kaggle_secret.ps1": ("Import-Clixml",),
            "scripts/manage_hpc_ssh_secret.ps1": (
                "Get-EvoMindHpcCredentialStorePaths",
                "Resolve-EvoMindHpcCredentialGeneration",
                "New-EvoMindHpcCredentialGeneration",
                "Remove-EvoMindHpcCredentialStore",
            ),
        }
        for manager, specific_markers in manager_markers.items():
            manager_source = archive.read(f"{root}/{manager}").decode("utf-8-sig")
            for required_token in (
                "dpapi_credential_store.ps1",
                "Enter-EvoMindCredentialStoreLock",
                "SecretFromStdin",
                *specific_markers,
            ):
                if required_token not in manager_source:
                    raise VerificationError(f"{bundle.name}:{manager} lacks {required_token}")

    return {
        "archive": bundle.name,
        "root": root,
        "files": len(file_names),
        "mle_bench_registered_tasks": len(registry_tasks),
        "mle_bench_official_tasks": split75_contract["official_tasks"],
        "mle_bench_local_official_overlap": split75_contract["local_official_overlap"],
        "critical_source_sha256": critical_source_hashes,
    }


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"invalid JSON evidence {path}: {exc}") from exc


def _dependency_coordinate(name: str, version: str, ecosystem: str) -> tuple[str, str]:
    normalized_name = name.strip()
    normalized_version = version.strip()
    if ecosystem == "pypi":
        normalized_name = _canonical_project_name(normalized_name)
    elif ecosystem != "npm":
        raise ValueError(f"unsupported package ecosystem: {ecosystem!r}")
    return normalized_name, normalized_version


def _purl_coordinate(purl: str, ecosystem: str) -> tuple[str, str]:
    prefix = f"pkg:{ecosystem}/"
    if not purl.startswith(prefix):
        raise ValueError(f"purl is not in the {ecosystem!r} ecosystem")
    package_and_version = purl[len(prefix):].split("?", 1)[0].split("#", 1)[0]
    if "@" not in package_and_version:
        raise ValueError("purl has no version")
    name, version = package_and_version.rsplit("@", 1)
    if not name or not version:
        raise ValueError("purl has an empty name or version")
    return _dependency_coordinate(unquote(name), unquote(version), ecosystem)


def _cyclonedx_component_name(component: dict[str, Any], ecosystem: str) -> str:
    name = component.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("component name is malformed")
    group = component.get("group")
    if group is not None and (not isinstance(group, str) or not group.strip()):
        raise ValueError("component group is malformed")
    if ecosystem == "npm" and group:
        return f"{group.strip()}/{name.strip()}"
    return name.strip()


def _flatten_cyclonedx_components(path: Path, components: list[Any]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    pending = list(reversed(components))
    while pending:
        component = pending.pop()
        if not isinstance(component, dict):
            raise VerificationError(f"{path} contains a malformed dependency component")
        nested = component.get("components", [])
        if not isinstance(nested, list):
            raise VerificationError(f"{path} contains a malformed nested component list")
        flattened.append(component)
        pending.extend(reversed(nested))
    return flattened


def python_site_packages_inventory(site_packages: Path) -> set[tuple[str, str]]:
    if not site_packages.is_dir():
        raise VerificationError(f"Python site-packages directory does not exist: {site_packages}")
    inventory: set[tuple[str, str]] = set()
    for distribution in importlib.metadata.distributions(path=[str(site_packages)]):
        name = str(distribution.metadata.get("Name") or "").strip()
        version = str(distribution.version or "").strip()
        if not name or not version:
            raise VerificationError(f"installed Python distribution has incomplete metadata under {site_packages}")
        inventory.add(_dependency_coordinate(name, version, "pypi"))
    if not inventory:
        raise VerificationError(f"Python site-packages inventory is empty: {site_packages}")
    return inventory


def npm_ls_inventory(
    path: Path,
    expected_name: str,
    expected_version: str,
) -> set[tuple[str, str]]:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise VerificationError(f"{path} is not an npm ls JSON object")
    if payload.get("name") != expected_name or payload.get("version") != expected_version:
        raise VerificationError(f"{path} root does not match {expected_name} {expected_version}")
    problems = payload.get("problems")
    if problems is not None:
        if not isinstance(problems, list) or any(not isinstance(problem, str) for problem in problems):
            raise VerificationError(f"{path} contains malformed npm problem records")
        if problems:
            raise VerificationError(f"{path} reports an invalid npm dependency tree: {problems[:10]}")
    inventory: set[tuple[str, str]] = set()

    def walk(node: dict[str, Any]) -> None:
        dependencies = node.get("dependencies", {})
        if not isinstance(dependencies, dict):
            raise VerificationError(f"{path} contains malformed npm dependencies")
        for name, child in dependencies.items():
            if not isinstance(name, str) or not name.strip() or not isinstance(child, dict):
                raise VerificationError(f"{path} contains malformed npm dependency entries")
            # npm ls represents optional or peer dependencies that are not installed as empty objects.
            if not child:
                continue
            version = child.get("version")
            if not isinstance(version, str) or not version.strip():
                raise VerificationError(f"{path} dependency {name!r} has no version")
            inventory.add(_dependency_coordinate(name, version, "npm"))
            walk(child)

    walk(payload)
    if not inventory:
        raise VerificationError(f"{path} npm inventory is empty")
    return inventory


def verify_pip_audit(
    path: Path,
    expected_packages: set[tuple[str, str]] | None = None,
) -> dict[str, int]:
    report = _read_json(path)
    if not isinstance(report, dict) or not isinstance(report.get("dependencies"), list):
        raise VerificationError(f"{path} is not a supported pip-audit JSON report")
    if not report["dependencies"]:
        raise VerificationError(f"{path} contains no audited Python dependencies")

    vulnerabilities = 0
    audited: set[tuple[str, str]] = set()
    for index, dependency in enumerate(report["dependencies"]):
        if not isinstance(dependency, dict):
            raise VerificationError(f"{path} contains a malformed dependency record at index {index}")
        name = dependency.get("name")
        version = dependency.get("version")
        vulns = dependency.get("vulns")
        if (
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(version, str)
            or not version.strip()
            or not isinstance(vulns, list)
        ):
            raise VerificationError(f"{path} contains a malformed dependency record at index {index}")
        for vulnerability in vulns:
            if not isinstance(vulnerability, dict) or not isinstance(vulnerability.get("id"), str):
                raise VerificationError(f"{path} contains a malformed vulnerability for dependency {name!r}")
            if not vulnerability["id"].strip():
                raise VerificationError(f"{path} contains a vulnerability with an empty id for dependency {name!r}")
        vulnerabilities += len(dependency["vulns"])
        audited.add(_dependency_coordinate(name, version, "pypi"))
    if vulnerabilities:
        raise VerificationError(f"{path} reports {vulnerabilities} Python vulnerabilities")
    if expected_packages is not None and audited != expected_packages:
        missing = sorted(expected_packages - audited)
        unexpected = sorted(audited - expected_packages)
        raise VerificationError(
            f"{path} does not cover the installed Python tree; missing={missing[:10]} unexpected={unexpected[:10]}"
        )

    return {"dependencies": len(report["dependencies"]), "vulnerabilities": vulnerabilities}


_NPM_AUDIT_SCOPES = {
    "full": {"schemaVersion": 1, "kind": "npm-audit", "scope": "full", "omit": []},
    "production": {
        "schemaVersion": 1,
        "kind": "npm-audit",
        "scope": "production",
        "omit": ["dev"],
    },
}


def verify_npm_audit(path: Path, expected_scope: str) -> dict[str, Any]:
    if expected_scope not in _NPM_AUDIT_SCOPES:
        raise ValueError(f"unsupported npm audit scope: {expected_scope!r}")
    report = _read_json(path)
    if not isinstance(report, dict):
        raise VerificationError(f"{path} is not an npm audit JSON object")
    if report.get("auditReportVersion") != 2:
        raise VerificationError(f"{path} is not an npm audit reportVersion 2 document")
    if report.get("_evomindEvidence") != _NPM_AUDIT_SCOPES[expected_scope]:
        raise VerificationError(f"{path} does not declare npm audit scope {expected_scope!r}")
    if not isinstance(report.get("vulnerabilities"), dict):
        raise VerificationError(f"{path} is missing the npm vulnerabilities object")
    metadata = report.get("metadata")
    counts = metadata.get("vulnerabilities") if isinstance(metadata, dict) else None
    required = ("info", "low", "moderate", "high", "critical", "total")
    if not isinstance(counts, dict) or any(
        not isinstance(counts.get(key), int) or isinstance(counts.get(key), bool) or counts[key] < 0 for key in required
    ):
        raise VerificationError(f"{path} is missing npm vulnerability counts")
    if counts["total"] != sum(counts[key] for key in required if key != "total"):
        raise VerificationError(f"{path} has inconsistent npm vulnerability counts")
    dependency_counts = metadata.get("dependencies") if isinstance(metadata, dict) else None
    dependency_keys = ("prod", "dev", "optional", "peer", "peerOptional", "total")
    if not isinstance(dependency_counts, dict) or any(
        not isinstance(dependency_counts.get(key), int)
        or isinstance(dependency_counts.get(key), bool)
        or dependency_counts[key] < 0
        for key in dependency_keys
    ):
        raise VerificationError(f"{path} is missing npm dependency counts")
    if counts["total"]:
        raise VerificationError(f"{path} reports {counts['total']} total npm vulnerabilities")
    if report["vulnerabilities"]:
        raise VerificationError(f"{path} has vulnerability records despite a zero total")
    return {
        "scope": expected_scope,
        "vulnerabilities": {key: counts[key] for key in required},
        "dependencies": {key: dependency_counts[key] for key in dependency_keys},
    }


def verify_npm_audits(full_path: Path, production_path: Path) -> dict[str, Any]:
    if full_path.resolve() == production_path.resolve():
        raise VerificationError("full and production npm audits must be different files")
    return {
        "full": verify_npm_audit(full_path, "full"),
        "production": verify_npm_audit(production_path, "production"),
    }


def verify_cyclonedx_sbom(
    path: Path,
    expected_name: str,
    expected_version: str,
    expected_purl_type: str,
    expected_components: set[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    report = _read_json(path)
    if not isinstance(report, dict) or report.get("bomFormat") != "CycloneDX":
        raise VerificationError(f"{path} is not a CycloneDX JSON BOM")
    spec_version = str(report.get("specVersion") or "")
    if re.fullmatch(r"1\.(?:[4-9]|[1-9][0-9]+)", spec_version) is None:
        raise VerificationError(f"{path} has unsupported CycloneDX specVersion={spec_version!r}")

    metadata = report.get("metadata")
    components = report.get("components")
    dependencies = report.get("dependencies")
    if not isinstance(metadata, dict) or not isinstance(metadata.get("component"), dict):
        raise VerificationError(f"{path} is missing the CycloneDX metadata root component")
    if not isinstance(components, list) or not components:
        raise VerificationError(f"{path} has no dependency components")
    if not isinstance(dependencies, list) or not dependencies:
        raise VerificationError(f"{path} has no dependency graph")

    root_component = metadata.get("component")
    root_name = root_component.get("name")
    root_version = root_component.get("version")
    root_ref = root_component.get("bom-ref")
    root_purl = root_component.get("purl")
    try:
        root_identity = _dependency_coordinate(
            _cyclonedx_component_name(root_component, expected_purl_type),
            str(root_version or ""),
            expected_purl_type,
        )
        expected_root_identity = _dependency_coordinate(
            expected_name,
            expected_version,
            expected_purl_type,
        )
    except ValueError as exc:
        raise VerificationError(f"{path} metadata root identity is malformed: {exc}") from exc
    if (
        not isinstance(root_name, str)
        or not isinstance(root_version, str)
        or root_identity != expected_root_identity
    ):
        raise VerificationError(f"{path} metadata root is not {expected_name} {expected_version}")
    purl_prefix = f"pkg:{expected_purl_type}/"
    if root_purl is not None and (not isinstance(root_purl, str) or not root_purl.startswith(purl_prefix)):
        raise VerificationError(f"{path} metadata root purl is not in the {expected_purl_type!r} ecosystem")
    if isinstance(root_purl, str):
        try:
            root_purl_identity = _purl_coordinate(root_purl, expected_purl_type)
        except ValueError as exc:
            raise VerificationError(f"{path} metadata root purl is malformed: {exc}") from exc
        if root_purl_identity != root_identity:
            raise VerificationError(f"{path} metadata root purl does not match its name/version")
    if not isinstance(root_ref, str) or not root_ref.strip():
        raise VerificationError(f"{path} metadata root has no bom-ref")

    flattened_components = _flatten_cyclonedx_components(path, components)
    malformed_components = [
        component
        for component in flattened_components
        if not isinstance(component.get("name"), str)
        or not component["name"].strip()
        or not isinstance(component.get("version"), str)
        or not component["version"].strip()
        or not isinstance(component.get("bom-ref"), str)
        or not component["bom-ref"].strip()
        or not isinstance(component.get("purl"), str)
        or not component["purl"].startswith(purl_prefix)
    ]
    if malformed_components:
        raise VerificationError(f"{path} contains malformed or non-{expected_purl_type} dependency components")
    component_inventory: set[tuple[str, str]] = set()
    for component in flattened_components:
        try:
            declared_coordinate = _dependency_coordinate(
                _cyclonedx_component_name(component, expected_purl_type),
                str(component["version"]),
                expected_purl_type,
            )
            purl_coordinate = _purl_coordinate(str(component["purl"]), expected_purl_type)
        except ValueError as exc:
            raise VerificationError(f"{path} contains a malformed dependency component: {exc}") from exc
        if purl_coordinate != declared_coordinate:
            raise VerificationError(f"{path} contains a component whose purl does not match its name/version")
        component_inventory.add(declared_coordinate)
    if expected_components is not None and component_inventory != expected_components:
        missing = sorted(expected_components - component_inventory)
        unexpected = sorted(component_inventory - expected_components)
        raise VerificationError(
            f"{path} does not cover the installed dependency tree; "
            f"missing={missing[:10]} unexpected={unexpected[:10]}"
        )

    graph: dict[str, list[str]] = {}
    for item in dependencies:
        if not isinstance(item, dict):
            raise VerificationError(f"{path} contains a malformed dependency graph entry")
        ref = item.get("ref")
        depends_on = item.get("dependsOn", [])
        if (
            not isinstance(ref, str)
            or not ref.strip()
            or not isinstance(depends_on, list)
            or any(not isinstance(value, str) or not value.strip() for value in depends_on)
        ):
            raise VerificationError(f"{path} contains a malformed dependency graph entry")
        if ref in graph:
            raise VerificationError(f"{path} contains duplicate dependency graph ref {ref!r}")
        graph[ref] = depends_on
    root_dependencies = graph.get(root_ref)
    if not root_dependencies:
        raise VerificationError(f"{path} root component has no dependency edges")

    component_ref_list = [str(component["bom-ref"]) for component in flattened_components]
    component_refs = set(component_ref_list)
    if len(component_refs) != len(component_ref_list):
        raise VerificationError(f"{path} contains duplicate dependency component bom-ref values")
    known_refs = component_refs | {root_ref}
    unknown_graph_refs = sorted(set(graph) - known_refs)
    if unknown_graph_refs:
        raise VerificationError(f"{path} dependency graph contains unknown refs: {unknown_graph_refs[:10]}")
    unresolved = sorted(
        {dependency_ref for depends_on in graph.values() for dependency_ref in depends_on} - known_refs
    )
    if unresolved:
        raise VerificationError(f"{path} dependency refs are missing components: {unresolved[:10]}")
    missing_graph_entries = sorted(component_refs - set(graph))
    if missing_graph_entries:
        raise VerificationError(f"{path} components are missing dependency graph entries: {missing_graph_entries[:10]}")

    reachable: set[str] = set()
    pending_refs = [root_ref]
    while pending_refs:
        ref = pending_refs.pop()
        if ref in reachable:
            continue
        reachable.add(ref)
        pending_refs.extend(graph.get(ref, []))
    unreachable = sorted(component_refs - reachable)
    if unreachable:
        raise VerificationError(f"{path} contains components unreachable from the root: {unreachable[:10]}")

    return {
        "format": "CycloneDX",
        "spec_version": spec_version,
        "components": len(flattened_components),
        "dependencies": len(dependencies),
        "edges": sum(len(item.get("dependsOn") or []) for item in dependencies),
        "root_dependencies": len(root_dependencies),
        "project": {
            "name": expected_name,
            "version": expected_version,
            "purl_type": expected_purl_type,
        },
    }


_CHECKSUM_LINE = re.compile(r"^([0-9a-fA-F]{64})  ([^\\/]+)$")


def verify_sha256_manifest(dist_dir: Path, manifest: Path) -> dict[str, int]:
    entries: dict[str, str] = {}
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8-sig").splitlines(), start=1):
        match = _CHECKSUM_LINE.fullmatch(line)
        if match is None:
            raise VerificationError(f"invalid checksum line {manifest}:{line_number}")
        digest, name = match.groups()
        if name in entries:
            raise VerificationError(f"duplicate checksum entry for {name}")
        entries[name] = digest.casefold()

    expected_names = {
        path.name for path in dist_dir.iterdir() if path.is_file() and path.resolve() != manifest.resolve()
    }
    if set(entries) != expected_names:
        missing = sorted(expected_names - set(entries))
        extra = sorted(set(entries) - expected_names)
        raise VerificationError(f"checksum coverage mismatch: missing={missing}, extra={extra}")

    for name, expected_digest in entries.items():
        actual_digest = hashlib.sha256((dist_dir / name).read_bytes()).hexdigest()
        if actual_digest != expected_digest:
            raise VerificationError(f"SHA256 mismatch for {name}")
    return {"files": len(entries)}


def verify_reproducible_distributions(dist_dir: Path, comparison_dir: Path) -> dict[str, Any]:
    primary = {
        _single(sorted(dist_dir.glob("*.whl")), "wheel").name,
        _single(sorted(dist_dir.glob("*.tar.gz")), "source distribution").name,
    }
    comparison = {
        _single(sorted(comparison_dir.glob("*.whl")), "comparison wheel").name,
        _single(sorted(comparison_dir.glob("*.tar.gz")), "comparison source distribution").name,
    }
    if primary != comparison:
        raise VerificationError(
            f"distribution names differ: primary={sorted(primary)}, comparison={sorted(comparison)}"
        )

    hashes: dict[str, str] = {}
    for name in sorted(primary):
        first = hashlib.sha256((dist_dir / name).read_bytes()).hexdigest()
        second = hashlib.sha256((comparison_dir / name).read_bytes()).hexdigest()
        if first != second:
            raise VerificationError(f"distribution is not reproducible: {name}")
        hashes[name] = first
    return {"files": len(hashes), "sha256": hashes}


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify release archives and security evidence.")
    parser.add_argument("--dist", type=Path, required=True)
    parser.add_argument("--source-license", type=Path, default=Path("LICENSE"))
    parser.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))
    parser.add_argument(
        "--npm-package",
        type=Path,
        default=Path("web/research-agent-workstation/package.json"),
    )
    parser.add_argument(
        "--npm-lock",
        type=Path,
        default=Path("web/research-agent-workstation/package-lock.json"),
    )
    parser.add_argument("--pip-audit", type=Path)
    parser.add_argument("--python-site-packages", type=Path)
    parser.add_argument("--npm-audit-full", type=Path)
    parser.add_argument("--npm-audit-production", type=Path)
    parser.add_argument("--python-sbom", type=Path)
    parser.add_argument("--npm-sbom", type=Path)
    parser.add_argument("--npm-inventory", type=Path)
    parser.add_argument("--sha256sums", type=Path)
    parser.add_argument("--require-workstation-source", action="store_true")
    parser.add_argument("--compare-dist", type=Path)
    args = parser.parse_args()

    if not args.dist.is_dir():
        raise VerificationError(f"distribution directory does not exist: {args.dist}")
    if not args.source_license.is_file():
        raise VerificationError(f"source LICENSE does not exist: {args.source_license}")
    if not args.pyproject.is_file():
        raise VerificationError(f"pyproject does not exist: {args.pyproject}")

    npm_audit_paths = (args.npm_audit_full, args.npm_audit_production)
    if any(npm_audit_paths) and not all(npm_audit_paths):
        raise VerificationError("both --npm-audit-full and --npm-audit-production are required")
    sbom_paths = (args.python_sbom, args.npm_sbom)
    if any(sbom_paths) and not all(sbom_paths):
        raise VerificationError("both --python-sbom and --npm-sbom are required")
    if all(sbom_paths) and args.python_sbom.resolve() == args.npm_sbom.resolve():
        raise VerificationError("Python and npm SBOMs must be different files")

    expected_name, expected_version = _project_identity(args.pyproject)
    python_inventory = None
    if args.pip_audit or args.python_sbom:
        if args.python_site_packages is None:
            raise VerificationError("--python-site-packages is required with Python audit/SBOM evidence")
        python_inventory = python_site_packages_inventory(args.python_site_packages)

    summary: dict[str, Any] = {
        "licenses": verify_distribution_licenses(
            args.dist,
            args.source_license,
            expected_name,
            expected_version,
        ),
    }
    workstation_bundles = sorted(args.dist.glob("*-workstation-source.zip"))
    if workstation_bundles or args.require_workstation_source:
        summary["workstation_source"] = verify_workstation_source_bundle(args.dist, args.source_license)
    if args.pip_audit:
        if args.pip_audit.resolve().parent != args.dist.resolve():
            raise VerificationError(f"pip audit must be inside the distribution directory: {args.pip_audit}")
        summary["pip_audit"] = verify_pip_audit(args.pip_audit, python_inventory)
    if all(npm_audit_paths):
        for audit_path in npm_audit_paths:
            if audit_path.resolve().parent != args.dist.resolve():
                raise VerificationError(f"npm audit must be inside the distribution directory: {audit_path}")
        summary["npm_audit"] = verify_npm_audits(*npm_audit_paths)
    if all(sbom_paths):
        for sbom_path in sbom_paths:
            if sbom_path.resolve().parent != args.dist.resolve():
                raise VerificationError(f"SBOM must be inside the distribution directory: {sbom_path}")
        if not args.npm_package.is_file():
            raise VerificationError(f"npm package does not exist: {args.npm_package}")
        if not args.npm_lock.is_file():
            raise VerificationError(f"npm package lock does not exist: {args.npm_lock}")
        npm_name, npm_version = _npm_project_identity(args.npm_package)
        if args.npm_inventory is None:
            raise VerificationError("--npm-inventory is required with npm SBOM evidence")
        npm_inventory = npm_ls_inventory(
            args.npm_inventory,
            npm_name,
            npm_version,
        )
        summary["sbom"] = {
            "python": verify_cyclonedx_sbom(
                args.python_sbom,
                expected_name,
                expected_version,
                "pypi",
                python_inventory,
            ),
            "npm": verify_cyclonedx_sbom(
                args.npm_sbom,
                npm_name,
                npm_version,
                "npm",
                npm_inventory,
            ),
        }
    if args.sha256sums:
        summary["sha256"] = verify_sha256_manifest(args.dist, args.sha256sums)
    if args.compare_dist:
        if not args.compare_dist.is_dir():
            raise VerificationError(f"comparison distribution directory does not exist: {args.compare_dist}")
        summary["reproducible"] = verify_reproducible_distributions(args.dist, args.compare_dist)

    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationError as exc:
        raise SystemExit(f"release artifact verification failed: {exc}") from exc
