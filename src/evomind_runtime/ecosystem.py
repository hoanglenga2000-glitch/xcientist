"""Safe capability discovery and manifest bridges for the EvoMind agent kernel.

This module deliberately stops at *description*.  It can parse and rank declared
capabilities, and it can track the verification state of a run-scoped tool
package, but it cannot install a package, spawn a command, or contact a provider.
Runtime execution remains the responsibility of separately permissioned
connectors.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from jsonschema import Draft202012Validator

try:  # PyYAML is a project dependency, but JSON front matter remains supported.
    import yaml
except ImportError:  # pragma: no cover - the project installation includes it
    yaml = None


MAX_MANIFEST_BYTES = 1_048_576
MAX_QUERY_CHARS = 8_192
DEFAULT_OBJECT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}

_HTTP_METHODS = {"get", "put", "post", "delete", "patch", "head", "options", "trace"}
_IDENTIFIER_RE = re.compile(r"[^a-z0-9._-]+")
_SECRET_KEY_RE = re.compile(
    r"(?i)(?:password|passwd|pwd|api[_-]?key|access[_-]?token|refresh[_-]?token|"
    r"private[_-]?key|authorization|cookie|credential|"
    r"(?:^|[_-])(?:secret|token)(?:$|[_-])|口令|密码|凭据)"
)
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(password|passwd|pwd|token|api[_-]?key|secret|credential|authorization)"
    r"\s*[:=]\s*([^\s,;]+)"
)
_CHINESE_SECRET_ASSIGNMENT_RE = re.compile(r"(密码|口令|凭据|令牌)\s*[:：=]\s*([^\s,;，；]+)")
_JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")
_PEM_RE = re.compile(r"-----BEGIN [^-]+-----.*?-----END [^-]+-----", re.DOTALL)
_URL_CREDENTIAL_RE = re.compile(r"(?i)([a-z][a-z0-9+.-]*://)[^/@\s:]+:[^/@\s]+@")


class ManifestError(ValueError):
    """Raised when a declarative bridge manifest is malformed or unsafe."""


class ExternalExecutionDisabled(RuntimeError):
    """Raised when callers try to use this metadata-only module as an executor."""


class PackageCheckStatus(str, Enum):
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"


class PackagePromotionStatus(str, Enum):
    DRAFT = "draft"
    BLOCKED = "blocked"
    PROMOTED = "promoted"
    REJECTED = "rejected"


def canonical_sha256(value: Any) -> str:
    """Hash a JSON-compatible value using a stable, UTF-8 representation."""

    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _bounded_text(value: Any, *, limit: int = 4_096) -> str:
    return str(value or "")[:limit]


def redact_text(value: Any, *, limit: int = 4_096) -> str:
    """Return bounded display text with common secret material removed."""

    text = _bounded_text(value, limit=limit)
    text = _PEM_RE.sub("<redacted-pem>", text)
    text = _JWT_RE.sub("<redacted-token>", text)
    text = _URL_CREDENTIAL_RE.sub(r"\1<redacted>@", text)
    text = _SECRET_ASSIGNMENT_RE.sub(lambda item: f"{item.group(1)}=<redacted>", text)
    text = _CHINESE_SECRET_ASSIGNMENT_RE.sub(lambda item: f"{item.group(1)}=<redacted>", text)
    return text


def _safe_identifier(value: Any, *, fallback: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold().strip()
    normalized = _IDENTIFIER_RE.sub("-", normalized).strip("-._")
    return (normalized or fallback)[:128]


def _sanitize_schema(value: Any, *, property_name: str = "") -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        sensitive_property = bool(_SECRET_KEY_RE.search(property_name))
        for key, item in value.items():
            key_text = str(key)
            if sensitive_property and key_text.casefold() in {"const", "default", "example", "examples", "enum"}:
                continue
            if key_text in {"description", "title"} and isinstance(item, str):
                result[key_text] = redact_text(item)
            elif key_text == "properties" and isinstance(item, Mapping):
                result[key_text] = {
                    str(name): _sanitize_schema(schema, property_name=str(name))
                    for name, schema in item.items()
                }
            else:
                result[key_text] = _sanitize_schema(item, property_name=property_name)
        return result
    if isinstance(value, list):
        return [_sanitize_schema(item, property_name=property_name) for item in value]
    return copy.deepcopy(value)


def _normalize_schema(value: Any) -> dict[str, Any]:
    schema = _sanitize_schema(value if isinstance(value, Mapping) else DEFAULT_OBJECT_SCHEMA)
    if not schema:
        schema = copy.deepcopy(DEFAULT_OBJECT_SCHEMA)
    if schema.get("type") is None:
        schema["type"] = "object"
    if schema.get("type") != "object":
        schema = {
            "type": "object",
            "properties": {"value": schema},
            "required": ["value"],
            "additionalProperties": False,
        }
    schema.setdefault("properties", {})
    schema.setdefault("additionalProperties", False)
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        raise ManifestError(f"invalid input schema: {type(exc).__name__}") from exc
    return schema


def _normalize_operations(value: Any, *, fallback: str) -> tuple[str, ...]:
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        items = [str(item) for item in value]
    else:
        items = [fallback]
    normalized = tuple(dict.fromkeys(_safe_identifier(item, fallback=fallback) for item in items if str(item).strip()))
    return normalized or (fallback,)


def _health_evidence(value: Any, *, default_status: str = "unknown") -> dict[str, Any]:
    if isinstance(value, Mapping):
        # Health evidence must be compact and must never carry authentication data.
        result: dict[str, Any] = {}
        for key, item in value.items():
            safe_key = _safe_identifier(key, fallback="field")
            if _SECRET_KEY_RE.search(str(key)):
                continue
            if isinstance(item, (str, int, float, bool)) or item is None:
                result[safe_key] = redact_text(item, limit=512) if isinstance(item, str) else item
        result.setdefault("status", default_status)
        return result
    return {"status": _bounded_text(value, limit=64) or default_status}


def _manifest_has_secret_material(value: Any, *, parent_key: str = "") -> bool:
    """Detect secret values while allowing schemas and opaque reference fields."""

    if isinstance(value, Mapping):
        if _SECRET_KEY_RE.search(parent_key):
            for value_key in ("const", "default", "example", "examples", "enum"):
                if value.get(value_key) not in (None, "", False, [], {}):
                    return True
        for key, item in value.items():
            key_text = str(key)
            lowered = key_text.casefold()
            # Input schemas may legitimately describe a password field.  The schema
            # contains no value and therefore is not secret material.
            if parent_key in {"properties", "input_schema", "inputschema", "schema"}:
                if _manifest_has_secret_material(item, parent_key=lowered):
                    return True
                continue
            if _SECRET_KEY_RE.search(key_text) and not lowered.endswith(("_ref", "-ref")):
                if item not in (None, "", False, [], {}):
                    return True
            if _manifest_has_secret_material(item, parent_key=lowered):
                return True
        return False
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_manifest_has_secret_material(item, parent_key=parent_key) for item in value)
    if isinstance(value, str):
        return bool(
            _JWT_RE.search(value)
            or _PEM_RE.search(value)
            or _URL_CREDENTIAL_RE.search(value)
            or _SECRET_ASSIGNMENT_RE.search(value)
            or _CHINESE_SECRET_ASSIGNMENT_RE.search(value)
        )
    return False


def _load_json_manifest(source: Mapping[str, Any] | str | bytes | Path) -> tuple[dict[str, Any], str]:
    if isinstance(source, Mapping):
        data = copy.deepcopy(dict(source))
        try:
            raw = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ManifestError(f"manifest is not JSON-compatible: {type(exc).__name__}") from exc
        if len(raw) > MAX_MANIFEST_BYTES:
            raise ManifestError("manifest exceeds the bounded size limit")
        return data, _sha256_bytes(raw)
    if isinstance(source, Path):
        if source.is_symlink() or not source.is_file():
            raise ManifestError("manifest path must be a regular non-symlink file")
        size = source.stat().st_size
        if size > MAX_MANIFEST_BYTES:
            raise ManifestError("manifest exceeds the bounded size limit")
        raw = source.read_bytes()
    elif isinstance(source, bytes):
        raw = bytes(source)
    elif isinstance(source, str):
        raw = source.encode("utf-8")
    else:
        raise ManifestError("manifest must be a mapping, JSON text, bytes, or an explicit Path")
    if len(raw) > MAX_MANIFEST_BYTES:
        raise ManifestError("manifest exceeds the bounded size limit")
    try:
        parsed = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"manifest is not valid UTF-8 JSON: {type(exc).__name__}") from exc
    if not isinstance(parsed, dict):
        raise ManifestError("manifest root must be an object")
    return parsed, _sha256_bytes(raw)


@dataclass(frozen=True)
class CapabilityDescriptor:
    capability_id: str
    provider_id: str
    version: str
    operations: tuple[str, ...]
    input_schema: dict[str, Any]
    risk_class: str
    idempotency: str
    health_evidence: dict[str, Any]
    tool_package_sha256: str
    description: str = ""
    source_kind: str = "manifest"
    tags: tuple[str, ...] = ()
    source_ref: str = ""

    def __post_init__(self) -> None:
        if not self.capability_id or not self.provider_id:
            raise ManifestError("capability_id and provider_id are required")
        digest = self.tool_package_sha256.casefold()
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ManifestError("tool_package_sha256 must be a SHA-256 hex digest")
        object.__setattr__(self, "tool_package_sha256", digest)
        object.__setattr__(self, "version", str(self.version or "0.0.0")[:64])
        object.__setattr__(self, "operations", _normalize_operations(self.operations, fallback="describe"))
        object.__setattr__(self, "input_schema", _normalize_schema(self.input_schema))
        object.__setattr__(self, "risk_class", _safe_identifier(self.risk_class, fallback="unknown"))
        object.__setattr__(self, "idempotency", _safe_identifier(self.idempotency, fallback="unknown"))
        object.__setattr__(self, "health_evidence", _health_evidence(self.health_evidence))
        object.__setattr__(self, "description", redact_text(self.description))
        object.__setattr__(self, "tags", tuple(dict.fromkeys(redact_text(item, limit=128) for item in self.tags)))
        object.__setattr__(self, "source_ref", redact_text(self.source_ref, limit=512))

    @property
    def healthy(self) -> bool:
        return str(self.health_evidence.get("status", "unknown")).casefold() in {
            "healthy", "ready", "ok", "available", "validated",
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CapabilityMatch:
    descriptor: CapabilityDescriptor
    score: float
    matched_terms: tuple[str, ...] = ()
    matched_operations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "descriptor": self.descriptor.to_dict(),
            "score": self.score,
            "matched_terms": list(self.matched_terms),
            "matched_operations": list(self.matched_operations),
        }


def _search_tokens(value: Any) -> set[str]:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()[:MAX_QUERY_CHARS]
    tokens = set(re.findall(r"[a-z0-9][a-z0-9._-]*", text))
    for group in re.findall(r"[\u3400-\u9fff]+", text):
        tokens.update(group)
        tokens.update(group[index:index + 2] for index in range(max(0, len(group) - 1)))
        tokens.update(group[index:index + 3] for index in range(max(0, len(group) - 2)))
    return {token for token in tokens if token}


def _descriptor_search_text(descriptor: CapabilityDescriptor) -> str:
    schema_properties = " ".join(str(key) for key in descriptor.input_schema.get("properties", {}))
    return " ".join(
        [descriptor.capability_id, descriptor.provider_id, descriptor.description, *descriptor.operations,
         *descriptor.tags, schema_properties]
    )


class CapabilityCatalog:
    """In-memory, metadata-only catalog with descriptor-driven retrieval."""

    def __init__(self, descriptors: Iterable[CapabilityDescriptor] = ()) -> None:
        self._descriptors: dict[str, CapabilityDescriptor] = {}
        self._packages: dict[tuple[str, str], ToolPackage] = {}
        for descriptor in descriptors:
            self.register(descriptor)

    def register(self, descriptor: CapabilityDescriptor, *, replace: bool = False) -> None:
        if descriptor.capability_id in self._descriptors and not replace:
            raise ManifestError(f"duplicate capability: {descriptor.capability_id}")
        self._descriptors[descriptor.capability_id] = descriptor

    def get(self, capability_id: str, *, run_id: str | None = None) -> CapabilityDescriptor | None:
        descriptor = self._descriptors.get(capability_id)
        if descriptor is not None:
            return descriptor
        if run_id is not None:
            for package in self._packages.values():
                if package.run_id != run_id or package.promotion_status != PackagePromotionStatus.PROMOTED.value:
                    continue
                for item in package.descriptors:
                    if item.capability_id == capability_id:
                        return item
        return None

    def all(self, *, run_id: str | None = None) -> list[CapabilityDescriptor]:
        descriptors = list(self._descriptors.values())
        if run_id is not None:
            descriptors.extend(
                descriptor
                for package in self._packages.values()
                if package.run_id == run_id and package.promotion_status == PackagePromotionStatus.PROMOTED.value
                for descriptor in package.descriptors
            )
        return sorted(descriptors, key=lambda item: item.capability_id)

    def discover_ranked(
        self,
        objective: str = "",
        *,
        required_operations: Iterable[str] = (),
        provider_ids: Iterable[str] | None = None,
        healthy_only: bool = False,
        max_risk: str | None = None,
        run_id: str | None = None,
        limit: int = 20,
    ) -> list[CapabilityMatch]:
        if limit < 1 or limit > 1_000:
            raise ValueError("limit must be between 1 and 1000")
        query = unicodedata.normalize("NFKC", str(objective or ""))[:MAX_QUERY_CHARS]
        query_tokens = _search_tokens(query)
        required = {_safe_identifier(item, fallback="operation") for item in required_operations}
        allowed_providers = {str(item) for item in provider_ids} if provider_ids is not None else None
        risk_order = {
            "observe": 0, "read": 0, "low": 0,
            "write": 1, "medium": 1,
            "destructive": 2, "high": 2,
            "privileged": 3, "critical": 3,
        }
        risk_limit = risk_order.get(_safe_identifier(max_risk, fallback="critical"), 3) if max_risk else None
        matches: list[CapabilityMatch] = []
        for descriptor in self.all(run_id=run_id):
            if allowed_providers is not None and descriptor.provider_id not in allowed_providers:
                continue
            if healthy_only and not descriptor.healthy:
                continue
            if risk_limit is not None and risk_order.get(descriptor.risk_class, 3) > risk_limit:
                continue
            operations = set(descriptor.operations)
            operation_matches = tuple(sorted(required & operations))
            if required and len(operation_matches) != len(required):
                continue
            candidate_text = _descriptor_search_text(descriptor)
            candidate_tokens = _search_tokens(candidate_text)
            term_matches = tuple(sorted(query_tokens & candidate_tokens))
            if query_tokens and not term_matches and not operation_matches:
                continue
            normalized_query = query.casefold().strip()
            exact_phrase = bool(normalized_query and normalized_query in candidate_text.casefold())
            score = (
                len(operation_matches) * 20.0
                + len(term_matches) * 2.0
                + (10.0 if exact_phrase else 0.0)
                + (1.0 if descriptor.healthy else 0.0)
            )
            if query_tokens:
                score += len(term_matches) / len(query_tokens)
            matches.append(CapabilityMatch(descriptor, round(score, 6), term_matches, operation_matches))
        matches.sort(key=lambda item: (-item.score, item.descriptor.capability_id))
        return matches[:limit]

    def discover(self, objective: str = "", **kwargs: Any) -> list[CapabilityDescriptor]:
        return [item.descriptor for item in self.discover_ranked(objective, **kwargs)]

    def ingest(self, descriptors: Iterable[CapabilityDescriptor], *, replace: bool = False) -> None:
        for descriptor in descriptors:
            self.register(descriptor, replace=replace)

    def ingest_manifest(
        self,
        bridge_kind: str,
        source: Mapping[str, Any] | str | bytes | Path,
        *,
        provider_id: str | None = None,
        replace: bool = False,
    ) -> list[CapabilityDescriptor]:
        descriptors = parse_bridge_manifest(bridge_kind, source, provider_id=provider_id)
        self.ingest(descriptors, replace=replace)
        return descriptors

    def stage_tool_package(self, package: "ToolPackage") -> None:
        key = (package.run_id, package.package_id)
        if key in self._packages:
            raise ManifestError("duplicate run-scoped tool package")
        self._packages[key] = package

    def tool_package(self, run_id: str, package_id: str) -> "ToolPackage | None":
        return self._packages.get((run_id, package_id))

    def promote_tool_package(self, run_id: str, package_id: str) -> "ToolPackage":
        package = self._packages.get((run_id, package_id))
        if package is None:
            raise ManifestError("unknown run-scoped tool package")
        package.promote()
        return package


def _descriptor_from_entry(
    entry: Mapping[str, Any],
    *,
    provider_id: str,
    version: str,
    package_sha256: str,
    source_kind: str,
    fallback_name: str,
    default_risk: str = "unknown",
    default_idempotency: str = "unknown",
    health: Mapping[str, Any] | None = None,
    source_ref: str = "",
) -> CapabilityDescriptor:
    name = entry.get("capability_id") or entry.get("operationId") or entry.get("name") or fallback_name
    name_id = _safe_identifier(name, fallback=fallback_name)
    capability_id = str(entry.get("capability_id") or f"{source_kind}:{provider_id}:{name_id}")
    schema = entry.get("input_schema", entry.get("inputSchema", entry.get("schema", DEFAULT_OBJECT_SCHEMA)))
    operations = entry.get("operations", entry.get("operation", name_id))
    return CapabilityDescriptor(
        capability_id=capability_id,
        provider_id=provider_id,
        version=str(entry.get("version") or version or "0.0.0"),
        operations=_normalize_operations(operations, fallback=name_id),
        input_schema=_normalize_schema(schema),
        risk_class=str(entry.get("risk_class") or entry.get("riskClass") or default_risk),
        idempotency=str(entry.get("idempotency") or default_idempotency),
        health_evidence=_health_evidence(entry.get("health_evidence") or entry.get("healthEvidence") or health),
        tool_package_sha256=package_sha256,
        description=redact_text(entry.get("description") or entry.get("summary") or name_id),
        source_kind=source_kind,
        tags=tuple(str(item) for item in entry.get("tags", ()) if isinstance(item, (str, int, float))),
        source_ref=source_ref,
    )


def parse_mcp_manifest(
    source: Mapping[str, Any] | str | bytes | Path,
    *,
    provider_id: str | None = None,
) -> list[CapabilityDescriptor]:
    manifest, digest = _load_json_manifest(source)
    server = manifest.get("serverInfo") if isinstance(manifest.get("serverInfo"), Mapping) else {}
    provider = _safe_identifier(provider_id or server.get("name") or manifest.get("name"), fallback="mcp-provider")
    version = str(server.get("version") or manifest.get("version") or "0.0.0")
    tools = manifest.get("tools")
    if tools is None and isinstance(manifest.get("capabilities"), Mapping):
        tools = manifest["capabilities"].get("tools")
    if not isinstance(tools, list):
        raise ManifestError("MCP manifest must declare a tools array")
    result: list[CapabilityDescriptor] = []
    for index, item in enumerate(tools):
        if not isinstance(item, Mapping) or not str(item.get("name") or "").strip():
            raise ManifestError(f"MCP tool at index {index} is invalid")
        annotations = item.get("annotations") if isinstance(item.get("annotations"), Mapping) else {}
        risk = item.get("risk_class")
        if not risk:
            risk = "destructive" if annotations.get("destructiveHint") is True else (
                "observe" if annotations.get("readOnlyHint") is True else "write"
            )
        idempotency = item.get("idempotency") or (
            "idempotent" if annotations.get("idempotentHint") is True else "unknown"
        )
        entry = dict(item)
        entry["risk_class"] = risk
        entry["idempotency"] = idempotency
        result.append(_descriptor_from_entry(
            entry,
            provider_id=provider,
            version=version,
            package_sha256=digest,
            source_kind="mcp",
            fallback_name=f"tool-{index}",
            health=manifest.get("health_evidence"),
        ))
    return result


def _openapi_input_schema(path_item: Mapping[str, Any], operation: Mapping[str, Any]) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []
    parameters: list[Any] = []
    for source in (path_item.get("parameters"), operation.get("parameters")):
        if isinstance(source, list):
            parameters.extend(source)
    for parameter in parameters:
        if not isinstance(parameter, Mapping) or "$ref" in parameter:
            continue
        name = str(parameter.get("name") or "").strip()
        if not name or name in properties:
            continue
        properties[name] = copy.deepcopy(parameter.get("schema") or {"type": "string"})
        if parameter.get("required") is True:
            required.append(name)
    request_body = operation.get("requestBody")
    if isinstance(request_body, Mapping):
        content = request_body.get("content") if isinstance(request_body.get("content"), Mapping) else {}
        media = next((value for key, value in content.items() if "json" in str(key).casefold()), None)
        if not isinstance(media, Mapping):
            media = next((value for value in content.values() if isinstance(value, Mapping)), None)
        properties["body"] = copy.deepcopy(media.get("schema") if isinstance(media, Mapping) else {}) or {}
        if request_body.get("required") is True:
            required.append("body")
    schema: dict[str, Any] = {"type": "object", "properties": properties, "additionalProperties": False}
    if required:
        schema["required"] = list(dict.fromkeys(required))
    return schema


def parse_openapi_manifest(
    source: Mapping[str, Any] | str | bytes | Path,
    *,
    provider_id: str | None = None,
) -> list[CapabilityDescriptor]:
    manifest, digest = _load_json_manifest(source)
    if not str(manifest.get("openapi") or "").startswith("3."):
        raise ManifestError("only OpenAPI 3.x manifests are supported")
    info = manifest.get("info") if isinstance(manifest.get("info"), Mapping) else {}
    provider = _safe_identifier(provider_id or info.get("title"), fallback="openapi-provider")
    version = str(info.get("version") or "0.0.0")
    paths = manifest.get("paths")
    if not isinstance(paths, Mapping):
        raise ManifestError("OpenAPI manifest must declare paths")
    result: list[CapabilityDescriptor] = []
    for path_name in sorted(paths):
        path_item = paths[path_name]
        if not isinstance(path_item, Mapping):
            continue
        for method in sorted(_HTTP_METHODS & {str(key).casefold() for key in path_item}):
            operation = path_item.get(method)
            if not isinstance(operation, Mapping):
                continue
            generated = f"{method}-{_safe_identifier(path_name, fallback='root')}"
            operation_id = _safe_identifier(operation.get("operationId"), fallback=generated)
            extension_risk = operation.get("x-evomind-risk-class") or operation.get("x-risk-class")
            default_risk = extension_risk or (
                "observe" if method in {"get", "head", "options"} else (
                    "destructive" if method == "delete" else "write"
                )
            )
            idempotency = operation.get("x-evomind-idempotency") or operation.get("x-idempotency") or (
                "idempotent" if method in {"get", "put", "delete", "head", "options"} else "conditional"
            )
            entry = {
                "name": operation_id,
                "description": operation.get("summary") or operation.get("description") or f"{method.upper()} {path_name}",
                "operations": [operation_id, method],
                "input_schema": _openapi_input_schema(path_item, operation),
                "risk_class": default_risk,
                "idempotency": idempotency,
                "health_evidence": operation.get("x-health-evidence") or manifest.get("x-health-evidence"),
                "tags": operation.get("tags") or (),
            }
            result.append(_descriptor_from_entry(
                entry,
                provider_id=provider,
                version=version,
                package_sha256=digest,
                source_kind="openapi",
                fallback_name=generated,
                source_ref=f"{method.upper()} {path_name}",
            ))
    return result


def _cli_schema(command: Mapping[str, Any]) -> dict[str, Any]:
    explicit = command.get("input_schema", command.get("inputSchema"))
    if isinstance(explicit, Mapping):
        return _normalize_schema(explicit)
    arguments = command.get("arguments")
    if not isinstance(arguments, list):
        return copy.deepcopy(DEFAULT_OBJECT_SCHEMA)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for argument in arguments:
        if not isinstance(argument, Mapping):
            continue
        name = _safe_identifier(argument.get("name"), fallback="argument")
        properties[name] = copy.deepcopy(argument.get("schema") or {"type": argument.get("type") or "string"})
        if argument.get("required") is True:
            required.append(name)
    schema: dict[str, Any] = {"type": "object", "properties": properties, "additionalProperties": False}
    if required:
        schema["required"] = required
    return schema


def parse_cli_manifest(
    source: Mapping[str, Any] | str | bytes | Path,
    *,
    provider_id: str | None = None,
) -> list[CapabilityDescriptor]:
    manifest, digest = _load_json_manifest(source)
    provider = _safe_identifier(provider_id or manifest.get("provider_id") or manifest.get("name"), fallback="cli-provider")
    version = str(manifest.get("version") or "0.0.0")
    commands = manifest.get("commands")
    if not isinstance(commands, list):
        raise ManifestError("CLI manifest must declare a commands array")
    health = manifest.get("health_evidence")
    result: list[CapabilityDescriptor] = []
    for index, command in enumerate(commands):
        if not isinstance(command, Mapping) or not str(command.get("name") or "").strip():
            raise ManifestError(f"CLI command at index {index} is invalid")
        entry = dict(command)
        entry["input_schema"] = _cli_schema(command)
        result.append(_descriptor_from_entry(
            entry,
            provider_id=provider,
            version=version,
            package_sha256=digest,
            source_kind="cli",
            fallback_name=f"command-{index}",
            default_risk="unknown",
            default_idempotency="unknown",
            health=health,
        ))
    return result


@dataclass(frozen=True)
class SkillScanPolicy:
    max_files: int = 256
    max_file_bytes: int = 65_536
    max_total_bytes: int = 2_097_152
    max_depth: int = 8
    max_entries: int = 4_096

    def __post_init__(self) -> None:
        if not 1 <= self.max_files <= 10_000:
            raise ValueError("max_files is outside the safe range")
        if not 1 <= self.max_file_bytes <= MAX_MANIFEST_BYTES:
            raise ValueError("max_file_bytes is outside the safe range")
        if not self.max_file_bytes <= self.max_total_bytes <= 32 * MAX_MANIFEST_BYTES:
            raise ValueError("max_total_bytes is outside the safe range")
        if not 1 <= self.max_depth <= 32:
            raise ValueError("max_depth is outside the safe range")
        if not 1 <= self.max_entries <= 100_000:
            raise ValueError("max_entries is outside the safe range")


@dataclass(frozen=True)
class SkillScanReport:
    descriptors: tuple[CapabilityDescriptor, ...]
    scanned_files: int
    skipped_files: int
    total_bytes: int


def _skill_front_matter(raw: bytes) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ManifestError("SKILL.md is not valid UTF-8") from exc
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ManifestError("SKILL.md must start with YAML front matter")
    try:
        end = next(index for index in range(1, min(len(lines), 256)) if lines[index].strip() == "---")
    except StopIteration as exc:
        raise ManifestError("SKILL.md front matter is not bounded or terminated") from exc
    front = "\n".join(lines[1:end])
    if re.search(r"(?:^|\s)[&*][A-Za-z0-9_-]+", front) or "!!" in front:
        raise ManifestError("SKILL.md aliases and explicit YAML tags are not allowed")
    if yaml is not None:
        value = yaml.safe_load(front)
    else:  # minimal fallback for name/description/version-only manifests
        value = {}
        for line in front.splitlines():
            if ":" in line:
                key, item = line.split(":", 1)
                value[key.strip()] = item.strip().strip("\"'")
    if not isinstance(value, Mapping):
        raise ManifestError("SKILL.md front matter must be an object")
    return dict(value)


def parse_skill_manifest(
    source: Mapping[str, Any] | str | bytes | Path,
    *,
    provider_id: str = "local-skill-library",
    source_ref: str = "",
    max_bytes: int = 65_536,
) -> CapabilityDescriptor:
    """Parse only trusted-size SKILL metadata; the instruction body is ignored."""

    if not 1 <= max_bytes <= MAX_MANIFEST_BYTES:
        raise ValueError("max_bytes is outside the safe range")
    if isinstance(source, Mapping):
        front = copy.deepcopy(dict(source))
        try:
            encoded = json.dumps(front, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ManifestError(f"skill metadata is not JSON-compatible: {type(exc).__name__}") from exc
        if len(encoded) > max_bytes:
            raise ManifestError("SKILL.md exceeds the bounded size limit")
        digest = _sha256_bytes(encoded)
    else:
        if isinstance(source, Path):
            if source.is_symlink() or not source.is_file():
                raise ManifestError("SKILL.md path must be a regular non-symlink file")
            if source.stat().st_size > max_bytes:
                raise ManifestError("SKILL.md exceeds the bounded size limit")
            raw = source.read_bytes()
        elif isinstance(source, bytes):
            raw = bytes(source)
        elif isinstance(source, str):
            raw = source.encode("utf-8")
        else:
            raise ManifestError("skill manifest must be metadata, UTF-8 text, bytes, or an explicit Path")
        if len(raw) > max_bytes:
            raise ManifestError("SKILL.md exceeds the bounded size limit")
        front = _skill_front_matter(raw)
        digest = _sha256_bytes(raw)
    name = str(front.get("name") or "").strip()
    if not name:
        raise ManifestError("SKILL.md name is empty")
    provider = _safe_identifier(provider_id, fallback="skill-provider")
    slug = _safe_identifier(name, fallback="skill")
    entry = {
        "name": slug,
        "description": redact_text(front.get("description") or name),
        "operations": front.get("operations") or ["instructions"],
        "input_schema": front.get("input_schema") or DEFAULT_OBJECT_SCHEMA,
        "risk_class": front.get("risk_class") or "observe",
        "idempotency": front.get("idempotency") or "idempotent",
        "health_evidence": {"status": "validated", "manifest": "bounded-front-matter"},
        "version": front.get("version") or "0.0.0",
        "tags": front.get("tags") if isinstance(front.get("tags"), list) else (),
    }
    return _descriptor_from_entry(
        entry,
        provider_id=provider,
        version=str(entry["version"]),
        package_sha256=digest,
        source_kind="skill",
        fallback_name=slug,
        source_ref=source_ref,
    )


def _bounded_skill_candidates(base: Path, policy: SkillScanPolicy) -> tuple[list[Path], int]:
    """Walk without following links and stop before an adversarial tree can grow unbounded."""

    candidates: list[Path] = []
    skipped = 0
    entries_seen = 0
    stack: list[tuple[Path, int]] = [(base, 0)]
    while stack:
        directory, depth = stack.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    entries_seen += 1
                    if entries_seen > policy.max_entries:
                        return candidates, skipped + 1
                    try:
                        if entry.is_symlink():
                            skipped += 1
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            if depth < policy.max_depth:
                                stack.append((Path(entry.path), depth + 1))
                            else:
                                skipped += 1
                            continue
                        if entry.name == "SKILL.md" and entry.is_file(follow_symlinks=False):
                            if len(candidates) >= policy.max_files:
                                return candidates, skipped + 1
                            candidates.append(Path(entry.path))
                    except OSError:
                        skipped += 1
        except OSError:
            skipped += 1
    candidates.sort(key=lambda item: str(item).casefold())
    return candidates, skipped


def scan_skill_manifests(
    root: str | Path,
    *,
    provider_id: str = "local-skill-library",
    policy: SkillScanPolicy | None = None,
) -> SkillScanReport:
    scan_policy = policy or SkillScanPolicy()
    base = Path(root).expanduser()
    if base.is_symlink() or not base.is_dir():
        raise ManifestError("skill root must be a regular non-symlink directory")
    resolved_base = base.resolve()
    descriptors: list[CapabilityDescriptor] = []
    skipped = 0
    total_bytes = 0
    candidates, traversal_skipped = _bounded_skill_candidates(base, scan_policy)
    skipped += traversal_skipped
    for candidate in candidates:
        try:
            relative = candidate.relative_to(base)
            if len(relative.parts) - 1 > scan_policy.max_depth:
                skipped += 1
                continue
            if candidate.is_symlink() or any(parent.is_symlink() for parent in candidate.parents if parent != base.parent):
                skipped += 1
                continue
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(resolved_base)
            size = resolved.stat().st_size
            if size > scan_policy.max_file_bytes or total_bytes + size > scan_policy.max_total_bytes:
                skipped += 1
                continue
            total_bytes += size
            raw = resolved.read_bytes()
            descriptors.append(parse_skill_manifest(
                raw,
                provider_id=provider_id,
                source_ref=relative.as_posix(),
                max_bytes=scan_policy.max_file_bytes,
            ))
        except (OSError, ValueError, ManifestError):
            skipped += 1
    return SkillScanReport(tuple(descriptors), len(descriptors), skipped, total_bytes)


class MCPManifestBridge:
    parse = staticmethod(parse_mcp_manifest)


class OpenAPIManifestBridge:
    parse = staticmethod(parse_openapi_manifest)


class CLIManifestBridge:
    parse = staticmethod(parse_cli_manifest)


class SkillManifestBridge:
    parse = staticmethod(parse_skill_manifest)
    scan = staticmethod(scan_skill_manifests)


# Concise aliases keep bridge composition readable while retaining the explicit
# ManifestBridge names for callers that want to emphasize the no-execution boundary.
MCPBridge = MCPManifestBridge
OpenAPIBridge = OpenAPIManifestBridge
CLIBridge = CLIManifestBridge
SkillBridge = SkillManifestBridge


def parse_bridge_manifest(
    bridge_kind: str,
    source: Mapping[str, Any] | str | bytes | Path,
    *,
    provider_id: str | None = None,
) -> list[CapabilityDescriptor]:
    """Parse one supported bridge kind without probing or executing its provider."""

    kind = _safe_identifier(bridge_kind, fallback="unknown")
    if kind == "mcp":
        return parse_mcp_manifest(source, provider_id=provider_id)
    if kind == "openapi":
        return parse_openapi_manifest(source, provider_id=provider_id)
    if kind == "cli":
        return parse_cli_manifest(source, provider_id=provider_id)
    if kind == "skill":
        return [parse_skill_manifest(source, provider_id=provider_id or "local-skill-library")]
    raise ManifestError(f"unsupported bridge kind: {kind}")


TOOL_PACKAGE_MANIFEST_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": [
        "schema", "package_id", "run_id", "version", "provider_id", "capabilities",
        "permissions", "tests", "canary", "rollback",
    ],
    "properties": {
        "schema": {"const": "evomind.tool_package.v1"},
        "package_id": {"type": "string", "minLength": 1, "maxLength": 128},
        "run_id": {"type": "string", "minLength": 1, "maxLength": 128},
        "version": {"type": "string", "minLength": 1, "maxLength": 64},
        "provider_id": {"type": "string", "minLength": 1, "maxLength": 128},
        "capabilities": {"type": "array", "minItems": 1, "maxItems": 256, "items": {"type": "object"}},
        "permissions": {"type": ["object", "array"]},
        "tests": {"type": "array", "minItems": 1, "maxItems": 1_000},
        "canary": {"type": "object"},
        "rollback": {"type": "object", "minProperties": 1},
    },
    "additionalProperties": True,
}


@dataclass
class ToolPackage:
    package_id: str
    run_id: str
    version: str
    provider_id: str
    manifest: dict[str, Any]
    manifest_sha256: str
    descriptors: tuple[CapabilityDescriptor, ...]
    test_status: str = PackageCheckStatus.PENDING.value
    canary_status: str = PackageCheckStatus.PENDING.value
    promotion_status: str = PackagePromotionStatus.DRAFT.value
    test_evidence: dict[str, Any] = field(default_factory=dict)
    canary_evidence: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_manifest(
        cls,
        source: Mapping[str, Any] | str | bytes | Path,
        *,
        run_id: str | None = None,
    ) -> "ToolPackage":
        manifest, _source_digest = _load_json_manifest(source)
        errors = sorted(Draft202012Validator(TOOL_PACKAGE_MANIFEST_SCHEMA).iter_errors(manifest), key=lambda item: list(item.path))
        if errors:
            raise ManifestError("invalid tool package manifest: " + "; ".join(item.message for item in errors[:4]))
        if run_id is not None and str(manifest["run_id"]) != str(run_id):
            raise ManifestError("tool package is scoped to a different run")
        if _manifest_has_secret_material(manifest):
            raise ManifestError("tool package manifest contains secret material")
        package_id = _safe_identifier(manifest["package_id"], fallback="tool-package")
        package_run_id = str(manifest["run_id"])
        provider = _safe_identifier(manifest["provider_id"], fallback="tool-package-provider")
        version = str(manifest["version"])
        digest = canonical_sha256(manifest)
        descriptors: list[CapabilityDescriptor] = []
        for index, entry in enumerate(manifest["capabilities"]):
            if not isinstance(entry, Mapping):
                raise ManifestError(f"tool package capability at index {index} is invalid")
            descriptors.append(_descriptor_from_entry(
                entry,
                provider_id=provider,
                version=version,
                package_sha256=digest,
                source_kind="tool-package",
                fallback_name=f"capability-{index}",
                health={"status": "unknown", "package_state": "staged"},
            ))
        return cls(
            package_id=package_id,
            run_id=package_run_id,
            version=version,
            provider_id=provider,
            manifest=copy.deepcopy(manifest),
            manifest_sha256=digest,
            descriptors=tuple(descriptors),
        )

    def record_tests(self, *, passed: bool, evidence: Mapping[str, Any] | None = None) -> None:
        if self.promotion_status == PackagePromotionStatus.PROMOTED.value:
            raise ManifestError("promoted package verification is immutable")
        self.test_status = PackageCheckStatus.PASSED.value if passed else PackageCheckStatus.FAILED.value
        self.test_evidence = _health_evidence(evidence, default_status=self.test_status)
        # Any new test result invalidates an earlier canary, even when the tests
        # pass again, because that canary did not observe this verification set.
        self.canary_status = PackageCheckStatus.PENDING.value
        self.canary_evidence = {}
        if not passed:
            self.promotion_status = PackagePromotionStatus.BLOCKED.value
        elif self.promotion_status == PackagePromotionStatus.BLOCKED.value:
            self.promotion_status = PackagePromotionStatus.DRAFT.value

    def record_canary(self, *, passed: bool, evidence: Mapping[str, Any] | None = None) -> None:
        if self.test_status != PackageCheckStatus.PASSED.value:
            raise ManifestError("canary verification requires passing declared tests")
        if self.promotion_status == PackagePromotionStatus.PROMOTED.value:
            raise ManifestError("promoted package verification is immutable")
        self.canary_status = PackageCheckStatus.PASSED.value if passed else PackageCheckStatus.FAILED.value
        self.canary_evidence = _health_evidence(evidence, default_status=self.canary_status)
        self.promotion_status = (
            PackagePromotionStatus.DRAFT.value if passed else PackagePromotionStatus.BLOCKED.value
        )

    def promote(self) -> None:
        if self.test_status != PackageCheckStatus.PASSED.value:
            self.promotion_status = PackagePromotionStatus.BLOCKED.value
            raise ManifestError("promotion requires passing declared tests")
        if self.canary_status != PackageCheckStatus.PASSED.value:
            self.promotion_status = PackagePromotionStatus.BLOCKED.value
            raise ManifestError("promotion requires a passing canary")
        self.promotion_status = PackagePromotionStatus.PROMOTED.value

    def reject(self) -> None:
        if self.promotion_status == PackagePromotionStatus.PROMOTED.value:
            raise ManifestError("a promoted package cannot be rejected by metadata mutation")
        self.promotion_status = PackagePromotionStatus.REJECTED.value

    def install(self, *_args: Any, **_kwargs: Any) -> None:
        raise ExternalExecutionDisabled("tool packages are declarative; installation is disabled")

    def execute(self, *_args: Any, **_kwargs: Any) -> None:
        raise ExternalExecutionDisabled("tool packages are declarative; execution is disabled")

    def to_dict(self, *, include_manifest: bool = False) -> dict[str, Any]:
        value: dict[str, Any] = {
            "package_id": self.package_id,
            "run_id": self.run_id,
            "version": self.version,
            "provider_id": self.provider_id,
            "manifest_sha256": self.manifest_sha256,
            "descriptors": [item.to_dict() for item in self.descriptors],
            "test_status": self.test_status,
            "canary_status": self.canary_status,
            "promotion_status": self.promotion_status,
            "test_evidence": copy.deepcopy(self.test_evidence),
            "canary_evidence": copy.deepcopy(self.canary_evidence),
        }
        if include_manifest:
            value["manifest"] = copy.deepcopy(self.manifest)
        return value


__all__ = [
    "CLIManifestBridge",
    "CLIBridge",
    "CapabilityCatalog",
    "CapabilityDescriptor",
    "CapabilityMatch",
    "ExternalExecutionDisabled",
    "MCPManifestBridge",
    "MCPBridge",
    "ManifestError",
    "OpenAPIManifestBridge",
    "OpenAPIBridge",
    "PackageCheckStatus",
    "PackagePromotionStatus",
    "SkillManifestBridge",
    "SkillBridge",
    "SkillScanPolicy",
    "SkillScanReport",
    "TOOL_PACKAGE_MANIFEST_SCHEMA",
    "ToolPackage",
    "canonical_sha256",
    "parse_cli_manifest",
    "parse_bridge_manifest",
    "parse_mcp_manifest",
    "parse_openapi_manifest",
    "parse_skill_manifest",
    "redact_text",
    "scan_skill_manifests",
]
