from __future__ import annotations

import hashlib
import json
import os
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from .agent_kernel_v2 import AgentKernel, RunSupervisor, TaskGraph, TaskNode
from .capabilities import DirectoryCapability
from .capabilities import CapabilityNotFoundError, CapabilityPermissionError
from .connectors import LocalDirectoryConnector
from .credential_leases import CredentialLeaseManager, JsonLeaseBackend
from .directory_broker import DirectoryBroker
from .ecosystem import (
    CapabilityCatalog,
    CapabilityDescriptor as EcosystemDescriptor,
    CapabilityMatch,
    SkillScanPolicy,
    scan_skill_manifests,
)
from .evidence_memory import EvidenceMemory, EvidenceRef, EvidenceScope, EvidenceStatus, JsonEvidenceBackend
from .recovery import JsonRunStateRepository
from .remote_connectors import HttpSourceConnector, RestrictedHttpOpener
from .super_agent_store import SuperAgentStore
from .tool_package_runtime import ToolPackageRuntime


class SuperAgentMode(str, Enum):
    OFF = "off"
    SHADOW = "shadow"
    ENABLED = "enabled"


class SuperAgentRuntime:
    """Composition root for the additive Super Agent V1 kernel.

    V1 defaults to shadow mode: capabilities and durable task graphs are
    available, while legacy tool execution remains authoritative until a
    release gate explicitly selects ``enabled``.
    """

    def __init__(
        self,
        *,
        workspace_root: str | Path,
        runtime_root: str | Path,
        tenant_id: str = "local",
        project_id: str = "default",
        mode: str | SuperAgentMode | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve(strict=False)
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.runtime_root = Path(runtime_root).resolve(strict=False)
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        self.tenant_id = str(tenant_id or "local")
        self.project_id = str(project_id or "default")
        selected = mode or os.getenv("EVOMIND_SUPER_AGENT_MODE", SuperAgentMode.SHADOW.value)
        self.mode = SuperAgentMode(str(selected))

        self.store = SuperAgentStore(self.runtime_root / "runtime.sqlite3")
        self.broker = DirectoryBroker()
        self.catalog = CapabilityCatalog()
        self.repository = JsonRunStateRepository(self.runtime_root / "agent-kernel-v2" / "runs")
        self.supervisor = RunSupervisor(self.repository)
        self.memory = EvidenceMemory(JsonEvidenceBackend(self.runtime_root / "evidence-memory-v2.json"))
        self.leases = CredentialLeaseManager(JsonLeaseBackend(self.runtime_root / "credential-leases-v2"))
        self._tool_package_runtimes: dict[str, ToolPackageRuntime] = {}
        self._source_connectors: dict[str, Any] = {}
        self._execution_connectors: dict[str, tuple[Any, DirectoryCapability]] = {}
        self._session_directories: dict[str, str] = {}
        self._skill_scan = {"scanned_files": 0, "skipped_files": 0, "total_bytes": 0, "root_count": 0}
        self._mount_workspace()
        self._register_public_https_source()
        if self.enabled or os.getenv("EVOMIND_SUPER_AGENT_SCAN_SKILLS", "0") == "1":
            self.refresh_installed_skills()

    @property
    def enabled(self) -> bool:
        return self.mode is SuperAgentMode.ENABLED

    @property
    def shadow(self) -> bool:
        return self.mode is SuperAgentMode.SHADOW

    def _mount_workspace(self) -> None:
        connector = LocalDirectoryConnector(
            "local-workspace",
            version="1.0.0",
            allowed_roots=(self.workspace_root,),
        )
        self.broker.register_connector(connector)
        capability = DirectoryCapability(
            directory_id="workspace",
            connector_id=connector.connector_id,
            opaque_root_ref=str(self.workspace_root),
            operations=("list", "stat", "read", "hash", "mkdir", "write", "copy", "sync", "delete"),
            tenant_id=self.tenant_id,
            project_id=self.project_id,
            run_id="",
            connector_version=connector.version,
            max_files=1_000_000,
            max_bytes=1 << 40,
        )
        self.broker.mount(capability)
        self.store.put_connector({
            "connector_id": connector.connector_id,
            "connector_type": "local_directory",
            "version": connector.version,
        })
        self.store.put_directory(capability.to_dict(include_opaque_root=True))
        self.refresh_catalog()

    def _register_public_https_source(self) -> None:
        allowed_hosts = ("raw.githubusercontent.com",)
        try:
            timeout_seconds = float(os.getenv("EVOMIND_PUBLIC_HTTPS_TIMEOUT_SECONDS", "120"))
        except (TypeError, ValueError):
            timeout_seconds = 120.0
        timeout_seconds = max(30.0, min(timeout_seconds, 300.0))
        connector = HttpSourceConnector(
            RestrictedHttpOpener(allowed_hosts=allowed_hosts, timeout_seconds=timeout_seconds),
            lambda capability: self.broker.transfer_target(capability.directory_id),
            connector_id="public-https-source",
            version="1.1.0",
            allowed_hosts=allowed_hosts,
        )
        self.register_source_connector(connector)

    def ensure_session_workspace(
        self,
        *,
        session_id: str,
        workspace_root: str | Path,
        tenant_id: str = "",
        project_id: str = "",
    ) -> str:
        existing = self._session_directories.get(session_id)
        if existing:
            return existing
        root = Path(workspace_root).resolve(strict=False)
        root.mkdir(parents=True, exist_ok=True)
        requested_tenant_id = tenant_id or self.tenant_id
        requested_project_id = project_id or self.project_id
        if (
            root == self.workspace_root
            and requested_tenant_id == self.tenant_id
            and requested_project_id == self.project_id
        ):
            self._session_directories[session_id] = "workspace"
            return "workspace"
        digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:16]
        connector_id = f"local-workspace-{digest}"
        directory_id = f"workspace-{digest}"
        connector = LocalDirectoryConnector(
            connector_id,
            version="1.0.0",
            allowed_roots=(root,),
        )
        capability = DirectoryCapability(
            directory_id=directory_id,
            connector_id=connector_id,
            opaque_root_ref=str(root),
            operations=("list", "stat", "read", "hash", "mkdir", "write", "copy", "sync", "delete"),
            tenant_id=requested_tenant_id,
            project_id=requested_project_id,
            run_id=session_id,
            connector_version=connector.version,
            max_files=1_000_000,
            max_bytes=1 << 40,
        )
        self.register_directory_connector(connector, capability)
        self._session_directories[session_id] = directory_id
        return directory_id

    @staticmethod
    def _metadata_identity(metadata: dict[str, Any] | None, key: str) -> str:
        source = metadata if isinstance(metadata, dict) else {}
        direct = str(source.get(key) or "")
        if direct:
            return direct
        managed = source.get("managed_hpc_identity")
        return str(managed.get(key) or "") if isinstance(managed, dict) else ""

    def session_identity(self, metadata: dict[str, Any] | None) -> tuple[str, str]:
        return (
            self._metadata_identity(metadata, "tenant_id") or self.tenant_id,
            self._metadata_identity(metadata, "project_id") or self.project_id,
        )

    def authorize_directory(
        self,
        directory_id: str,
        *,
        session_id: str,
        workspace_root: str | Path,
        metadata: dict[str, Any] | None = None,
    ) -> DirectoryCapability:
        try:
            capability = self.broker.capability(directory_id)
        except CapabilityNotFoundError:
            execution = self._execution_connectors.get(str(directory_id))
            if execution is None:
                raise
            capability = execution[1]
        tenant_id, project_id = self.session_identity(metadata)
        if capability.tenant_id != tenant_id or capability.project_id != project_id:
            raise CapabilityPermissionError("directory capability is bound to a different tenant or project")
        if capability.run_id and capability.run_id != session_id:
            raise CapabilityPermissionError("directory capability is bound to a different Run")
        if not capability.run_id and capability.directory_id == "workspace":
            if Path(workspace_root).resolve(strict=False) != self.workspace_root:
                raise CapabilityPermissionError("default workspace capability is outside the current session workspace")
        return capability

    def _register_directory_descriptor(self, descriptor: Any) -> None:
        payload = descriptor.to_dict()
        ecosystem = EcosystemDescriptor(
            capability_id=descriptor.capability_id,
            provider_id=descriptor.provider_id,
            version=descriptor.version,
            operations=tuple(descriptor.operations),
            input_schema=dict(descriptor.input_schema),
            risk_class=descriptor.risk_class,
            idempotency=descriptor.idempotency,
            health_evidence={
                "status": "ready" if descriptor.health_evidence.get("ok") else "degraded",
                **descriptor.health_evidence,
            },
            tool_package_sha256=descriptor.fingerprint(),
            description="Authorized directory capability",
            source_kind="directory-capability",
            tags=("directory", "filesystem", *descriptor.operations),
            source_ref=descriptor.capability_id,
        )
        self.catalog.register(ecosystem, replace=True)
        self.store.put_capability(payload)

    def refresh_catalog(self) -> None:
        for descriptor in self.broker.discover(
            tenant_id=self.tenant_id,
            project_id=self.project_id,
        ):
            self._register_directory_descriptor(descriptor)

    def ingest_runtime_tools(self, specs: Iterable[Any]) -> int:
        count = 0
        for spec in specs:
            payload = spec.to_dict() if callable(getattr(spec, "to_dict", None)) else dict(spec)
            name = str(payload.get("name") or "")
            if not name:
                continue
            encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
            capability = str(payload.get("capability") or "tool.invoke")
            descriptor = EcosystemDescriptor(
                capability_id=name,
                provider_id="evomind-runtime",
                version=str(payload.get("version") or "1.0.0"),
                operations=("invoke", capability, capability.rsplit(".", 1)[-1]),
                input_schema=dict(payload.get("input_schema") or {"type": "object"}),
                risk_class="read" if bool(payload.get("read_only", True)) else "write",
                idempotency="declared" if payload.get("supports_cancel") else "tool-call-key",
                health_evidence={
                    "status": "ready" if bool(payload.get("available", True)) else "unavailable",
                    "available": bool(payload.get("available", True)),
                },
                tool_package_sha256=hashlib.sha256(encoded).hexdigest(),
                description=str(payload.get("description") or name),
                source_kind="runtime-tool",
                tags=(name, *name.replace("-", "_").split("_"), capability, *capability.split(".")),
                source_ref=name,
            )
            self.catalog.register(descriptor, replace=True)
            self.store.put_capability(descriptor.to_dict())
            count += 1
        return count

    def discover(
        self,
        objective: str,
        *,
        required_operations: Iterable[str] = (),
        healthy_only: bool = True,
        run_id: str | None = None,
        workspace_root: str | Path | None = None,
        metadata: dict[str, Any] | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        rows = self.catalog.discover_ranked(
                objective,
                required_operations=required_operations,
                healthy_only=healthy_only,
                run_id=run_id,
                limit=limit,
            )
        required = tuple(dict.fromkeys(str(value) for value in required_operations if str(value)))
        session_directory_id = self._session_directories.get(str(run_id or ""))
        if session_directory_id:
            descriptor = self.catalog.get(session_directory_id, run_id=run_id)
            if descriptor is not None and set(required).issubset(set(descriptor.operations)):
                try:
                    self.authorize_directory(
                        session_directory_id,
                        session_id=str(run_id),
                        workspace_root=workspace_root or self.workspace_root,
                        metadata=metadata,
                    )
                except CapabilityPermissionError:
                    pass
                else:
                    rows = [
                        CapabilityMatch(
                            descriptor,
                            score=1_000_000.0,
                            matched_operations=tuple(sorted(required)),
                        ),
                        *(item for item in rows if item.descriptor.capability_id != session_directory_id),
                    ][:limit]
        result: list[dict[str, Any]] = []
        for item in rows:
            try:
                self.broker.capability(item.descriptor.capability_id)
            except Exception:
                result.append(item.to_dict())
                continue
            if run_id and workspace_root is not None:
                try:
                    self.authorize_directory(
                        item.descriptor.capability_id,
                        session_id=run_id,
                        workspace_root=workspace_root,
                        metadata=metadata,
                    )
                except CapabilityPermissionError:
                    continue
            result.append(item.to_dict())
        return result

    def refresh_installed_skills(self, roots: Iterable[str | Path] | None = None) -> dict[str, int]:
        selected = list(roots or (Path.home() / ".codex" / "skills", Path.home() / ".agents" / "skills"))
        totals = {"scanned_files": 0, "skipped_files": 0, "total_bytes": 0, "root_count": 0}
        policy = SkillScanPolicy(max_files=256, max_file_bytes=65_536, max_total_bytes=2_097_152)
        for index, root in enumerate(selected):
            candidate = Path(root).expanduser()
            if candidate.is_symlink() or not candidate.is_dir():
                continue
            report = scan_skill_manifests(
                candidate,
                provider_id=f"installed-skill-root-{index}",
                policy=policy,
            )
            self.catalog.ingest(report.descriptors, replace=True)
            totals["root_count"] += 1
            totals["scanned_files"] += report.scanned_files
            totals["skipped_files"] += report.skipped_files
            totals["total_bytes"] += report.total_bytes
            for descriptor in report.descriptors:
                self.store.put_capability(descriptor.to_dict())
        self._skill_scan = totals
        return dict(totals)

    def shadow_plan(
        self,
        *,
        run_id: str,
        objective: str,
        workspace_root: str | Path | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentKernel:
        existing = self.repository.load(run_id)
        if existing is not None:
            return self.supervisor.get(run_id)
        capability_ids = [
            item["descriptor"]["capability_id"]
            for item in self.discover(
                objective,
                run_id=run_id,
                workspace_root=workspace_root,
                metadata=metadata,
            )
        ]
        graph = TaskGraph(
            id=run_id,
            objective=objective,
            tenant_id=self.tenant_id,
            project_id=self.project_id,
            nodes=[
                TaskNode(
                    "discover",
                    "discover authorized capabilities",
                    acceptance_criteria=["capability catalog queried with health evidence"],
                    capability_ids=capability_ids,
                ),
                TaskNode(
                    "execute",
                    "execute the bounded objective",
                    dependencies=["discover"],
                    acceptance_criteria=["operation receipt produced"],
                    capability_ids=capability_ids,
                ),
                TaskNode(
                    "verify",
                    "independently verify receipts and deliverables",
                    dependencies=["execute"],
                    acceptance_criteria=["all selected receipts independently verified"],
                    capability_ids=capability_ids,
                ),
            ],
        )
        kernel = self.supervisor.register(AgentKernel(graph, repository=self.repository))
        self.store.put_task_graph(run_id, graph.status, graph.to_dict())
        return kernel

    def tool_packages(self, run_id: str) -> ToolPackageRuntime:
        key = str(run_id)
        current = self._tool_package_runtimes.get(key)
        if current is None:
            current = ToolPackageRuntime(self.store, run_id=key)
            self._tool_package_runtimes[key] = current
        return current

    def register_directory_connector(
        self,
        connector: Any,
        capability: DirectoryCapability,
        *,
        require_healthy: bool = True,
    ) -> None:
        self.broker.register_connector(connector)
        self.broker.mount(capability, require_healthy=require_healthy)
        self.store.put_connector({
            "connector_id": connector.connector_id,
            "connector_type": type(connector).__name__,
            "version": connector.version,
        })
        self.store.put_directory(capability.to_dict(include_opaque_root=True))
        descriptors = [
            descriptor
            for descriptor in self.broker.discover(
                tenant_id=capability.tenant_id,
                project_id=capability.project_id,
            )
            if descriptor.capability_id == capability.directory_id
        ]
        if len(descriptors) != 1:
            raise CapabilityNotFoundError("mounted directory capability is not discoverable")
        self._register_directory_descriptor(descriptors[0])

    def register_source_connector(self, connector: Any) -> None:
        connector_id = str(getattr(connector, "connector_id", ""))
        descriptor_factory = getattr(connector, "descriptor", None)
        if not connector_id or not callable(descriptor_factory) or not callable(getattr(connector, "fetch", None)):
            raise TypeError("source connector must expose connector_id, descriptor(), and fetch()")
        descriptor = descriptor_factory()
        payload = descriptor.to_dict()
        ecosystem = EcosystemDescriptor(
            capability_id=descriptor.capability_id,
            provider_id=descriptor.provider_id,
            version=descriptor.version,
            operations=tuple(descriptor.operations),
            input_schema=dict(descriptor.input_schema),
            risk_class=descriptor.risk_class,
            idempotency=descriptor.idempotency,
            health_evidence={"status": "ready", **descriptor.health_evidence},
            tool_package_sha256=descriptor.fingerprint(),
            description="Resumable source transfer connector",
            source_kind="source-connector",
            tags=("source", "transfer", "fetch", connector_id),
            source_ref=connector_id,
        )
        self._source_connectors[connector_id] = connector
        self.catalog.register(ecosystem, replace=True)
        self.store.put_capability(ecosystem.to_dict())

    def transfer_fetch(
        self,
        *,
        source_connector_id: str,
        source_url: str,
        directory_id: str,
        relative_path: str,
        resume: bool = True,
        expected_sha256: str = "",
    ):
        connector = self._source_connectors.get(str(source_connector_id))
        if connector is None:
            raise KeyError(source_connector_id)
        capability = self.broker.capability(directory_id)
        if not capability.allows("write"):
            raise ValueError("destination directory capability is not writable")
        return connector.fetch(
            source_url,
            capability,
            relative_path,
            resume=resume,
            expected_sha256=expected_sha256,
        )

    def register_execution_connector(self, connector: Any, capability: DirectoryCapability) -> None:
        connector_id = str(getattr(connector, "connector_id", ""))
        if (
            not connector_id
            or capability.connector_id != connector_id
            or not capability.allows("execute")
            or (capability.connector_version and capability.connector_version != str(getattr(connector, "version", "")))
        ):
            raise ValueError("execution connector capability binding is invalid")
        self._execution_connectors[capability.directory_id] = (connector, capability)
        payload = {
            "capability_id": capability.directory_id,
            "provider_id": connector_id,
            "version": str(getattr(connector, "version", "1.0.0")),
            "operations": ["execute", "status", "cancel"],
            "input_schema": {"type": "object"},
            "risk_class": "high",
            "idempotency": "job-ref",
            "health_evidence": {"status": "unknown", "requires_live_identity_gate": True},
            "tool_package_sha256": hashlib.sha256(
                f"{connector_id}:{getattr(connector, 'version', '1.0.0')}".encode("utf-8")
            ).hexdigest(),
        }
        self.catalog.register(EcosystemDescriptor(**payload), replace=True)
        self.store.put_connector({
            "connector_id": connector_id,
            "connector_type": type(connector).__name__,
            "version": payload["version"],
        })
        self.store.put_capability(payload)

    def execution_connector(self, directory_id: str) -> tuple[Any, DirectoryCapability]:
        current = self._execution_connectors.get(str(directory_id))
        if current is None:
            raise KeyError(directory_id)
        return current

    def remember_verified_experience(
        self,
        *,
        claim: str,
        evidence_sha256: str,
        payload: dict[str, Any] | None = None,
        category: str = "general",
        failure_signature: str = "",
        promote_anonymous_global: bool = False,
    ) -> list[dict[str, Any]]:
        record = self.memory.remember(
            claim=claim,
            status=EvidenceStatus.PROVEN.value,
            payload=payload or {},
            source_evidence=[EvidenceRef(evidence_sha256, kind="evidence")],
            tenant_id=self.tenant_id,
            project_id=self.project_id,
            category=category,
            failure_signature=failure_signature,
            scope=EvidenceScope.TENANT_PROJECT.value,
        )
        records = [record]
        if promote_anonymous_global:
            records.append(self.memory.promote_anonymous_global(record.id))
        for item in records:
            self.store.put_memory({
                "record_id": item.id,
                "tenant_id": item.tenant_id,
                "project_id": item.project_id,
                "memory_scope": item.scope,
                "evidence_status": item.status,
                "failure_signature": item.failure_signature,
                "evidence_sha256": item.content_sha256,
                "claim": item.claim,
                "payload": item.payload,
                "source_evidence": [entry.to_dict() for entry in item.source_evidence],
            })
        return [item.to_dict() for item in records]

    def status(self) -> dict[str, Any]:
        directories = self.store.list_directories(
            tenant_id=self.tenant_id,
            project_id=self.project_id,
        )
        public_directories = [
            {key: value for key, value in item.items() if key != "opaque_root_ref"}
            for item in directories
        ]
        digest = hashlib.sha256(
            "\n".join(item["directory_id"] for item in public_directories).encode("utf-8")
        ).hexdigest()
        active_run_ids = self.supervisor.active_run_ids()
        run_summaries: list[dict[str, Any]] = []
        for run_id in active_run_ids[:100]:
            snapshot = self.repository.load(run_id) or {}
            graph = snapshot.get("graph") if isinstance(snapshot.get("graph"), dict) else {}
            nodes = graph.get("nodes") if isinstance(graph.get("nodes"), list) else []
            run_summaries.append(
                {
                    "run_id": run_id,
                    "status": str(graph.get("status") or "unknown"),
                    "objective": str(graph.get("objective") or "")[:500],
                    "nodes": [
                        {
                            "id": str(item.get("id") or ""),
                            "action": str(item.get("action") or "")[:300],
                            "status": str(item.get("status") or "unknown"),
                            "dependencies": list(item.get("dependencies") or []),
                            "capability_ids": list(item.get("capability_ids") or []),
                        }
                        for item in nodes[:200]
                        if isinstance(item, dict)
                    ],
                    "failure_count": len(snapshot.get("failures") or []),
                    "repair_count": len(snapshot.get("repairs") or []),
                    "exact_gate": snapshot.get("exact_gate") if isinstance(snapshot.get("exact_gate"), dict) else None,
                }
            )
        return {
            "schema": "evomind.super_agent_runtime.v1",
            "mode": self.mode.value,
            "enabled": self.enabled,
            "shadow": self.shadow,
            "migration_applied": self.store.migration_applied(),
            "capability_count": len(self.catalog.all()),
            "skill_scan": dict(self._skill_scan),
            "directory_count": len(public_directories),
            "directory_ids_sha256": digest,
            "active_run_ids": active_run_ids,
            "runs": run_summaries,
            "directories": public_directories,
        }

    def close(self) -> None:
        self.store.close()


__all__ = ["SuperAgentMode", "SuperAgentRuntime"]
