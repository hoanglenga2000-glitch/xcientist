from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGENTS_PATH = PROJECT_ROOT / "AGENTS.md"
MEMORY_PATH = PROJECT_ROOT / "docs" / "HPC_CONNECTION_MEMORY_CORE.md"
CONTRACT_PATH = PROJECT_ROOT / "configs" / "hpc_connection_memory_core.json"
BRIDGE_MANAGER_PATH = PROJECT_ROOT / "scripts" / "manage_hpc_proxy_bridge.ps1"


def test_hpc_connection_core_is_a_mandatory_project_entrypoint() -> None:
    agents = AGENTS_PATH.read_text(encoding="utf-8")
    normalized_agents = " ".join(agents.split())

    assert "HPC_CONNECTION_MEMORY_CORE.md" in agents
    assert "hpc_connection_memory_core.json" in agents
    assert "NOT connected until the designated proxy path has been verified" in normalized_agents
    assert "entered and verified the intended job container" in normalized_agents
    assert "proxy first -> HPC SSH gateway" in normalized_agents
    assert "job container" in normalized_agents
    assert "profile name is not the role account" in normalized_agents
    assert "a forbidden off-campus direct target" in normalized_agents
    assert "Accounts/profiles are never shared" in normalized_agents
    assert "must never be reused or retargeted" in normalized_agents
    assert "new/rebuilt job-scoped profile" in normalized_agents


def test_hpc_connection_contract_preserves_proxy_gateway_container_order() -> None:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    route = contract["connection_state_machine"]

    assert contract["schema"] == "evomind.hpc.connection_memory_core.v1"
    assert contract["status"] == "mandatory"
    assert contract["gateway"]["proxy_required"] is True
    assert contract["gateway"]["ssh_host"] == "100.85.169.63"
    assert contract["gateway"]["ssh_port"] == 1235
    assert contract["gateway"]["off_campus_allocation_private_ip_direct_allowed"] is False
    assert route.index("local_socks_proxy") < route.index("hpc_ssh_gateway")
    assert route.index("local_socks_proxy") < route.index("designated_upstream_socks5_proxy")
    assert route.index("designated_upstream_socks5_proxy") < route.index("hpc_ssh_gateway")
    assert route.index("hpc_ssh_gateway") < route.index("allocation_role_routing")
    assert route.index("allocation_role_routing") < route.index("job_container")
    assert "off_campus_direct_to_allocation_private_ip" in contract["forbidden_connection_modes"]
    assert "job_profile_treated_as_role_account" in contract["forbidden_connection_modes"]
    assert contract["secrets_stored"] is False


def test_hpc_connection_core_pins_the_guide_proxy_chain_without_secrets() -> None:
    memory = MEMORY_PATH.read_text(encoding="utf-8")
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    proxy = contract["proxy_chain"]

    assert contract["preflight_read_order"] == [
        "AGENTS.md",
        "docs/HPC_CONNECTION_MEMORY_CORE.md",
        "configs/hpc_connection_memory_core.json",
    ]
    assert contract["connection_ready_invariant"] == (
        "designated_proxy_path_verified AND job_container_verified"
    )
    assert proxy == {
        "manual_client": "Clash",
        "guide_local_socks_host": "127.0.0.1",
        "guide_local_socks_port": 7890,
        "designated_upstream_socks5_host": "8.163.52.223",
        "designated_upstream_socks5_port": 1080,
        "gateway_route_rule": "IP-CIDR,100.85.169.63/32,Proxy",
        "managed_profile_local_port_allowed": True,
        "managed_profile_must_preserve_designated_upstream": True,
        "direct_route_allowed": False,
        "other_upstream_proxy_allowed": False,
        "credentials_embedded": False,
    }
    assert "proxy_path_verified AND job_container_verified" in memory
    assert "代理账号、代理密码和 HPC 角色凭据均属于秘密" in memory


def test_managed_hpc_bridge_defaults_to_the_designated_upstream_proxy() -> None:
    manager = BRIDGE_MANAGER_PATH.read_text(encoding="utf-8")

    assert '[ValidateSet("upstream", "direct")]' in manager
    assert '[string]$RouteMode = "upstream"' in manager
    assert '[string]$RouteMode = "direct"' not in manager


def test_hpc_connection_is_not_ready_until_target_container_is_verified() -> None:
    memory = MEMORY_PATH.read_text(encoding="utf-8")
    agents = AGENTS_PATH.read_text(encoding="utf-8")
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    gate = contract["execution_gate"]

    assert "永久记忆口令" in memory
    assert "没有完成目标容器身份核验，就不算已经连接 HPC" in memory
    assert "Reaching the SOCKS proxy or SSH gateway alone is not a usable HPC connection" in agents
    assert gate["ready_state"] == "job_container_verified"
    assert gate["container_entry_required"] is True
    assert gate["proxy_only_is_ready"] is False
    assert gate["gateway_only_is_ready"] is False
    assert gate["allocation_private_ip_metadata_is_ready"] is False
    assert gate["required_container_evidence"] == [
        "designated_proxy_path_verified",
        "pinned_gateway_host_key_verified",
        "allocation_role_authenticated",
        "expected_host_uuid_match",
        "expected_gpu_uuid_match",
        "expected_gpu_model_and_memory_match",
        "allowed_remote_root_match",
    ]
    assert gate["forbidden_before_container_verified"] == [
        "gpu_command",
        "training",
        "monitoring",
        "artifact_collection",
        "grader_execution",
        "remote_write",
    ]
    assert gate["fail_closed"] is True


def test_hpc_connection_core_distinguishes_source_policy_from_project_gate() -> None:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    source = contract["source_policy"]

    assert contract["source"] == {
        "path": "D:/桌面/AIMSLAB HPC校外连接指南（实习生版本）.pdf",
        "sha256": "7850dea325dd61cccf9ae19c5b11dcc75f840d12ec881d86259ffe3fe45a7b51",
        "page_count": 4,
        "visually_verified_pages": 4,
    }
    assert source["origin"] == "AIMSLAB_off_campus_connection_guide"
    assert source["required_sequence"] == [
        "designated_proxy",
        "hpc_ssh_gateway",
        "allocation_role_account",
    ]
    assert source["manual_second_ssh_required"] is False
    assert source["designated_proxy_only"] is True
    assert source["direct_access_forbidden"] is True
    assert source["other_proxy_forbidden"] is True
    assert source["role_account_personal_only"] is True
    assert source["role_account_sharing_allowed"] is False
    assert source["role_account_reclaimed_on_assignment_end"] is True
    assert source["bulk_file_transfer_over_ssh_forbidden"] is True
    assert contract["guide_evidence"]["page_4"] == [
        "bulk_file_transfer_over_ssh_forbidden",
        "proxy_configuration_mutation_forbidden",
        "role_account_sharing_forbidden",
        "role_account_reclaimed_after_assignment",
        "direct_or_other_proxy_access_forbidden",
    ]
    assert contract["project_identity_gate"]["origin"] == "evomind_fail_closed_extension"
    assert contract["project_identity_gate"]["required_bindings"] == [
        "job_id",
        "named_job_dpapi_profile",
        "profile_state_active",
        "allocation_binding_id",
        "allocation_generation",
        "profile_instance_id",
        "profile_bound_allocation_role_account",
        "pinned_host_key",
        "expected_host_uuid",
        "expected_gpu_uuid",
        "allowed_remote_root",
    ]


def test_hpc_connection_core_requires_fail_closed_profile_lifecycle() -> None:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    lifecycle = contract["account_lifecycle"]

    assert lifecycle["metadata_schema"] == "evomind.hpc.dpapi_profile.v2"
    assert lifecycle["runtime_required_state"] == "active"
    assert lifecycle["runtime_credential_decryption_before_active_allowed"] is False
    assert lifecycle["provisioning_credential_use"] == "one_shot_identity_bootstrap_only"
    assert lifecycle["allocation_binding_required"] is True
    assert lifecycle["allocation_generation_must_be_positive"] is True
    assert lifecycle["profile_instance_identity_required"] is True
    assert lifecycle["legacy_metadata_migration"] == (
        "fail_closed_secure_reenrollment_required"
    )
    assert lifecycle["lifecycle_lock_file"] == ".hpc_profile_lifecycle.lock"
    assert lifecycle["serialized_operations"] == [
        "dpapi_decryption",
        "strict_ssh_connection",
        "identity_bootstrap_activation_commit",
        "freeze",
        "retire",
    ]
    assert lifecycle["tombstone_files"] == {
        "frozen": "hpc_profile_frozen.tombstone.json",
        "retired": "hpc_profile_retired.tombstone.json",
        "runtime_loader_rejects_if_present": True,
        "automatic_removal_allowed": False,
    }
    assert {
        "schema",
        "profile_state",
        "allocation_binding_id",
        "allocation_generation",
        "profile_instance_id",
        "lifecycle_revision",
        "expected_host_uuid",
        "expected_gpu_uuid",
        "container_binding_sha256",
    } <= set(contract["required_profile_fields"])


def test_hpc_job_profile_is_not_a_role_account_and_has_fail_closed_lifecycle() -> None:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    semantics = contract["profile_semantics"]
    lifecycle = contract["account_lifecycle"]
    current = contract["current_binding"]

    assert semantics["name_pattern"] == "job<job_id>"
    assert semantics["profile_is_role_account"] is False
    assert semantics["profile_is_ssh_username"] is False
    assert semantics["role_account_source"] == "windows_dpapi_profile_credential"
    assert semantics["role_account_value_embedded_in_contract"] is False
    assert current["credential_profile_is_role_account"] is False
    assert current["role_account_storage"] == "windows_dpapi_profile_credential"
    assert current["role_account_value_embedded"] is False
    assert lifecycle["account_sharing_allowed"] is False
    assert lifecycle["profile_sharing_allowed"] is False
    assert lifecycle["on_allocation_or_account_revoked"] == "freeze_profile"
    assert "allocation_expired" in lifecycle["freeze_triggers"]
    assert "allocation_reclaimed" in lifecycle["freeze_triggers"]
    assert "role_account_reclaimed" in lifecycle["freeze_triggers"]
    assert lifecycle["frozen_actions"] == {
        "connection_allowed": False,
        "training_allowed": False,
        "monitoring_allowed": False,
        "collection_allowed": False,
        "grader_allowed": False,
        "retarget_allowed": False,
    }
    assert lifecycle["new_or_reissued_allocation_requires"] == [
        "new_or_rebuilt_job_scoped_profile",
        "secure_reenrollment",
        "pinned_host_key",
        "expected_host_uuid",
        "expected_gpu_uuid",
        "allowed_remote_root",
    ]
    assert lifecycle["old_profile_reuse_allowed"] is False
    assert lifecycle["old_profile_retarget_allowed"] is False


def test_allocation_private_ip_is_never_an_off_campus_target_or_fallback() -> None:
    memory = MEMORY_PATH.read_text(encoding="utf-8")
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    private_network = contract["allocation_private_network"]

    assert "10.120.x.x" in memory
    assert "禁止的直连目标" in memory
    assert "不是首选但可作备用" in memory
    assert private_network["address_pattern"] == "10.120.0.0/16"
    assert private_network["off_campus_direct_target"] == "forbidden"
    assert private_network["off_campus_fallback_target"] == "forbidden"
    assert private_network["direct_probe_as_route_discovery"] == "forbidden"
    assert private_network["tcp_reachability_changes_route_policy"] is False


def test_hpc_connection_memory_records_identity_and_process_guards() -> None:
    memory = MEMORY_PATH.read_text(encoding="utf-8")
    agents = AGENTS_PATH.read_text(encoding="utf-8")
    raw_contract = CONTRACT_PATH.read_text(encoding="utf-8")
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

    assert "SOCKS5" in memory
    assert "HPC SSH 网关" in memory
    assert "作业容器" in memory
    assert "DPAPI" in memory
    assert "RejectPolicy" in memory
    assert "Host UUID" in memory
    assert "GPU UUID" in memory
    assert "是禁止的直连目标" in memory
    assert "PDF 强制链路" in memory
    assert "EvoMind 项目增强门禁" in memory
    assert "不声称由 PDF 原文提出" in memory
    assert "profile 名称，不是 HPC 角色账号" in memory
    assert "禁止共享" in memory
    assert "立即冻结旧 profile" in memory
    assert "旧 profile 不得改指或复用" in memory
    assert "新建/重建" in memory
    assert "aimslab-" not in memory + agents + raw_contract
    assert contract["required_audit_values"] == {
        "signals_sent": 0,
        "other_processes_modified": False,
    }
