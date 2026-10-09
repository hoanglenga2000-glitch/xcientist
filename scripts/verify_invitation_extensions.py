"""Isolated project HTTP and tensor-contract checks; never access HPC or a provider."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import secrets
import shutil
import sqlite3
import struct
import sys
import time
from types import SimpleNamespace


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--web-sha256", required=True)
    parser.add_argument("--runtime-sha256", required=True)
    parser.add_argument("--build-id", required=True)
    args = parser.parse_args()
    stage = args.stage_root.resolve()
    stage.relative_to(Path("C:/ProgramData/EvoMind/staging").resolve())
    helper = stage / "verify_invitation_server_candidate.py"
    spec = importlib.util.spec_from_file_location("extension_harness", helper)
    v = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v)
    flow = stage / "extension-flow"
    flow.mkdir(exist_ok=False)
    v.extract(stage / "web.zip", args.web_sha256, flow / "web", "operational-overlay-manifest.json")
    v.extract(stage / "runtime.zip", args.runtime_sha256, flow / "runtime", "runtime-hotfix-manifest.json")
    (flow / "workspace").mkdir()
    shutil.copyfile(stage / "fixture-schema.sqlite", flow / "workspace/workstation.sqlite")
    secret, password = secrets.token_urlsafe(48), secrets.token_urlsafe(32)
    identity = json.loads((flow / "web/runtime-build-manifest.json").read_text())["source_tree_sha256"]
    environment = v.fixture_environment(flow, secret, password, args.build_id, identity)
    services = v.FixtureServices(flow, environment, "C:/Program Files/nodejs/node.exe")
    before, checks = v.production_snapshot(), []
    result = {"schema": "evomind.invitation_project_acceptance.v1", "status": "running", "build_id": args.build_id, "web_sha256": args.web_sha256, "runtime_sha256": args.runtime_sha256, "fixture_only": True, "hpc_accessed": False, "real_training_verified": False}

    def check(name, passed, **evidence):
        checks.append({"name": name, "passed": bool(passed), **evidence})
        if not passed:
            raise RuntimeError(name)

    try:
        services.start()
        port = services.web_port
        origin = "http://127.0.0.1:" + str(port)
        status, _, _ = v.request(port, "GET", "/api/assistant/projects")
        check("project_unauthenticated_denied", status == 401)
        status, headers, _ = v.request(port, "POST", "/api/auth/login", body={"username": "invitation_acceptance", "password": password}, headers={"Origin": origin})
        if status != 200:
            raise RuntimeError("fixture_login_failed")
        cookie = headers["set-cookie"].split(";", 1)[0]
        auth = {"Cookie": cookie, "Origin": origin}
        _, _, session = v.request(port, "GET", "/api/session/status", headers=auth)
        auth[v.CSRF] = session["csrf_token"]
        body = {"name": "Isolated project acceptance", "idempotency_key": "project-create-once"}
        status, _, _ = v.request(port, "POST", "/api/assistant/projects", body=body, headers={"Cookie": cookie, "Origin": origin})
        check("project_csrf_required", status == 403)
        status, _, created = v.request(port, "POST", "/api/assistant/projects", body=body, headers=auth)
        project = created.get("project") or {}
        check("project_create", status == 201 and bool(project.get("id")))
        status, _, repeated = v.request(port, "POST", "/api/assistant/projects", body=body, headers=auth)
        check("project_replay", status == 200 and repeated.get("project") == project)
        status, _, _ = v.request(port, "POST", "/api/assistant/projects", body={**body, "name": "Changed"}, headers=auth)
        check("project_payload_conflict", status == 409)
        status, _, _ = v.request(port, "POST", "/api/assistant/projects", body={**body, "owner_id": "foreign"}, headers=auth)
        check("project_owner_spoof_denied", status == 400)
        foreign = {"Cookie": v.COOKIE + "=" + v.signed_principal(secret, "foreign_fixture", "tenant_" + "b" * 24), "Origin": origin}
        _, _, foreign_session = v.request(port, "GET", "/api/session/status", headers=foreign)
        foreign[v.CSRF] = foreign_session["csrf_token"]
        status, _, _ = v.request(port, "GET", "/api/assistant/projects/" + project["id"], headers=foreign)
        check("project_foreign_read_denied", status == 404)
        status, _, listing = v.request(port, "GET", "/api/assistant/projects", headers=foreign)
        check("project_foreign_list_empty", status == 200 and listing.get("projects") == [] and listing.get("runs") == [])
        run_body = {"prompt": "ISOLATED_ACCEPTANCE_ECHO project", "project_id": project["id"], "idempotency_key": "project-run-once"}
        status, _, _ = v.request(port, "POST", "/api/assistant/runs", body=run_body, headers=foreign)
        check("project_foreign_run_denied", status == 404)
        status, _, run = v.request(port, "POST", "/api/assistant/runs", body=run_body, headers=auth)
        run_id = run.get("id")
        check("project_run_binding", status == 201 and run.get("project_id") == project["id"] and bool(run_id))
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            _, _, snapshot = v.request(port, "GET", "/api/assistant/runs/" + run_id, headers=auth)
            if snapshot.get("terminal"):
                break
            time.sleep(0.2)
        if snapshot.get("status") != "completed":
            raise RuntimeError("project_fixture_run_not_completed")
        _, _, replay = v.request(port, "POST", "/api/assistant/runs", body=run_body, headers=auth)
        check("project_run_replay", replay.get("id") == run_id)
        _, _, other = v.request(port, "POST", "/api/assistant/projects", body={"name": "Second", "idempotency_key": "project-second"}, headers=auth)
        status, _, _ = v.request(port, "POST", "/api/assistant/runs", body={**run_body, "project_id": other["project"]["id"]}, headers=auth)
        check("project_run_fingerprint", status == 409)
        services.stop()
        services.start()
        status, _, restored = v.request(services.web_port, "GET", "/api/assistant/projects/" + project["id"], headers=auth)
        check("project_survives_restart", status == 200 and restored.get("project") == project)
        check("project_run_survives_restart", len(restored.get("runs", [])) == 1 and restored["runs"][0]["id"] == run_id)
        connection = sqlite3.connect((flow / "workspace/workspace/runtime/runtime.sqlite3").as_uri() + "?mode=ro", uri=True)
        try:
            count = connection.execute("SELECT count(*) FROM events WHERE session_id=? AND event_type='acceptance_fixture_invoked'", (run_id,)).fetchone()[0]
        finally:
            connection.close()
        check("project_single_invocation", count == 1)
    except Exception as error:
        result["error_class"] = type(error).__name__
    finally:
        try:
            services.stop()
        except Exception:
            result["cleanup_failed"] = True
        checks.append({"name": "production_unchanged", "passed": v.production_snapshot() == before})
        result["checks"] = checks
        result["status"] = "passed" if len(checks) == 16 and all(row["passed"] for row in checks) and "error_class" not in result and not result.get("cleanup_failed") else "failed"
        v.write_json(stage / "project-acceptance.json", result)
        print(json.dumps(result), flush=True)
    if result["status"] != "passed":
        return 1
    if not (flow / "runtime/evomind_runtime/managed_tensor_train.py").is_file():
        return 0
    return tensor_checks(v, stage, flow, args)


def tensor_checks(v, stage, flow, args):
    import numpy as np
    sys.path.insert(0, str(flow / "runtime"))
    from evomind_runtime.training_control import digest, load_protocol
    from evomind_runtime.model_evidence import verify_managed_tensor_bundle
    from evomind_runtime.research_budget import GpuExecutionBudget
    from evomind_runtime.policy import PolicyEngine
    from evomind_runtime.models import PermissionLevel

    root = flow / "tensor-contract"
    root.mkdir()
    runtime_root = root / "runtime"
    runtime_root.mkdir()
    protocols = root / "protocols"
    protocols.mkdir()
    metadata = {"tenant_id": "tenant_" + "c" * 24, "owner_principal_id": "tensor_fixture", "project_id": "project_" + "c" * 32}
    for name, x, y, ids in (("train", [[0, 0], [1, 1]], [0, 1], [0, 1]), ("eval", [[2, 1], [3, 1]], [3, 5], [2, 3])):
        np.savez(protocols / (name + ".npz"), x=np.array(x, dtype=np.float32), y=np.array(y, dtype=np.float32), ids=np.array(ids))
    protocol = {"schema": "evomind.tensor_protocol.v1", "tenant_id": metadata["tenant_id"], "owner_id": metadata["owner_principal_id"], "project_id": metadata["project_id"], "scope": "engineering", "task": "regression", "metric": "rmse", "training": {"path": "train.npz", "sha256": digest(protocols / "train.npz")}, "evaluation": {"path": "eval.npz", "sha256": digest(protocols / "eval.npz")}, "source": {"uri": "urn:fixture:synthetic", "version": "1", "license": "CC0-1.0"}}
    temporary = protocols / "temporary.json"
    v.write_json(temporary, protocol)
    protocol_id = digest(temporary)
    temporary.rename(protocols / (protocol_id + ".json"))
    v.write_json(root / "policy.json", {"schema": "evomind.training_control.v1", "runtime_root": str(runtime_root), "enabled": True, "tenants": [metadata["tenant_id"]], "owners": [metadata["owner_principal_id"]], "projects": [metadata["project_id"]], "study_id": "fixture"})
    os.environ["EVOMIND_AIBUILD_POLICY_FILE"] = str(root / "policy.json")
    output = root / "task/outputs/hpc/solution_01"
    output.mkdir(parents=True)
    weights = {"normalizer_mean": [0, 0], "normalizer_scale": [1, 1], "layers.0.weight": [[2, -1]], "layers.0.bias": [0]}
    offset, header, data = 0, {}, []
    for name, value in sorted(weights.items()):
        array = np.asarray(value, dtype="<f4")
        blob = array.tobytes()
        header[name] = {"dtype": "F32", "shape": list(array.shape), "data_offsets": [offset, offset+len(blob)]}
        offset += len(blob); data.append(blob)
    encoded = json.dumps(header).encode()
    encoded += b" " * ((-len(encoded)) % 8)
    (output / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(data))
    v.write_json(output / "model-config.json", {"schema": "evomind.tensor_training.v1", "protocol_id": protocol_id, "task": "regression", "input_features": 2, "outputs": 1, "hidden_sizes": []})
    np.savez(output / "predictions.npz", predictions=np.array([[3], [5]], dtype=np.float32))
    for name in ("training-receipt.json", "environment.lock.json", "data-protocol.json"):
        v.write_json(output / name, {"fixture_only": True})
    (output / "predict.py").write_text("# fixture-only inference identity\n")
    (output / "training-log.jsonl").write_text('{"fixture_only":true}\n')
    evidence = {"schema": "evomind.managed_training_evidence.v1", "candidate": "solution_01", "protocol_id": protocol_id, "relative_root": "outputs/hpc/solution_01", "files": {p.name: digest(p) for p in output.iterdir()}, "fit_steps": 1, "adapter_sha256": "a" * 64}
    calls = [{"tool_name": "managed_tensor_train", "status": "completed", "result": {"ok": True, "content": {"managed_training": evidence}}}]
    runtime = SimpleNamespace(runtime_root=runtime_root, store=SimpleNamespace(list_tool_calls=lambda _: calls))
    session = {"id": "fixture", "workspace_root": str(root / "task"), "metadata": metadata}
    checked = verify_managed_tensor_bundle(runtime, session, "solution_01")
    checks = [{"name": "independent_tensor_reload", "passed": checked.get("verified") is True and checked.get("value") == 0 and checked.get("official_score") is False}]
    (output / "model.safetensors").write_bytes(b"tampered")
    checks.append({"name": "tampered_model_rejected", "passed": verify_managed_tensor_bundle(runtime, session, "solution_01").get("verified") is False})
    calls.clear()
    v.write_json(output / "independent-verification.json", {"passed": True})
    checks.append({"name": "self_assertion_rejected", "passed": verify_managed_tensor_bundle(runtime, session, "solution_01").get("verified") is False})
    try:
        load_protocol(runtime_root, {**metadata, "owner_principal_id": "other"}, protocol_id)
        foreign_rejected = False
    except ValueError:
        foreign_rejected = True
    checks.append({"name": "foreign_protocol_rejected", "passed": foreign_rejected})
    budget = GpuExecutionBudget(str(root / "budget.sqlite3"), "fixture")
    reservation = budget.reserve("fixture", 60)
    budget.settle(reservation["id"], 1, uncertain=True)
    try:
        budget.reserve("replacement", 60)
        uncertain_rejected = False
    except RuntimeError:
        uncertain_rejected = True
    checks.append({"name": "uncertain_gpu_budget_blocks_replacement", "passed": uncertain_rejected})
    policy = PolicyEngine()
    protocol_read = policy.evaluate(tool_name="model_protocols", arguments={}, permission_level=PermissionLevel.OBSERVE.value, workspace_root=root)
    training = policy.evaluate(tool_name="managed_tensor_train", arguments={"configuration_path": "config.json", "configuration_sha256": "b"*64, "protocol_id": protocol_id, "solution_id": "solution_01"}, permission_level=PermissionLevel.FULL_AUTO.value, workspace_root=root)
    checks.append({"name": "managed_training_permission_contract", "passed": protocol_read.allowed and not protocol_read.requires_approval and not training.allowed and training.requires_approval})
    report = {"schema": "evomind.invitation_tensor_contract.v1", "status": "passed" if all(row["passed"] for row in checks) else "failed", "checks": checks, "web_sha256": args.web_sha256, "runtime_sha256": args.runtime_sha256, "fixture_only": True, "real_training_verified": False, "hpc_accessed": False}
    v.write_json(stage / "tensor-acceptance.json", report)
    print(json.dumps(report), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
