from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json

import pytest

from evomind_runtime.run_requests import RunRequests, fingerprint
from evomind_runtime.runtime import AgentRuntime
from evomind_runtime.tenant_access import AccessError, AccessStore, Principal, current_principal
from evomind_runtime.user_projects import ProjectStore


ALICE = Principal("tenant_" + "a" * 24, "alice")
BOB = Principal("tenant_" + "a" * 24, "bob")
OTHER = Principal("tenant_" + "b" * 24, "alice")


def test_projects_are_private_persistent_and_idempotent(tmp_path):
    store = ProjectStore(tmp_path)
    body = {"name": "Owned project", "idempotency_key": "project_request_once"}
    first = store.create(ALICE, body)
    assert first["replayed"] is False
    assert store.create(ALICE, body)["project"] == first["project"]
    assert store.create(ALICE, body)["replayed"] is True
    assert ProjectStore(tmp_path).list_owned(ALICE) == [first["project"]]
    assert store.list_owned(BOB) == store.list_owned(OTHER) == []
    for principal in (BOB, OTHER):
        with pytest.raises(AccessError) as failure:
            store.get_owned(principal, first["project"]["id"])
        assert failure.value.status == 404
    with pytest.raises(AccessError):
        store.create(ALICE, {**body, "name": "Changed"})
    assert store.create(BOB, body)["project"]["id"] != first["project"]["id"]


@pytest.mark.parametrize("body", [{"name": "", "idempotency_key": "request_12345"}, {"name": "x\nsecret", "idempotency_key": "request_12345"}, {"name": "x", "idempotency_key": "short"}, {"name": "x", "idempotency_key": "request_12345", "owner_id": "alice"}])
def test_invalid_project_requests_do_not_create_records(tmp_path, body):
    store = ProjectStore(tmp_path)
    with pytest.raises(AccessError):
        store.create(ALICE, body)
    assert store.list_owned(ALICE) == []
    with pytest.raises(AccessError):
        store.create(None, {"name": "No owner", "idempotency_key": "request_12345"})


def test_simultaneous_project_creation_has_one_record(tmp_path):
    store = ProjectStore(tmp_path)
    with ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(lambda _: store.create(ALICE, {"name": "One project", "idempotency_key": "request_concurrent"}), range(5)))
    assert len({item["project"]["id"] for item in results}) == 1
    assert sum(not item["replayed"] for item in results) == 1


def test_project_run_is_bound_before_worker_and_cross_owner_is_denied(tmp_path, monkeypatch):
    runtime = AgentRuntime(tmp_path)
    store = ProjectStore(runtime.runtime_root)
    project = store.create(ALICE, {"name": "Project", "idempotency_key": "request_project"})["project"]
    access, requests, starts = AccessStore(runtime.runtime_root), RunRequests(runtime.runtime_root), []

    def start(run_id):
        assert runtime.store.get_session(run_id)["metadata"]["project_id"] == project["id"]
        assert store.runs(runtime, access, ALICE, project["id"])[0]["id"] == run_id
        starts.append(run_id)
        return True

    monkeypatch.setattr(runtime.assistant, "start", start)
    body = {"prompt": "No execution fixture", "project_id": project["id"], "idempotency_key": "request_project_run"}
    token = current_principal.set(ALICE)
    try:
        first = requests.create(runtime, access, ALICE, body)
        assert first["project_id"] == project["id"]
        assert requests.create(runtime, access, ALICE, body)["id"] == first["id"]
        assert starts == [first["id"]]
        current_principal.set(BOB)
        with pytest.raises(AccessError):
            requests.create(runtime, access, BOB, body)
        assert store.runs(runtime, access, BOB) == []
        assert len(runtime.store.list_assistant_runs()) == 1
    finally:
        current_principal.reset(token)
        runtime.close()


def test_project_titles_do_not_expose_sensitive_prompts(tmp_path, monkeypatch):
    runtime = AgentRuntime(tmp_path)
    monkeypatch.setattr(runtime.assistant, "start", lambda *_: True)
    token = current_principal.set(ALICE)
    try:
        requests, access = RunRequests(runtime.runtime_root), AccessStore(runtime.runtime_root)
        requests.create(runtime, access, ALICE, {"prompt": "password: fixture_sensitive_value", "idempotency_key": "request_redaction"})
        runs = ProjectStore(runtime.runtime_root).runs(runtime, access, ALICE)
        assert "fixture_sensitive_value" not in json.dumps(runs)
    finally:
        current_principal.reset(token)
        runtime.close()


def test_legacy_request_fingerprint_does_not_change_without_a_project():
    normalized = {"prompt": "Existing request", "conversation_id": "", "selected_task": "", "attachment_ids": [], "secret_refs": []}
    expected = hashlib.sha256(json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    assert fingerprint({"prompt": "Existing request"}) == expected
    assert fingerprint({"prompt": "Existing request", "project_id": ""}) == expected
    assert fingerprint({"prompt": "Existing request", "project_id": "project_" + "a" * 32}) != expected
