from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from evomind_runtime.runtime import AgentRuntime
from evomind_runtime.run_requests import RunRequests
from evomind_runtime.tenant_access import AccessError, AccessStore, Principal, current_principal


ALICE = Principal("tenant_" + "a" * 24, "alice")
BOB = Principal("tenant_" + "b" * 24, "bob")


@pytest.fixture
def admission(tmp_path, monkeypatch):
    runtime = AgentRuntime(tmp_path)
    requests = RunRequests(runtime.runtime_root)
    access = AccessStore(runtime.runtime_root)
    starts = []
    monkeypatch.setattr(runtime.assistant, "start", lambda run_id, **_: starts.append(run_id) or True)

    def create(principal=ALICE, key="request_fixture_once", **changes):
        token = current_principal.set(principal)
        try:
            return requests.create(runtime, access, principal, {"prompt": "Admission fixture", "idempotency_key": key, **changes})
        finally:
            current_principal.reset(token)

    yield runtime, requests, access, starts, create
    runtime.close()


def test_repeated_request_has_one_run_and_one_worker(admission):
    runtime, requests, access, starts, create = admission
    first = create()
    second = create()
    assert first["id"] == second["id"]
    assert second["creation_request"]["replayed"] is True
    assert starts == [first["id"]]
    assert len(runtime.store.list_assistant_runs()) == 1
    assert requests.lookup(runtime, access, ALICE, "request_fixture_once")["id"] == first["id"]


def test_changed_payload_does_not_reuse_an_idempotency_key(admission):
    runtime, _, _, starts, create = admission
    first = create()
    with pytest.raises(AccessError) as failure:
        create(prompt="Changed request")
    assert failure.value.status == 409 and failure.value.code == "idempotency_payload_changed"
    assert starts == [first["id"]]
    assert len(runtime.store.list_assistant_runs()) == 1


def test_request_keys_and_blank_conversations_are_principal_scoped(admission):
    _, requests, access, starts, create = admission
    first = create(ALICE)
    second = create(BOB)
    assert first["id"] != second["id"]
    assert first["conversation_id"] != second["conversation_id"]
    assert "conversation_default" not in {first["conversation_id"], second["conversation_id"]}
    assert len(starts) == 2


def test_foreign_conversation_is_rejected_before_run_or_worker_creation(admission):
    runtime, _, _, starts, create = admission
    first = create(conversation_id="owned_conversation")
    with pytest.raises(AccessError):
        create(BOB, conversation_id="owned_conversation")
    assert len(runtime.store.list_assistant_runs()) == 1
    assert starts == [first["id"]]


def test_history_filters_legacy_mixed_ownership(admission):
    runtime, _, _, _, create = admission
    create(key="request_owner_earlier", prompt="ALICE_ONLY", conversation_id="mixed_legacy")
    token = current_principal.set(BOB)
    try:
        runtime.assistant.create_run(prompt="BOB_PRIVATE", conversation_id="mixed_legacy", start=False)
    finally:
        current_principal.reset(token)
    runtime.assistant.create_run(prompt="UNOWNED_PRIVATE", conversation_id="mixed_legacy", start=False)
    latest = create(key="request_owner_latest", conversation_id="mixed_legacy")
    history = runtime.assistant._conversation_context(runtime.store.get_assistant_run(latest["id"]))
    assert "ALICE_ONLY" in history
    assert "BOB_PRIVATE" not in history and "UNOWNED_PRIVATE" not in history


def test_parallel_duplicate_is_rejected_while_original_admission_is_live(admission, monkeypatch):
    runtime, _, _, starts, create = admission
    entered, release = threading.Event(), threading.Event()
    original = runtime.assistant.create_run

    def delayed(**kwargs):
        entered.set()
        assert release.wait(5)
        return original(**kwargs)

    monkeypatch.setattr(runtime.assistant, "create_run", delayed)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(create)
        assert entered.wait(2)
        try:
            with pytest.raises(AccessError) as failure:
                create()
            assert failure.value.code == "run_creation_in_progress"
        finally:
            release.set()
        first = future.result(timeout=5)
    assert starts == [first["id"]]
    assert create()["id"] == first["id"]


def test_restart_replays_without_starting_another_worker(tmp_path, monkeypatch):
    runtime = AgentRuntime(tmp_path)
    access = AccessStore(runtime.runtime_root)
    requests = RunRequests(runtime.runtime_root)
    monkeypatch.setattr(runtime.assistant, "start", lambda *_args, **_kwargs: True)
    body = {"prompt": "Restart fixture", "idempotency_key": "request_restart_once"}
    token = current_principal.set(ALICE)
    try:
        first = requests.create(runtime, access, ALICE, body)
    finally:
        current_principal.reset(token)
        runtime.close()
    restored = AgentRuntime(tmp_path)
    monkeypatch.setattr(restored.assistant, "start", lambda *_args, **_kwargs: pytest.fail("replay cannot start a worker"))
    try:
        result = RunRequests(restored.runtime_root).create(restored, AccessStore(restored.runtime_root), ALICE, body)
        assert result["id"] == first["id"]
    finally:
        restored.close()


def test_creation_failure_remains_durable_and_never_reexecutes(admission, monkeypatch):
    runtime, requests, access, starts, create = admission
    monkeypatch.setattr(runtime.assistant, "create_run", lambda **_: (_ for _ in ()).throw(ValueError("fixture_input_failure")))
    with pytest.raises(ValueError, match="fixture_input_failure"):
        create()
    with pytest.raises(AccessError) as failure:
        create()
    assert failure.value.code == "run_creation_reconciliation_required"
    assert starts == []
    with pytest.raises(AccessError):
        requests.lookup(runtime, access, BOB, "request_fixture_once")


def test_partial_run_is_blocked_before_worker_admission(admission, monkeypatch):
    runtime, requests, access, starts, create = admission
    original = access.created_run
    monkeypatch.setattr(access, "created_run", lambda *_: (_ for _ in ()).throw(RuntimeError("fixture_index_failure")))
    with pytest.raises(RuntimeError, match="fixture_index_failure"):
        create()
    monkeypatch.setattr(access, "created_run", original)
    restored = create()
    assert restored["status"] == "blocked"
    assert restored["error_class"] == "creation_interrupted"
    assert starts == []


@pytest.mark.parametrize("key", ["short", "../unsafe_key", "x" * 129])
def test_invalid_request_key_has_no_side_effects(admission, key):
    runtime, _, _, starts, create = admission
    with pytest.raises(AccessError) as failure:
        create(key=key)
    assert failure.value.status == 400
    assert runtime.store.list_assistant_runs() == [] and starts == []
