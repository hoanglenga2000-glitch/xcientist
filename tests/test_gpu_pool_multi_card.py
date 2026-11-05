"""2026-11-05: several cards in one pool (gpu-pool.json ``nodes``): best-fit grants, concurrency, per-card breakers."""
from __future__ import annotations

import importlib
import json
import uuid
from types import SimpleNamespace

import pytest

TENANT = "tenant_" + "a" * 24
OTHER = "tenant_" + "b" * 24
PRIMARY = {"node_id": "job98456", "hpc_job_id": 98456, "service_tenant_id": TENANT, "tenant_training": True,
           "gpu_model": "A40", "gpu_vram_mb": 49140}
EXTRA = {"node_id": "job98229", "hpc_job_id": 98229, "gpu_model": "A800 MIG 3g.40gb", "gpu_vram_mb": 40192, "priority": 1}


def _reload(monkeypatch, tmp_path, block):
    path = tmp_path / "gpu-pool.json"
    path.write_text(json.dumps({"gpu_pool": block}), encoding="utf-8")
    monkeypatch.setenv("EVOMIND_GPU_POOL_CONFIG", str(path))
    from evomind_runtime import gpu_admission, gpu_scheduler, tenant_training
    s = importlib.reload(gpu_scheduler)
    a = importlib.reload(gpu_admission)
    t = importlib.reload(tenant_training)
    return s, a, t


@pytest.fixture(autouse=True)
def _restore(monkeypatch):
    yield
    monkeypatch.delenv("EVOMIND_GPU_POOL_CONFIG", raising=False)
    from evomind_runtime import gpu_admission, gpu_scheduler, tenant_training
    importlib.reload(gpu_scheduler)
    importlib.reload(gpu_admission)
    importlib.reload(tenant_training)


def _two_card_sched(monkeypatch, tmp_path, extra=EXTRA):
    s, a, t = _reload(monkeypatch, tmp_path, {**PRIMARY, "nodes": [extra]})
    sched = s.GpuScheduler(tmp_path / "sched.sqlite3")
    sched.ensure_node(profile=f"{TENANT}_job98456_g47")
    sched.ensure_extra_nodes({"job98229": f"{TENANT}_job98229_g48"})
    return s, a, t, sched


def test_nodes_list_is_parsed_and_invalid_entries_are_ignored(monkeypatch, tmp_path):
    bad = [{"node_id": "job1", "hpc_job_id": 98229, "gpu_vram_mb": 1}, {"node_id": "job98456", "hpc_job_id": 98456, "gpu_vram_mb": 1},
           {"node_id": "job98300", "hpc_job_id": 98300}, {"node_id": "job98301", "hpc_job_id": True, "gpu_vram_mb": 1}, "x"]
    s, _a, _t = _reload(monkeypatch, tmp_path, {**PRIMARY, "nodes": [*bad, EXTRA]})
    assert s.POOL_NODE_IDS == ("job98456", "job98229")
    assert s.POOL_HPC_JOB_IDS == frozenset({98456, 98229})
    assert s.NODE_PRIORITY == {"job98456": 0, "job98229": 1}


def test_single_card_file_is_unchanged(monkeypatch, tmp_path):
    s, a, _t = _reload(monkeypatch, tmp_path, PRIMARY)
    assert (s.POOL_NODE_IDS, s.EXTRA_POOL_NODES) == (("job98456",), ())
    sched = s.GpuScheduler(tmp_path / "sched.sqlite3")
    try:
        sched.ensure_node(profile=f"{TENANT}_job98456_g47")
        job = sched.submit(tenant_id=TENANT, owner_id="admin", run_id="r1", requested_seconds=600, vram_mb=4096)
        assert job["required_node"] == "job98456"
        grants = sched.schedule(free_vram=40000)
        assert [g.node_id for g in grants] == ["job98456"]
        assert a.pool_breaker_state(sched) == "closed"
    finally:
        sched.close()


def test_two_tenants_run_at_once_on_two_cards(monkeypatch, tmp_path):
    _s, _a, _t, sched = _two_card_sched(monkeypatch, tmp_path)
    try:
        first = sched.submit(tenant_id=TENANT, owner_id="admin", run_id="r1", requested_seconds=600, vram_mb=4096)
        second = sched.submit(tenant_id=OTHER, owner_id="u", run_id="r2", requested_seconds=600, vram_mb=4096)
        grants = sched.schedule(free_vram={"job98456": 48000, "job98229": 40000})
        assert {g.job_id: g.node_id for g in grants} == {first["id"]: "job98456", second["id"]: "job98229"}
        assert {g.profile for g in grants} == {f"{TENANT}_job98456_g47", f"{TENANT}_job98229_g48"}
        assert sched.idle_nodes() == []
        # a third job waits for whichever card frees first
        third = sched.submit(tenant_id="tenant_" + "c" * 24, owner_id="u", run_id="r3", requested_seconds=600, vram_mb=4096)
        assert sched.schedule(free_vram={"job98456": 48000, "job98229": 40000}) == []
        assert sched.queue_status(third["id"])["position"] == 1
        sched.finish(second["id"], "succeeded")
        assert [g.node_id for g in sched.schedule(free_vram={"job98456": 48000, "job98229": 40000})] == ["job98229"]
    finally:
        sched.close()


def test_same_tenant_still_runs_one_job_at_a_time(monkeypatch, tmp_path):
    _s, _a, _t, sched = _two_card_sched(monkeypatch, tmp_path)
    try:
        sched.submit(tenant_id=TENANT, owner_id="admin", run_id="r1", requested_seconds=600, vram_mb=4096)
        sched.submit(tenant_id=TENANT, owner_id="admin", run_id="r2", requested_seconds=600, vram_mb=4096)
        assert len(sched.schedule(free_vram={"job98456": 48000, "job98229": 40000})) == 1
    finally:
        sched.close()


def test_unhealthy_card_fails_over_to_the_other(monkeypatch, tmp_path):
    _s, _a, _t, sched = _two_card_sched(monkeypatch, tmp_path)
    try:
        sched.submit(tenant_id=TENANT, owner_id="admin", run_id="r1", requested_seconds=600, vram_mb=4096)
        grants = sched.schedule(free_vram={"job98456": None, "job98229": 40000})
        assert [g.node_id for g in grants] == ["job98229"]
    finally:
        sched.close()


def test_preference_then_most_free_memory(monkeypatch, tmp_path):
    _s, _a, _t, sched = _two_card_sched(monkeypatch, tmp_path, {**EXTRA, "priority": 0})
    try:
        sched.submit(tenant_id=TENANT, owner_id="admin", run_id="r1", requested_seconds=600, vram_mb=4096)
        assert [g.node_id for g in sched.schedule(free_vram={"job98456": 20000, "job98229": 39000})] == ["job98229"]
    finally:
        sched.close()


def test_big_head_job_keeps_its_card_but_small_card_serves_others(monkeypatch, tmp_path):
    _s, _a, _t, sched = _two_card_sched(monkeypatch, tmp_path)
    try:
        big = sched.submit(tenant_id=TENANT, owner_id="admin", run_id="r1", requested_seconds=600, vram_mb=45000)
        small = sched.submit(tenant_id=OTHER, owner_id="u", run_id="r2", requested_seconds=600, vram_mb=4096)
        # A40 is short right now (another process uses memory); the head job keeps its claim on it,
        # the 40 GB slice can never hold the head job, so the small job runs there meanwhile.
        grants = sched.schedule(free_vram={"job98456": 30000, "job98229": 40000})
        assert [(g.job_id, g.node_id) for g in grants] == [(small["id"], "job98229")]
        assert sched.job(big["id"])["reason"] == "insufficient_gpu_memory"
        assert [g.node_id for g in sched.schedule(free_vram={"job98456": 48000})] == ["job98456"]
    finally:
        sched.close()


def test_request_larger_than_every_card_is_refused(monkeypatch, tmp_path):
    s, _a, _t, sched = _two_card_sched(monkeypatch, tmp_path)
    try:
        with pytest.raises(s.SchedulerError, match="insufficient_gpu_memory"):
            sched.submit(tenant_id=TENANT, owner_id="admin", run_id="r1", requested_seconds=600, vram_mb=60000)
    finally:
        sched.close()


def test_breakers_are_per_card(monkeypatch, tmp_path):
    _s, a, _t, sched = _two_card_sched(monkeypatch, tmp_path)
    try:
        primary, extra = a.CircuitBreaker(sched), a.CircuitBreaker(sched, node_id="job98229")
        for _ in range(3):
            extra.record(False)
        assert (primary.state(), extra.state(), a.pool_breaker_state(sched)) == ("closed", "open", "closed")
        for _ in range(3):
            primary.record(False)
        assert a.pool_breaker_state(sched) == "open"
    finally:
        sched.close()


class _Out:
    def __init__(self, text, code=0):
        self._text, self.channel = text.encode(), SimpleNamespace(recv_exit_status=lambda: code)

    def read(self):
        return self._text


class _Client:
    QUERY = "GPU-db6886e5-f9fc-ddb5-a0d8-1220e43d97d6, [Insufficient Permissions]\n"
    TABLE = ("GPU 0: NVIDIA A800-SXM4-80GB (UUID: GPU-db6886e5-f9fc-ddb5-a0d8-1220e43d97d6)\n"
             "  MIG 3g.40gb     Device  0: (UUID: MIG-34f06a42-5a3a-501a-ba7c-aac912057ddd)\n__EVOMIND_MIG_TABLE__\n"
             "|==================+==================================+===========+=======================|\n"
             "|  0    2   0   0  |             102MiB / 40192MiB    | 42      0 |  3   0    2    0    0 |\n"
             "|                  |               0MiB / 24531MiB    |           |                       |\n")

    def exec_command(self, command, timeout=None):
        return None, _Out(self.QUERY if command.startswith("nvidia-smi --query-gpu") else self.TABLE), None

    def close(self):
        pass


def test_mig_slice_free_memory_is_read_from_the_table(monkeypatch, tmp_path):
    _s, a, _t = _reload(monkeypatch, tmp_path, PRIMARY)
    result = a.ssh_free_vram_probe(_Client, "mig-34f06a42-5a3a-501a-ba7c-aac912057ddd")()
    assert (result.ok, result.free_vram_mb) == (True, 40090)
    assert a.ssh_free_vram_probe(_Client, "mig-00000000-0000-0000-0000-000000000000")().ok is False


def test_gates_of_two_tenants_get_different_cards(monkeypatch, tmp_path):
    _s, a, _t = _reload(monkeypatch, tmp_path, {**PRIMARY, "nodes": [EXTRA]})
    config = {**a.pool_config(), "credential_profile": f"{TENANT}_job98456_g47", "node_profiles": {"job98229": f"{TENANT}_job98229_g48"}}
    ok = lambda mb: (lambda: a.ProbeResult(True, mb))
    gates = [a.GpuGate(tmp_path, config, probe=ok(48000), node_probes={"job98229": ok(40000)}, sleep=lambda _s: None) for _ in range(2)]
    try:
        first = gates[0].acquire(tenant_id=TENANT, owner_id="admin", run_id="r1", requested_seconds=600, vram_mb=4096)
        second = gates[1].acquire(tenant_id=OTHER, owner_id="u", run_id="r2", requested_seconds=600, vram_mb=4096)
        assert (first.node_id, second.node_id) == ("job98456", "job98229")
        first.release(True)
        second.release(True)
    finally:
        for gate in gates:
            gate.close()


def test_gate_with_one_dead_card_uses_the_other(monkeypatch, tmp_path):
    _s, a, _t = _reload(monkeypatch, tmp_path, {**PRIMARY, "nodes": [EXTRA]})
    config = {**a.pool_config(), "credential_profile": f"{TENANT}_job98456_g47", "node_profiles": {"job98229": f"{TENANT}_job98229_g48"}}
    gate = a.GpuGate(tmp_path, config, probe=lambda: a.ProbeResult(False), node_probes={"job98229": lambda: a.ProbeResult(True, 40000)},
                     sleep=lambda _s: None)
    try:
        lease = gate.acquire(tenant_id=TENANT, owner_id="admin", run_id="r1", requested_seconds=600, vram_mb=4096)
        assert lease.node_id == "job98229"
        lease.release(True)
    finally:
        gate.close()


def _pool_record(tmp_path, **override):
    record = {"schema": "evomind.tenant_hpc_pool_node.v1", "tenant_id": TENANT, "job_id": 98229, "state": "active",
              "credential_profile": f"{TENANT}_job98229_g48", "allocation_generation": 48,
              "profile_instance_id": str(uuid.uuid4()), "allocation_binding_id": "job-98229", **override}
    path = tmp_path / "byoa" / "tenants" / TENANT / "pool-nodes" / "job98229.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8")


def test_pool_node_identity_is_strict(monkeypatch, tmp_path):
    _s, a, t = _reload(monkeypatch, tmp_path, {**PRIMARY, "nodes": [EXTRA]})
    monkeypatch.setenv("WORKSTATION_BYOA_STATE_ROOT", str(tmp_path / "byoa"))
    config = a.pool_config()
    _pool_record(tmp_path)
    assert t.pool_node_identity(config, 98229)["credential_profile"] == f"{TENANT}_job98229_g48"
    assert set(a.pool_node_bindings(config)) == {"job98229"}
    for bad in ({"state": "retired"}, {"job_id": 98228}, {"credential_profile": f"{TENANT}_job98229_g47"}, {"tenant_id": OTHER}):
        _pool_record(tmp_path, **bad)
        with pytest.raises(t.TenantTrainingError):
            t.pool_node_identity(config, 98229)
        assert a.pool_node_bindings(config) == {}
    with pytest.raises(t.TenantTrainingError):
        t.pool_node_identity(config, 98456)  # the primary is bound by hpc-binding.json only


def test_every_pool_job_id_is_scrubbed_for_tenants(monkeypatch, tmp_path):
    _s, _a, t = _reload(monkeypatch, tmp_path, {**PRIMARY, "nodes": [EXTRA]})
    text = t.scrub_text("卡 98229 和 98456 都在忙")
    assert "98229" not in text and "98456" not in text


def test_job_moved_to_another_card_is_verified_first(monkeypatch, tmp_path):
    _s, a, _t = _reload(monkeypatch, tmp_path, {**PRIMARY, "nodes": [EXTRA]})
    from evomind_runtime import tools

    sched = a.scheduler(tmp_path)
    sched.ensure_node(profile=f"{TENANT}_job98456_g47")
    sched.ensure_extra_nodes({"job98229": f"{TENANT}_job98229_g48"})
    job = sched.submit(tenant_id=TENANT, owner_id="admin", run_id="r1", requested_seconds=600, vram_mb=4096)
    sched.schedule(free_vram={"job98229": 40000})
    lease = a.LeaseHandle(sched, sched.job(job["id"]), sched.lease_for(job["id"]), heartbeat_seconds=30)
    runtime = SimpleNamespace(reports_dispatch=True)
    configs = {"job98229": (None, f"{TENANT}_job98229_g48", 98229)}
    assert tools._move_runtime_to_card(SimpleNamespace(gpu_node_verify=lambda _n: True), runtime, lease, configs) is None
    assert runtime.gpu_slot_lock_name == ".evomind-gpu-slot-job98229.lock"
    lease.release(True)

    job = sched.submit(tenant_id=TENANT, owner_id="admin", run_id="r2", requested_seconds=600, vram_mb=4096)
    sched.schedule(free_vram={"job98229": 40000})
    lease = a.LeaseHandle(sched, sched.job(job["id"]), sched.lease_for(job["id"]), heartbeat_seconds=30)
    refused = tools._move_runtime_to_card(SimpleNamespace(gpu_node_verify=lambda _n: False), SimpleNamespace(reports_dispatch=True), lease, configs)
    assert refused is not None and refused.error == "gpu_pool_unavailable"
    row = sched.job(job["id"])
    assert row["status"] == "failed" and sched.job_billed_seconds(job["id"]) == 0.0
    sched.close()


def test_kills_and_checks_go_to_the_card_the_job_ran_on(monkeypatch, tmp_path):
    _s, a, _t = _reload(monkeypatch, tmp_path, {**PRIMARY, "nodes": [EXTRA]})
    seen = []
    route = a.per_node({"job98456": "c1", "job98229": "c2"}, lambda connect: (lambda job: seen.append(connect) or True), missing=None)
    route({"node_id": "job98229"})
    route({"node_id": None})
    assert seen == ["c2", "c1"]
    assert route({"node_id": "job1"}) is None
