"""2026-11-05: the schedulable card follows gpu-pool.json (job 98270 expired; new allocation 98456, an A40)."""
from __future__ import annotations

import importlib
import json

import pytest


def _reload(monkeypatch, tmp_path, block):
    path = tmp_path / "gpu-pool.json"
    if block is not None:
        path.write_text(json.dumps({"gpu_pool": block}), encoding="utf-8")
    monkeypatch.setenv("EVOMIND_GPU_POOL_CONFIG", str(path))
    from evomind_runtime import gpu_scheduler
    return importlib.reload(gpu_scheduler)


@pytest.fixture(autouse=True)
def _restore(monkeypatch):
    yield
    monkeypatch.delenv("EVOMIND_GPU_POOL_CONFIG", raising=False)
    from evomind_runtime import gpu_scheduler
    importlib.reload(gpu_scheduler)


def test_new_job_and_card_come_from_the_pool_file(monkeypatch, tmp_path):
    s = _reload(monkeypatch, tmp_path, {"node_id": "job98456", "hpc_job_id": 98456, "gpu_model": "A40", "gpu_vram_mb": 49140})
    assert (s.ALLOWED_HPC_JOB_ID, s.ALLOWED_NODE_ID) == (98456, "job98456")
    assert (s.SERVICE_NODE_LABEL, s.SERVICE_NODE_VRAM_MB, s.GpuPolicy().node_vram_mb) == ("A40", 49140, 49140)


def test_missing_or_inconsistent_pool_file_keeps_the_previous_default(monkeypatch, tmp_path):
    s = _reload(monkeypatch, tmp_path, None)
    assert (s.ALLOWED_HPC_JOB_ID, s.ALLOWED_NODE_ID, s.SERVICE_NODE_LABEL) == (98270, "job98270", "A800")
    s = _reload(monkeypatch, tmp_path, {"node_id": "job1", "hpc_job_id": 98456})
    assert s.ALLOWED_HPC_JOB_ID == 98270
    s = _reload(monkeypatch, tmp_path, {"node_id": "job98456", "hpc_job_id": True})
    assert s.ALLOWED_HPC_JOB_ID == 98270


def test_scheduler_registers_and_grants_the_new_card(monkeypatch, tmp_path):
    s = _reload(monkeypatch, tmp_path, {"node_id": "job98456", "hpc_job_id": 98456, "gpu_model": "A40", "gpu_vram_mb": 49140})
    sched = s.GpuScheduler(tmp_path / "sched.sqlite3")
    try:
        sched.ensure_node(profile="tenant_" + "a" * 24 + "_job98456_g47")
        row = sched._db().execute("SELECT id, gpu_model, vram_mb FROM pool_nodes").fetchone()
        assert tuple(row) == ("job98456", "A40", 49140)
        job = sched.submit(tenant_id="tenant_" + "a" * 24, owner_id="admin", run_id="run_x", requested_seconds=600, vram_mb=4096)
        grants = sched.schedule(free_vram=40000)
        assert grants and grants[0].job_id == job["id"] and grants[0].node_id == "job98456"
    finally:
        sched.close()
