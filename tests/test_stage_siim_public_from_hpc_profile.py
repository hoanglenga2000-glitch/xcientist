from __future__ import annotations

import importlib.util

import pytest


def _load_stage_module():
    spec = importlib.util.spec_from_file_location(
        "stage_siim_public_from_hpc",
        "scripts/stage_siim_public_from_hpc.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_siim_hpc_staging_requires_job90353_profile_before_network(monkeypatch):
    module = _load_stage_module()
    called = False

    def fake_loader(**_kwargs):
        nonlocal called
        called = True
        return object()

    monkeypatch.delenv("EVOMIND_HPC_CREDENTIAL_PROFILE", raising=False)
    monkeypatch.setattr(module, "load_gpu_ssh_config", fake_loader)

    with pytest.raises(RuntimeError, match="EVOMIND_HPC_CREDENTIAL_PROFILE=job90353"):
        module.load_bound_siim_hpc_config()

    assert called is False


def test_siim_hpc_staging_uses_strict_named_profile(monkeypatch):
    module = _load_stage_module()
    seen: dict[str, object] = {}

    def fake_loader(**kwargs):
        seen.update(kwargs)
        return {"ok": True}

    monkeypatch.setenv("EVOMIND_HPC_CREDENTIAL_PROFILE", "job90353")
    monkeypatch.setattr(module, "load_gpu_ssh_config", fake_loader)

    assert module.load_bound_siim_hpc_config() == {"ok": True}
    assert seen == {"strict_named_profile": True}
