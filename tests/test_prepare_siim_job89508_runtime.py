from __future__ import annotations

import inspect

import pytest

from scripts import prepare_siim_job89508_runtime as runtime


def test_runtime_paths_are_content_addressed_and_contained():
    digest = "a" * 64
    paths = runtime.runtime_paths(digest)
    assert paths["root"].endswith(digest)
    assert all(path.startswith(runtime.REMOTE_RUNTIME_ROOT + "/") for path in paths.values())
    with pytest.raises(runtime.RuntimeErrorContract, match="invalid requirements"):
        runtime.runtime_paths("../escape")
    with pytest.raises(runtime.RuntimeErrorContract, match="escaped"):
        runtime.ensure_remote(runtime.ALLOWED_GPU_REMOTE_ROOT + "/../escape")


def test_runtime_installer_and_verifier_never_signal_or_modify_other_processes():
    source = inspect.getsource(runtime)
    assert "os.kill(" not in source
    assert "terminate(" not in source
    assert "signals_sent\": 0" in source
    assert "other_processes_modified\": False" in source
    assert "strict_named_profile=True" in source
    assert "verify_job_container_identity" in source
    assert "cwd=ROOT" in source
    assert "PYTHONNOUSERSITE" in source
    for name in (
        "HOME",
        "TMPDIR",
        "XDG_CACHE_HOME",
        "PIP_CACHE_DIR",
        "PYTHONPYCACHEPREFIX",
        "KAGGLE_CONFIG_DIR",
    ):
        assert name in source
    assert 'environment.pop("KAGGLE_USERNAME",None)' in source
    assert 'environment.pop("KAGGLE_KEY",None)' in source
    assert "2.5.1+cu118" in source
    assert "0.20.1+cu118" in source
