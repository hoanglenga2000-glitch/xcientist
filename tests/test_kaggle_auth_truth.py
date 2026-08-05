from __future__ import annotations

import json
from pathlib import Path

from xsci.config import Config
from xsci.doctor import FAIL, OK, WARN, _check_compute, _check_kaggle
from xsci.kaggle_session import (
    KAGGLE_AUTHENTICATED,
    KAGGLE_CONFIGURED_UNVERIFIED,
    KAGGLE_NOT_CONFIGURED,
    SessionState,
    kaggle_auth_state,
)


def write_report(root: Path, payload: dict[str, object]) -> None:
    path = root / "workspace" / "verification" / "kaggle_dpapi_readiness.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_kaggle_not_configured_without_credentials_or_evidence(tmp_path: Path) -> None:
    cfg = Config()

    state = kaggle_auth_state(cfg, tmp_path)

    assert state.status == KAGGLE_NOT_CONFIGURED
    assert state.configured is False
    assert state.authenticated is False
    assert _check_kaggle(cfg, tmp_path)[0] == WARN


def test_token_presence_is_configured_unverified_not_ready(tmp_path: Path) -> None:
    cfg = Config(data={"secrets": {"kaggle_api_token": "fake-token"}})

    state = SessionState.from_root(tmp_path, cfg=cfg)

    assert state.kaggle_status == KAGGLE_CONFIGURED_UNVERIFIED
    assert state.kaggle_authenticated is False
    assert state.kaggle_ready is False
    assert ("kaggle", KAGGLE_CONFIGURED_UNVERIFIED) in state.status_rows()
    assert _check_kaggle(cfg, tmp_path)[0] == WARN


def test_authenticated_flag_without_real_smoke_contract_is_unverified(tmp_path: Path) -> None:
    write_report(
        tmp_path,
        {
            "status": "passed",
            "authenticated": True,
            "credential_installed": True,
            "credential_status": "configured_dpapi_unverified",
            "verification_method": "dpapi_status_only",
        },
    )

    state = kaggle_auth_state(Config(), tmp_path)

    assert state.status == KAGGLE_CONFIGURED_UNVERIFIED
    assert state.authenticated is False


def test_only_complete_real_api_smoke_contract_is_authenticated(tmp_path: Path) -> None:
    write_report(
        tmp_path,
        {
            "status": "passed",
            "authenticated": True,
            "credential_installed": True,
            "credential_status": "authenticated_real_api",
            "verification_method": "dpapi_status_and_real_api_smoke",
        },
    )
    cfg = Config()

    state = SessionState.from_root(tmp_path, cfg=cfg)

    assert state.kaggle_status == KAGGLE_AUTHENTICATED
    assert state.kaggle_authenticated is True
    assert state.kaggle_ready is True
    assert _check_kaggle(cfg, tmp_path)[0] == OK


def test_session_defaults_to_gpu_and_blocks_local_override() -> None:
    state = SessionState(llm_ready=True, selected_task="task", gpu_ready=True)

    assert state.compute_backend == "gpu"
    assert state.can_execute(compute_override="local") is False
    assert any("local training is disabled" in gap for gap in state.blocking_setup(compute_override="local"))
    assert _check_compute(Config(data={"compute": {"backend": "local"}}))[0] == FAIL
