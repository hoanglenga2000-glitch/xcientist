from __future__ import annotations

from pathlib import Path

import pytest

from research_agent_workstation.server.strategy import harness_optimizer


def test_harness_paths_are_repository_relative() -> None:
    expected_root = Path(__file__).resolve().parents[1]
    assert harness_optimizer.ROOT == expected_root
    config = harness_optimizer.HarnessEngine("titanic").task_config
    sample = Path(config["sample_submission"])
    assert sample == expected_root / "tasks" / "titanic" / "data" / "sample_submission.csv"


def test_harness_local_training_is_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EVOMIND_ALLOW_LOCAL_TRAINING", raising=False)

    with pytest.raises(RuntimeError, match="Local harness training is disabled"):
        harness_optimizer.train_ensemble_for_island("titanic", {})


def test_harness_local_training_cannot_be_enabled_by_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EVOMIND_ALLOW_LOCAL_TRAINING", "1")

    with pytest.raises(RuntimeError, match="Local harness training is disabled"):
        harness_optimizer.train_ensemble_for_island("titanic", {})


def test_harness_source_has_no_machine_specific_root() -> None:
    source = Path(harness_optimizer.__file__).read_text(encoding="utf-8")
    assert "D:/" not in source
    assert "D:\\" not in source
