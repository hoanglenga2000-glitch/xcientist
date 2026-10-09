"""Regression: local MLE-bench prepared sources resolve without touching the
shipped six-competition contract."""
from __future__ import annotations

from evomind_runtime.competition_data import (
    COMPETITION_CATALOG,
    LOCAL_PREPARED_SOURCES,
    MLEBENCH_PREPARED_ROOT,
    build_adapter_script,
    normalize_competition,
    persistent_root,
    source_catalog_sha256,
    validate_receipt,
)

SIX_COMPETITION_CONTRACT_SHA256 = "5540fffdea8741766783b18b2915f3c6feac7d5174c1368c4ab7e8902b919a8b"


def test_six_competition_contract_is_unchanged() -> None:
    assert list(COMPETITION_CATALOG) == [
        "cure_bench",
        "e2lmc",
        "mindgames",
        "ariel_2025",
        "weather4cast",
        "open_polymer",
    ]
    assert source_catalog_sha256() == SIX_COMPETITION_CONTRACT_SHA256


def test_local_prepared_source_resolves_and_builds_read_only_adapters() -> None:
    assert "histopathologic_cancer" in LOCAL_PREPARED_SOURCES
    assert normalize_competition("histopathologic-cancer-detection") == "histopathologic_cancer"
    assert normalize_competition("Pathology") == "histopathologic_cancer"
    assert persistent_root("histopathologic_cancer").endswith("/histopathologic_cancer")

    status = build_adapter_script("histopathologic_cancer", "status")
    prepare = build_adapter_script("histopathologic_cancer", "prepare")
    assert "FULL_DATA_READY" in status
    assert f"{MLEBENCH_PREPARED_ROOT}/histopathologic-cancer-detection/prepared" in prepare
    assert "ln -sfn" in prepare
    assert "FULL_DATA_READY" in prepare
    # Preparation is read-only: no Kaggle archive download, no credential gate.
    assert "setup_kaggle" not in prepare
    assert "rules-accept" not in prepare

    receipt = {
        "schema": "evomind.competition_data_receipt.v1",
        "competition": "histopathologic_cancer",
        "status": "FULL_DATA_READY",
        "persistent_root": persistent_root("histopathologic_cancer"),
        "secret_values_logged": False,
    }
    assert validate_receipt("histopathologic_cancer", receipt)["status"] == "FULL_DATA_READY"
