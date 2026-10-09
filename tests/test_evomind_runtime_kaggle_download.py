from __future__ import annotations

import io
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from evomind_runtime import tools


def _zip_payload() -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("train.csv", "id,target\n1,0\n")
        archive.writestr("test.csv", "id\n2\n")
        archive.writestr("sample_submission.csv", "id,target\n2,0.5\n")
    return stream.getvalue()


def _context(tmp_path: Path) -> tools.ToolContext:
    task_root = tmp_path / "assistant_tasks" / "run_fixture"
    for name in ("inputs", "work", "outputs", "logs", "evidence"):
        (task_root / name).mkdir(parents=True, exist_ok=True)
    return tools.ToolContext(
        session_id="run_fixture",
        workspace_root=task_root,
        project_root=tmp_path,
        runtime_root=tmp_path,
        artifact_root=tmp_path / "artifacts",
        store=None,
        metadata={},
    )


@pytest.mark.parametrize("wrapped", [False, True])
def test_kaggle_list_accepts_legacy_sequences_and_kaggle_2_2_response_wrappers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, wrapped: bool,
) -> None:
    rows = [
        SimpleNamespace(
            ref="porto-seguro-safe-driver-prediction",
            title="Porto Seguro Safe Driver Prediction",
            deadline="2017-11-29T23:59:00Z",
            evaluationMetric="Normalized Gini",
        )
    ]
    response = SimpleNamespace(competitions=rows) if wrapped else rows
    api = SimpleNamespace(competitions_list=lambda **_kwargs: response)
    monkeypatch.setattr(tools, "_load_kaggle_api", lambda _context: api)

    result = tools._kaggle_list({}, _context(tmp_path))

    assert result.ok is True
    assert result.content["count"] == 1
    assert result.content["attempts"] == 1
    assert result.content["competitions"] == [
        {
            "slug": "porto-seguro-safe-driver-prediction",
            "title": "Porto Seguro Safe Driver Prediction",
            "deadline": "2017-11-29T23:59:00Z",
            "metric": "Normalized Gini",
            "url": "https://www.kaggle.com/competitions/porto-seguro-safe-driver-prediction",
        }
    ]


def test_kaggle_list_retries_transient_connection_resets_then_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    sleeps: list[float] = []
    row = SimpleNamespace(
        ref="porto-seguro-safe-driver-prediction",
        title="Porto Seguro Safe Driver Prediction",
        deadline="2017-11-29T23:59:00Z",
        evaluationMetric="Normalized Gini",
    )

    def competitions_list(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ConnectionError("fixture reset")
        return SimpleNamespace(competitions=[row])

    monkeypatch.setenv("EVOMIND_KAGGLE_LIST_RETRY_BASE_SECONDS", "0.5")
    loads = 0

    def load_api(_context):
        nonlocal loads
        loads += 1
        return SimpleNamespace(competitions_list=competitions_list)

    monkeypatch.setattr(tools, "_load_kaggle_api", load_api)
    monkeypatch.setattr(tools.time, "sleep", sleeps.append)

    result = tools._kaggle_list({}, _context(tmp_path))

    assert result.ok is True
    assert result.content["count"] == 1
    assert result.content["attempts"] == 2
    assert calls == 2
    assert loads == 1
    assert sleeps == [0.5]


def test_kaggle_list_falls_back_to_fixed_catalog_after_three_transient_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    sleeps: list[float] = []

    def competitions_list(**_kwargs):
        nonlocal calls
        calls += 1
        raise ConnectionError("fixture reset")

    monkeypatch.setenv("EVOMIND_KAGGLE_LIST_RETRY_BASE_SECONDS", "0.5")
    monkeypatch.setattr(tools, "_load_kaggle_api", lambda _context: SimpleNamespace(competitions_list=competitions_list))
    monkeypatch.setattr(tools.time, "sleep", sleeps.append)

    result = tools._kaggle_list({}, _context(tmp_path))

    assert result.ok is True
    assert result.error == ""
    assert result.content["attempts"] == 3
    assert result.content["source"] == "managed_catalog_fallback"
    assert result.content["live_listing_ok"] is False
    assert result.content["listing_error_class"] == "ConnectionError"
    assert {item["slug"] for item in result.content["competitions"]} == {
        "cure-bench",
        "ariel-data-challenge-2025",
        "neurips-open-polymer-prediction-2025",
    }
    assert {item["id"] for item in result.content["competition_data_catalog"]} == {
        "cure_bench", "e2lmc", "mindgames", "ariel_2025", "weather4cast", "open_polymer",
    }
    assert calls == 3
    assert sleeps == [0.5, 1.0]


def test_kaggle_list_retries_transient_authentication_then_reuses_authenticated_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    loads = 0
    list_calls = 0
    sleeps: list[float] = []
    row = SimpleNamespace(
        ref="porto-seguro-safe-driver-prediction",
        title="Porto Seguro Safe Driver Prediction",
        deadline="2017-11-29T23:59:00Z",
        evaluationMetric="Normalized Gini",
    )

    def competitions_list(**_kwargs):
        nonlocal list_calls
        list_calls += 1
        return SimpleNamespace(competitions=[row])

    def load_api(_context):
        nonlocal loads
        loads += 1
        if loads == 1:
            raise ConnectionError("fixture authentication reset")
        return SimpleNamespace(competitions_list=competitions_list)

    monkeypatch.setenv("EVOMIND_KAGGLE_LIST_RETRY_BASE_SECONDS", "0.5")
    monkeypatch.setattr(tools, "_load_kaggle_api", load_api)
    monkeypatch.setattr(tools.time, "sleep", sleeps.append)

    result = tools._kaggle_list({}, _context(tmp_path))

    assert result.ok is True
    assert result.content["attempts"] == 2
    assert result.content["count"] == 1
    assert loads == 2
    assert list_calls == 1
    assert sleeps == [0.5]


def test_kaggle_list_does_not_retry_non_transient_contract_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    sleeps: list[float] = []

    def competitions_list(**_kwargs):
        nonlocal calls
        calls += 1
        raise TypeError("fixture contract error")

    monkeypatch.setattr(tools, "_load_kaggle_api", lambda _context: SimpleNamespace(competitions_list=competitions_list))
    monkeypatch.setattr(tools.time, "sleep", sleeps.append)

    result = tools._kaggle_list({}, _context(tmp_path))

    assert result.ok is False
    assert result.error == "TypeError"
    assert result.content["attempts"] == 1
    assert calls == 1
    assert sleeps == []


def test_kaggle_download_is_atomic_bounded_and_hashes_extracted_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = {}
    monkeypatch.setenv("KAGGLE_API_TOKEN", "fixture-token")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-kaggle-worker")
    monkeypatch.setenv("EVOMIND_KAGGLE_TOTAL_TIMEOUT_SECONDS", "123")
    monkeypatch.setattr("xsci.config.load_config", lambda _root: SimpleNamespace())
    monkeypatch.setattr("xsci.config.inject_engine_env", lambda *_args, **_kwargs: [])

    def fake_run(argv, **kwargs):
        observed.update(argv=argv, kwargs=kwargs)
        destination = Path(argv[-1])
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "porto-seguro-safe-driver-prediction.zip").write_bytes(_zip_payload())
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(tools.subprocess, "run", fake_run)

    result = tools._kaggle_download(
        {"competition": "porto-seguro-safe-driver-prediction"},
        _context(tmp_path),
    )

    assert result.ok is True
    assert result.content["schema"] == "evomind.kaggle.download.v2"
    assert result.content["download_mode"] == "bounded_atomic"
    names = {Path(item["path"]).name for item in result.content["files"]}
    assert {"train.csv", "test.csv", "sample_submission.csv"} <= names
    assert all(len(item["sha256"]) == 64 for item in result.content["files"])
    assert 1 <= observed["kwargs"]["timeout"] <= 41
    assert observed["kwargs"]["env"]["KAGGLE_API_TOKEN"] == "fixture-token"
    assert "OPENAI_API_KEY" not in observed["kwargs"]["env"]
    assert observed["argv"][:3] == [tools.sys.executable, "-X", "utf8"]
    target = Path(_context(tmp_path).workspace_root) / "inputs" / "kaggle" / "porto-seguro-safe-driver-prediction"
    assert not list(target.glob("*.part"))
    assert not list(target.parent.glob(".*.download"))


def test_kaggle_download_preserves_redacted_worker_error_and_classifies_human_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KAGGLE_API_TOKEN", "fixture-token")
    monkeypatch.setattr("xsci.config.load_config", lambda _root: SimpleNamespace())
    monkeypatch.setattr("xsci.config.inject_engine_env", lambda *_args, **_kwargs: [])

    calls = 0
    sleeps: list[float] = []

    def forbidden_run(_argv, **_kwargs):
        nonlocal calls
        calls += 1
        return SimpleNamespace(
            returncode=1,
            stderr=(
                b"403 Forbidden: You must accept the competition rules\n"
                b"Authorization: Bearer fixture-secret-value\n"
                b"proxy=https://proxy-user:proxy-password@example.invalid\n"
            ),
        )

    monkeypatch.setattr(tools.subprocess, "run", forbidden_run)
    monkeypatch.setattr(tools.time, "sleep", sleeps.append)

    result = tools._kaggle_download({"competition": "cure-bench"}, _context(tmp_path))

    assert result.ok is False
    assert result.error == "RuntimeError"
    assert result.content["error_category"] == "human_gate_or_auth"
    assert result.content["human_gate_required"] is True
    assert "403 Forbidden" in result.content["error_detail"]
    assert "fixture-secret-value" not in result.content["error_detail"]
    assert "proxy-user" not in result.content["error_detail"]
    assert "proxy-password" not in result.content["error_detail"]
    assert "[redacted]" in result.content["error_detail"]
    assert calls == 1
    assert sleeps == []


def test_kaggle_download_retries_connection_reset_with_same_staging_then_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KAGGLE_API_TOKEN", "fixture-token")
    monkeypatch.setenv("EVOMIND_KAGGLE_DOWNLOAD_RETRY_BASE_SECONDS", "1")
    monkeypatch.setattr("xsci.config.load_config", lambda _root: SimpleNamespace())
    monkeypatch.setattr("xsci.config.inject_engine_env", lambda *_args, **_kwargs: [])
    calls = 0
    destinations: list[Path] = []
    sleeps: list[float] = []

    def reset_then_succeed(argv, **_kwargs):
        nonlocal calls
        calls += 1
        destination = Path(argv[-1])
        destinations.append(destination)
        destination.mkdir(parents=True, exist_ok=True)
        partial = destination / "cure-bench.zip.kaggle-partial"
        if calls < 3:
            partial.write_bytes(f"partial-{calls}".encode("ascii"))
            return SimpleNamespace(
                returncode=1,
                stderr=(
                    "requests.exceptions.ConnectionError: "
                    "ConnectionResetError(10054) via "
                    "https://proxy-user:proxy-secret@example.invalid\n"
                ).encode("utf-8"),
            )
        assert partial.read_bytes() == b"partial-2"
        (destination / "cure-bench.zip").write_bytes(_zip_payload())
        return SimpleNamespace(returncode=0, stderr=b"")

    monkeypatch.setattr(tools.subprocess, "run", reset_then_succeed)
    monkeypatch.setattr(tools.time, "sleep", sleeps.append)

    context = _context(tmp_path)
    result = tools._kaggle_download({"competition": "cure-bench"}, context)

    assert result.ok is True
    assert calls == 3
    assert len(set(destinations)) == 1
    assert sleeps == [1.0, 2.0]
    assert all(len(item["sha256"]) == 64 for item in result.content["files"])
    assert "proxy-user" not in str(result.content)
    assert "proxy-secret" not in str(result.content)
    target = context.workspace_root / "inputs" / "kaggle" / "cure-bench"
    assert zipfile.is_zipfile(target / "cure-bench.zip")
    assert not (target.parent / ".cure-bench.download").exists()


def test_kaggle_download_timeout_preserves_only_resumable_staging_and_next_call_promotes_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KAGGLE_API_TOKEN", "fixture-token")
    monkeypatch.setenv("EVOMIND_KAGGLE_DOWNLOAD_RETRY_BASE_SECONDS", "0")
    monkeypatch.setattr("xsci.config.load_config", lambda _root: SimpleNamespace())
    monkeypatch.setattr("xsci.config.inject_engine_env", lambda *_args, **_kwargs: [])

    def timeout_run(argv, **_kwargs):
        destination = Path(argv[-1])
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "porto-seguro-safe-driver-prediction.zip").write_bytes(b"partial")
        (destination / "porto-seguro-safe-driver-prediction.zip.kaggle-partial").write_text("fixture", encoding="utf-8")
        raise tools.subprocess.TimeoutExpired(argv, 300)

    monkeypatch.setattr(tools.subprocess, "run", timeout_run)
    context = _context(tmp_path)

    result = tools._kaggle_download(
        {"competition": "porto-seguro-safe-driver-prediction"},
        context,
    )

    assert result.ok is False
    assert result.error == "TimeoutError"
    target = context.workspace_root / "inputs" / "kaggle" / "porto-seguro-safe-driver-prediction"
    assert not list(target.glob("*.part"))
    assert not list(target.glob("*.zip"))
    staging = target.parent / ".porto-seguro-safe-driver-prediction.download"
    assert (staging / "porto-seguro-safe-driver-prediction.zip").read_bytes() == b"partial"

    def resume_run(argv, **_kwargs):
        destination = Path(argv[-1])
        assert (destination / "porto-seguro-safe-driver-prediction.zip").read_bytes() == b"partial"
        (destination / "porto-seguro-safe-driver-prediction.zip").write_bytes(_zip_payload())
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(tools.subprocess, "run", resume_run)
    resumed = tools._kaggle_download(
        {"competition": "porto-seguro-safe-driver-prediction"},
        context,
    )

    assert resumed.ok is True
    assert resumed.content["download_mode"] == "bounded_atomic"
    assert resumed.content["resume_supported"] is True
    assert not staging.exists()
    assert zipfile.is_zipfile(target / "porto-seguro-safe-driver-prediction.zip")
