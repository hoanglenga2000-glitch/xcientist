from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import verify_siim_job90353_goal as verifier
from scripts.verify_siim_job90353_goal import RUN_ID, evaluate, rank_roc_auc

ROOT = Path(__file__).resolve().parents[1]


def test_rank_roc_auc_is_tie_aware() -> None:
    assert rank_roc_auc([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8]) == 1.0
    assert rank_roc_auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5


def test_ui_check_exchanges_one_time_fragment_without_exposing_token(monkeypatch, tmp_path) -> None:
    token = "test-bootstrap-token-with-at-least-24-characters"
    bootstrap = tmp_path / "dashboard.bootstrap.once"
    bootstrap.write_text(
        f"http://127.0.0.1:8088/?page=assistant#bootstrap={token}",
        encoding="utf-8",
    )
    requests: list[object] = []

    class FakeResponse:
        def __init__(self, status: int, payload: dict) -> None:
            self.status = status
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self) -> bytes:
            return json.dumps(self.payload).encode("utf-8")

    class FakeOpener:
        def open(self, request, timeout):
            assert timeout == 4.0
            requests.append(request)
            if not isinstance(request, str):
                assert request.full_url.endswith("/api/session/bootstrap")
                assert json.loads(request.data) == {"token": token}
                return FakeResponse(200, {"ok": True, "csrf_token": "csrf-not-printed"})
            return FakeResponse(200, {"ok": True, "run": {"run_id": RUN_ID, "status": "completed"}})

    monkeypatch.setattr(verifier.urllib.request, "build_opener", lambda *_handlers: FakeOpener())

    result = verifier._check_ui(
        RUN_ID,
        "http://127.0.0.1:8088",
        bootstrap_url_file=bootstrap,
    )

    assert result["passed"] is True
    assert result["authentication"] == "one_time_fragment_exchange"
    assert not bootstrap.exists()
    assert token not in json.dumps(result)


def test_job90353_goal_verifier_covers_local_artifacts_when_present() -> None:
    run_dir = ROOT / "workspace" / "evomind_runs" / RUN_ID
    video = (
        ROOT
        / "video-production"
        / "siim-isic-melanoma-commercial-v2-user-journey"
        / "final"
        / "evomind-siim-isic-medical-research-user-journey-zh-92s.mp4"
    )
    if not run_dir.is_dir() or not video.is_file():
        pytest.skip("job90353 Run/video acceptance artifacts are not present in this checkout")

    report = evaluate(ROOT, require_ui=False)

    assert report["schema"] == "evomind.siim.job90353_goal_verification.v1"
    assert report["run_id"] == RUN_ID
    assert report["achievable_requirements_status"] == "passed"
    assert report["checks_passed"] == report["checks_total"]
    assert report["medal_outcome"]["status"] == "not_established"
    assert report["goal_complete"] is False
    assert report["goal_status_recommendation"] == "blocked"
