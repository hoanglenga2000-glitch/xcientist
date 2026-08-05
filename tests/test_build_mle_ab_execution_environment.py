from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from research_os.mle_ab_campaign import SCREEN_TASKS, build_config, preregister, sha256_file
from scripts.verify_mle_ab_campaign_readiness import _read_execution_environment


def _load_builder():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_mle_ab_execution_environment.py"
    spec = importlib.util.spec_from_file_location("build_mle_ab_execution_environment", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _campaign(tmp_path: Path) -> tuple[Path, dict]:
    canonical = tmp_path / "canonical.json"
    canonical.write_text('{"schema":"test"}\n', encoding="utf-8")
    source = Path(__file__).resolve().parents[1] / "src" / "research_os" / "experience_mcgs.py"
    config = build_config(
        namespace="evomind-mle-screen-env-test",
        phase="screen",
        tasks=SCREEN_TASKS,
        provider="local-model-gateway",
        source_hashes={"src/research_os/experience_mcgs.py": sha256_file(source)},
        canonical_manifest_sha256=sha256_file(canonical),
        created_at="2026-08-02T00:00:00+00:00",
    )
    campaign = Path(preregister(tmp_path / "campaigns", config)["campaign_dir"])
    prereg = json.loads((campaign / "preregistration.json").read_text(encoding="utf-8"))
    return campaign, prereg


def test_execution_environment_builder_binds_gateway_without_completion(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("EVOMIND_LOCAL_GATEWAY_CONFIG", raising=False)
    builder = _load_builder()
    campaign, prereg = _campaign(tmp_path)
    config = tmp_path / "gateway.json"
    config.write_text(json.dumps({"api-keys": ["secret-token"]}), encoding="utf-8")
    captured: dict[str, object] = {}

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _limit=-1):
            return b'{"object":"list","data":[]}'

    def fake_urlopen(request, *, timeout):
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["headers"] = dict(request.header_items())
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(builder.urllib.request, "urlopen", fake_urlopen)

    payload = builder.build_execution_environment(
        campaign,
        base_url="http://127.0.0.1:65068/v1",
        gateway_config=config,
    )
    output = builder.write_json_atomic(tmp_path / "execution-environment.json", payload)
    verified = _read_execution_environment(
        output,
        prereg,
        campaign_lock_sha256=payload["campaign_lock_sha256"],
    )

    assert captured["url"] == "http://127.0.0.1:65068/v1/models"
    assert captured["method"] == "GET"
    assert "Bearer secret-token" in captured["headers"].values()
    assert "chat/completions" not in json.dumps(captured)
    assert payload["completion_requested"] is False
    assert payload["task_runs_started"] == 0
    assert payload["grader_calls"] == 0
    assert payload["kaggle_submissions"] == 0
    assert payload["authenticated_no_completion_probe"]["authentication_succeeded"] is True
    assert payload["same_hardware_policy"]["receipts_required_for_every_run"] is True
    assert "secret-token" not in json.dumps(payload)
    assert verified["valid"] is True


def test_execution_environment_fails_closed_on_auth_probe_failure(monkeypatch, tmp_path):
    builder = _load_builder()
    campaign, _prereg = _campaign(tmp_path)
    config = tmp_path / "gateway.json"
    config.write_text(json.dumps({"api-keys": ["secret-token"]}), encoding="utf-8")

    class FakeError(builder.urllib.error.HTTPError):
        def read(self, _limit=-1):
            return b'{"error":"unauthorized"}'

    def fake_urlopen(request, *, timeout):
        del request, timeout
        raise FakeError("http://127.0.0.1:65068/v1/models", 401, "unauthorized", {}, None)

    monkeypatch.setattr(builder.urllib.request, "urlopen", fake_urlopen)

    payload = builder.build_execution_environment(campaign, gateway_config=config)

    assert payload["status"] == "failed_closed"
    assert payload["authenticated_no_completion_probe"]["http_status"] == 401
    assert payload["authenticated_no_completion_probe"]["completion_requested"] is False
