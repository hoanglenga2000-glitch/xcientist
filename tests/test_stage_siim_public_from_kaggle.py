from __future__ import annotations

import importlib.util
import json
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "stage_siim_public_from_kaggle.py"
SPEC = importlib.util.spec_from_file_location("stage_siim_public_from_kaggle", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_safe_target_accepts_only_frozen_public_shapes(tmp_path: Path):
    assert MODULE.safe_target(tmp_path, "train.csv") == tmp_path / "train.csv"
    assert (
        MODULE.safe_target(tmp_path, "jpeg/train/ISIC_0000001.jpg")
        == tmp_path / "jpeg" / "train" / "ISIC_0000001.jpg"
    )
    with pytest.raises(ValueError):
        MODULE.safe_target(tmp_path, "../private.csv")
    with pytest.raises(ValueError):
        MODULE.safe_target(tmp_path, "private/labels.csv")


def test_inventory_and_scan_require_exact_sizes(tmp_path: Path):
    destination = tmp_path / "public"
    destination.mkdir()
    entries = [
        {"path": "train.csv", "size": 3},
        {"path": "jpeg/test/ISIC_1.jpg", "size": 4},
    ]
    inventory = {
        "schema": "evomind.siim.public_staging_inventory.v1",
        "competition_id": MODULE.COMPETITION,
        "file_count": 2,
        "total_bytes": 7,
        "manifest_sha256": "fixture",
        "entries": entries,
    }
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    loaded, parsed = MODULE.load_inventory(inventory_path, destination)
    assert loaded["file_count"] == 2
    (destination / "train.csv").write_bytes(b"abc")
    image = destination / "jpeg" / "test" / "ISIC_1.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"xx")
    scan = MODULE.scan_entries(destination, parsed)
    assert scan["completed_files"] == 1
    assert scan["completed_bytes"] == 3
    assert [entry.name for entry in scan["missing"]] == ["jpeg/test/ISIC_1.jpg"]
    assert scan["wrong_size"][0]["actual_bytes"] == 2


def test_extract_kaggle_single_file_archive(tmp_path: Path):
    target = tmp_path / "jpeg" / "train" / "ISIC_1.jpg"
    target.parent.mkdir(parents=True)
    archive = target.with_name(target.name + ".zip")
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("ISIC_1.jpg", b"abcd")
    MODULE.extract_download_archive(archive, target, 4)
    assert target.read_bytes() == b"abcd"


def test_bounded_kaggle_download_sets_timeout_and_streams_signed_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    class Response:
        def __init__(self, status_code, headers=None, chunks=()):
            self.status_code = status_code
            self.headers = headers or {}
            self.chunks = chunks

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        @staticmethod
        def raise_for_status():
            return None

        def iter_content(self, chunk_size):
            assert chunk_size == 1024 * 1024
            return iter(self.chunks)

    requests = []
    def fake_get(url, **kwargs):
        requests.append((url, kwargs))
        if len(requests) == 1:
            return Response(
                302,
                {"Location": "https://files.test/ISIC_1.jpg.zip?sig=x"},
            )
        return Response(200, chunks=(b"zip-payload",))

    monkeypatch.setattr(MODULE.requests, "get", fake_get)
    monkeypatch.setenv("KAGGLE_API_TOKEN", "fixture-token")

    output = MODULE.competition_download_file_bounded(
        MODULE.COMPETITION,
        "jpeg/train/ISIC_1.jpg",
        tmp_path,
        force=False,
        quiet=True,
        request_timeout_seconds=90,
        proxy_url="http://127.0.0.1:7892",
    )
    assert output == tmp_path / "ISIC_1.jpg.zip"
    assert output.read_bytes() == b"zip-payload"
    assert len(requests) == 2
    assert requests[0][0].endswith("jpeg%2Ftrain%2FISIC_1.jpg")
    assert requests[0][1]["allow_redirects"] is False
    assert requests[1][1]["timeout"] == (30, 90)
    assert requests[1][1]["proxies"] == {
        "http": "http://127.0.0.1:7892",
        "https": "http://127.0.0.1:7892",
    }


def test_bounded_kaggle_download_surfaces_rate_limit_without_signed_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    class Response:
        status_code = 429
        headers = {"Retry-After": "120"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setenv("KAGGLE_API_TOKEN", "fixture-token")
    monkeypatch.setattr(MODULE.requests, "get", lambda *_args, **_kwargs: Response())
    with pytest.raises(MODULE.KaggleRateLimitError) as raised:
        MODULE.competition_download_file_bounded(
            MODULE.COMPETITION,
            "train.csv",
            tmp_path,
            force=False,
            quiet=True,
            request_timeout_seconds=90,
            proxy_url="http://127.0.0.1:7892",
        )
    assert raised.value.retry_after_seconds == 120


def test_bounded_future_results_caps_outstanding_submissions():
    entries = [MODULE.Entry(f"jpeg/train/ISIC_{index}.jpg", index) for index in range(12)]
    lock = threading.Lock()
    state = {"outstanding": 0, "maximum": 0}

    def submit(pool, entry):
        with lock:
            state["outstanding"] += 1
            state["maximum"] = max(state["maximum"], state["outstanding"])

        def work():
            time.sleep(0.01)
            return entry.name

        future = pool.submit(work)

        def finished(_future):
            with lock:
                state["outstanding"] -= 1

        future.add_done_callback(finished)
        return future

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = [
            future.result()
            for future, _entry in MODULE.bounded_future_results(
                executor,
                entries,
                max_in_flight=3,
                submit=submit,
            )
        ]
    assert sorted(results) == sorted(entry.name for entry in entries)
    assert state["maximum"] <= 3
