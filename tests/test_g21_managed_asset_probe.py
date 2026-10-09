from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/g21_managed_asset_probe.py"
SPEC = importlib.util.spec_from_file_location("g21_managed_asset_probe_fixture", MODULE_PATH)
assert SPEC and SPEC.loader
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def _write_package(site: Path, name: str, version: str) -> None:
    package = site / name
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("# fixture\n", encoding="utf-8")
    dist = site / f"{name}-{version}.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    allowed = tmp_path / "allowed"
    data = allowed / "competition-data/mindgames"
    data.mkdir(parents=True)
    runtime = data.parent / ".runtime/mindgames"
    site = runtime / "lib/python3.12/site-packages"
    site.mkdir(parents=True)
    _write_package(site, "textarena", "0.7.4")
    _write_package(site, "trueskill", "0.4.5")
    for model in ("Qwen3-8B", "STARS", "tungsten"):
        root = runtime / "models" / model
        root.mkdir(parents=True)
        (root / "config.json").write_text("{}\n", encoding="utf-8")
        (root / "weights.bin").write_bytes(model.encode("utf-8"))
    (runtime / "formal-protocol.json").write_text("{}\n", encoding="utf-8")
    cure = data / ".evomind/cure-bench"
    cure.mkdir(parents=True)
    (cure / "frozen-encoder.json").write_text("{}\n", encoding="utf-8")
    (cure / "holdout-ledger.json").write_text("{}\n", encoding="utf-8")
    return allowed, data, allowed / "outputs/probe"


def test_probe_reports_only_bounded_metadata_without_file_contents(tmp_path: Path) -> None:
    allowed, data, output = _fixture(tmp_path)

    result = probe.probe(data, output, allowed)

    packages = {row["name"]: row for row in result["packages"]}
    assert packages["textarena"]["available"] is True
    assert packages["textarena"]["version"] == "0.7.4"
    assert packages["trueskill"]["version"] == "0.4.5"
    assert packages["transformers"]["available"] is False
    assert {Path(row["path"]).name for row in result["models"]} == {"Qwen3-8B", "STARS", "tungsten"}
    assert all(row["content_sha256_computed"] is False for row in result["models"])
    assert all(file["content_read"] is False for row in result["models"] for file in row["files"])
    assert len(result["manifest_candidates"]["mindgames_formal"]) == 1
    assert len(result["manifest_candidates"]["cure_encoder"]) == 1
    assert len(result["manifest_candidates"]["holdout_ledger"]) == 1
    assert result["network_access"] is False
    assert result["test_labels_used"] is False
    assert result["signals_sent"] == 0
    assert result["other_processes_modified"] is False


def test_cli_accepts_hpc_execute_solution_data_and_output_flags(tmp_path: Path) -> None:
    allowed, data, output = _fixture(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            "--data-dir",
            str(data),
            "--out-dir",
            str(output),
            "--allowed-remote-root",
            str(allowed),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    result = json.loads((output / "managed-asset-probe.json").read_text(encoding="utf-8"))
    assert result["status"] == "probe_completed"
    assert result["file_contents_read"] is False


def test_path_escape_and_nonempty_output_fail_closed(tmp_path: Path) -> None:
    allowed, data, output = _fixture(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(probe.ProbeGate, match="PATH_ESCAPE"):
        probe.probe(outside, output, allowed)
    output.mkdir(parents=True)
    (output / "existing.txt").write_text("preserve\n", encoding="utf-8")
    with pytest.raises(probe.ProbeGate, match="OUTPUT_DIRECTORY_NOT_EMPTY"):
        probe.probe(data, output, allowed)


def test_symlink_or_reparse_asset_is_rejected_when_supported(tmp_path: Path) -> None:
    allowed, data, output = _fixture(tmp_path)
    model = data.parent / ".runtime/mindgames/models/Qwen3-8B"
    target = model / "config.json"
    target.unlink()
    outside = tmp_path / "outside-config.json"
    outside.write_text("{}\n", encoding="utf-8")
    try:
        target.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable on this host")
    with pytest.raises(probe.ProbeGate, match="REPARSE_PATH_REJECTED"):
        probe.probe(data, output, allowed)


def test_source_has_no_network_process_signal_delete_or_content_read_surface() -> None:
    text = MODULE_PATH.read_text(encoding="utf-8")
    for forbidden in (
        "subprocess",
        "socket",
        "urllib",
        "requests",
        "os.kill",
        "signal.",
        "unlink(",
        "rmtree",
        "remove(",
        "read_text(",
        "read_bytes(",
        "open(\"rb\"",
        "open('rb'",
    ):
        assert forbidden not in text
