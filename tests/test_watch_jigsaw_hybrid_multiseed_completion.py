from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.watch_jigsaw_hybrid_multiseed_completion import (
    build_aggregation_plan,
    download_remote_file,
    file_record,
    source_stable_key,
)


class _LocalSftp:
    def open(self, path: str, mode: str):
        return Path(path).open(mode)


def _source_record(tmp_path: Path, name: str) -> dict:
    path = tmp_path / name
    path.write_text(name, encoding="utf-8")
    return file_record(path)


def test_source_stability_requires_every_seed() -> None:
    snapshots = [
        {
            "ready": True,
            "run_id": f"run-{seed}",
            **{
                name: {"bytes": seed, "sha256": f"{seed}-{name}"}
                for name in (
                    "run_plan",
                    "summary",
                    "independent_verification",
                    "prediction_bundle",
                )
            },
        }
        for seed in (40, 41, 42)
    ]
    assert source_stable_key(snapshots) is not None
    snapshots[1]["ready"] = False
    assert source_stable_key(snapshots) is None


def test_resume_safe_download_verifies_hash(tmp_path: Path) -> None:
    remote_path = tmp_path / "remote.bin"
    payload = b"0123456789" * 1000
    remote_path.write_bytes(payload)
    remote = {
        "path": str(remote_path),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    local_path = tmp_path / "download" / "local.bin"
    partial = local_path.with_suffix(local_path.suffix + ".part")
    partial.parent.mkdir(parents=True)
    partial.write_bytes(payload[:123])

    record = download_remote_file(
        _LocalSftp(),
        remote,
        local_path,
        progress_path=tmp_path / "progress.json",
    )
    assert local_path.read_bytes() == payload
    assert record["sha256"] == remote["sha256"]
    assert not partial.exists()


def test_build_aggregation_plan_keeps_mixed_topology(tmp_path: Path) -> None:
    public = {}
    for name in ("public_train", "public_test", "public_sample"):
        public[name] = _source_record(tmp_path, f"{name}.csv")
    control_path = tmp_path / "control.json"
    control = {
        "schema": "evomind.jigsaw.confirmation_queue_frozen_plan.v1",
        "inputs": public,
        "confirmation_gate": {
            "minimum_seed_auc": 0.987,
            "minimum_mean_auc": 0.987,
            "minimum_seed_gain": 0.0003,
            "maximum_population_std": 0.0025,
        },
    }
    control_path.write_text(json.dumps(control), encoding="utf-8")
    evidence = tmp_path / "source_evidence.json"
    evidence.write_text("{}", encoding="utf-8")
    seeds = []
    for seed in (40, 41, 42):
        seeds.append(
            {
                "model_seed": seed,
                "run_id": f"run-{seed}",
                "run_plan": _source_record(tmp_path, f"seed{seed}_plan.json"),
                "summary": _source_record(tmp_path, f"seed{seed}_summary.json"),
                "independent_verification": _source_record(
                    tmp_path, f"seed{seed}_verification.json"
                ),
                "prediction_bundle": _source_record(
                    tmp_path, f"seed{seed}_bundle.npz"
                ),
            }
        )

    plan_path = tmp_path / "aggregation_plan.json"
    plan = build_aggregation_plan(
        control_plan_path=control_path,
        control_plan=control,
        seed40=seeds[0],
        seed41=seeds[1],
        seed42=seeds[2],
        source_evidence_path=evidence,
        aggregation_plan_path=plan_path,
        output_dir=tmp_path / "output",
    )
    assert [record["model_seed"] for record in plan["seed_runs"]] == [40, 41, 42]
    assert plan["driver"]["execution_topology"] == "local_seed40+hpc_seed41+local_seed42"
    assert plan["boundaries"]["process_signals_sent"] == 0
    assert plan_path.is_file()
