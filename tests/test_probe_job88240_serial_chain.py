from __future__ import annotations

import json
from pathlib import Path

from scripts import probe_job88240_serial_chain as probe


def test_remote_command_is_read_only_and_confined() -> None:
    command = probe.build_remote_command()
    assert "nvidia-smi" not in command  # source is base64 encoded
    assert "rm " not in command
    assert "kill " not in command
    assert all(path.startswith(probe.REMOTE_ROOT + "/") for path in probe.STATUS_PATHS)


def test_probe_writes_normalized_live_evidence(tmp_path: Path, monkeypatch) -> None:
    class Client:
        closed = False

        def close(self) -> None:
            self.closed = True

    client = Client()
    remote = {
        "statuses": [{"path": probe.STATUS_PATHS[0], "exists": True}],
        "gpu_exit_code": 0,
        "gpu_stdout": "0, NVIDIA A40, 13021, 100",
        "gpu_stderr": "",
        "apps_exit_code": 0,
        "apps_stdout": "123, python, 13012",
        "apps_stderr": "",
    }
    monkeypatch.setattr(probe.ops, "_connect", lambda: client)
    monkeypatch.setattr(
        probe.ops,
        "_run_remote",
        lambda *args, **kwargs: (0, json.dumps(remote), ""),
    )
    monkeypatch.setattr(
        probe,
        "inspect_local_successors",
        lambda *args, **kwargs: {"passed": True, "successors": []},
    )
    output = tmp_path / "serial.json"

    report = probe.probe(output)

    assert client.closed is True
    assert report["schema"] == "evomind.hpc88240.serial_chain_readonly_probe.v3"
    assert report["local_successor_chain"]["passed"] is True
    assert report["process_signals_sent"] == 0
    assert report["other_processes_modified"] is False
    assert json.loads(output.read_text(encoding="utf-8"))["gpu_exit_code"] == 0


def test_local_successor_inspection_requires_all_six_live_workers(
    tmp_path: Path, monkeypatch
) -> None:
    manifests = []
    for index in range(6):
        evidence = tmp_path / f"evidence-{index}"
        evidence.mkdir()
        (evidence / "status_current.json").write_text(
            json.dumps({"schema": "status.v1", "status": "waiting"}),
            encoding="utf-8",
        )
        launcher = tmp_path / f"launcher-{index}.json"
        launcher.write_text(
            json.dumps(
                {
                    "status": "running",
                    "wrapper_pid": 100 + index,
                    "worker_pid": 200 + index,
                }
            ),
            encoding="utf-8",
        )
        manifest = tmp_path / f"manifest-{index}.json"
        manifest.write_text(
            json.dumps(
                {
                    "task_id": f"task-{index}",
                    "arguments": ["script.py", "--evidence-dir", str(evidence)],
                }
            ),
            encoding="utf-8",
        )
        manifests.append(
            {
                "name": f"successor-{index}",
                "manifest": str(manifest),
                "launcher_status": str(launcher),
            }
        )
    index_path = tmp_path / "index.json"
    index_path.write_text(
        json.dumps(
            {
                "schema": "evomind.job88240.final_successor_persistent_chain.v1",
                "manifests": manifests,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        probe,
        "_process_record",
        lambda pid: {"pid": int(pid), "exists": True},
    )

    report = probe.inspect_local_successors(index_path)

    assert report["passed"] is True
    assert report["observed_successor_count"] == 6
    assert report["all_workers_live"] is True
