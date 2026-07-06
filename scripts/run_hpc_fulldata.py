# -*- coding: utf-8 -*-
"""Workstation full-data HPC driver. Reuses tested SSH/SFTP helpers.
Writes only under gpu_tra remote root. Official Kaggle stays behind human gate.
Modes: sync (upload+run+wait+download) | async (upload+nohup) | poll (download)."""
from __future__ import annotations
import argparse, json, time, socket
from datetime import datetime
from pathlib import Path
import paramiko
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("_hpc_single", ROOT / "scripts" / "run_hpc_kaggle_single_model.py")
_H = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_H)

REMOTE_ROOT = "/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/workstation_fulldata"
DATA_ROOT = "/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_raw_data"


def connect(a):
    return _H.connect(a)


def run_remote(a, cmd, timeout):
    c = connect(a)
    try:
        _, so, se = c.exec_command(cmd, timeout=timeout)
        code = so.channel.recv_exit_status()
        return code, so.read().decode("utf-8", "replace"), se.read().decode("utf-8", "replace")
    finally:
        c.close()
# __DRIVER_MAIN__


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task-id", required=True)
    p.add_argument("--data-subdir", required=True)
    p.add_argument("--exp-id", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--task-type", required=True)
    p.add_argument("--submission-format", required=True)
    p.add_argument("--metric", required=True)
    p.add_argument("--targets", required=True)
    p.add_argument("--id-col", required=True)
    p.add_argument("--drop-cols", default="")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seeds", default="42")
    p.add_argument("--sample-rows", type=int, default=0)
    p.add_argument("--trainer-file", default="generic_trainer_v2.py")
    p.add_argument("--mode", choices=["sync", "async", "poll"], default="sync")
    p.add_argument("--timeout-seconds", type=int, default=560)
    p.add_argument("--host", default="100.85.169.63")
    p.add_argument("--port", type=int, default=1235)
    p.add_argument("--user", required=True)
    p.add_argument("--proxy-host", default="127.0.0.1")
    p.add_argument("--proxy-port", type=int, default=7890)
    p.add_argument("--password-env", default="GPU_SSH_PASSWORD")
    a = p.parse_args()

    rdir = f"{REMOTE_ROOT}/{a.task_id}/{a.exp_id}_{a.model}"
    local = ROOT / "workspace" / "workstation_runs" / a.task_id / f"{a.exp_id}_{a.model}"
    local.mkdir(parents=True, exist_ok=True)

    if a.mode == "poll":
        names = ["metrics.json", "submission.csv", "oof_predictions.npz", "run.log"]
        got = _H.download_outputs_with_retries(a, rdir, local, names)
        mp = local / "metrics.json"
        metrics = json.loads(mp.read_text(encoding="utf-8")) if mp.is_file() and mp.stat().st_size else None
        print(json.dumps({"event": "poll", "downloaded": got, "metrics": metrics}, ensure_ascii=False))
        return

    cfg = {"task_id": a.task_id, "data_dir": f"{DATA_ROOT}/{a.data_subdir}",
           "id_col": a.id_col, "task_type": a.task_type, "submission_format": a.submission_format,
           "metric": a.metric, "targets": [t for t in a.targets.split(",") if t],
           "drop_cols": [c for c in a.drop_cols.split(",") if c]}
    cfg_local = local / "task_config.json"
    cfg_local.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    c = connect(a); sftp = c.open_sftp()
    try:
        _H.sftp_mkdirs(sftp, f"{rdir}/outputs")
        _H.upload_file(sftp, ROOT / "workspace" / "hpc_fulldata" / a.trainer_file, f"{rdir}/{a.trainer_file}")
        _H.upload_file(sftp, cfg_local, f"{rdir}/task_config.json")
    finally:
        sftp.close(); c.close()

    args_str = (f"--config task_config.json --model {a.model} --exp-id {a.exp_id} "
                f"--folds {a.folds} --seeds '{a.seeds}' --sample-rows {a.sample_rows} --out outputs")
    py = "PY=$(command -v python3 || command -v python)"
    if a.mode == "async":
        cmd = (f"cd '{rdir}' && {py} && nohup \"$PY\" {a.trainer_file} {args_str} "
               f"> outputs/run.log 2>&1 & echo STARTED_PID $!")
        code, so, se = run_remote(a, cmd, 60)
        (local / "async_launch.log").write_text(so + "\n" + se, encoding="utf-8")
        print(json.dumps({"event": "async_launched", "task": a.task_id, "exp": a.exp_id,
                          "model": a.model, "remote_dir": rdir, "stdout": so.strip()}, ensure_ascii=False))
        return

    cmd = f"cd '{rdir}' && {py} && \"$PY\" {a.trainer_file} {args_str} > outputs/run.log 2>&1; echo EXIT $?"
    t0 = time.time()
    code, so, se = run_remote(a, cmd, a.timeout_seconds)
    names = ["metrics.json", "submission.csv", "oof_predictions.npz", "run.log"]
    got = _H.download_outputs_with_retries(a, rdir, local, names)
    mp = local / "metrics.json"
    metrics = json.loads(mp.read_text(encoding="utf-8")) if mp.is_file() and mp.stat().st_size else None
    manifest = {"schema": "workstation.hpc_fulldata_manifest.v1",
                "status": "passed" if (got.get("metrics.json") and got.get("submission.csv")) else "failed",
                "task_id": a.task_id, "exp_id": a.exp_id, "model": a.model, "remote_dir": rdir,
                "local_dir": str(local.relative_to(ROOT)).replace("\\", "/"),
                "seconds": round(time.time() - t0, 2), "downloaded": got, "metrics": metrics,
                "stdout_tail": so[-1500:], "human_gate_required_for_official_submission": True}
    (local / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"event": "sync_done", "status": manifest["status"],
                      "primary_metric": (metrics or {}).get("primary_metric"),
                      "primary_score": (metrics or {}).get("primary_score"),
                      "seconds": manifest["seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
