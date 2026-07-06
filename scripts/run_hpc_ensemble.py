# -*- coding: utf-8 -*-
"""Launch EXP002 ensemble on HPC over already-computed remote OOF. Human gate stays on."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import importlib.util
ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("_hpc_single", ROOT / "scripts" / "run_hpc_kaggle_single_model.py")
_H = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_H)
BASE = "/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/workstation_fulldata"
DATA = "/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_raw_data"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task-id", required=True); p.add_argument("--data-subdir", required=True)
    p.add_argument("--fmt", required=True); p.add_argument("--id-col", required=True)
    p.add_argument("--targets", required=True); p.add_argument("--members", required=True)
    p.add_argument("--user", required=True); p.add_argument("--host", default="100.85.169.63")
    p.add_argument("--port", type=int, default=1235); p.add_argument("--proxy-host", default="127.0.0.1")
    p.add_argument("--proxy-port", type=int, default=7890); p.add_argument("--password-env", default="GPU_SSH_PASSWORD")
    p.add_argument("--timeout-seconds", type=int, default=300)
    p.add_argument("--runner-file", default="ensemble_runner_v2.py")
    p.add_argument("--metric", default="")
    a = p.parse_args()
    rdir = f"{BASE}/{a.task_id}/EXP002_ensemble"
    local = ROOT / "workspace" / "workstation_runs" / a.task_id / "EXP002_ensemble"
    local.mkdir(parents=True, exist_ok=True)
    member_dirs = ",".join(f"{BASE}/{a.task_id}/{m}" for m in a.members.split(","))
    c = _H.connect(a); sftp = c.open_sftp()
    try:
        _H.sftp_mkdirs(sftp, f"{rdir}/outputs")
        _H.upload_file(sftp, ROOT / "workspace" / "hpc_fulldata" / a.runner_file, f"{rdir}/{a.runner_file}")
    finally:
        sftp.close(); c.close()
    args_str = (f"--fmt {a.fmt} --metric '{a.metric}' --members '{member_dirs}' --data-dir '{DATA}/{a.data_subdir}' "
                f"--id-col {a.id_col} --targets '{a.targets}' --out outputs")
    cmd = (f"cd '{rdir}' && PY=$(command -v python3 || command -v python) && "
           f"\"$PY\" {a.runner_file} {args_str} > outputs/run.log 2>&1; echo EXIT $?")
    c = _H.connect(a)
    try:
        _, so, se = c.exec_command(cmd, timeout=a.timeout_seconds)
        so.channel.recv_exit_status(); out_txt = so.read().decode("utf-8", "replace")
    finally:
        c.close()
    got = _H.download_outputs_with_retries(a, rdir, local, ["score_promotion_gate.json", "submission.csv", "run.log"])
    gp = local / "score_promotion_gate.json"
    gate = json.loads(gp.read_text(encoding="utf-8")) if gp.is_file() and gp.stat().st_size else None
    (local / "manifest.json").write_text(json.dumps({"schema": "workstation.hpc_ensemble_manifest.v1",
        "task_id": a.task_id, "remote_dir": rdir, "downloaded": got, "gate": gate,
        "human_gate_required_for_official_submission": True}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"event": "ensemble_launch_done", "downloaded": got, "gate": gate}, ensure_ascii=False))


if __name__ == "__main__":
    main()
