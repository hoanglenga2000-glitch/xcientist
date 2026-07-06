from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import paramiko

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_hpc_exp003_lightgbm_cv import (  # noqa: E402
    connect,
    download_tree,
    remote_file_exists,
    sftp_mkdirs,
    upload_file,
)


def close_sftp_quietly(sftp: paramiko.SFTPClient | None) -> None:
    if sftp is None:
        return
    try:
        sftp.close()
    except Exception:
        pass


def download_outputs_with_retries(
    args: argparse.Namespace,
    remote_dir: str,
    local_artifact_dir: Path,
    attempts: int = 3,
) -> list[str]:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        download_client: paramiko.SSHClient | None = None
        sftp: paramiko.SFTPClient | None = None
        try:
            download_client = connect(args)
            sftp = download_client.open_sftp()
            return download_tree(sftp, f"{remote_dir}/outputs", local_artifact_dir)
        except FileNotFoundError:
            return []
        except (OSError, EOFError, paramiko.SSHException, socket.error) as exc:
            last_error = exc
            time.sleep(min(2 * attempt, 6))
        finally:
            close_sftp_quietly(sftp)
            if download_client is not None:
                download_client.close()
    if last_error is not None:
        raise RuntimeError(f"Failed to download EXP016 outputs after {attempts} attempts: {last_error}") from last_error
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EXP016 LightGBM Optuna search on the HPC SSH path.")
    parser.add_argument("--host", default="100.85.169.63")
    parser.add_argument("--port", type=int, default=1235)
    parser.add_argument("--user", required=True)
    parser.add_argument("--proxy-host", default="127.0.0.1")
    parser.add_argument("--proxy-port", type=int, default=7890)
    parser.add_argument("--password-env", default="GPU_SSH_PASSWORD")
    parser.add_argument("--remote-root", default="/hpc2ssd/JH_DATA/spooler/aimslab/research_agent_workstation")
    parser.add_argument("--python-executable", default="/hpc2ssd/JH_DATA/spooler/aimslab/research_agent_workstation/pyenvs/tabular_s6e6_lightgbm/bin/python")
    parser.add_argument("--timeout-seconds", type=int, default=28800)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=260612)
    parser.add_argument("--sample-rows", type=int, default=0)
    parser.add_argument("--n-estimators", type=int, default=2500)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "dryrun" if args.sample_rows else "full"
    remote_dir = f"{args.remote_root.rstrip('/')}/playground_series_s6e6/EXP016_lgbm_optuna_{mode}_{run_id}"
    local_artifact_dir = ROOT / "workspace" / "hpc_experiments" / "playground_series_s6e6" / f"EXP016_lgbm_optuna_{mode}_{run_id}"
    local_artifact_dir.mkdir(parents=True, exist_ok=True)

    client = connect(args)
    sftp = client.open_sftp()
    try:
        sftp_mkdirs(sftp, f"{remote_dir}/data")
        data_names = ["train.csv", "test.csv", "sample_submission.csv"]
        remote_data_candidates = [
            f"{args.remote_root.rstrip('/')}/playground_series_s6e6/20260614_183531/data",
            "/hpc2ssd/JH_DATA/spooler/aimslab/research_agent_workstation/playground_series_s6e6/20260614_183531/data",
            "/hpc2hdd/home/aimslab/playground_series_s6e6/20260614_183531/data",
        ]
        remote_existing_data = next(
            (
                candidate
                for candidate in remote_data_candidates
                if all(remote_file_exists(sftp, f"{candidate}/{name}") for name in data_names)
            ),
            "",
        )
        if remote_existing_data:
            copy_command = " && ".join(
                [f"cp '{remote_existing_data}/{name}' '{remote_dir}/data/{name}'" for name in data_names]
            )
            _, stdout, stderr = client.exec_command(copy_command, timeout=600)
            exit_status = stdout.channel.recv_exit_status()
            if exit_status != 0:
                raise RuntimeError(stderr.read().decode("utf-8", "replace"))
        else:
            for name in data_names:
                upload_file(sftp, ROOT / "tasks" / "playground_series_s6e6" / "data" / name, f"{remote_dir}/data/{name}")

        upload_file(sftp, ROOT / "notebooks_or_scripts" / "exp003_lightgbm_cv.py", f"{remote_dir}/exp003_lightgbm_cv.py")
        upload_file(sftp, ROOT / "notebooks_or_scripts" / "exp016_lgbm_optuna_search.py", f"{remote_dir}/exp016_lgbm_optuna_search.py")
        command = (
            f"cd '{remote_dir}' && '{args.python_executable}' exp016_lgbm_optuna_search.py "
            f"--data-dir data --out-dir outputs --trials {args.trials} --folds {args.folds} "
            f"--seed {args.seed} --sample-rows {args.sample_rows} --n-estimators {args.n_estimators} "
            f"--early-stopping-rounds {args.early_stopping_rounds} --n-jobs {args.n_jobs}"
        )
        started = time.time()
        _, stdout, stderr = client.exec_command(command, timeout=args.timeout_seconds)
        exit_status = stdout.channel.recv_exit_status()
        stdout_text = stdout.read().decode("utf-8", "replace")
        stderr_text = stderr.read().decode("utf-8", "replace")
        (local_artifact_dir / "remote_stdout.log").write_text(stdout_text, encoding="utf-8")
        (local_artifact_dir / "remote_stderr.log").write_text(stderr_text, encoding="utf-8")
        close_sftp_quietly(sftp)
        sftp = None
        downloaded = download_outputs_with_retries(args, remote_dir, local_artifact_dir)
        metrics: dict[str, Any] | None = None
        metrics_path = local_artifact_dir / "metrics.json"
        if metrics_path.is_file():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        manifest = {
            "status": "passed" if exit_status == 0 and metrics_path.is_file() else "failed",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "experiment_id": "EXP016",
            "mode": mode,
            "run_id": run_id,
            "remote_dir": remote_dir,
            "local_artifact_dir": str(local_artifact_dir.relative_to(ROOT)).replace("\\", "/"),
            "exit_status": exit_status,
            "seconds": round(time.time() - started, 3),
            "downloaded": downloaded,
            "metrics": metrics,
            "stdout_tail": stdout_text[-4000:],
            "stderr_tail": stderr_text[-4000:],
            "official_submission_run": False,
        }
        (local_artifact_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        if manifest["status"] != "passed":
            raise SystemExit(1)
    finally:
        close_sftp_quietly(sftp)
        client.close()


if __name__ == "__main__":
    main()
