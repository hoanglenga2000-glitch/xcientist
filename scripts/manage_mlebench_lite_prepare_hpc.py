#!/usr/bin/env python3
"""Start or inspect the resumable official MLE-bench Lite preparation job."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from research_agent_workstation.server.core.gpu_credentials import (  # noqa: E402
    ALLOWED_GPU_REMOTE_ROOT,
    connect_ssh,
    load_gpu_ssh_config,
)

OFFICIAL_COMMIT = "507f92e1138bb6e40dac5c6ee7a6758e6424bf97"
RUNTIME = f"{ALLOWED_GPU_REMOTE_ROOT}/mlebench_lite_runtime"
STATUS_PATH = f"{RUNTIME}/status.json"
LOCAL_STATUS = ROOT / "workspace" / "mlebench_lite_prepare_status.json"
LOCAL_ACCESS = ROOT / "workspace" / "mlebench_lite_access_probe.json"
LITE_COMPETITION_IDS = (
    "aerial-cactus-identification",
    "aptos2019-blindness-detection",
    "denoising-dirty-documents",
    "detecting-insults-in-social-commentary",
    "dog-breed-identification",
    "dogs-vs-cats-redux-kernels-edition",
    "histopathologic-cancer-detection",
    "jigsaw-toxic-comment-classification-challenge",
    "leaf-classification",
    "mlsp-2013-birds",
    "new-york-city-taxi-fare-prediction",
    "nomad2018-predict-transparent-conductors",
    "plant-pathology-2020-fgvc7",
    "random-acts-of-pizza",
    "ranzcr-clip-catheter-line-classification",
    "siim-isic-melanoma-classification",
    "spooky-author-identification",
    "tabular-playground-series-dec-2021",
    "tabular-playground-series-may-2022",
    "text-normalization-challenge-english-language",
    "text-normalization-challenge-russian-language",
    "the-icml-2013-whale-challenge-right-whale-redux",
)


def _decode(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16-le", "utf-16", "cp936", "gb18030"):
        try:
            text = data.decode(encoding)
        except UnicodeError:
            continue
        if "\ufffd" not in text:
            return text.strip()
    return data.decode("utf-8", "replace").strip()


def _load_kaggle_secret() -> dict[str, str]:
    credential_path = Path(os.environ.get("APPDATA", "")) / "ResearchAgentWorkstation" / "kaggle_api_token.xml"
    if not credential_path.is_file():
        raise RuntimeError("Kaggle DPAPI credential is not installed")
    script = (
        "$ErrorActionPreference='Stop';"
        "$c=Import-Clixml -LiteralPath $env:EVOMIND_KAGGLE_CREDENTIAL_PATH;"
        "$s=$c.GetNetworkCredential().Password;"
        "$kind=if($c.UserName -eq '__KAGGLE_API_TOKEN__' -or $s -match '^KGAT_'){'access_token'}else{'legacy'};"
        "@{kind=$kind;username=$c.UserName;secret=$s}|ConvertTo-Json -Compress"
    )
    env = dict(os.environ)
    env["EVOMIND_KAGGLE_CREDENTIAL_PATH"] = str(credential_path)
    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
        check=True,
        capture_output=True,
        env=env,
        timeout=20,
    )
    payload = json.loads(_decode(completed.stdout).lstrip("\ufeff"))
    secret = str(payload.get("secret") or "")
    if not secret:
        raise RuntimeError("Kaggle DPAPI credential is empty")
    if payload.get("kind") == "access_token":
        return {"KAGGLE_API_TOKEN": secret}
    return {
        "KAGGLE_USERNAME": str(payload.get("username") or ""),
        "KAGGLE_KEY": secret,
    }


def _runner_source() -> str:
    return f'''from __future__ import annotations
import fcntl
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

ROOT = pathlib.Path({ALLOWED_GPU_REMOTE_ROOT!r})
RUNTIME = ROOT / "mlebench_lite_runtime"
SOURCE = RUNTIME / "source"
VENV = RUNTIME / "venv-uv"
BIN = RUNTIME / "bin"
TOOLS = RUNTIME / "tools"
DATA = ROOT / "mlebench_official_data"
LOG = RUNTIME / "prepare.log"
STATUS = RUNTIME / "status.json"
LOCK = RUNTIME / "prepare.lock"
COMMIT = {OFFICIAL_COMMIT!r}

RUNTIME.mkdir(parents=True, exist_ok=True)
(RUNTIME / "cache").mkdir(exist_ok=True)
(RUNTIME / "tmp").mkdir(exist_ok=True)
(RUNTIME / "conda_pkgs").mkdir(exist_ok=True)
BIN.mkdir(exist_ok=True)
TOOLS.mkdir(exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)


def now():
    return datetime.now(timezone.utc).isoformat()


def write_status(phase, state="running", **extra):
    payload = {{
        "schema": "evomind.mlebench_lite_prepare.v1",
        "updated_at": now(),
        "pid": os.getpid(),
        "state": state,
        "phase": phase,
        "official_commit": COMMIT,
        "runtime": str(RUNTIME),
        "data_root": str(DATA),
        "log": str(LOG),
        **extra,
    }}
    tmp = STATUS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(STATUS)


env = dict(os.environ)
env.update({{
    "XDG_CACHE_HOME": str(RUNTIME / "cache"),
    "PIP_CACHE_DIR": str(RUNTIME / "cache" / "pip"),
    "CONDA_PKGS_DIRS": str(RUNTIME / "conda_pkgs"),
    "UV_CACHE_DIR": str(RUNTIME / "cache" / "uv"),
    "UV_PYTHON_INSTALL_DIR": str(RUNTIME / "python"),
    "TMPDIR": str(RUNTIME / "tmp"),
    "PYTHONUNBUFFERED": "1",
    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    "PATH": str(BIN) + os.pathsep + str(VENV / "bin") + os.pathsep + env.get("PATH", ""),
}})


def run(command, phase):
    write_status(phase)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"\\n[{{now()}}] PHASE={{phase}}\\n")
        handle.flush()
        completed = subprocess.run(command, cwd=RUNTIME, env=env, stdout=handle, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f"phase {{phase}} failed with exit code {{completed.returncode}}")


def run_with_retries(command, phase, attempts=96):
    """Keep the resumable data preparation alive across transient network failures."""
    for attempt in range(1, attempts + 1):
        attempt_phase = f"{{phase}}_attempt_{{attempt}}"
        write_status(
            attempt_phase,
            retry_attempt=attempt,
            retry_limit=attempts,
        )
        with LOG.open("a", encoding="utf-8") as handle:
            handle.write(f"\\n[{{now()}}] PHASE={{attempt_phase}}\\n")
            handle.flush()
            completed = subprocess.run(
                command,
                cwd=RUNTIME,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        if completed.returncode == 0:
            return
        if attempt == attempts:
            break
        delay = min(300, 30 * attempt)
        write_status(
            f"{{phase}}_retry_wait",
            retry_attempt=attempt,
            retry_limit=attempts,
            last_exit_code=completed.returncode,
            next_retry_seconds=delay,
        )
        with LOG.open("a", encoding="utf-8") as handle:
            handle.write(
                f"[{{now()}}] RETRY phase={{phase}} attempt={{attempt}} "
                f"exit={{completed.returncode}} sleep={{delay}}s\\n"
            )
            handle.flush()
        time.sleep(delay)
    raise RuntimeError(
        f"phase {{phase}} failed after {{attempts}} attempts; "
        f"last exit code {{completed.returncode}}"
    )


with LOCK.open("a+") as lock_handle:
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        write_status("lock", state="blocked", error="another preparation job is already running")
        raise SystemExit(3)
    try:
        write_status("starting", started_at=now())
        python = VENV / "bin" / "python"
        if not python.is_file():
            run([
                "/hpc2hdd/home/aimslab/.local/bin/uv", "venv", "--seed",
                "--python", "3.11", "--python-preference", "only-managed", str(VENV),
            ], "create_python311_environment")
        git_lfs = BIN / "git-lfs"
        if not git_lfs.is_file():
            archive = RUNTIME / "tmp" / "git-lfs-linux-amd64-v3.6.1.tar.gz"
            run([
                "/usr/bin/curl", "-fL", "--retry", "3",
                "https://github.com/git-lfs/git-lfs/releases/download/v3.6.1/git-lfs-linux-amd64-v3.6.1.tar.gz",
                "-o", str(archive),
            ], "download_git_lfs")
            run(["/usr/bin/tar", "-xzf", str(archive), "-C", str(TOOLS)], "extract_git_lfs")
            candidates = list(TOOLS.glob("git-lfs-*/git-lfs"))
            if len(candidates) != 1:
                raise RuntimeError(f"expected one git-lfs binary, found {{len(candidates)}}")
            shutil.copy2(candidates[0], git_lfs)
            git_lfs.chmod(0o755)
        if not (SOURCE / ".git").is_dir():
            run(["git", "clone", "--no-checkout", "https://github.com/openai/mle-bench.git", str(SOURCE)], "clone_official_source")
        run(["git", "-C", str(SOURCE), "lfs", "install", "--local"], "initialize_git_lfs")
        run(["git", "-C", str(SOURCE), "fetch", "origin", COMMIT], "fetch_official_commit")
        run(["git", "-C", str(SOURCE), "checkout", "--force", "--detach", COMMIT], "checkout_official_commit")
        run(["git", "-C", str(SOURCE), "lfs", "pull"], "git_lfs_pull")
        run([
            str(python), "-c",
            "from pathlib import Path; import sys; "
            "p=Path(sys.argv[1]); s=p.read_text(encoding='utf-8'); "
            "old='        api.competitions_list()  # a cheap op that requires authentication'; "
            "new='        # Access-token validity is proven by the actual dataset request.'; "
            "assert old in s or new in s; "
            "p.write_text(s.replace(old, new), encoding='utf-8')",
            str(SOURCE / "mlebench" / "utils.py"),
        ], "install_mlebench_token_network_compatibility")
        run([
            str(python), "-c",
            "from pathlib import Path; import sys; "
            "p=Path(sys.argv[1]); lines=p.read_text(encoding='utf-8').splitlines(); "
            "tag='Using existing dataset archive'; "
            "marker=next(i for i,x in enumerate(lines) if 'logger.info(f\\\"Downloading the dataset for' in x); "
            "block=['    existing_archives = list(download_dir.glob(\\\"*.zip\\\"))', "
            "'    if not force and len(existing_archives) == 1:', "
            "'        logger.info(\\\"Using existing dataset archive `%s`.\\\", existing_archives[0])', "
            "'        return existing_archives[0]', '']; "
            "lines[marker:marker]=([] if any(tag in x for x in lines) else block); "
            "p.write_text(chr(10).join(lines) + chr(10), encoding='utf-8')",
            str(SOURCE / "mlebench" / "data.py"),
        ], "install_existing_archive_resume_compatibility")
        run([str(python), "-m", "pip", "install", "-e", str(SOURCE)], "install_mlebench")
        run([str(python), "-m", "pip", "install", "--upgrade", "kaggle>=2.2.2"], "upgrade_kaggle_token_runtime")
        run([
            str(python), "-c",
            "import kaggle; from pathlib import Path; "
            "Path(kaggle.__file__).with_name('rest.py').write_text("
            "'from requests.exceptions import HTTPError\\\\nApiException = HTTPError\\\\n', "
            "encoding='utf-8')",
        ], "install_kaggle_rest_compatibility")
        run([
            str(python), "-c",
            "from pathlib import Path; import importlib; "
            "k=importlib.import_module('kaggle.api.kaggle_api_extended'); "
            "p=Path(k.__file__); s=p.read_text(encoding='utf-8'); "
            "old='username = self._introspect_token(access_token)'; "
            "new='username = os.environ.get(\\\"KAGGLE_USERNAME\\\") or \\\"access-token-user\\\"'; "
            "assert old in s or new in s; "
            "p.write_text(s.replace(old, new), encoding='utf-8')",
        ], "install_kaggle_token_offline_compatibility")
        run_with_retries([
            str(python), "-c",
            "from kaggle.api.kaggle_api_extended import KaggleApi; a=KaggleApi(); a.authenticate(); print('KAGGLE_AUTH_OK')",
        ], "verify_kaggle_auth", attempts=24)
        run_with_retries([
            str(VENV / "bin" / "mlebench"), "prepare", "--lite",
            "--data-dir", str(DATA), "--keep-raw",
        ], "prepare_lite_22")
        write_status("complete", state="completed", completed_at=now())
    except Exception as exc:
        with LOG.open("a", encoding="utf-8") as handle:
            handle.write("\\n" + traceback.format_exc() + "\\n")
        write_status("failed", state="needs_continuation", error=str(exc), failed_at=now())
        raise
'''


def _launcher_source() -> str:
    return f'''from __future__ import annotations
import json
import os
import pathlib
import sys

runtime = pathlib.Path({RUNTIME!r})
runtime.mkdir(parents=True, exist_ok=True)
payload = json.loads(sys.stdin.readline())
pid = os.fork()
if pid:
    print(json.dumps({{"pid": pid}}), flush=True)
    raise SystemExit(0)
os.setsid()
os.chdir(runtime)
log = open(runtime / "supervisor.log", "ab", buffering=0)
os.dup2(log.fileno(), 0)
os.dup2(log.fileno(), 1)
os.dup2(log.fileno(), 2)
for key in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"):
    os.environ.pop(key, None)
for key, value in payload.items():
    if key in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY") and value:
        os.environ[key] = value
os.execv("/usr/bin/python3", ["python3", str(runtime / "prepare_lite.py")])
'''


def _connect():
    config = load_gpu_ssh_config()
    config.jump_host = None
    return connect_ssh(config, timeout=20)


def _read_remote_status(client) -> dict:
    command = f"test -f {STATUS_PATH} && cat {STATUS_PATH} || printf '{{}}'"
    _stdin, stdout, stderr = client.exec_command(command, timeout=30)
    text = stdout.read().decode("utf-8", "replace")
    error = stderr.read().decode("utf-8", "replace")
    code = stdout.channel.recv_exit_status()
    if code:
        raise RuntimeError(error or "failed to read remote status")
    return json.loads(text or "{}")


def _persist_status(status: dict) -> None:
    LOCAL_STATUS.parent.mkdir(parents=True, exist_ok=True)
    temp = LOCAL_STATUS.with_suffix(".json.tmp")
    temp.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(LOCAL_STATUS)


def access_command() -> int:
    secrets = _load_kaggle_secret()
    old_values = {key: os.environ.get(key) for key in secrets}
    results: list[dict[str, object]] = []
    try:
        os.environ.update(secrets)
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        for competition_id in LITE_COMPETITION_IDS:
            try:
                response = api.competition_list_files(competition_id)
                files = list(getattr(response, "files", None) or [])
                results.append(
                    {
                        "competition_id": competition_id,
                        "accessible": True,
                        "file_count": len(files),
                        "declared_bytes": sum(int(getattr(item, "total_bytes", 0) or 0) for item in files),
                    }
                )
            except Exception as exc:  # noqa: BLE001 - preserve per-competition access evidence
                message = str(exc)
                for secret in secrets.values():
                    message = message.replace(secret, "***")
                results.append(
                    {
                        "competition_id": competition_id,
                        "accessible": False,
                        "error_type": type(exc).__name__,
                        "error": message[:500],
                    }
                )
    finally:
        for key, previous in old_values.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous
        for key in list(secrets):
            secrets[key] = ""

    payload = {
        "schema": "evomind.mlebench_lite_access_probe.v1",
        "official_commit": OFFICIAL_COMMIT,
        "competition_count": len(LITE_COMPETITION_IDS),
        "accessible": sum(bool(item["accessible"]) for item in results),
        "blocked": sum(not bool(item["accessible"]) for item in results),
        "results": results,
    }
    LOCAL_ACCESS.parent.mkdir(parents=True, exist_ok=True)
    temp = LOCAL_ACCESS.with_suffix(".json.tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(LOCAL_ACCESS)
    print(json.dumps({"local_access": str(LOCAL_ACCESS.resolve()), **payload}, ensure_ascii=False, indent=2))
    return 0 if not payload["blocked"] else 2


def status_command() -> int:
    client = _connect()
    try:
        status = _read_remote_status(client)
        if status.get("pid"):
            command = f"kill -0 {int(status['pid'])} 2>/dev/null && echo running || echo stopped"
            _stdin, stdout, _stderr = client.exec_command(command, timeout=15)
            status["process"] = stdout.read().decode("utf-8", "replace").strip()
    finally:
        client.close()
    _persist_status(status)
    print(json.dumps({"local_status": str(LOCAL_STATUS.resolve()), "status": status}, ensure_ascii=False, indent=2))
    return 0


def start_command() -> int:
    secrets = _load_kaggle_secret()
    client = _connect()
    try:
        current = _read_remote_status(client)
        if current.get("pid") and current.get("state") == "running":
            _stdin, stdout, _stderr = client.exec_command(
                f"kill -0 {int(current['pid'])} 2>/dev/null && echo running || echo stopped",
                timeout=15,
            )
            if stdout.read().decode("utf-8", "replace").strip() == "running":
                current["process"] = "running"
                _persist_status(current)
                print(json.dumps({"started": False, "reason": "already_running", "status": current}, ensure_ascii=False, indent=2))
                return 0

        _stdin, stdout, stderr = client.exec_command(f"mkdir -p {RUNTIME}", timeout=30)
        mkdir_error = stderr.read().decode("utf-8", "replace")
        if stdout.channel.recv_exit_status():
            raise RuntimeError(mkdir_error or "failed to create remote runtime directory")
        with client.open_sftp() as sftp:
            for remote_name, source in (
                (f"{RUNTIME}/prepare_lite.py", _runner_source()),
                (f"{RUNTIME}/start_detached.py", _launcher_source()),
            ):
                with sftp.file(remote_name, "wb") as handle:
                    handle.write(source.encode("utf-8"))

        stdin, stdout, stderr = client.exec_command(f"python3 {RUNTIME}/start_detached.py", timeout=30)
        stdin.write(json.dumps(secrets, separators=(",", ":")) + "\n")
        stdin.flush()
        stdin.channel.shutdown_write()
        output = stdout.read().decode("utf-8", "replace")
        error = stderr.read().decode("utf-8", "replace")
        code = stdout.channel.recv_exit_status()
        if code:
            raise RuntimeError(error or output or "detached launcher failed")
        launch = json.loads(output)
    finally:
        for key in list(secrets):
            secrets[key] = ""
        client.close()

    print(
        json.dumps(
            {
                "started": True,
                "pid": launch["pid"],
                "runtime": RUNTIME,
                "data_root": f"{ALLOWED_GPU_REMOTE_ROOT}/mlebench_official_data",
                "status_path": STATUS_PATH,
                "credential_transport": "encrypted SSH stdin; memory only",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("access", "start", "status"))
    args = parser.parse_args()
    if args.command == "access":
        return access_command()
    return start_command() if args.command == "start" else status_command()


if __name__ == "__main__":
    raise SystemExit(main())
