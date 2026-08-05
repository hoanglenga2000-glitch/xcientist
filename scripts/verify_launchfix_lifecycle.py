from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OWNED_FILES = [
    "install.ps1", "start.ps1", "stop.ps1", "status.ps1", "upgrade.ps1", "rollback.ps1", "uninstall.ps1", "migrate.ps1",
    "scripts/manage_workstation_dashboard.py", "scripts/manage_local_gateway.py", "scripts/apply_workstation_migrations.py",
    "scripts/reconcile_action_log_mirror.py",
    "scripts/workstation_lifecycle.py", "scripts/manage_workstation_lifecycle.ps1", "scripts/start_verified_workstation.ps1",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest(package: Path, version: str) -> None:
    files: list[dict[str, str | int]] = []
    for path in sorted(package.rglob("*")):
        if not path.is_file() or path.name == "release-manifest.json" or any(part in {".venv", "user-data", "__pycache__"} for part in path.relative_to(package).parts):
            continue
        relative = path.relative_to(package).as_posix()
        files.append({"path": relative, "sha256": sha256(path), "size": path.stat().st_size})
    payload = {"format_version": 1, "product": "research-workstation", "version": version, "release_id": f"lifecycle-e2e-{version}", "files": files}
    (package / "release-manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def make_package(package: Path, version: str) -> None:
    package.mkdir(parents=True)
    for relative in OWNED_FILES:
        source = ROOT / relative
        destination = package / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    migrations = ROOT / "web" / "research-agent-workstation" / "prisma" / "migrations"
    shutil.copytree(migrations, package / "app" / "prisma" / "migrations")
    server = package / "app" / "server.js"
    server.parent.mkdir(parents=True, exist_ok=True)
    server.write_text(
        "const http=require('http');\n"
        "const host=process.env.HOSTNAME||'127.0.0.1'; const port=Number(process.env.PORT||8088);\n"
        f"const version={json.dumps(version)};\n"
        "const server=http.createServer((req,res)=>{"
        "res.setHeader('content-type','application/json; charset=utf-8');"
        "if(req.url.startsWith('/api/workstation-summary')){res.end(JSON.stringify({tasks:[],connector_status:{},runtime:{version}}));return;}"
        "res.setHeader('content-type','text/html; charset=utf-8');res.end('<!doctype html><title>Workstation '+version+'</title>');});\n"
        "server.listen(port,host); const stop=()=>server.close(()=>process.exit(0)); process.on('SIGTERM',stop); process.on('SIGINT',stop);\n",
        encoding="utf-8",
    )
    (package / "requirements.txt").write_text("", encoding="utf-8")
    (package / "src" / "xsci").mkdir(parents=True)
    (package / "src" / "research_os").mkdir(parents=True)
    (package / "src" / "xsci" / "__init__.py").write_text("__version__='1.0'\n", encoding="utf-8")
    (package / "src" / "research_os" / "__init__.py").write_text("__version__='1.0'\n", encoding="utf-8")
    (package / "pyproject.toml").write_text(
        "[build-system]\nrequires=['setuptools']\nbuild-backend='setuptools.build_meta'\n"
        "[project]\nname='xcientist'\nversion='1.0.0'\nrequires-python='>=3.10'\n"
        "[tool.setuptools]\npackage-dir={''='src'}\n[tool.setuptools.packages.find]\nwhere=['src']\n",
        encoding="utf-8",
    )
    wheel_dir = package / "runtime" / "wheels"
    wheel_dir.mkdir(parents=True)
    result = subprocess.run([sys.executable, "-m", "pip", "wheel", str(package), "--no-deps", "--no-build-isolation", "--wheel-dir", str(wheel_dir)], text=True, capture_output=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"fixture wheel build failed: {result.stdout}\n{result.stderr}")
    manifest(package, version)


def run(command: list[str], cwd: Path, env: dict[str, str], log: Path, timeout: int = 180) -> subprocess.CompletedProcess[str]:
    started = time.perf_counter()
    result = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True, encoding="utf-8", errors="replace", timeout=timeout)
    log.write_text(json.dumps({"command": command, "cwd": str(cwd), "exit_code": result.returncode, "duration_s": round(time.perf_counter() - started, 3), "stdout": result.stdout, "stderr": result.stderr}, ensure_ascii=False, indent=2), encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"command failed: {command}\n{result.stdout}\n{result.stderr}")
    return result


def port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18192)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    fixture = output / "e2e-中文 空格"
    if fixture.exists():
        shutil.rmtree(fixture)
    fixture.mkdir(parents=True)
    install_root = fixture / "安装目录 工作站"
    package_v2 = fixture / "升级包 v2"
    logs = output / "e2e-logs"
    logs.mkdir(parents=True, exist_ok=True)
    make_package(install_root, "1.0.0")
    make_package(package_v2, "2.0.0")
    env = os.environ.copy()
    env["APPDATA"] = str(install_root / "user-data" / "appdata")
    env["WORKSTATION_HOST"] = "127.0.0.1"
    env["WORKSTATION_PORT"] = str(args.port)
    env.pop("OPENAI_API_KEY", None)
    powershell = shutil.which("powershell.exe") or shutil.which("powershell") or shutil.which("pwsh.exe") or shutil.which("pwsh")
    if not powershell:
        raise RuntimeError("PowerShell not found")
    steps: list[dict[str, object]] = []
    try:
        run([powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(install_root / "install.ps1"), "-OfflineOnly", "-SkipSecretPrompt", "-SkipVerify", "-Port", str(args.port)], install_root, env, logs / "01-install.json", 240)
        steps.append({"step": "install", "ok": True})
        sentinel = install_root / "user-data" / "workspace" / "preserve-me.txt"
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.write_text("persistent-user-data", encoding="utf-8")
        venv_python = install_root / ".venv" / "Scripts" / "python.exe"
        env["WORKSTATION_PYTHON"] = str(venv_python)
        run([powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(install_root / "install.ps1"), "-OfflineOnly", "-SkipSecretPrompt", "-SkipVerify", "-Port", str(args.port)], install_root, env, logs / "02-install-idempotent.json", 240)
        env_text = (install_root / ".env").read_text(encoding="utf-8-sig")
        if not sentinel.is_file() or env_text.count("WORKSTATION_PORT=") != 1 or env_text.count("DATABASE_URL=") != 1:
            raise RuntimeError("idempotent install did not preserve data or produced duplicate managed env keys")
        steps.append({"step": "install_idempotent", "ok": True})
        run([powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(install_root / "migrate.ps1")], install_root, env, logs / "03-migrate.json", 120)
        steps.append({"step": "migrate_idempotent", "ok": True})

        sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], cwd=install_root)
        stale_pid = install_root / "user-data" / "logs" / f"dashboard.{args.port}.pid"
        stale_pid.parent.mkdir(parents=True, exist_ok=True)
        stale_pid.write_text(str(sleeper.pid), encoding="utf-8")
        run([str(venv_python), str(install_root / "scripts" / "manage_workstation_dashboard.py"), "stop", "--port", str(args.port), "--timeout", "5"], install_root, env, logs / "04-stale-pid-safety.json", 30)
        if sleeper.poll() is not None:
            raise RuntimeError("dashboard manager terminated an unrelated process referenced by a stale PID file")
        sleeper.terminate()
        sleeper.wait(timeout=10)
        steps.append({"step": "stale_pid_safety", "ok": True})

        run([powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(install_root / "start.ps1"), "-Port", str(args.port)], install_root, env, logs / "05-start.json", 180)
        if not port_open(args.port):
            raise RuntimeError("dashboard did not listen after start")
        steps.append({"step": "start", "ok": True})
        status = run([powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(install_root / "status.ps1"), "-Port", str(args.port)], install_root, env, logs / "06-status.json", 60)
        status_outer = json.loads(status.stdout)
        status_dashboard = json.loads(status_outer["dashboard"]["output"])
        if status_dashboard.get("status") != "running" or not status_dashboard.get("process_port_consistent"):
            raise RuntimeError("status did not report running dashboard")
        steps.append({"step": "status", "ok": True})
        run([str(venv_python), str(install_root / "scripts" / "manage_workstation_dashboard.py"), "restart", "--port", str(args.port), "--timeout", "30"], install_root, env, logs / "07-restart.json", 90)
        if not port_open(args.port):
            raise RuntimeError("dashboard did not listen after restart")
        steps.append({"step": "restart", "ok": True})
        run([powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(install_root / "upgrade.ps1"), "-PackagePath", str(package_v2), "-Port", str(args.port), "-NoRestart"], install_root, env, logs / "08-upgrade.json", 180)
        marker = json.loads((install_root / ".workstation-install.json").read_text(encoding="utf-8-sig"))
        if marker.get("version") != "2.0.0" or not sentinel.is_file():
            raise RuntimeError("upgrade did not preserve data or apply v2")
        steps.append({"step": "upgrade", "ok": True, "version": marker.get("version")})
        backup_count = len(list((install_root / "user-data" / "backups").glob("upgrade-*")))
        run([powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(install_root / "upgrade.ps1"), "-PackagePath", str(package_v2), "-Port", str(args.port), "-NoRestart"], install_root, env, logs / "09-upgrade-idempotent.json", 180)
        if len(list((install_root / "user-data" / "backups").glob("upgrade-*"))) != backup_count:
            raise RuntimeError("same-release upgrade created an unnecessary rollback backup")
        steps.append({"step": "upgrade_idempotent", "ok": True})
        env["WORKSTATION_PYTHON"] = str(install_root / ".venv" / "Scripts" / "python.exe")
        run([powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(install_root / "rollback.ps1"), "-Port", str(args.port), "-NoRestart"], install_root, env, logs / "10-rollback.json", 180)
        marker = json.loads((install_root / ".workstation-install.json").read_text(encoding="utf-8-sig"))
        if marker.get("version") != "1.0.0" or not sentinel.is_file():
            raise RuntimeError("rollback did not restore v1 or preserve data")
        steps.append({"step": "rollback", "ok": True, "version": marker.get("version")})
        run([powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(install_root / "stop.ps1"), "-Port", str(args.port)], install_root, env, logs / "11-stop.json", 90)
        if port_open(args.port):
            raise RuntimeError("port remains open after stop")
        steps.append({"step": "stop", "ok": True})
        env.pop("WORKSTATION_PYTHON", None)
        run([powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(install_root / "uninstall.ps1"), "-Confirm"], install_root, env, logs / "12-uninstall.json", 180)
        if not sentinel.is_file() or (install_root / "app" / "server.js").exists() or (install_root / ".workstation-install.json").exists():
            raise RuntimeError("uninstall preservation/removal contract failed")
        steps.append({"step": "uninstall", "ok": True, "user_data_preserved": True})
        gateway = json.loads((install_root / "user-data" / "gateway-status.json").read_text(encoding="utf-8-sig"))
        result = {"status": "passed", "ok": True, "port": args.port, "steps": steps, "gateway": {"status": gateway.get("status"), "local_fallback_available": gateway.get("local_fallback_available")}, "chinese_space_path": str(install_root), "port_released": not port_open(args.port), "user_data_preserved": sentinel.is_file()}
    except BaseException as error:
        subprocess.run([sys.executable, str(ROOT / "scripts" / "manage_workstation_dashboard.py"), "stop", "--port", str(args.port), "--timeout", "20"], cwd=ROOT, env=env, capture_output=True)
        result = {"status": "failed", "ok": False, "error_type": type(error).__name__, "message": str(error), "steps": steps, "port_released": not port_open(args.port)}
    cleanup = {"fixture": str(fixture), "removed": False}
    if not port_open(args.port):
        try:
            resolved_fixture = fixture.resolve()
            if resolved_fixture.parent == output.resolve() and resolved_fixture.name == "e2e-中文 空格":
                shutil.rmtree(resolved_fixture)
                cleanup["removed"] = not resolved_fixture.exists()
        except OSError as error:
            cleanup["error_type"] = type(error).__name__
    result["cleanup"] = cleanup
    result_path = output / "lifecycle-e2e-result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
