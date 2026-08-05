from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import psutil


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web" / "research-agent-workstation"
MANAGER = ROOT / "scripts" / "manage_workstation_dashboard.py"
CONTRACTS = {
    "report": WEB / "scripts" / "verify-report-studio-contract.mjs",
    "security": WEB / "scripts" / "verify-localhost-security-contract.mjs",
}
PROTECTED_PORTS = (8088, 8765, 17897, 65068)


class ContractRunnerError(RuntimeError):
    pass


@dataclass(frozen=True)
class StartedIdentity:
    port: int
    runtime_port: int
    host: str
    mode: str
    dashboard_pid: int
    runtime_pid: int
    release_nonce: str
    started_at: float
    dashboard_record: dict[str, Any]
    runtime_record: dict[str, Any]
    launcher_pids: tuple[int, ...]
    state_file: Path

    @property
    def recorded_pids(self) -> frozenset[int]:
        return frozenset((self.dashboard_pid, self.runtime_pid, *self.launcher_pids))


def json_object(value: str, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ContractRunnerError(f"{label} did not return JSON") from exc
    if not isinstance(payload, dict):
        raise ContractRunnerError(f"{label} did not return an object")
    return payload


def run_manager(arguments: list[str], env: dict[str, str], *, check: bool = True) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
    result = subprocess.run(
        [sys.executable, str(MANAGER), *arguments],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )
    payload = json_object(result.stdout, f"dashboard manager {' '.join(arguments[:1])}") if result.stdout.strip() else {}
    if check and result.returncode != 0:
        detail = (result.stderr.strip() or result.stdout.strip() or "no diagnostics")[-2000:]
        raise ContractRunnerError(f"dashboard manager {arguments[0]} failed: {detail}")
    return result, payload


def listener_pids(port: int) -> set[int]:
    return {
        int(connection.pid)
        for connection in psutil.net_connections(kind="tcp")
        if connection.pid
        and connection.status == psutil.CONN_LISTEN
        and connection.laddr
        and int(connection.laddr.port) == port
    }


def listener_bindings(port: int) -> set[tuple[str, int]]:
    return {
        (str(connection.laddr.ip), int(connection.pid))
        for connection in psutil.net_connections(kind="tcp")
        if connection.pid
        and connection.status == psutil.CONN_LISTEN
        and connection.laddr
        and int(connection.laddr.port) == port
    }


def assert_port_free(port: int) -> None:
    if listener_pids(port):
        raise ContractRunnerError(f"loopback contract port {port} is already in use")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        probe.bind(("127.0.0.1", port))


def protected_snapshot() -> dict[int, set[tuple[str, int]]]:
    return {port: listener_bindings(port) for port in PROTECTED_PORTS}


def expected_runtime_file(runtime_dir: Path, port: int, suffix: str) -> Path:
    port_suffix = "" if port == 8088 else f".{port}"
    return runtime_dir / f"dashboard{port_suffix}.{suffix}"


def read_started_identity(payload: dict[str, Any], runtime_dir: Path, port: int, runtime_port: int) -> StartedIdentity:
    if payload.get("status") != "started":
        raise ContractRunnerError("dashboard manager did not create the requested isolated instance")
    if payload.get("port") not in (None, port) or payload.get("runtime_port") not in (None, runtime_port):
        raise ContractRunnerError("isolated start response contains a conflicting port binding")
    if payload.get("host") not in (None, "127.0.0.1"):
        raise ContractRunnerError("isolated start response contains a conflicting host binding")
    state_file = expected_runtime_file(runtime_dir, port, "process.json")
    if state_file.is_symlink() or not state_file.is_file():
        raise ContractRunnerError("isolated dashboard process state is missing")
    state = json_object(state_file.read_text(encoding="utf-8"), "isolated dashboard state")
    if (
        state.get("schema") != "evomind.lifecycle_state.v2"
        or state.get("port") != port
        or state.get("runtime_port") != runtime_port
        or state.get("host") != "127.0.0.1"
        or state.get("mode") != payload.get("mode")
    ):
        raise ContractRunnerError("isolated dashboard state binding drift was detected")
    processes = state.get("processes")
    if not isinstance(processes, dict):
        raise ContractRunnerError("isolated lifecycle state has no bound process records")
    dashboard = processes.get("dashboard")
    runtime = processes.get("runtime")
    if not isinstance(dashboard, dict) or not isinstance(runtime, dict):
        raise ContractRunnerError("isolated lifecycle state is missing dashboard/runtime identities")
    release_nonce = state.get("release_nonce")
    if not isinstance(release_nonce, str) or not 32 <= len(release_nonce) <= 128:
        raise ContractRunnerError("isolated lifecycle release nonce is invalid")

    def record_pid(record: dict[str, Any], role: str, expected_port: int) -> int:
        pid = record.get("pid")
        if (
            record.get("schema") != "evomind.process_identity.v1"
            or record.get("role") != role
            or record.get("port") != expected_port
            or record.get("release_nonce") != release_nonce
            or not isinstance(pid, int)
            or pid <= 0
        ):
            raise ContractRunnerError(f"isolated {role} process identity is invalid")
        for field in ("creation_time", "executable", "command_line", "cwd", "install_dir"):
            if not isinstance(record.get(field), str) or not str(record[field]).strip():
                raise ContractRunnerError(f"isolated {role} identity is missing {field}")
        return pid

    dashboard_pid = record_pid(dashboard, "dashboard", port)
    runtime_pid = record_pid(runtime, "runtime", runtime_port)
    launcher_pids: list[int] = []
    for role, expected_port in (("dashboard_launcher", port), ("runtime_launcher", runtime_port)):
        record = processes.get(role)
        if record is not None:
            if not isinstance(record, dict):
                raise ContractRunnerError(f"isolated {role} identity is malformed")
            launcher_pids.append(record_pid(record, role, expected_port))
    if payload.get("pid") != dashboard_pid or payload.get("runtime_pid") != runtime_pid:
        raise ContractRunnerError("isolated start response does not match its process identity records")
    return StartedIdentity(
        port=port,
        runtime_port=runtime_port,
        host="127.0.0.1",
        mode=str(state["mode"]),
        dashboard_pid=dashboard_pid,
        runtime_pid=runtime_pid,
        release_nonce=release_nonce,
        started_at=float(state.get("started_at") or 0.0),
        dashboard_record=dashboard,
        runtime_record=runtime,
        launcher_pids=tuple(launcher_pids),
        state_file=state_file,
    )


def expected_process_cwd(mode: str) -> Path:
    if mode == "source-standalone":
        return (WEB / ".next" / "standalone").resolve()
    if mode == "production":
        return WEB.resolve()
    if mode == "standalone":
        return (ROOT / "app").resolve()
    raise ContractRunnerError(f"unexpected dashboard launch mode: {mode}")


def verify_identity(identity: StartedIdentity, env: dict[str, str]) -> None:
    _, status = run_manager(
        ["status", "--host", identity.host, "--port", str(identity.port)],
        env,
    )
    current_listeners = listener_pids(identity.port)
    runtime_listeners = listener_pids(identity.runtime_port)
    status_listeners = {
        value for value in status.get("listener_pids", []) if isinstance(value, int) and value > 0
    }
    if (
        status.get("status") != "running"
        or status.get("pid") != identity.dashboard_pid
        or status.get("runtime_pid") != identity.runtime_pid
        or status.get("runtime_port") != identity.runtime_port
        or status.get("process_port_consistent") is not True
        or status.get("runtime_process_consistent") is not True
        or current_listeners != {identity.dashboard_pid}
        or runtime_listeners != {identity.runtime_pid}
        or current_listeners != status_listeners
        or listener_bindings(identity.port) != {("127.0.0.1", identity.dashboard_pid)}
        or listener_bindings(identity.runtime_port) != {("127.0.0.1", identity.runtime_pid)}
    ):
        raise ContractRunnerError("isolated dashboard/runtime PID and listener verification failed")

    root_marker = str(ROOT.resolve()).casefold()
    expected = (
        (identity.dashboard_pid, "dashboard", expected_process_cwd(identity.mode), identity.dashboard_record),
        (identity.runtime_pid, "runtime", ROOT.resolve(), identity.runtime_record),
    )
    for pid, role, expected_cwd, record in expected:
        try:
            process = psutil.Process(pid)
            command = " ".join(process.cmdline()).casefold()
            cwd = Path(process.cwd()).resolve()
            executable = Path(process.exe()).resolve()
            created_at = process.create_time()
        except (psutil.Error, OSError) as exc:
            raise ContractRunnerError(f"could not inspect isolated {role} PID {pid}") from exc
        if cwd != expected_cwd:
            raise ContractRunnerError(f"isolated {role} PID {pid} has an unexpected working directory")
        recorded_executable = str(record["executable"]).replace("\\", "/").casefold()
        if executable.as_posix().casefold() != recorded_executable:
            raise ContractRunnerError(f"isolated {role} PID {pid} executable identity drifted")
        if not identity.started_at or abs(created_at - identity.started_at) > 30:
            raise ContractRunnerError(f"isolated {role} PID {pid} creation time is outside the launch transaction")
        if root_marker not in command or f"evomind-{identity.release_nonce}".casefold() not in command:
            raise ContractRunnerError(f"isolated {role} PID {pid} command line is not root/nonce bound")
        if role == "dashboard" and not any(marker in command for marker in ("server.js", "next\\dist\\bin\\next", "next/dist/bin/next")):
            raise ContractRunnerError(f"isolated dashboard PID {pid} command line does not match EvoMind")
        if role == "runtime" and "evomind_runtime.http_server" not in command:
            raise ContractRunnerError(f"isolated runtime PID {pid} command line does not match EvoMind")


def claim_bootstrap_token(payload: dict[str, Any], runtime_dir: Path, port: int) -> str:
    expected = expected_runtime_file(runtime_dir, port, "bootstrap.once").resolve()
    raw_path = payload.get("bootstrap_url_file")
    if not isinstance(raw_path, str) or Path(raw_path).resolve() != expected:
        raise ContractRunnerError("dashboard bootstrap once-file is outside the isolated runtime")
    source = Path(raw_path)
    if source.is_symlink() or not source.is_file() or source.stat().st_size > 1024:
        raise ContractRunnerError("dashboard bootstrap once-file is malformed")
    claimed = runtime_dir / f".{source.name}.claimed-{os.getpid()}"
    os.replace(source, claimed)
    try:
        value = claimed.read_text(encoding="utf-8").strip()
    finally:
        claimed.unlink(missing_ok=True)

    parsed = urlsplit(value)
    fragment = parse_qs(parsed.fragment, strict_parsing=True)
    tokens = fragment.get("bootstrap", [])
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.port != port
        or parsed.path != "/"
        or parsed.query != "page=assistant"
        or set(fragment) != {"bootstrap"}
        or len(tokens) != 1
        or not 24 <= len(tokens[0]) <= 256
    ):
        raise ContractRunnerError("dashboard bootstrap once-file did not contain the expected fragment URL")
    return tokens[0]


def run_contract(port: int, token: str, env: dict[str, str], contract: Path) -> dict[str, Any]:
    node = os.environ.get("WORKSTATION_NODE") or shutil.which("node.exe") or shutil.which("node")
    if not node:
        raise ContractRunnerError("Node.js is unavailable")
    contract_env = env.copy()
    contract_env.update({
        "REPORT_TEST_BASE_URL": f"http://127.0.0.1:{port}",
        "REPORT_TEST_BOOTSTRAP_TOKEN": token,
        "WORKSTATION_ROOT": str(ROOT),
    })
    result = subprocess.run(
        [node, str(contract)],
        cwd=WEB,
        env=contract_env,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )
    stdout = result.stdout.replace(token, "[redacted]")
    stderr = result.stderr.replace(token, "[redacted]")
    token = ""
    if result.returncode != 0:
        detail = stderr.strip() or stdout.strip() or "contract exited without diagnostics"
        raise ContractRunnerError(f"authenticated report contract failed: {detail[-3000:]}")
    return json_object(stdout, "authenticated report contract")


def stop_started(identity: StartedIdentity, env: dict[str, str]) -> None:
    verify_identity(identity, env)
    _, payload = run_manager(
        ["stop", "--host", identity.host, "--port", str(identity.port), "--timeout", "30"],
        env,
    )
    stopped = {value for value in payload.get("stopped_pids", []) if isinstance(value, int) and value > 0}
    if not stopped.issubset(identity.recorded_pids):
        raise ContractRunnerError("dashboard manager stopped a PID outside the isolated identity")
    if (
        payload.get("port_released") is not True
        or payload.get("runtime_port_released") is not True
        or listener_pids(identity.port)
        or listener_pids(identity.runtime_port)
        or any(psutil.pid_exists(pid) for pid in identity.recorded_pids)
    ):
        raise ContractRunnerError("isolated dashboard/runtime processes or ports were not released")


def stop_unparsed_started(port: int, runtime_port: int, env: dict[str, str]) -> None:
    """Ask the strict manager to clean a start whose response could not be parsed.

    The manager performs its own full identity verification and treats every
    mismatch as a conflict, so this fallback never signals an unbound listener.
    """

    _, payload = run_manager(
        ["stop", "--host", "127.0.0.1", "--port", str(port), "--timeout", "30"],
        env,
        check=False,
    )
    if (
        payload.get("conflicts")
        or listener_pids(port)
        or listener_pids(runtime_port)
        or payload.get("port_released") is not True
        or payload.get("runtime_port_released") is not True
    ):
        raise ContractRunnerError("strict cleanup could not close the unparsed isolated lifecycle state")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the report contract against a fresh authenticated loopback session.")
    parser.add_argument("--port", type=int, default=18088)
    parser.add_argument("--runtime-port", type=int, default=None)
    parser.add_argument("--suite", choices=sorted(CONTRACTS), default="report")
    parser.add_argument("--output", default="", help="optional JSON evidence path below artifacts/")
    return parser.parse_args()


def write_evidence(path_value: str, payload: dict[str, Any]) -> Path | None:
    if not path_value:
        return None
    destination = Path(path_value).expanduser().resolve()
    allowed = (ROOT / "artifacts").resolve()
    if destination == allowed or not destination.is_relative_to(allowed):
        raise ContractRunnerError("contract evidence output must remain below artifacts/")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise ContractRunnerError("contract evidence output already exists")
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def main() -> int:
    args = parse_args()
    contract = CONTRACTS[args.suite]
    if contract.is_symlink() or not contract.is_file():
        raise ContractRunnerError(f"contract suite is missing: {args.suite}")
    if not 1024 <= args.port <= 65535 or args.port in PROTECTED_PORTS:
        raise ContractRunnerError("contract port is invalid or reserved by a protected EvoMind service")
    runtime_port = args.runtime_port if args.runtime_port is not None else args.port + 1
    if (
        not 1024 <= runtime_port <= 65535
        or runtime_port == args.port
        or runtime_port in PROTECTED_PORTS
    ):
        raise ContractRunnerError("contract runtime port is invalid or reserved by a protected EvoMind service")
    assert_port_free(args.port)
    assert_port_free(runtime_port)
    protected_before = protected_snapshot()
    identity: StartedIdentity | None = None
    isolated_start_returned = False
    contract_result: dict[str, Any] | None = None
    primary_error: BaseException | None = None
    cleanup_error: BaseException | None = None
    protected_changed = False

    try:
        with tempfile.TemporaryDirectory(prefix="evomind-report-contract-") as temporary:
            runtime_dir = (Path(temporary) / "runtime").resolve()
            data_dir = (Path(temporary) / "data").resolve()
            runtime_dir.mkdir(parents=True)
            data_dir.mkdir(parents=True)
            if any(path == ROOT.resolve() or path.is_relative_to(ROOT.resolve()) for path in (runtime_dir, data_dir)):
                raise ContractRunnerError("isolated runtime and data directories must remain outside the repository")
            env = os.environ.copy()
            env["WORKSTATION_RUNTIME_DIR"] = str(runtime_dir)
            # Keep the auxiliary runtime token, SQLite store and artifacts out
            # of the live 8088/8765 data root.  WORKSTATION_ROOT remains the
            # real repository because this contract deliberately verifies the
            # existing reviewed report artifacts without mutating them.
            env["WORKSTATION_DATA_DIR"] = str(data_dir)
            env["WORKSTATION_ROOT"] = str(ROOT)
            env["EVOMIND_RUNTIME_PORT"] = str(runtime_port)
            source_path = str((ROOT / "src").resolve())
            env["PYTHONPATH"] = source_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            for key in ("WORKSTATION_SESSION_SECRET", "WORKSTATION_BOOTSTRAP_TOKEN_HASH", "REPORT_TEST_SESSION_COOKIE", "REPORT_TEST_CSRF_TOKEN"):
                env.pop(key, None)

            try:
                _, start_payload = run_manager(
                    [
                        "start",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        str(args.port),
                        "--timeout",
                        "90",
                        "--no-auto-build",
                    ],
                    env,
                )
                isolated_start_returned = start_payload.get("status") == "started"
                identity = read_started_identity(start_payload, runtime_dir, args.port, runtime_port)
                verify_identity(identity, env)
                token = claim_bootstrap_token(start_payload, runtime_dir, args.port)
                try:
                    contract_result = run_contract(args.port, token, env, contract)
                finally:
                    token = ""
            except BaseException as exc:
                primary_error = exc
            finally:
                if identity is not None:
                    try:
                        stop_started(identity, env)
                    except BaseException as exc:  # Preserve the contract error while still surfacing cleanup failure.
                        cleanup_error = exc
                elif isolated_start_returned or expected_runtime_file(runtime_dir, args.port, "process.json").is_file():
                    try:
                        stop_unparsed_started(args.port, runtime_port, env)
                    except BaseException as exc:
                        cleanup_error = exc
    finally:
        protected_changed = protected_snapshot() != protected_before

    if cleanup_error is not None:
        raise ContractRunnerError(f"isolated dashboard cleanup failed: {cleanup_error}") from cleanup_error
    if protected_changed:
        raise ContractRunnerError("a protected EvoMind service listener changed during the isolated contract")
    if primary_error is not None:
        raise primary_error
    if contract_result is None:
        raise ContractRunnerError("authenticated report contract did not produce a result")
    result_payload = {
        "ok": True,
        "suite": args.suite,
        "port": args.port,
        "runtime_port": runtime_port,
        "isolated_runtime": True,
        "loopback_bindings_verified": True,
        "protected_ports_unchanged": True,
        "contract": contract_result,
    }
    evidence_path = write_evidence(args.output, result_payload)
    if evidence_path:
        result_payload["evidence_path"] = str(evidence_path)
    print(json.dumps(result_payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractRunnerError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)
