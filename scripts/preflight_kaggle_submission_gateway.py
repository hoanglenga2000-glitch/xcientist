from __future__ import annotations

import argparse
import concurrent.futures
import json
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "workspace" / "kaggle_submissions"


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def http_probe(url: str, timeout: float) -> dict[str, Any]:
    started = datetime.now()
    req = urllib.request.Request(url, method="GET", headers={"User-Agent": "ResearchAgentGatewayPreflight/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = response.read(16)
            return {
                "name": f"http_get:{url}",
                "passed": True,
                "url": url,
                "status_code": response.status,
                "bytes_read": len(body),
                "elapsed_seconds": (datetime.now() - started).total_seconds(),
            }
    except Exception as exc:  # noqa: BLE001 - diagnostic tool records exact failure class.
        return {
            "name": f"http_get:{url}",
            "passed": False,
            "url": url,
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
            "elapsed_seconds": (datetime.now() - started).total_seconds(),
        }


def tcp_probe(host: str, port: int, timeout: float) -> dict[str, Any]:
    started = datetime.now()
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return {
                "name": f"tcp_connect:{host}:{port}",
                "passed": True,
                "host": host,
                "port": port,
                "elapsed_seconds": (datetime.now() - started).total_seconds(),
            }
    except OSError as exc:
        return {
            "name": f"tcp_connect:{host}:{port}",
            "passed": False,
            "host": host,
            "port": port,
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
            "elapsed_seconds": (datetime.now() - started).total_seconds(),
        }


def http_connect_probe(proxy_host: str, proxy_port: int, dest_host: str, dest_port: int, timeout: float) -> dict[str, Any]:
    started = datetime.now()
    try:
        with socket.create_connection((proxy_host, proxy_port), timeout=timeout) as sock:
            sock.settimeout(timeout)
            request = f"CONNECT {dest_host}:{dest_port} HTTP/1.1\r\nHost: {dest_host}:{dest_port}\r\n\r\n"
            sock.sendall(request.encode("ascii"))
            response = sock.recv(256).decode("latin1", errors="replace")
        return {
            "name": f"http_connect:{proxy_host}:{proxy_port}->{dest_host}:{dest_port}",
            "passed": response.startswith("HTTP/1.1 200") or response.startswith("HTTP/1.0 200"),
            "proxy_host": proxy_host,
            "proxy_port": proxy_port,
            "dest_host": dest_host,
            "dest_port": dest_port,
            "response_head": response[:120],
            "elapsed_seconds": (datetime.now() - started).total_seconds(),
        }
    except OSError as exc:
        return {
            "name": f"http_connect:{proxy_host}:{proxy_port}->{dest_host}:{dest_port}",
            "passed": False,
            "proxy_host": proxy_host,
            "proxy_port": proxy_port,
            "dest_host": dest_host,
            "dest_port": dest_port,
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
            "elapsed_seconds": (datetime.now() - started).total_seconds(),
        }


def kaggle_list_probe(competition: str, timeout: int) -> dict[str, Any]:
    started = datetime.now()
    command = ["kaggle", "competitions", "submissions", "-c", competition]
    try:
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=timeout)
        stdout_tail = completed.stdout[-4000:]
        return {
            "name": "kaggle_submissions_list",
            "passed": completed.returncode == 0,
            "command": command,
            "returncode": completed.returncode,
            "stdout_tail": stdout_tail,
            "stderr_tail": completed.stderr[-2000:],
            "contains_completed_submission": "SubmissionStatus.COMPLETE" in stdout_tail,
            "elapsed_seconds": (datetime.now() - started).total_seconds(),
        }
    except Exception as exc:  # noqa: BLE001 - diagnostic tool records exact failure class.
        return {
            "name": "kaggle_submissions_list",
            "passed": False,
            "command": command,
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
            "elapsed_seconds": (datetime.now() - started).total_seconds(),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail-fast preflight for Kaggle submission upload gateway readiness.")
    parser.add_argument("--competition", default="playground-series-s6e6")
    parser.add_argument("--timeout-seconds", type=float, default=4.0)
    parser.add_argument("--kaggle-timeout-seconds", type=int, default=30)
    parser.add_argument("--proxy-host", default="127.0.0.1")
    parser.add_argument("--proxy-port", type=int, default=7890)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    socket.setdefaulttimeout(args.timeout_seconds)
    checks = [kaggle_list_probe(args.competition, args.kaggle_timeout_seconds)]
    network_jobs = [
        (http_probe, ("https://www.kaggle.com/", args.timeout_seconds)),
        (http_probe, ("https://www.googleapis.com/generate_204", args.timeout_seconds)),
        (tcp_probe, ("www.googleapis.com", 443, args.timeout_seconds)),
        (tcp_probe, (args.proxy_host, args.proxy_port, args.timeout_seconds)),
        (http_connect_probe, (args.proxy_host, args.proxy_port, "www.googleapis.com", 443, args.timeout_seconds)),
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(network_jobs)) as executor:
        futures = [executor.submit(func, *job_args) for func, job_args in network_jobs]
        for future in futures:
            checks.append(future.result())
    required_names = {
        "kaggle_submissions_list",
        "http_get:https://www.kaggle.com/",
        "http_get:https://www.googleapis.com/generate_204",
        "tcp_connect:www.googleapis.com:443",
    }
    required = [check for check in checks if check["name"] in required_names]
    gateway_ready = all(check.get("passed") for check in required)
    status = "passed" if gateway_ready else "blocked"

    report = {
        "status": status,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "competition": args.competition,
        "interpretation": (
            "Kaggle read/list and Google object-storage reachability are ready for a guarded submit."
            if gateway_ready
            else "Do not start an official CLI submit from this environment; the upload gateway is not fully reachable."
        ),
        "required_checks": [check["name"] for check in required],
        "checks": checks,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else OUT_DIR / f"gateway_preflight_{now_stamp()}.json"
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["artifact_path"] = rel(out_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if gateway_ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
