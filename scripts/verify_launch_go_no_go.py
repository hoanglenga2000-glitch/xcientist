from __future__ import annotations

import argparse
from datetime import datetime
import json
import socket
import struct
import urllib.request
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "external_resources.yaml"
PROBE = ROOT / "workspace" / "hpc" / "web_terminal_probe.txt"
JSON_REPORT = ROOT / "docs" / "launch_go_no_go_20260613.json"
MD_REPORT = ROOT / "docs" / "上线Go-No-Go判定-20260613.md"


def fail(message: str, evidence: dict[str, Any] | None = None) -> None:
    raise SystemExit(json.dumps({"status": "failed", "message": message, "evidence": evidence or {}}, ensure_ascii=False, indent=2))


def get_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def socks5_banner(proxy_host: str, proxy_port: int, dest_host: str, dest_port: int) -> str:
    sock = socket.create_connection((proxy_host, proxy_port), timeout=15)
    try:
        sock.sendall(b"\x05\x01\x00")
        if sock.recv(2) != b"\x05\x00":
            raise RuntimeError("local SOCKS bridge rejected no-auth client mode")
        host = dest_host.encode("utf-8")
        sock.sendall(b"\x05\x01\x00\x03" + bytes([len(host)]) + host + struct.pack("!H", dest_port))
        head = sock.recv(4)
        if len(head) < 4 or head[1] != 0:
            raise RuntimeError(f"SOCKS connect failed: {head!r}")
        if head[3] == 1:
            sock.recv(4)
        elif head[3] == 3:
            sock.recv(sock.recv(1)[0])
        elif head[3] == 4:
            sock.recv(16)
        sock.recv(2)
        return sock.recv(64).decode("ascii", "replace").strip()
    finally:
        sock.close()


def probe_status() -> dict[str, Any]:
    if not PROBE.is_file():
        return {
            "status": "missing",
            "path": str(PROBE.relative_to(ROOT)),
            "fully_ready_allowed": False,
            "reason": "Web Terminal nvidia-smi evidence has not been pasted yet.",
        }
    text = PROBE.read_text(encoding="utf-8", errors="replace")
    required = ["whoami", "hostname", "pwd", "Python", "NVIDIA-SMI", "Filesystem", "Mem:"]
    missing = [term for term in required if term not in text]
    a800_count = text.count("NVIDIA A800") + text.count("NVIDIAA800") + text.count("A800-SXM4")
    return {
        "status": "passed" if not missing else "incomplete",
        "path": str(PROBE.relative_to(ROOT)),
        "missing_terms": missing,
        "fully_ready_allowed": not missing and a800_count >= 4,
        "a800_text_hits": a800_count,
    }


def write_markdown(report: dict[str, Any]) -> None:
    gpu_ready = report["hpc_web_terminal_probe"].get("fully_ready_allowed")
    gpu_scope = (
        "- HKUST(GZ) GPU 容器：已通过登录节点 + 环境 SSH 用户进入，`nvidia-smi` 证明 `4 x NVIDIA A800-SXM4-80GB`。"
        if gpu_ready
        else "- HKUST(GZ) GPU 容器：登录节点可达，但仍需 Web Terminal `nvidia-smi` 证据后才能升级。"
    )
    next_steps = (
        [
            "1. 在平台 Web UI 或让管理员把本机生成的 `research_agent_hpc_ed25519.pub` 绑定到该 GPU 环境账号。",
            "2. 再将 `GPU_SSH_HOST`、`GPU_SSH_USER`、`GPU_SSH_PASSWORD` 或 `GPU_SSH_KEY_PATH`、`GPU_REMOTE_WORKSPACE` 放入环境变量、`*_FILE` 或 `WORKSTATION_SECRET_DIR`。",
            "3. 重新运行 `python scripts\\verify_external_resource_gateways.py --url http://127.0.0.1:8088 --allow-real-external`。",
            "4. 保持 GPU Job 入口只允许白名单训练模板，不开放任意 shell。",
        ]
        if gpu_ready
        else [
            "1. 在平台 Web Terminal/JupyterLab 运行 `whoami; hostname; pwd; python --version; nvidia-smi; df -hT; free -h`。",
            "2. 将输出保存为 `workspace\\hpc\\web_terminal_probe.txt`。",
            "3. 运行 `python scripts\\verify_hpc_web_terminal_probe.py workspace\\hpc\\web_terminal_probe.txt`。",
            "4. 只有 4 x A800 被 `nvidia-smi` 证明后，才允许升级 GPU verified。",
        ]
    )
    lines = [
        "# 上线 Go/No-Go 判定 - 2026-06-13",
        "",
        f"- 判定时间：{report['generated_at']}",
        f"- 总体判定：`{report['decision']}`",
        f"- Dashboard：{report['dashboard_url']}",
        "",
        "## 当前可上线范围",
        "",
        "- 本地科研工作站、Kaggle 风格本地训练闭环、报告/Gate/Action Log：可演示、可运行、可验收。",
        "- HKUST(GZ) HPC 网络层：`127.0.0.1:7890` 到 `100.85.169.63:1235` 可达。",
        gpu_scope,
        "- 外部增强资源：未提供运行时密钥/自动作业凭据的项目必须继续显示待配置，不能伪造 ready。",
        "",
        "## 关键状态",
        "",
        f"- HPC SSH banner：`{report['hpc']['banner']}`",
        f"- GPU live state：`{report['live_connectors']['gpu']['state']}`",
        f"- DeepSeek live state：`{report['live_connectors']['deepseek']['state']}`",
        f"- Claude Code live state：`{report['live_connectors']['code_agent']['state']}`",
        f"- Kaggle live state：`{report['live_connectors']['kaggle']['state']}`",
        f"- Web Terminal proof：`{report['hpc_web_terminal_probe']['status']}`",
        "",
        "## 待办但不伪造成 Ready",
        "",
    ]
    if report["pending_hardening_items"]:
        for item in report["pending_hardening_items"]:
            lines.append(f"- {item}")
    else:
        lines.append("- 无。")
    lines.extend([
        "",
        "## No-Go 条件",
        "",
    ])
    for item in report["no_go_conditions"]:
        lines.append(f"- {item}")
    if not report["no_go_conditions"]:
        lines.append("- 无。")
    lines.extend(
        [
            "",
            "## 下一步",
            "",
            *next_steps,
            "",
        ]
    )
    MD_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and verify a launch Go/No-Go decision from live dashboard and resource evidence.")
    parser.add_argument("--dashboard-url", default="http://127.0.0.1:8088")
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()

    if not MANIFEST.is_file():
        fail("external resource manifest is missing", {"path": str(MANIFEST.relative_to(ROOT))})
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8")) or {}
    summary = get_json(f"{args.dashboard_url.rstrip('/')}/api/workstation-summary")
    connectors = summary.get("connector_status") or {}

    required_connector_keys = ["gpu", "deepseek", "code_agent", "kaggle"]
    missing_connectors = [key for key in required_connector_keys if key not in connectors]
    if missing_connectors:
        fail("live dashboard summary is missing connector statuses", {"missing": missing_connectors})

    hpc_manifest = ((manifest.get("resources") or {}).get("hpc_gpu_ssh") or {})
    bridge = hpc_manifest.get("local_bridge") or {}
    listen = str(bridge.get("listen") or "127.0.0.1:7890")
    proxy_host, proxy_port_text = listen.rsplit(":", 1)
    banner = socks5_banner(proxy_host, int(proxy_port_text), str(hpc_manifest.get("ssh_host")), int(hpc_manifest.get("ssh_port")))
    if "SSH-2.0-SSHPiper" not in banner:
        fail("HPC login node banner is not verified", {"banner": banner})

    probe = probe_status()
    gpu_state = str(connectors["gpu"].get("state", ""))
    unsafe_claims: list[str] = []
    if connectors["gpu"].get("configured") and not (probe.get("fully_ready_allowed") or "SSH Gateway Ready" in gpu_state):
        unsafe_claims.append("GPU is configured/ready in live summary without Web Terminal proof or SSH gateway readiness.")
    for key in ["deepseek", "code_agent", "kaggle"]:
        state = str(connectors[key].get("state", ""))
        configured = bool(connectors[key].get("configured"))
        if not configured and "Ready" in state:
            unsafe_claims.append(f"{key} reports Ready while configured=false.")
    if unsafe_claims:
        fail("unsafe launch readiness claims detected", {"unsafe_claims": unsafe_claims})

    no_go_conditions = []
    pending_hardening_items = []
    if not connectors["deepseek"].get("configured"):
        no_go_conditions.append("DeepSeek runtime key is not configured in the current 8088 process.")
    if not connectors["code_agent"].get("configured"):
        no_go_conditions.append("Code Agent is not configured through DeepSeek or Anthropic.")
    if not connectors["kaggle"].get("configured"):
        no_go_conditions.append("Kaggle official API token is not configured.")
    if not probe.get("fully_ready_allowed") and "SSH Gateway Ready" not in gpu_state:
        no_go_conditions.append("GPU cannot be marked fully ready until Web Terminal nvidia-smi proves 4 x A800.")
    if "Permission denied" in str(hpc_manifest.get("current_blocker")):
        no_go_conditions.append("HPC SSH login node still rejects the provided account authentication.")
    normalized_blocker = str(hpc_manifest.get("current_blocker")).lower().replace("public key", "publickey")
    if probe.get("fully_ready_allowed") and "publickey" in normalized_blocker:
        pending_hardening_items.append("GPU hardware/container access is verified, but the remote environment still requires an already-authorized public key before automated SSH jobs can run.")

    decision = "go_fully_ready" if not no_go_conditions and not pending_hardening_items else "go_local_ready_external_pending"
    report = {
        "status": "passed",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dashboard_url": args.dashboard_url,
        "decision": decision,
        "live_connectors": {key: connectors[key] for key in required_connector_keys},
        "hpc": {
            "local_bridge": listen,
            "ssh_destination": f"{hpc_manifest.get('ssh_host')}:{hpc_manifest.get('ssh_port')}",
            "banner": banner,
            "current_blocker": hpc_manifest.get("current_blocker"),
        },
        "hpc_web_terminal_probe": probe,
        "no_go_conditions": no_go_conditions,
        "pending_hardening_items": pending_hardening_items,
        "safe_to_launch_local_workstation": True,
        "safe_to_mark_external_resources_fully_ready": decision == "go_fully_ready",
    }

    if args.write_report:
        JSON_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_markdown(report)
        report["report_paths"] = {
            "json": str(JSON_REPORT.relative_to(ROOT)),
            "markdown": str(MD_REPORT.relative_to(ROOT)),
        }

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
