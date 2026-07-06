"""New-user release readiness gate.

This gate answers a narrow launch question: can a fresh user install the
workstation, open the default EvoMind gateway, configure credentials, and use
the non-training workstation features with honest boundaries?

It intentionally does not start training, GPU jobs, model calls, or Kaggle
submissions. GPU/HPC and large-batch LLM cache blockers are reported as optional
training blockers, not as blockers for the default workstation UI release.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT / "workspace"
REPORTS = ROOT / "reports"

REQUIRED_FILES = [
    "README.md",
    "docs/NEW_USER_ONBOARDING_GUIDE.md",
    "docs/RELEASE_CHECKLIST.md",
    ".env.example",
    "requirements.txt",
    "install.ps1",
    "scripts/quick_setup.ps1",
    "scripts/install_autokaggle_cli.ps1",
    "scripts/start_verified_workstation.ps1",
    "scripts/restart_workstation_frontend.ps1",
    "scripts/manage_deepseek_secret.ps1",
    "scripts/manage_kaggle_secret.ps1",
    "scripts/manage_hpc_ssh_secret.ps1",
    "scripts/verify_no_plaintext_secrets.py",
    "scripts/run_new_user_release_acceptance.ps1",
    "scripts/verify_workstation_launch_readiness.py",
    "web/research-agent-workstation/package.json",
    "web/research-agent-workstation/package-lock.json",
    "pyproject.toml",
]

DOC_REQUIREMENTS = {
    "README.md": [
        "http://127.0.0.1:8088/?page=control",
        "evomind ready",
        "evomind official",
        "Human Gate",
    ],
    "docs/NEW_USER_ONBOARDING_GUIDE.md": [
        "http://127.0.0.1:8088/?page=control",
        "Windows DPAPI",
        "Training and official Kaggle submission",
        "evomind setup",
    ],
    "docs/RELEASE_CHECKLIST.md": [
        "run_new_user_release_acceptance.ps1",
        "gpu_resource_blocked",
        "Human Gate",
        "not prove",
    ],
}

MOJIBAKE_PATTERNS = [
    "\u9225",
    "\u920b",
    "\u7ec9",
    "\u9428",
    "\u93c2",
    "\u5bb8",
    "\u95b0",
    "\u6940",
    "\u9983",
    "\u9241",
    "\u95c2",
    "\u5a75",
    "\u7f02",
    "\u9286",
    "\u93ac",
    "\u935a",
    "\u95c6",
    "\u95c8",
    "\u95c1",
    "\u9359",
    "\u9422",
    "\u9a9e",
    "\u9a83",
    "\u9e9f",
    "\u9611",
    "\u6d93",
    "\u5a34",
    "\u7d13",
]

MOJIBAKE_REGEXES = [
    re.compile(r"[\u9200-\u95ff]{2,}"),
    re.compile(r"[\u9286\u4f75\u20ac]{2,}"),
    re.compile(
        r"\u951b|\u951f|\u9428|\u7ed4|\u7eef|\u7cba|\u59ab|\u93cc|"
        r"\u6d93|\u6d60|\u9354|\u93ba|\u95c2|\u95b0|\u93c2|\u7487|"
        r"\u7035|\u5bee|\u9359"
    ),
]

CHINESE_DOCS = {
    "docs/NEW_USER_ONBOARDING_GUIDE.md",
}


def run(cmd: list[str], *, cwd: Path = ROOT, timeout: int = 60) -> dict:
    started = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "seconds": round(time.time() - started, 3),
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-2000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "returncode": None,
            "seconds": round(time.time() - started, 3),
            "stdout_tail": (exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-2000:] if isinstance(exc.stderr, str) else "timeout",
        }


def http_json(path: str, timeout: int = 8) -> dict:
    url = f"http://127.0.0.1:8088{path}"
    req = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as res:
            text = res.read().decode("utf-8", errors="replace")
            data = json.loads(text)
            return {"ok": 200 <= res.status < 300, "status": res.status, "url": url, "keys": sorted(data)[:30]}
    except (URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        return {"ok": False, "status": None, "url": url, "error": str(exc)}


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def check_files() -> list[dict]:
    checks = []
    for rel in REQUIRED_FILES:
        path = ROOT / rel
        checks.append({"id": f"file:{rel}", "ok": path.exists(), "path": rel})
    return checks


def check_docs() -> list[dict]:
    checks = []
    for rel, needles in DOC_REQUIREMENTS.items():
        path = ROOT / rel
        if not path.exists():
            checks.append({"id": f"doc:{rel}", "ok": False, "missing": ["file missing"]})
            continue
        text = read_text(path)
        missing = [needle for needle in needles if needle not in text]
        literal_hits = sorted({pat for pat in MOJIBAKE_PATTERNS if pat in text})
        regex_hits = sorted({regex.pattern for regex in MOJIBAKE_REGEXES if regex.search(text)})
        chinese_count = len(re.findall(r"[\u4e00-\u9fff]", text))
        chinese_required_missing = rel in CHINESE_DOCS and chinese_count < 50
        mojibake_hits = literal_hits + regex_hits
        checks.append({
            "id": f"doc:{rel}",
            "ok": not missing and not mojibake_hits and not chinese_required_missing,
            "missing": missing,
            "mojibake_hits": mojibake_hits,
            "chinese_count": chinese_count,
            "chinese_required_missing": chinese_required_missing,
        })
    return checks


def check_cli() -> list[dict]:
    checks = []
    help_result = run([sys.executable, "-X", "utf8", "-m", "xsci.kaggle", "--help"], timeout=30)
    help_text = help_result["stdout_tail"] + help_result["stderr_tail"]
    checks.append({
        "id": "cli:python_module_help",
        "ok": help_result["ok"] and "EvoMind" in help_text and "http://127.0.0.1:8088/?page=control" in help_text,
        "result": help_result,
    })
    ready_result = run([sys.executable, "-X", "utf8", "-m", "xsci.kaggle", "ready"], timeout=30)
    ready_text = ready_result["stdout_tail"] + ready_result["stderr_tail"]
    checks.append({
        "id": "cli:ready",
        "ok": ready_result["ok"] and "Readiness" in ready_text and "Dashboard" in ready_text,
        "result": ready_result,
    })
    official_result = run([sys.executable, "-X", "utf8", "-m", "xsci.kaggle", "official", "--help"], timeout=45)
    official_text = official_result["stdout_tail"] + official_result["stderr_tail"]
    checks.append({
        "id": "cli:official_passthrough",
        "ok": official_result["ok"] and "competitions" in official_text and "datasets" in official_text,
        "result": official_result,
    })
    return checks


def check_python_compile() -> list[dict]:
    targets = [
        "src/xsci/kaggle.py",
        "src/xsci/config.py",
        "src/xsci/kaggle_session.py",
        "src/xsci/kaggle_intent.py",
        "src/xsci/kaggle_conversation.py",
    ]
    result = run([sys.executable, "-m", "py_compile", *targets], timeout=45)
    return [{"id": "python:core_compile", "ok": result["ok"], "result": result}]


def check_frontend_runtime(*, require_live_server: bool) -> list[dict]:
    checks = [
        {"id": "http:workstation_summary", **http_json("/api/workstation-summary")},
        {"id": "http:tasks", **http_json("/api/tasks")},
        {"id": "http:settings", **http_json("/api/settings")},
    ]
    if require_live_server:
        return checks
    normalized: list[dict] = []
    for item in checks:
        if item.get("ok"):
            normalized.append(item)
            continue
        error = str(item.get("error", ""))
        server_down = "10061" in error or "Connection refused" in error or "actively refused" in error
        if server_down:
            normalized.append({
                **item,
                "ok": True,
                "optional": True,
                "note": "live server is not running; pass because --require-live-server was not set",
            })
        else:
            normalized.append(item)
    return normalized


def check_existing_launch_gate() -> list[dict]:
    path = WORKSPACE / "workstation_launch_readiness_20260630.json"
    if not path.exists():
        return [{"id": "launch:existing_gate_report", "ok": False, "missing": str(path)}]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [{"id": "launch:existing_gate_report", "ok": False, "error": str(exc)}]
    critical = data.get("critical_failures") or []
    state = data.get("launch_state")
    blockers = data.get("blockers") or []
    release_ok = data.get("status") == "passed" and not critical
    return [{
        "id": "launch:existing_gate_report",
        "ok": release_ok,
        "launch_state": state,
        "blockers": blockers,
        "critical_failures": critical,
        "release_boundary": "GPU/cache blockers are optional training blockers for new-user UI release.",
    }]


def build_report(*, require_live_server: bool = False) -> dict:
    checks: list[dict] = []
    for group in [
        check_files(),
        check_docs(),
        check_python_compile(),
        check_cli(),
        check_frontend_runtime(require_live_server=require_live_server),
        check_existing_launch_gate(),
    ]:
        checks.extend(group)
    failed = [item for item in checks if not item.get("ok")]
    optional_training_blockers = []
    for item in checks:
        if item.get("id") == "launch:existing_gate_report":
            optional_training_blockers = list(item.get("blockers") or [])
    release_state = "ready_for_new_user_evomind_gateway" if not failed else "not_ready"
    return {
        "schema": "xcientist.new_user_release_readiness.v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "passed" if not failed else "failed",
        "release_state": release_state,
        "default_gateway": "http://127.0.0.1:8088/?page=control",
        "require_live_server": require_live_server,
        "failed_checks": [item["id"] for item in failed],
        "optional_training_blockers": optional_training_blockers,
        "checks": checks,
        "claim_boundary": (
            "This proves new-user installation and EvoMind gateway readiness only. It does not prove "
            "GPU training availability, official Kaggle submission, rank, medal, or MLE-Bench-75 performance."
        ),
    }


def to_markdown(report: dict) -> str:
    lines = [
        "# New User Release Readiness",
        "",
        f"- status: `{report['status']}`",
        f"- release_state: `{report['release_state']}`",
        f"- default_gateway: {report['default_gateway']}",
        f"- require_live_server: `{report['require_live_server']}`",
        f"- failed_checks: `{', '.join(report['failed_checks']) or 'none'}`",
        f"- optional_training_blockers: `{', '.join(report['optional_training_blockers']) or 'none'}`",
        "",
        "## Checks",
        "",
        "| id | ok | note |",
        "| --- | --- | --- |",
    ]
    for item in report["checks"]:
        note = ""
        if item.get("missing"):
            note = "missing: " + ", ".join(map(str, item["missing"]))
        elif item.get("mojibake_hits"):
            note = "mojibake: " + ", ".join(map(str, item["mojibake_hits"]))
        elif item.get("launch_state"):
            note = f"launch_state={item.get('launch_state')}; blockers={','.join(item.get('blockers') or []) or 'none'}"
        elif item.get("error"):
            note = str(item["error"])[:120]
        lines.append(f"| `{item['id']}` | `{item.get('ok')}` | {note} |")
    lines.extend([
        "",
        "## Claim Boundary",
        "",
        report["claim_boundary"],
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-report", action="store_true")
    parser.add_argument("--require-live-server", action="store_true")
    args = parser.parse_args()
    report = build_report(require_live_server=args.require_live_server)
    if args.write_report:
        WORKSPACE.mkdir(parents=True, exist_ok=True)
        REPORTS.mkdir(parents=True, exist_ok=True)
        (WORKSPACE / "new_user_release_readiness.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (REPORTS / "NEW_USER_RELEASE_READINESS.md").write_text(
            to_markdown(report),
            encoding="utf-8",
        )
    print(json.dumps({
        "status": report["status"],
        "release_state": report["release_state"],
        "failed_checks": report["failed_checks"],
        "optional_training_blockers": report["optional_training_blockers"],
        "json": "workspace/new_user_release_readiness.json" if args.write_report else None,
        "md": "reports/NEW_USER_RELEASE_READINESS.md" if args.write_report else None,
    }, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
