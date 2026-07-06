from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "docs" / "academic_os_pages_20260614"

PAGES: dict[str, list[list[str]]] = {
    "overview": [["Academic Research OS"]],
    "mission": [["Academic Research OS"], ["科研操作系统能力地图"], ["Academic OS 质量评分卡", "Academic OS quality scorecard"]],
    "experiments": [["Experiment", "实验室"]],
    "data": [["Data & Kaggle", "数据与 Kaggle"], ["Kaggle"]],
    "report": [["Report Studio", "报告工作室"], ["报告"]],
    "code": [["Code Agent", "代码智能体"], ["Claude"]],
    "gpu": [["GPU / HPC"], ["SSH Credential", "GPU Compute", "SSH Gateway Ready"], ["NVIDIA A40", "A40", "NVIDIA A800", "A800", "NVIDIA"]],
    "evidence": [["Evidence Ledger", "证据账本"], ["Unverified", "未验证"]],
    "gates": [["Integrity", "完整性"], ["Gate"]],
    "literature": [["Literature", "文献知识"], ["No citation without source"]],
    "settings": [["Settings", "设置"], ["Kaggle"]],
}


def fail(message: str, evidence: dict | None = None) -> None:
    raise SystemExit(json.dumps({"status": "failed", "message": message, "evidence": evidence or {}}, ensure_ascii=False, indent=2))


def browser_candidates() -> list[Path]:
    return [
        Path("C:/Program Files/Google/Chrome/Application/chrome.exe"),
        Path("C:/Program Files (x86)/Google/Chrome/Application/chrome.exe"),
        Path("C:/Program Files/Microsoft/Edge/Application/msedge.exe"),
        Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"),
    ]


def find_browser() -> Path:
    for candidate in browser_candidates():
        if candidate.is_file():
            return candidate
    fail("No supported Chromium browser was found", {"candidates": [str(path) for path in browser_candidates()]})
    raise AssertionError("unreachable")


def check_reachable(url: str) -> None:
    with urllib.request.urlopen(url, timeout=20) as response:
        if response.status >= 400:
            fail("dashboard URL is not reachable", {"url": url, "status": response.status})


def run_browser(browser: Path, args: list[str], timeout: int = 45) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory(prefix="academic-os-chrome-") as profile:
        command = [
            str(browser),
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            "--no-first-run",
            "--no-default-browser-check",
            f"--user-data-dir={profile}",
            *args,
        ]
        return subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )


def dump_dom(browser: Path, url: str) -> str:
    completed = run_browser(browser, ["--virtual-time-budget=4500", "--dump-dom", url])
    if completed.returncode != 0:
        fail("browser DOM dump failed", {"url": url, "stderr": completed.stderr[-2000:]})
    return completed.stdout


def screenshot(browser: Path, url: str, path: Path, size: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    completed = run_browser(browser, ["--virtual-time-budget=4500", f"--window-size={size}", f"--screenshot={path}", url])
    if completed.returncode != 0:
        fail("browser screenshot failed", {"url": url, "path": str(path.relative_to(ROOT)), "stderr": completed.stderr[-2000:]})
    if not path.is_file() or path.stat().st_size < 10_000:
        fail("browser screenshot is missing or too small", {"url": url, "path": str(path.relative_to(ROOT)), "size": path.stat().st_size if path.exists() else 0})


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Academic Research OS page deep links with real Chromium screenshots.")
    parser.add_argument("--url", default="http://127.0.0.1:8088", help="Running dashboard URL.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Directory for screenshot and DOM artifacts.")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    check_reachable(base_url)
    browser = find_browser()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts: list[str] = []
    missing: dict[str, list[str]] = {}
    for page, terms in PAGES.items():
      page_url = f"{base_url}/?page={page}"
      dom = dump_dom(browser, page_url)
      page_missing = [" | ".join(group) for group in terms if not any(term in dom for term in group)]
      if page_missing:
          missing[page] = page_missing
      dom_path = out_dir / f"{page}.html"
      dom_path.write_text(dom, encoding="utf-8")
      artifacts.append(str(dom_path.relative_to(ROOT)))
      screenshot(browser, page_url, out_dir / f"{page}_desktop.png", "1440,1100")
      screenshot(browser, page_url, out_dir / f"{page}_mobile.png", "390,1000")
      artifacts.append(str((out_dir / f"{page}_desktop.png").relative_to(ROOT)))
      artifacts.append(str((out_dir / f"{page}_mobile.png").relative_to(ROOT)))

    if missing:
        fail("Academic OS page deep links are missing expected visible terms", {"missing": missing})

    index_path = out_dir / "acceptance_index.json"
    index_path.write_text(json.dumps({
        "status": "passed",
        "dashboard_url": base_url,
        "browser": str(browser),
        "pages": list(PAGES),
        "artifacts": artifacts,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    artifacts.append(str(index_path.relative_to(ROOT)))

    print(json.dumps({
        "status": "passed",
        "dashboard_url": base_url,
        "browser": str(browser),
        "page_count": len(PAGES),
        "artifact_count": len(artifacts),
        "out_dir": str(out_dir.relative_to(ROOT)),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
