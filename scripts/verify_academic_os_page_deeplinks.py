from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "docs" / "evomind_pages_20260708"

PAGES: dict[str, list[list[str]]] = {
    "overview": [
        ["EvoMind"],
        ["科研工作站运行态势与闭环总控"],
        ["实时运行证据", "Live Run Evidence"],
    ],
    "control": [
        ["EvoMind Gateway", "EvoMind 工作站入口"],
        ["Scientist Autopilot", "科学家诊断"],
        ["Scientist Action Queue", "科学家行动队列"],
    ],
    "mission": [
        ["Academic Research OS", "科研工作站运行态势与闭环总控"],
        ["Workstation-started runs", "工作站"],
        ["无官方 Kaggle response", "人工 Gate"],
    ],
    "experiments": [
        ["实验台账", "Experiment"],
        ["分支搜索", "Search"],
        ["分数提升门禁", "Gate"],
    ],
    "data": [
        ["Data & Kaggle", "数据与 Kaggle"],
        ["schema 审计", "schema"],
        ["submission 门禁", "submission"],
    ],
    "report": [
        ["Report Studio", "报告工作室"],
        ["AI 自动汇总实验结果", "AI 生成报告"],
        ["证据链", "风险审计"],
    ],
    "code": [
        ["Code Agent IDE", "代码 Agent"],
        ["可审计代码生成", "Algorithm code generation"],
        ["Diff", "quality gate"],
    ],
    "gpu": [
        ["GPU / HPC 控制台", "GPU / HPC"],
        ["算力资源状态", "远程算力"],
        ["远程训练", "产物回传"],
    ],
    "evidence": [
        ["证据台账", "Evidence"],
        ["统一归档 artifact", "artifact"],
        ["日志", "指标", "审计证据"],
    ],
    "gates": [
        ["完整性 Gate", "Integrity"],
        ["人工审批", "Human Gate"],
        ["提交阻断", "安全边界"],
    ],
    "literature": [
        ["Literature", "文献知识", "文献与知识库"],
        ["RAG"],
        ["研究上下文", "知识库"],
    ],
    "settings": [
        ["Settings", "系统设置"],
        ["账号", "语言", "主题"],
        ["凭据", "资源"],
    ],
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


def run_browser(browser: Path, args: list[str], timeout: int = 90) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory(prefix="academic-os-chrome-") as profile:
        command = [
            str(browser),
            "--headless=new",
            "--disable-gpu",
            "--disable-background-networking",
            "--disable-background-timer-throttling",
            "--disable-component-update",
            "--disable-dev-shm-usage",
            "--disable-extensions",
            "--disable-features=Translate,MediaRouter,BackForwardCache",
            "--hide-scrollbars",
            "--no-first-run",
            "--no-default-browser-check",
            f"--user-data-dir={profile}",
            *args,
        ]
        try:
            return subprocess.run(
                command,
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=timeout,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired as exc:
            fail(
                "browser command timed out",
                {
                    "command": command,
                    "timeout_seconds": timeout,
                    "stdout_tail": (exc.stdout or "")[-1000:] if isinstance(exc.stdout, str) else "",
                    "stderr_tail": (exc.stderr or "")[-1000:] if isinstance(exc.stderr, str) else "",
                },
            )
            raise AssertionError("unreachable")


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
    parser = argparse.ArgumentParser(description="Verify EvoMind page deep links with real Chromium screenshots.")
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
        fail("EvoMind page deep links are missing expected visible terms", {"missing": missing})

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
