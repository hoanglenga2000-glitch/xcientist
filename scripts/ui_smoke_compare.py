"""Compare candidate web overlay UI shell against live 514 baseline."""
from pathlib import Path
import hashlib, json, sys

def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main(baseline: Path, candidate: Path, out: Path) -> int:
    checks = []
    page_b, page_c = baseline / "src/app/page.tsx", candidate / "src/app/page.tsx"
    home_b, home_c = baseline / "src/app/home-client.tsx", candidate / "src/app/home-client.tsx"
    login_b, login_c = baseline / "src/app/login/page.tsx", candidate / "src/app/login/page.tsx"
    ws = candidate / "src/app/workspace/page.tsx"
    def check(label, path, must_contain=None, must_not=None):
        ok = path.is_file()
        text = path.read_text(encoding="utf-8", errors="replace") if ok else ""
        if must_contain and must_contain not in text: ok = False
        if must_not and must_not in text: ok = False
        checks.append({"id": label, "ok": ok, "sha256": sha(path) if path.is_file() else None})
    check("baseline_page_AppShell", page_b, "AppShell", "TaskWorkspace")
    check("candidate_page_AppShell", page_c, "AppShell", "TaskWorkspace")
    same_page = page_b.is_file() and page_c.is_file() and sha(page_b) == sha(page_c)
    checks.append({"id": "page_tsx_identical_to_baseline", "ok": same_page})
    same_home = home_b.is_file() and home_c.is_file() and sha(home_b) == sha(home_c)
    checks.append({"id": "home_client_identical_to_baseline", "ok": same_home})
    if login_b.is_file() and login_c.is_file():
        checks.append({"id": "login_identical_to_baseline", "ok": sha(login_b) == sha(login_c)})
    checks.append({"id": "no_workspace_page", "ok": not ws.exists()})
    ok_all = all(c["ok"] for c in checks)
    result = {"status": "passed" if ok_all else "failed", "checks": checks}
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if ok_all else 1

if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])))
