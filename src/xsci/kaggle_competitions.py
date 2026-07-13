
"""kaggle-competitions -- standalone Kaggle competition browser."""
import json, os, sys, textwrap, subprocess

def list_competitions(query="", category="all", page=1):
    code = textwrap.dedent(f"""
import json
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    result = api.competitions_list(page={page}, search="{query}", sort_by="latestDeadline")
    comps = result.competitions if hasattr(result, "competitions") else (result if isinstance(result, list) else [])
    out = []
    for c in (comps or []):
        d = c.__dict__ if hasattr(c, "__dict__") else (c if isinstance(c, dict) else {{}})
        entry = {{k: v for k, v in d.items() if not k.startswith("_") and not callable(v)}}
        out.append(entry)
    print(json.dumps({{"ok": True, "total": len(out), "competitions": out}}, ensure_ascii=False))
except Exception as exc:
    print(json.dumps({{"ok": False, "total": 0, "competitions": [], "message": str(exc)}}, ensure_ascii=False))
    """)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=30, text=False)
    stdout = r.stdout.decode("utf-8", errors="replace")
    if r.returncode != 0:
        return {{"ok": False, "message": (r.stderr.decode("utf-8", errors="replace") or stdout)[:500]}}
    try:
        return json.loads(stdout.strip().splitlines()[-1])
    except json.JSONDecodeError:
        return {{"ok": False, "message": "parse error"}}
