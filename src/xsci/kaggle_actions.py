"""Kaggle API actions: competition discovery, download, and task registration.

These are the real I/O operations the conversation agent can invoke — the
equivalent of Claude Code's tool layer. Every action returns a structured
result that the agent can present to the user or use as context for the
next decision.
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .config import Config, load_config, inject_engine_env


@dataclass
class Competition:
    slug: str
    title: str = ""
    description: str = ""
    url: str = ""
    category: str = ""
    reward: str = ""
    deadline: str = ""
    team_count: int = 0
    evaluation_metric: str = ""

    @classmethod
    def from_api_dict(cls, d: dict) -> "Competition":
        return cls(
            slug=d.get("ref", ""),
            title=d.get("title", d.get("ref", "")),
            description=(d.get("description", "") or "")[:300],
            url=f"https://www.kaggle.com/competitions/{d.get('ref', '')}",
            category=d.get("category", ""),
            reward=d.get("reward", ""),
            deadline=d.get("deadline", ""),
            team_count=int(d.get("teamCount", 0) or 0),
            evaluation_metric=d.get("evaluationMetric", ""),
        )


def _kaggle_api_auth(cfg: Optional[Config] = None) -> bool:
    """Ensure Kaggle API credentials are in the environment."""
    cfg = cfg or load_config()
    inject_engine_env(cfg)
    return bool(
        sys.modules.get("os") and __import__("os").environ.get("KAGGLE_API_TOKEN")
        or __import__("os").environ.get("KAGGLE_USERNAME")
    )


def _call_kaggle_python(code: str, timeout: int = 30) -> tuple[int, str]:
    """Run a one-shot Python snippet with the Kaggle API, return (exit_code, stdout)."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            timeout=timeout,
            text=False,
        )
        stdout = proc.stdout.decode("utf-8", errors="replace")
        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="replace")[:500]
            return proc.returncode, stderr or stdout
        return 0, stdout
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return 1, str(exc)


def list_competitions(
    query: str = "",
    sort_by: str = "latestDeadline",
    page: int = 1,
    category: str = "all",
    cfg: Optional[Config] = None,
) -> dict:
    """Fetch and optionally filter active Kaggle competitions.

    Returns a dict with {'ok': bool, 'competitions': [...], 'total': int, 'message': str}.
    When query is non-empty, filters locally by title/slug/tags.
    """
    if not _kaggle_api_auth(cfg):
        return {
            "ok": False,
            "competitions": [],
            "total": 0,
            "message": "Kaggle API is not configured. Run `evomind setup` to add your token, "
                       "or `kaggle official config view` to check the current configuration.",
        }

    code = textwrap.dedent(f"""
        import json, os, traceback
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            result = api.competitions_list(
                page={page},
                search="{query}",
                sort_by="{sort_by}",
                category="{category}" if "{category}" != "all" else None,
            )
            comps = []
            if hasattr(result, 'competitions'):
                comps = result.competitions or result
            elif isinstance(result, list):
                comps = result

            # Kaggle api returns objects with __dict__ or named fields
            out = []
            for c in (comps or []):
                if hasattr(c, '__dict__'):
                    d = {{k: v for k, v in c.__dict__.items() if not k.startswith('_')}}
                elif isinstance(c, dict):
                    d = c
                else:
                    continue
                # Normalize common field names
                entry = {{
                    "ref": d.get("ref", d.get("id", "")),
                    "title": d.get("title", d.get("title", "")),
                    "description": (d.get("description", "") or "")[:200],
                    "category": d.get("category", ""),
                    "reward": d.get("reward", ""),
                    "deadline": d.get("deadline", ""),
                    "teamCount": d.get("teamCount", d.get("team_count", 0)),
                    "evaluationMetric": d.get("evaluationMetric", d.get("evaluation_metric", "")),
                }}
                out.append(entry)
            print(json.dumps({{"ok": True, "total": len(out), "competitions": out}}, ensure_ascii=False))
        except Exception as exc:
            print(json.dumps({{
                "ok": False,
                "total": 0,
                "competitions": [],
                "message": f"Kaggle API error: {{exc}}",
            }}, ensure_ascii=False))
    """)
    exit_code, stdout = _call_kaggle_python(code, timeout=30)
    if exit_code != 0:
        return {"ok": False, "competitions": [], "total": 0, "message": stdout[:500]}
    try:
        return json.loads(stdout.strip().splitlines()[-1])
    except json.JSONDecodeError:
        return {"ok": False, "competitions": [], "total": 0, "message": "Failed to parse Kaggle API response"}


def download_competition_data(slug: str, force: bool = False, cfg: Optional[Config] = None) -> dict:
    """Download the data for a Kaggle competition.

    Returns {'ok': bool, 'data_dir': str, 'files': [...], 'message': str}.
    Files go to datasets/kaggle/<slug>/.
    """
    if not _kaggle_api_auth(cfg):
        return {"ok": False, "data_dir": "", "files": [], "message": "Kaggle API not configured."}

    # Resolve the project root for the data directory
    from .config import active_root
    root = active_root()
    data_dir = root / "datasets" / "kaggle" / slug
    data_dir.mkdir(parents=True, exist_ok=True)

    if not force and any(data_dir.iterdir()):
        existing = [p.name for p in data_dir.iterdir() if p.is_file()]
        return {
            "ok": True,
            "data_dir": str(data_dir),
            "files": existing,
            "message": f"Data already exists ({len(existing)} files). Use force=True to re-download.",
        }

    code = textwrap.dedent(f"""
        import json, os, sys, zipfile, io
        from pathlib import Path
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()

            # Use the Kaggle API to download competition files
            target = Path(r"{data_dir}")
            target.mkdir(parents=True, exist_ok=True)

            # Download files through the API
            api.competition_download_files("{slug}", path=str(target), quiet=False)

            # Find zip files and extract them
            files = []
            for p in sorted(target.iterdir()):
                if p.is_file():
                    files.append(p.name)
                    if p.suffix == '.zip':
                        with zipfile.ZipFile(p, 'r') as zf:
                            zf.extractall(target)
                        p.unlink()  # remove zip after extraction

            # Refresh file list after extraction
            final_files = sorted([p.name for p in target.iterdir() if p.is_file()])
            print(json.dumps({{
                "ok": True,
                "data_dir": str(target),
                "files": final_files,
                "message": f"Downloaded {{len(final_files)}} file(s) for {slug}",
            }}, ensure_ascii=False))
        except Exception as exc:
            print(json.dumps({{
                "ok": False,
                "data_dir": str(target) if 'target' in dir() else "",
                "files": [],
                "message": f"Download error: {{exc}}",
            }}, ensure_ascii=False))
    """)
    exit_code, stdout = _call_kaggle_python(code, timeout=120)
    if exit_code != 0:
        return {"ok": False, "data_dir": str(data_dir), "files": [], "message": stdout[:500]}
    try:
        return json.loads(stdout.strip().splitlines()[-1])
    except json.JSONDecodeError:
        return {"ok": False, "data_dir": str(data_dir), "files": [], "message": "Download output parse error"}


def register_from_url(url: str, root: Optional[Path] = None, force: bool = False) -> dict:
    """Register a competition from a Kaggle URL or slug.

    Returns {'ok': bool, 'slug': str, 'message': str}.
    """
    from .kaggle import _register_task

    slug = _register_task(url, root, force=force)
    if slug:
        return {"ok": True, "slug": slug, "message": f"Registered competition: {slug}"}
    return {"ok": False, "slug": "", "message": f"Failed to register: {url}"}


def quick_start(slug_or_url: str, root: Optional[Path] = None) -> dict:
    """One-shot: register, download data, and verify readiness.

    Returns {'ok': bool, 'slug': str, 'data_dir': str, 'files': [...], 'message': str, 'ready_to_train': bool}.
    """
    # Step 1: Register
    reg = register_from_url(slug_or_url, root)
    if not reg["ok"]:
        return {**reg, "data_dir": "", "files": [], "ready_to_train": False}

    slug = reg["slug"]

    # Step 2: Download data
    dl = download_competition_data(slug)
    if not dl["ok"]:
        return {
            "ok": False,
            "slug": slug,
            "data_dir": dl["data_dir"],
            "files": [],
            "message": f"Registered {slug} but download failed: {dl['message']}",
            "ready_to_train": False,
        }

    return {
        "ok": True,
        "slug": slug,
        "data_dir": dl["data_dir"],
        "files": dl["files"],
        "message": f"Ready: {slug} ({len(dl['files'])} files downloaded). Type `kaggle run {slug}` or just describe your goal.",
        "ready_to_train": True,
    }
