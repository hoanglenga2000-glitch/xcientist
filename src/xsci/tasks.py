"""`xsci task` — register and resolve research task configs.

A task config is the same small JSON the engine already understands (task_name,
modality, metric, data_schema, ...). `task add` copies/creates one under the
project's ``.xsci/tasks/<slug>.json``; `task list` enumerates them; `resolve`
loads one by slug for `xsci run`.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Optional

from .config import PROJECT_DIRNAME, find_project_dir

_SLUG_RE = re.compile(r"[^a-z0-9]+")

_TEMPLATE = {
    "task_name": "",
    "modality": "tabular",
    "task_type": "classification",
    "metric": "accuracy",
    "metric_direction": "maximize",
    "target_column": "",
    "id_column": "",
    "n_train": 0,
    "n_test": 0,
    "data_schema": "TODO: describe columns / files the agent will see.",
    "extra_notes": "TODO: task-specific guidance for the agent.",
    "local_data_dir": "",
    "remote_data_dirname": "",
}


def slugify(name: str) -> str:
    return _SLUG_RE.sub("-", name.strip().lower()).strip("-") or "task"


def tasks_dir(project_root: Optional[Path] = None) -> Path:
    root = project_root or find_project_dir()
    if root is None:
        raise FileNotFoundError("not inside an xsci project - run `xsci init` first")
    return root / PROJECT_DIRNAME / "tasks"


def _slug_from_kaggle_url(url: str) -> Optional[str]:
    m = re.search(r"kaggle\.com/(?:c/|competitions/)([^/?#]+)", url)
    return m.group(1) if m else None


def add_task(source: str, *, project_root: Optional[Path] = None, force: bool = False) -> Path:
    """Register a task from a JSON file path, a Kaggle URL, or a bare name.

    - existing .json file  -> validated + copied in
    - kaggle.com/c/<slug>  -> template scaffold seeded with the competition slug
    - bare name            -> template scaffold
    """
    tdir = tasks_dir(project_root)
    tdir.mkdir(parents=True, exist_ok=True)

    is_url = source.startswith(("http://", "https://"))
    src_path = Path(source).expanduser()
    # A URL is never a file path even though it contains "/".
    looks_like_path = not is_url and (
        src_path.suffix == ".json" or any(sep in source for sep in ("/", "\\"))
    )

    if looks_like_path:
        # The user clearly meant a file. If it isn't there, say so — never fall
        # through and scaffold a task named after the (broken) path.
        if not src_path.exists():
            raise FileNotFoundError(f"task file not found: {source}")
        data = json.loads(src_path.read_text(encoding="utf-8"))
        if "task_name" not in data:
            raise ValueError(f"{source}: missing required 'task_name'")
        slug = slugify(data["task_name"])
        dest = tdir / f"{slug}.json"
        if dest.exists() and not force:
            raise FileExistsError(f"task '{slug}' already exists (use --force)")
        shutil.copyfile(src_path, dest)
        return dest

    kaggle_slug = _slug_from_kaggle_url(source)
    name = kaggle_slug or source
    slug = slugify(name)
    dest = tdir / f"{slug}.json"
    if dest.exists() and not force:
        raise FileExistsError(f"task '{slug}' already exists (use --force)")

    scaffold = dict(_TEMPLATE)
    scaffold["task_name"] = name
    if kaggle_slug:
        scaffold["remote_data_dirname"] = kaggle_slug
        scaffold["extra_notes"] = f"Kaggle competition '{kaggle_slug}'. Fill in schema/metric before running."
    dest.write_text(json.dumps(scaffold, ensure_ascii=False, indent=2), encoding="utf-8")
    return dest


def list_tasks(project_root: Optional[Path] = None) -> list[tuple[str, Path]]:
    try:
        tdir = tasks_dir(project_root)
    except FileNotFoundError:
        return []
    return sorted((p.stem, p) for p in tdir.glob("*.json"))


def resolve_task(slug_or_path: str, project_root: Optional[Path] = None) -> Path:
    """Resolve a task by slug (in the project) or by direct path."""
    p = Path(slug_or_path)
    if p.suffix == ".json" and p.exists():
        return p
    tdir = tasks_dir(project_root)
    candidate = tdir / f"{slugify(slug_or_path)}.json"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"no task '{slug_or_path}' - see `xsci task list`")
