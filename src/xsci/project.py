"""`xsci init` — scaffold a research project in the current directory.

Creates ``.xsci/`` with a project-level config, an ``experiments/`` output dir,
and ensures the project ``.gitignore`` refuses to commit any local secret. This
is deliberately conservative: it never touches the global secret store and never
overwrites an existing project config unless ``--force`` is passed.
"""
from __future__ import annotations

from pathlib import Path

from .config import PROJECT_DIRNAME

# Patterns we guarantee are git-ignored inside a project so a stray key or a
# fetched dataset never lands in version control.
_GITIGNORE_BLOCK = [
    "# ── added by `xsci init` — never commit local secrets or heavy data ──",
    ".xsci/secrets*",
    ".xsci/*.local.*",
    "experiments/",
    "data/",
    "*.kaggle.json",
]

_PROJECT_CONFIG_TEMPLATE = """# xsci project config — non-secret settings only.
# Secrets use the global platform-protected user store, never this project.

[project]
name = "{name}"

[compute]
# Release builds require the gated remote HPC/GPU runtime.
backend = "{compute}"

[run]
# defaults for `xsci run` in this project
iterations = 20
metric_direction = "maximize"
"""


def _ensure_gitignore(root: Path) -> str:
    gi = root / ".gitignore"
    marker = _GITIGNORE_BLOCK[0]
    if gi.exists():
        current = gi.read_text(encoding="utf-8")
        if marker in current:
            return "gitignore: already guarded"
        sep = "" if current.endswith("\n") else "\n"
        gi.write_text(current + sep + "\n" + "\n".join(_GITIGNORE_BLOCK) + "\n", encoding="utf-8")
        return "gitignore: appended secret/data guards"
    gi.write_text("\n".join(_GITIGNORE_BLOCK) + "\n", encoding="utf-8")
    return "gitignore: created with secret/data guards"


def run_init(root: Path | None = None, *, compute: str = "gpu", force: bool = False) -> int:
    from research_os.hpc_policy import require_hpc_compute

    require_hpc_compute(compute)
    root = (root or Path.cwd()).resolve()
    xdir = root / PROJECT_DIRNAME
    cfg_path = xdir / "config.toml"

    if cfg_path.exists() and not force:
        print(f"project already initialized at {xdir} (use --force to overwrite)")
        return 1

    xdir.mkdir(parents=True, exist_ok=True)
    (root / "experiments").mkdir(exist_ok=True)
    (xdir / "tasks").mkdir(exist_ok=True)  # per-task configs land here (phase 3)

    cfg_path.write_text(
        _PROJECT_CONFIG_TEMPLATE.format(name=root.name, compute=compute),
        encoding="utf-8",
    )
    gi_msg = _ensure_gitignore(root)

    print(f"initialized xsci project at {root}")
    print(f"  - {PROJECT_DIRNAME}/config.toml   (project settings, compute={compute})")
    print(f"  - {PROJECT_DIRNAME}/tasks/         (task configs - add with `xsci task add`)")
    print("  - experiments/          (run outputs)")
    print(f"  - {gi_msg}")
    print("\nnext: `xsci login` to set your LLM / Kaggle keys, then `xsci doctor`.")
    return 0
