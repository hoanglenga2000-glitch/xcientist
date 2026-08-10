from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_web_isolated_build.py"


def load_verifier(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_build(verifier, tmp_path: Path, monkeypatch, *, audit_ok: bool):
    web = tmp_path / "source-web"
    web.mkdir()
    monkeypatch.setattr(verifier, "WEB", web)
    monkeypatch.setattr(verifier, "port_snapshot", lambda: {"8088": True, "8765": True})
    monkeypatch.setattr(verifier, "npm_command", lambda: ["npm"])

    commands: list[list[str]] = []

    def fake_copy_inputs(stage: Path) -> list[str]:
        (stage / "package-lock.json").write_text("{}\n", encoding="utf-8")
        return ["package-lock.json"]

    def fake_run(command: list[str], *, cwd: Path, timeout: int):
        commands.append(command)
        if command[1] == "audit":
            return {"ok": audit_ok, "returncode": 0 if audit_ok else 1, "seconds": 0.01, "output_tail": ""}
        if command[1:] == ["run", "build"]:
            build = cwd / ".next"
            (build / "standalone").mkdir(parents=True)
            (build / "BUILD_ID").write_text("isolated-build\n", encoding="utf-8")
            (build / "standalone" / "server.js").write_text("// verified\n", encoding="utf-8")
        return {"ok": True, "returncode": 0, "seconds": 0.01, "output_tail": ""}

    monkeypatch.setattr(verifier, "copy_inputs", fake_copy_inputs)
    monkeypatch.setattr(verifier, "run_command", fake_run)
    return commands


def test_isolated_build_requires_clean_full_dependency_audit(tmp_path: Path, monkeypatch) -> None:
    verifier = load_verifier("verify_web_isolated_build_pass")
    commands = configure_build(verifier, tmp_path, monkeypatch, audit_ok=True)

    report = verifier.build(keep_stage=True, npm_ci_timeout=30, build_timeout=30)

    assert report["status"] == "passed"
    assert report["npm_audit"]["ok"] is True
    assert [command[1:] for command in commands] == [
        ["ci", "--prefer-offline", "--no-audit", "--fund=false"],
        ["audit", "--audit-level=low", "--registry=https://registry.npmjs.org"],
        ["run", "db:generate"],
        ["run", "build"],
    ]


def test_isolated_build_fails_closed_before_prisma_or_build_when_audit_is_red(
    tmp_path: Path,
    monkeypatch,
) -> None:
    verifier = load_verifier("verify_web_isolated_build_audit_red")
    commands = configure_build(verifier, tmp_path, monkeypatch, audit_ok=False)

    report = verifier.build(keep_stage=True, npm_ci_timeout=30, build_timeout=30)

    assert report["status"] == "failed"
    assert report["npm_audit"]["ok"] is False
    assert [command[1:] for command in commands] == [
        ["ci", "--prefer-offline", "--no-audit", "--fund=false"],
        ["audit", "--audit-level=low", "--registry=https://registry.npmjs.org"],
    ]
