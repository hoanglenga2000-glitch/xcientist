from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str) -> ModuleType:
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_new_user_readiness_uses_requested_base_url(monkeypatch) -> None:
    module = load_script("verify_new_user_release_readiness")
    seen: list[str] = []

    def fake_http_json(path: str, *, base_url: str, timeout: int = 8) -> dict:
        del timeout
        seen.append(f"{base_url}{path}")
        return {"ok": True, "url": f"{base_url}{path}"}

    monkeypatch.setattr(module, "http_json", fake_http_json)
    checks = module.check_frontend_runtime(
        require_live_server=True,
        base_url="http://127.0.0.1:18089",
    )

    assert len(checks) == 3
    assert seen == [
        "http://127.0.0.1:18089/api/workstation-summary",
        "http://127.0.0.1:18089/api/tasks",
        "http://127.0.0.1:18089/api/settings",
    ]


def test_non_live_readiness_only_allows_connection_failures(monkeypatch) -> None:
    module = load_script("verify_new_user_release_readiness")

    monkeypatch.setattr(
        module,
        "http_json",
        lambda path, *, base_url, timeout=8: {
            "ok": False,
            "status": None,
            "url": f"{base_url}{path}",
            "error": "localized connection failure",
            "connection_error": True,
        },
    )
    offline = module.check_frontend_runtime(
        require_live_server=False,
        base_url="http://127.0.0.1:18089",
    )
    assert all(item["ok"] and item["optional"] for item in offline)

    monkeypatch.setattr(
        module,
        "http_json",
        lambda path, *, base_url, timeout=8: {
            "ok": False,
            "status": 500,
            "url": f"{base_url}{path}",
            "error": "HTTP 500",
        },
    )
    unhealthy = module.check_frontend_runtime(
        require_live_server=False,
        base_url="http://127.0.0.1:18089",
    )
    assert all(not item["ok"] and not item.get("optional", False) for item in unhealthy)


def test_launch_readiness_forwards_runtime_base_url() -> None:
    module = load_script("verify_workstation_launch_readiness")
    base_url = "http://127.0.0.1:18089"
    by_id = {item["id"]: item for item in module.COMMANDS}

    for command_id, flag in module.RUNTIME_BASE_URL_FLAGS.items():
        original = by_id[command_id]
        resolved = module.with_runtime_base_url(original, base_url)
        assert resolved["cmd"][-2:] == [flag, base_url] or command_id == "server_health"
        assert base_url not in original["cmd"]

    server = module.with_runtime_base_url(by_id["server_health"], base_url)
    assert server["cmd"][-4:] == ["--base-url", base_url, "--port", "18089"]


def test_backend_connector_contract_distinguishes_kaggle_auth_from_configuration() -> None:
    module = load_script("verify_backend_resource_status")

    assert module.configured_state_is_acceptable("kaggle", "configured_unverified") is True
    assert module.configured_state_is_acceptable("kaggle", "authenticated") is True
    assert module.configured_state_is_acceptable("kaggle", "failed") is False
    assert module.configured_state_is_acceptable("gpu", "configured_unverified") is False


def test_dashboard_source_digest_changes_with_build_input(tmp_path) -> None:
    module = load_script("manage_workstation_dashboard")
    app = tmp_path / "web"
    (app / "src").mkdir(parents=True)
    source = app / "src" / "page.ts"
    source.write_text("export const value = 1;\n", encoding="utf-8")

    before = module.source_tree_digest(app)
    source.write_text("export const value = 2;\n", encoding="utf-8")

    assert module.source_tree_digest(app) != before


def test_release_acceptance_threads_port_to_every_live_check() -> None:
    acceptance = (ROOT / "scripts" / "run_new_user_release_acceptance.ps1").read_text(encoding="utf-8-sig")
    restart = (ROOT / "scripts" / "restart_workstation_frontend.ps1").read_text(encoding="utf-8-sig")

    assert '$BaseUrl = "http://127.0.0.1:$Port"' in acceptance
    for script_name in (
        "verify_new_user_release_readiness.py",
        "verify_workstation_launch_readiness.py",
        "verify_workstation_browser_render_smoke.py",
        "verify_workstation_click_smoke.mjs",
        "verify_workstation_interactive_controls.mjs",
    ):
        invocation_lines = [
            line
            for line in acceptance.splitlines()
            if script_name in line and "--base-url" in line
        ]
        assert invocation_lines
        assert "--base-url $BaseUrl" in invocation_lines[0]

    assert '"--port",' in restart
    assert "[string]$Port" in restart
    assert '"node_modules\\next\\dist\\bin\\next"' in restart
    assert '[ValidateSet("development", "production")]' in restart
    assert '$nextCommand = if ($Mode -eq "production") { "start" } else { "dev" }' in restart
    assert '-Port $Port -Mode production' in acceptance
    assert 'Run-Check "start_production_workstation_frontend"' in acceptance
    assert '"--build"' in (ROOT / "scripts" / "start_verified_workstation.ps1").read_text(encoding="utf-8-sig")


def test_release_acceptance_uses_selected_python_and_initializes_database() -> None:
    acceptance = (ROOT / "scripts" / "run_new_user_release_acceptance.ps1").read_text(encoding="utf-8-sig")

    assert '[string]$PythonExecutable = ""' in acceptance
    assert '$PythonExe = (Get-Command python -ErrorAction Stop).Source' in acceptance
    assert '& $PythonExe -m pytest' in acceptance
    assert '& $PythonExe scripts\\verify_no_plaintext_secrets.py' in acceptance

    assert 'Join-Path $Root "install.ps1"' in acceptance
    installer_block = acceptance[
        acceptance.index('Run-Check "installer_smoke_no_secrets"'):
        acceptance.index('Run-Check "cli_tests"')
    ]
    assert "-SkipNpmInstall" not in installer_block
    assert '"-SkipBuild"' in installer_block
    assert "Start-Process" in installer_block
    assert "-RedirectStandardError $stderr" in installer_block
    assert "installer.ExitCode" in installer_block
    assert 'Run-Check "stop_existing_workstation_frontend"' in acceptance
    assert acceptance.index('Run-Check "stop_existing_workstation_frontend"') < acceptance.index(
        'Run-Check "installer_smoke_no_secrets"'
    )


def test_prisma_db_push_forwards_cli_arguments() -> None:
    source = (
        ROOT / "web" / "research-agent-workstation" / "scripts" / "prisma-db-push.mjs"
    ).read_text(encoding="utf-8-sig")

    assert 'runPrisma(["db", "push", ...process.argv.slice(2)])' in source


def test_security_runtime_initializes_unique_database_before_server_start() -> None:
    source = (
        ROOT
        / "web"
        / "research-agent-workstation"
        / "scripts"
        / "verify-production-security.ps1"
    ).read_text(encoding="utf-8-sig")

    assert '$DatabaseName = "ci-workstation-$PID.db"' in source
    assert '$env:DATABASE_URL = "file:./$DatabaseName"' in source
    assert source.index('& node $PrismaPush "--skip-generate"') < source.index("$server = Start-Process")
    for suffix in ("", "-journal", "-shm", "-wal"):
        assert f'$DatabasePath{suffix}' in source


def test_installers_initialize_database_and_generate_client_before_frontend_build() -> None:
    installers = (
        ROOT / "install.ps1",
        ROOT / "scripts" / "quick_setup.ps1",
        ROOT / "scripts" / "quick_setup.sh",
    )

    for installer in installers:
        source = installer.read_text(encoding="utf-8-sig")
        assert (
            source.index("npm run db:push")
            < source.index("npm run db:generate")
            < source.index("npm run build")
        )


def test_release_archives_keep_shell_scripts_lf() -> None:
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines()

    assert "*.sh text eol=lf" in attributes


def test_cdp_smokes_support_node20_without_a_global_websocket() -> None:
    package = json.loads(
        (ROOT / "web" / "research-agent-workstation" / "package.json").read_text(encoding="utf-8")
    )
    assert package["dependencies"]["ws"] == "8.21.0"

    for script_name in (
        "verify_workstation_click_smoke.mjs",
        "verify_workstation_interactive_controls.mjs",
    ):
        source = (ROOT / "scripts" / script_name).read_text(encoding="utf-8-sig")
        assert 'createRequire(new URL("../web/research-agent-workstation/package.json"' in source
        assert 'globalThis.WebSocket ?? requireFromWeb("ws")' in source
        assert "new WebSocketClient(this.wsUrl)" in source


def test_tasks_agent_logs_action_is_rendered_and_routed() -> None:
    screens = (
        ROOT / "web" / "research-agent-workstation" / "src" / "components" / "workstation" / "Screens.tsx"
    ).read_text(encoding="utf-8-sig")
    shell = (
        ROOT / "web" / "research-agent-workstation" / "src" / "components" / "workstation" / "AppShell.tsx"
    ).read_text(encoding="utf-8-sig")

    assert 'data-ui-action="tasks_view_agent_logs"' in screens
    assert 'tasks_view_agent_logs: { page: "runtime"' in shell


def test_release_sbom_environment_excludes_packaging_tools() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8-sig")

    assert "python -m venv --without-pip .release-sbom" in workflow
    assert "python -m pip --python .\\.release-sbom\\Scripts\\python.exe install $wheel.FullName" in workflow
    assert "python -m pip --python .\\.release-sbom\\Scripts\\python.exe check" in workflow
    assert "python -m pip --python .\\.release-sbom\\Scripts\\python.exe uninstall --yes xcientist" in workflow
    assert ".\\.release-sbom\\Scripts\\python.exe -m pip install --upgrade" not in workflow
