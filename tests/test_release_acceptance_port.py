from __future__ import annotations

import importlib.util
import io
import json
import urllib.error
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

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


def test_dashboard_source_digest_ignores_sqlite_runtime_files(tmp_path) -> None:
    module = load_script("manage_workstation_dashboard")
    app = tmp_path / "web"
    prisma = app / "prisma"
    prisma.mkdir(parents=True)
    (prisma / "schema.prisma").write_text("datasource db {}\n", encoding="utf-8")
    before = module.source_tree_digest(app)

    for suffix in (".db", ".db-journal", ".db-shm", ".db-wal"):
        (prisma / f"workstation{suffix}").write_bytes(b"runtime mutation")

    assert module.source_tree_digest(app) == before


def test_dashboard_environment_pins_database_and_sets_python(monkeypatch, tmp_path) -> None:
    module = load_script("manage_workstation_dashboard")
    database = tmp_path / "prisma" / "workstation.db"
    monkeypatch.setattr(module, "DEFAULT_DATABASE_PATH", database)
    monkeypatch.setenv("DATABASE_URL", "file:C:/unrelated-project.db")
    monkeypatch.delenv("WORKSTATION_PYTHON", raising=False)

    environment = module.dashboard_env()

    assert environment["DATABASE_URL"] == f"file:{database.as_posix()}"
    assert environment["WORKSTATION_ROOT"] == str(module.ROOT)
    assert environment["WORKSTATION_PYTHON"] == module.sys.executable


def test_dashboard_manager_initializes_database_with_runtime_environment(monkeypatch, tmp_path) -> None:
    module = load_script("manage_workstation_dashboard")
    app = tmp_path / "web"
    push_script = app / "scripts" / "prisma-db-push.mjs"
    push_script.parent.mkdir(parents=True)
    push_script.write_text("// fixture\n", encoding="utf-8")
    environment = {"DATABASE_URL": "file:fixture.db"}
    calls: list[dict] = []

    def fake_run(command, **kwargs):
        calls.append({"command": command, **kwargs})
        return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(module, "APP_DIR", app)
    monkeypatch.setattr(module, "PRISMA_PUSH_SCRIPT", push_script)
    monkeypatch.setattr(module, "node_command", lambda: "node")
    monkeypatch.setattr(module, "npm_command", lambda: "npm")
    monkeypatch.setattr(module.subprocess, "run", fake_run)

    assert module.ensure_database_schema(environment) == "synced"
    assert calls[0]["command"] == ["node", str(push_script), "--skip-generate"]
    assert calls[1]["command"] == ["npm", "run", "db:generate"]
    assert calls[0]["cwd"] == app
    assert calls[1]["cwd"] == app
    assert calls[0]["env"] is environment
    assert calls[1]["env"] is environment


def test_dashboard_manager_initializes_database_before_build_and_server_start() -> None:
    source = (ROOT / "scripts" / "manage_workstation_dashboard.py").read_text(encoding="utf-8-sig")

    schema_sync = source.index("database_schema_status = ensure_database_schema(environment)")
    build = source.index("if args.build:", schema_sync)
    server = source.index("process = subprocess.Popen(", build)
    assert schema_sync < build < server
    assert "except SystemExit as readiness_error:" in source
    assert "if stop_pid(process.pid):" in source
    assert "runtime metadata was preserved" in source
    assert "stop(args, emit=False)" in source


def test_dashboard_manager_does_not_treat_kill_error_as_success(monkeypatch) -> None:
    module = load_script("manage_workstation_dashboard")

    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError()))
    monkeypatch.setattr(module, "pid_running", lambda pid: True)

    assert module.stop_pid(7171, timeout=0.1) is False


def test_dashboard_manager_requires_exact_runtime_process_identity(monkeypatch) -> None:
    module = load_script("manage_workstation_dashboard")
    next_cli = Path("C:/fixture/workstation/node_modules/next/dist/bin/next")

    monkeypatch.setattr(module, "next_cli_path", lambda: str(next_cli))
    monkeypatch.setattr(
        module,
        "process_command_line",
        lambda pid: f'node.exe "{next_cli}" start --hostname 127.0.0.1 --port 8088' if pid == 42 else None,
    )

    assert module.process_matches_dashboard(42, 8088) is True
    assert module.process_matches_dashboard(42, 8089) is False
    assert module.process_matches_dashboard(99, 8088) is False


def test_dashboard_manager_refuses_to_stop_unowned_port_listener(monkeypatch) -> None:
    module = load_script("manage_workstation_dashboard")
    stopped: list[int] = []

    monkeypatch.setattr(module, "port_processes", lambda port, state=None: ([], [4242]))
    monkeypatch.setattr(module, "stop_pid", lambda pid: stopped.append(pid) or True)

    with pytest.raises(SystemExit) as error:
        module.stop_port(8088, {})

    payload = json.loads(str(error.value))
    assert payload["stage"] == "port_ownership"
    assert payload["evidence"] == {"port": 8088, "unowned_pids": [4242]}
    assert stopped == []


def test_dashboard_manager_fails_when_owned_listener_cannot_be_stopped(monkeypatch) -> None:
    module = load_script("manage_workstation_dashboard")

    monkeypatch.setattr(module, "port_processes", lambda port, state=None: ([4242, 4343], []))
    monkeypatch.setattr(module, "stop_pid", lambda pid: pid == 4242)

    with pytest.raises(SystemExit) as error:
        module.stop_port(8088, {})

    payload = json.loads(str(error.value))
    assert payload["stage"] == "port_cleanup"
    assert payload["evidence"] == {"port": 8088, "failed_pids": [4343]}


def test_dashboard_manager_start_refuses_unowned_listener_before_any_kill(monkeypatch, tmp_path) -> None:
    module = load_script("manage_workstation_dashboard")
    stopped: list[int] = []

    monkeypatch.setattr(module, "RUNTIME_DIR", tmp_path)
    monkeypatch.setattr(module, "read_runtime_state", lambda: {})
    monkeypatch.setattr(module, "fetch_status", lambda port, timeout=2: None)
    monkeypatch.setattr(module, "read_pid", lambda: None)
    monkeypatch.setattr(module, "port_processes", lambda port, state=None: ([], [5151]))
    monkeypatch.setattr(module, "stop_pid", lambda pid: stopped.append(pid) or True)

    args = SimpleNamespace(port=8088, force=True, build=False, timeout=1)
    with pytest.raises(SystemExit) as error:
        module.start(args)

    payload = json.loads(str(error.value))
    assert payload["stage"] == "port_ownership"
    assert payload["evidence"]["unowned_pids"] == [5151]
    assert stopped == []


def test_dashboard_manager_start_preserves_unverified_running_pid_metadata(monkeypatch, tmp_path) -> None:
    module = load_script("manage_workstation_dashboard")
    state = {"schema": "evomind.dashboard_runtime.v1", "pid": 6363, "port": 8088}

    monkeypatch.setattr(module, "RUNTIME_DIR", tmp_path)
    monkeypatch.setattr(module, "read_runtime_state", lambda: state)
    monkeypatch.setattr(module, "fetch_status", lambda port, timeout=2: None)
    monkeypatch.setattr(module, "read_pid", lambda: 6363)
    monkeypatch.setattr(module, "port_processes", lambda port, runtime_state=None: ([], []))
    monkeypatch.setattr(module, "pid_running", lambda pid: True)
    monkeypatch.setattr(module, "runtime_state_matches_process", lambda pid, port, runtime_state=None: False)

    with pytest.raises(SystemExit) as error:
        module.start(SimpleNamespace(port=8088, force=True, build=False, timeout=1))

    assert json.loads(str(error.value))["stage"] == "pid_ownership"


def test_dashboard_manager_removes_stale_pid_metadata_without_killing_reused_pid(monkeypatch, tmp_path) -> None:
    module = load_script("manage_workstation_dashboard")
    pid_file = tmp_path / "dashboard.pid"
    state_file = tmp_path / "dashboard.state.json"
    pid_file.write_text("6161", encoding="utf-8")
    state_file.write_text('{"schema":"evomind.dashboard_runtime.v1","pid":6161,"port":8088}', encoding="utf-8")
    stopped: list[int] = []

    monkeypatch.setattr(module, "PID_FILE", pid_file)
    monkeypatch.setattr(module, "STATE_FILE", state_file)
    monkeypatch.setattr(module, "port_processes", lambda port, state=None: ([], []))
    monkeypatch.setattr(module, "runtime_state_matches_process", lambda pid, port, state=None: False)
    monkeypatch.setattr(module, "pid_running", lambda pid: False)
    monkeypatch.setattr(module, "stop_pid", lambda pid: stopped.append(pid) or True)

    module.stop(SimpleNamespace(port=8088, force=True), emit=False)

    assert stopped == []
    assert not pid_file.exists()
    assert not state_file.exists()


def test_dashboard_manager_preserves_metadata_for_unverified_running_pid(monkeypatch, tmp_path) -> None:
    module = load_script("manage_workstation_dashboard")
    pid_file = tmp_path / "dashboard.pid"
    state_file = tmp_path / "dashboard.state.json"
    pid_file.write_text("6262", encoding="utf-8")
    state_file.write_text('{"schema":"evomind.dashboard_runtime.v1","pid":6262,"port":8088}', encoding="utf-8")

    monkeypatch.setattr(module, "PID_FILE", pid_file)
    monkeypatch.setattr(module, "STATE_FILE", state_file)
    monkeypatch.setattr(module, "port_processes", lambda port, state=None: ([], []))
    monkeypatch.setattr(module, "runtime_state_matches_process", lambda pid, port, state=None: False)
    monkeypatch.setattr(module, "pid_running", lambda pid: True)

    with pytest.raises(SystemExit) as error:
        module.stop(SimpleNamespace(port=8088, force=True), emit=False)

    assert json.loads(str(error.value))["stage"] == "pid_ownership"
    assert pid_file.exists()
    assert state_file.exists()


def test_verified_launch_runtime_reports_do_not_dirty_the_worktree() -> None:
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8-sig").splitlines()

    assert "docs/verified_workstation_launch_audit.json" in ignored
    assert "docs/verified_workstation_launch_audit.md" in ignored


def test_verified_launcher_refreshes_kaggle_readiness_before_backend_status() -> None:
    launcher = (ROOT / "scripts" / "start_verified_workstation.ps1").read_text(encoding="utf-8-sig")

    kaggle_args = launcher.index("$kaggleReadinessArgs = @(")
    kaggle_readiness = launcher.index('-Label "kaggle_dpapi_readiness"', kaggle_args)
    backend_status = launcher.index('-Label "backend_resource_status"')
    assert kaggle_args < kaggle_readiness < backend_status
    assert '"--write-report"' in launcher[kaggle_args:kaggle_readiness]
    assert '$kaggleReadinessArgs += "--allow-real-external"' in launcher[kaggle_args:kaggle_readiness]
    kaggle_smoke = launcher.index("$kaggleArgs = @(", backend_status)
    plaintext_scan = launcher.index('-Label "plaintext_secret_scan"', kaggle_smoke)
    assert "-AllowRealExternal" not in launcher[kaggle_smoke:plaintext_scan]


def test_backend_summary_exposes_required_local_connector_contract() -> None:
    summary = (ROOT / "web" / "research-agent-workstation" / "src" / "lib" / "server" / "summary.ts").read_text(
        encoding="utf-8-sig"
    )

    for provider, state in (("llm", "rule_based"), ("storage", "local_workspace")):
        provider_index = summary.index(f'"{provider}"')
        state_index = summary.index(f'state: "{state}"', provider_index)
        configured_index = summary.index("configured: true", state_index)
        assert provider_index < state_index < configured_index
    python_index = summary.index('"python_runner"')
    assert 'state: workstationPythonConfigured ? "local" : "not_configured"' in summary[python_index:]
    assert "configured: workstationPythonConfigured" in summary[python_index:]


def test_external_gateway_preserves_structured_http_rejection(monkeypatch) -> None:
    module = load_script("verify_external_resource_gateways")
    url = "http://127.0.0.1:8088/api/gpu/jobs"
    response = io.BytesIO(b'{"status":"rejected","artifact_path":"workspace/rejection.json","http_status":999}')
    rejection = urllib.error.HTTPError(url, 403, "Forbidden", hdrs=None, fp=response)
    captured: dict[str, str | None] = {}

    def reject(request, **kwargs):
        del kwargs
        captured["origin"] = request.get_header("Origin")
        raise rejection

    monkeypatch.setattr(module.urllib.request, "urlopen", reject)

    payload = module.post_json(url, {"template": "not_allowed_shell"})
    assert captured["origin"] == "http://127.0.0.1:8088"
    assert payload == {
        "status": "rejected",
        "artifact_path": "workspace/rejection.json",
        "http_status": 403,
    }


def test_release_acceptance_threads_port_to_every_live_check() -> None:
    acceptance = (ROOT / "scripts" / "run_new_user_release_acceptance.ps1").read_text(encoding="utf-8-sig")
    restart = (ROOT / "scripts" / "restart_workstation_frontend.ps1").read_text(encoding="utf-8-sig")

    assert '$BaseUrl = "http://127.0.0.1:$Port"' in acceptance
    assert "Set-Location -LiteralPath $Root" in acceptance
    for script_name in (
        "verify_new_user_release_readiness.py",
        "verify_workstation_launch_readiness.py",
        "verify_workstation_browser_render_smoke.py",
        "verify_workstation_click_smoke.mjs",
        "verify_workstation_interactive_controls.mjs",
    ):
        invocation_lines = [line for line in acceptance.splitlines() if script_name in line and "--base-url" in line]
        assert invocation_lines
        assert "--base-url $BaseUrl" in invocation_lines[0]

    assert '"--port",' in restart
    assert "[string]$Port" in restart
    assert '"node_modules\\next\\dist\\bin\\next"' in restart
    assert '[ValidateSet("development", "production")]' in restart
    assert '$nextCommand = if ($Mode -eq "production") { "start" } else { "dev" }' in restart
    assert "-Port $Port `" in acceptance
    assert "-Mode production `" in acceptance
    assert 'Run-Check "start_production_workstation_frontend"' in acceptance
    assert '"--build"' in (ROOT / "scripts" / "start_verified_workstation.ps1").read_text(encoding="utf-8-sig")


def test_release_acceptance_uses_selected_python_and_initializes_database() -> None:
    acceptance = (ROOT / "scripts" / "run_new_user_release_acceptance.ps1").read_text(encoding="utf-8-sig")

    assert '[string]$PythonExecutable = ""' in acceptance
    assert "$PythonExe = (Get-Command python -ErrorAction Stop).Source" in acceptance
    assert 'Run-Check "acceptance_test_dependencies"' in acceptance
    assert 'Join-Path $Root "requirements-dev.txt"' in acceptance
    assert "-m pip install $pytestRequirement --quiet" in acceptance
    assert "& $PythonExe -m pytest" in acceptance
    assert "& $PythonExe scripts\\verify_no_plaintext_secrets.py" in acceptance
    assert "$env:WORKSTATION_PYTHON = $PythonExe" in acceptance
    assert "-PythonExecutable $PythonExe" in acceptance
    assert acceptance.index('Run-Check "acceptance_test_dependencies"') < acceptance.index('Run-Check "cli_tests"')

    assert 'Join-Path $Root "install.ps1"' in acceptance
    installer_block = acceptance[
        acceptance.index('Run-Check "installer_smoke_no_secrets"') : acceptance.index('Run-Check "cli_tests"')
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


def test_release_acceptance_supports_a_fully_isolated_windows_profile() -> None:
    acceptance = (ROOT / "scripts" / "run_new_user_release_acceptance.ps1").read_text(encoding="utf-8-sig")

    assert '[string]$ProfileRoot = ""' in acceptance
    for variable in (
        "USERPROFILE",
        "HOME",
        "HOMEDRIVE",
        "HOMEPATH",
        "APPDATA",
        "LOCALAPPDATA",
        "XSCI_SHIM_DIR",
    ):
        assert f"$env:{variable}" in acceptance
    assert "$env:PSModuleAnalysisCachePath" in acceptance
    assert "evomind-powershell-cache" in acceptance


def test_restart_frontend_propagates_workspace_and_python_runtime() -> None:
    restart = (ROOT / "scripts" / "restart_workstation_frontend.ps1").read_text(encoding="utf-8-sig")

    assert '[string]$PythonExecutable = ""' in restart
    assert '[string]$DatabaseUrl = ""' in restart
    assert "$env:WORKSTATION_ROOT = $workspaceRoot.Path" in restart
    assert "$env:WORKSTATION_PYTHON" in restart
    assert "Get-Command python -ErrorAction SilentlyContinue" in restart
    assert '$env:DATABASE_URL = if ($DatabaseUrl)' in restart
    assert '"prisma\\workstation.db"' in restart
    assert '"scripts\\prisma-db-push.mjs"' in restart
    assert '$previousErrorActionPreference = $ErrorActionPreference' in restart
    assert '$ErrorActionPreference = "Continue"' in restart
    assert '$prismaOutput = @(& node.exe $prismaPush "--skip-generate" 2>&1)' in restart
    assert "$prismaExitCode = $LASTEXITCODE" in restart
    assert restart.index('$prismaOutput = @(& node.exe $prismaPush "--skip-generate" 2>&1)') < restart.index(
        "$server = Start-Process"
    )


def test_prisma_db_push_forwards_cli_arguments() -> None:
    source = (ROOT / "web" / "research-agent-workstation" / "scripts" / "prisma-db-push.mjs").read_text(
        encoding="utf-8-sig"
    )

    assert 'runPrisma(["db", "push", ...process.argv.slice(2)])' in source


def test_security_runtime_initializes_unique_database_before_server_start() -> None:
    source = (ROOT / "web" / "research-agent-workstation" / "scripts" / "verify-production-security.ps1").read_text(
        encoding="utf-8-sig"
    )

    assert '$DatabaseName = "ci-workstation-$PID.db"' in source
    assert '$env:DATABASE_URL = "file:./$DatabaseName"' in source
    assert source.index('& node $PrismaPush "--skip-generate"') < source.index("$server = Start-Process")
    for suffix in ("", "-journal", "-shm", "-wal"):
        assert f"$DatabasePath{suffix}" in source


def test_installers_initialize_database_and_generate_client_before_frontend_build() -> None:
    installers = (
        ROOT / "install.ps1",
        ROOT / "scripts" / "quick_setup.ps1",
        ROOT / "scripts" / "quick_setup.sh",
    )

    for installer in installers:
        source = installer.read_text(encoding="utf-8-sig")
        assert source.index("npm run db:push") < source.index("npm run db:generate") < source.index("npm run build")


def test_release_archives_keep_shell_scripts_lf() -> None:
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines()

    assert "*.sh text eol=lf" in attributes


def test_cdp_smokes_support_node20_without_a_global_websocket() -> None:
    package = json.loads((ROOT / "web" / "research-agent-workstation" / "package.json").read_text(encoding="utf-8"))
    assert package["dependencies"]["ws"] == "8.21.0"

    for script_name in (
        "verify_workstation_click_smoke.mjs",
        "verify_workstation_interactive_controls.mjs",
    ):
        source = (ROOT / "scripts" / script_name).read_text(encoding="utf-8-sig")
        assert 'createRequire(new URL("../web/research-agent-workstation/package.json"' in source
        assert 'globalThis.WebSocket ?? requireFromWeb("ws")' in source
        assert "new WebSocketClient(this.wsUrl)" in source
        assert '"--disable-extensions"' in source
        assert 'import { createServer } from "node:net"' in source
        assert "async function allocateCdpPort" in source
        assert 'stdio: ["ignore", "ignore", "pipe"]' in source
        assert 'blocker: "browser_cdp_unavailable"' in source
        assert "chrome_stderr_tail" in source

    click_smoke = (ROOT / "scripts" / "verify_workstation_click_smoke.mjs").read_text(encoding="utf-8-sig")
    controls_smoke = (ROOT / "scripts" / "verify_workstation_interactive_controls.mjs").read_text(
        encoding="utf-8-sig"
    )
    assert "9223 + (process.pid % 1000)" not in click_smoke
    assert '?? "9224"' not in controls_smoke


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
