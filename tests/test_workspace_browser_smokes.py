"""Contracts for the browser release smokes of the shipped /workspace UI.

The smokes (verify_workstation_click_smoke.mjs, verify_workstation_interactive_controls.mjs
and the render wrapper verify_workstation_browser_render_smoke.py) must sign in with the
release-check session and assert markers that the shipped TaskWorkspace components actually
render. These tests keep the two in lock-step: if a component marker changes, the smoke
contract fails here instead of silently passing or failing in release acceptance.
"""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
COMPONENTS = ROOT / "web" / "research-agent-workstation" / "src" / "components" / "workstation" / "task-workspace"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def _component_sources() -> str:
    return "\n".join(_read(path) for path in sorted(COMPONENTS.glob("*.tsx")))


def _workspace_view_ids() -> list[str]:
    probe = _read(SCRIPTS / "workstation_workspace_probe.mjs")
    block = probe[probe.index("export const WORKSPACE_VIEWS = [") : probe.index("];", probe.index("export const WORKSPACE_VIEWS = ["))]
    return re.findall(r'\{ id: "([a-z_]+)"', block)


def test_workspace_views_cover_every_shipped_section_and_settings_group() -> None:
    ids = _workspace_view_ids()
    assert ids == [
        "tasks", "new_task", "data", "literature", "results", "projects",
        "settings_models", "settings_account", "settings_resources",
    ]
    settings = _read(COMPONENTS / "WorkspaceSettings.tsx")
    groups = re.findall(r"\['([a-z]+)', '[^']+'\]", settings[settings.index("const groups"):settings.index("as const")])
    assert sorted(f"settings_{group}" for group in groups) == sorted(i for i in ids if i.startswith("settings_"))


@pytest.mark.parametrize(
    "marker",
    [
        'className="task-workspace"',
        'className="tw-skip" href="#task-main"',
        'id="workspace-navigation"',
        'aria-label="主导航"',
        'className="tw-brand" href="/workspace"',
        'className="tw-new" href="/workspace?new=1"',
        'className="tw-topbar"',
        'className="tw-menu" aria-controls="workspace-navigation"',
        'id="task-main"',
        'aria-label="搜索任务"',
        'className="tw-task-list"',
        'id="task-draft"',
        'className="tw-composer"',
        'aria-label="任务内容"',
        "aria-label={onSelect ? '本任务资料' : '我的资料'}",
        'aria-label="上传资料"',
        "<h1>文献检索</h1>",
        "{projects ? '管理项目' : '成果'}",
        'id="new-project"',
        'className="tw-settings-form"',
        "<h1>设置</h1>",
        'aria-label="设置分组"',
        "当前账户：",
        'id="workspace-theme"',
        "撤销未保存的外观",
        'id="task-model"',
        'id="task-message"',
        'className="tw-details"',
        "没有匹配的任务",
        "尚未开始",
        "关联已有记录",
        "正在读取你的任务",
    ],
)
def test_smoke_markers_exist_in_shipped_components(marker: str) -> None:
    assert marker in _component_sources()


def test_probe_and_smokes_use_only_current_markers() -> None:
    for name in (
        "workstation_workspace_probe.mjs",
        "verify_workstation_click_smoke.mjs",
        "verify_workstation_interactive_controls.mjs",
    ):
        source = _read(SCRIPTS / name)
        # The retired AppShell dashboard markers no longer exist in the routed UI.
        assert 'data-ui-component="workstation-page"' not in source
        assert "data-ui-page" not in source
        assert "data-ui-action=" not in source
        assert "?page=${" not in source


def test_smokes_sign_in_and_require_a_real_session() -> None:
    for name in ("verify_workstation_click_smoke.mjs", "verify_workstation_interactive_controls.mjs"):
        source = _read(SCRIPTS / name)
        assert "applyReleaseCheckAuth(client, root, baseUrl)" in source
        assert 'from "./workstation_workspace_probe.mjs"' in source
        # Without the signed-in principal the personal views cannot render; the
        # automation-header fallback must never be reported as a pass.
        assert 'status: authMode === "local_session" &&' in source
        assert 'blocker: authMode === "local_session" ? null : "release_check_session_missing"' in source
    probe = _read(SCRIPTS / "workstation_workspace_probe.mjs")
    for error in ("task_principal_required", "session_required", "Application error", "Hydration failed"):
        assert error in probe


def test_click_smoke_exercises_navigation_draft_creation_and_safety_gates() -> None:
    source = _read(SCRIPTS / "verify_workstation_click_smoke.mjs")
    assert 'const pageTargets = [...WORKSPACE_VIEWS.map((view) => view.id), "legacy_redirect"];' in source
    assert "`${baseUrl}/?page=overview`" in source
    assert source.count("{ page: ") >= 10 + 3
    assert "createDraftTask(client, baseUrl, draftTitle)" in source
    assert "model_run_requires_explicit_model_choice" in source
    assert "empty_draft_cannot_be_saved" in source
    # Real mouse input, never element.click().
    probe = _read(SCRIPTS / "workstation_workspace_probe.mjs")
    assert '"Input.dispatchMouseEvent"' in probe
    assert "covered_by_other_element" in probe
    assert ".click()" not in probe


def test_interactive_audit_has_behaviour_probes_and_never_commits_changes() -> None:
    source = _read(SCRIPTS / "verify_workstation_interactive_controls.mjs")
    for probe in (
        "skip_link_keyboard",
        "search_filters_task_list",
        "mobile_menu_toggle",
        "task_session_tabs",
        "start_requires_explicit_model",
        "details_disclosure",
        "project_name_gate",
        "theme_preview_and_undo",
    ):
        assert f'probe("{probe}"' in source
    for check in ("missing_accessible_name", "not_keyboard_reachable", "link_leaves_product_or_unknown_route"):
        assert check in source
    # Probes must not click the controls that send, persist or sign out.
    clicked = re.findall(r'clickSelector\(client, "([^"]+)"\)', source)
    assert clicked
    assert not any("submit" in selector or "tw-primary" in selector for selector in clicked)
    for forbidden in ("退出登录", "保存外观", "创建项目", "开始任务", "发送需求"):
        assert f"textContent === '{forbidden}'" not in source


def test_waitfor_fails_loudly_on_broken_probe_expressions() -> None:
    probe = _read(SCRIPTS / "workstation_workspace_probe.mjs")
    assert "if (/SyntaxError/.test(String(error?.message ?? error))) throw error;" in probe


def _render_module():
    spec = importlib.util.spec_from_file_location("render_smoke", SCRIPTS / "verify_workstation_browser_render_smoke.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_render_smoke_pages_match_click_smoke_targets() -> None:
    module = _render_module()
    assert module.DEFAULT_PAGES == [*_workspace_view_ids(), "legacy_redirect", "task_session"]


def _click_report(auth_mode: str, *, ok: bool = True) -> dict:
    shell = {"workspace": True, "topbar": True, "main": True, "sidebar": True, "navLinkCount": 5}
    module = _render_module()
    return {
        "status": "passed" if ok and auth_mode == "local_session" else "failed",
        "auth_mode": auth_mode,
        "ui": "workspace",
        "chrome": "chrome",
        "runtime_error_count": 0,
        "page_results": [
            {"page": page, "ok": ok, "activePage": page, "shell": shell, "actionCount": 14, "buttonCount": 2, "textSize": 300, "hasErrorText": False, "alerts": []}
            for page in module.DEFAULT_PAGES
        ],
    }


@pytest.mark.parametrize(
    ("auth_mode", "ok", "returncode", "expected_status", "expected_blocker"),
    [
        ("local_session", True, 0, "passed", None),
        ("local_automation_header", True, 0, "failed", "release_check_session_missing"),
        ("local_session", False, 1, "failed", None),
    ],
)
def test_render_smoke_status(tmp_path, monkeypatch, auth_mode, ok, returncode, expected_status, expected_blocker) -> None:
    module = _render_module()
    click_json = tmp_path / "click.json"
    click_json.write_text(json.dumps(_click_report(auth_mode, ok=ok)), encoding="utf-8")
    monkeypatch.setattr(module, "CLICK_JSON", click_json)
    monkeypatch.setattr(module, "run_click_smoke", lambda base_url, timeout: (returncode, "", ""))
    report = module.build_report("http://127.0.0.1:18088", module.DEFAULT_PAGES, 60)
    assert report["status"] == expected_status
    assert report["blocker"] == expected_blocker
    assert all(item["has_shell"] and item["has_sidebar"] for item in report["page_results"])
    if not ok:
        assert report["failed_pages"] == module.DEFAULT_PAGES


def test_release_browser_e2e_signs_in_before_smokes_and_cleans_up() -> None:
    script = _read(SCRIPTS / "verify_release_browser_e2e.ps1")
    establish = script.index("workstation_local_auth.py\") establish --base-url $BaseUrl")
    assert establish < script.index("verify_workstation_click_smoke.mjs")
    assert script.index("workstation_local_auth.py\") remove") < script.index("manage_workstation_dashboard.py\") stop")
