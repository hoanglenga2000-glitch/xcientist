"""A signed-in local install must not be told its login expired on the resources tab.

Local installs authenticate with the one-time bootstrap session (principal
"local", authentication "local_bootstrap", no tenant). The personal compute
binding is tenant-scoped, so GET /api/hpc/byoa/binding used to answer 401 and the
workspace rendered "登录已过期，请重新登录。" for a valid session. The route now
reports an explicit local-mode state; mutations still require a tenant session.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = ROOT / "web" / "research-agent-workstation" / "src" / "components" / "workstation" / "task-workspace"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def test_byoa_binding_reports_local_mode_instead_of_expired_login() -> None:
    route = _read(ROOT / "web" / "research-agent-workstation" / "src" / "app" / "api" / "hpc" / "byoa" / "binding" / "route.ts")
    get_block = route[route.index("export async function GET()") : route.index("export async function POST(")]
    assert get_block.index('principal?.authentication === "local_bootstrap"') < get_block.index("tenantPrincipal(principal)")
    assert 'personal_resources: "unavailable_in_local_mode"' in get_block
    assert "binding: null" in get_block and "tenant_id: null" in get_block
    # Mutations still require a tenant (public_password) session.
    post_block = route[route.index("export async function POST(") :]
    assert "tenantPrincipal(sessionPrincipal(" in post_block
    assert "local_bootstrap" not in post_block
    settings = _read(COMPONENTS / "WorkspaceSettings.tsx")
    assert "result.personal_resources === 'unavailable_in_local_mode'" in settings
    assert "本机模式不接入个人计算资源" in settings


def test_local_mode_branch_cannot_trigger_with_an_account_registry() -> None:
    # With the account registry enabled (public deployments) the bootstrap cookie
    # resolves to no principal at all, so the local-mode answer is unreachable there.
    session = _read(ROOT / "web" / "research-agent-workstation" / "src" / "lib" / "server" / "local-session.ts")
    block = session[session.index("if (safeEqual(value, expectedSessionCookie()))") :]
    block = block[: block.index("}") + 1]
    assert "if (accountRegistryEnabled()) return null;" in block
    assert 'authentication: "local_bootstrap"' in session
