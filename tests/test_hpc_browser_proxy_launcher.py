from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_browser_launcher_uses_dedicated_proxy_manager_and_port() -> None:
    launcher = (ROOT / "scripts" / "open_hpc_browser.ps1").read_text(encoding="utf-8")

    assert "manage_hpc_browser_proxy.ps1" in launcher
    assert "[int]$ProxyPort = 17897" in launcher
    assert 'Invoke-Manager -Command "test"' in launcher
    assert "--proxy-server=socks5://127.0.0.1:$ProxyPort" in launcher
    assert "manage_hpc_proxy_bridge.ps1" not in launcher


def test_browser_launcher_repairs_only_rejected_portal_credentials() -> None:
    launcher = (ROOT / "scripts" / "open_hpc_browser.ps1").read_text(encoding="utf-8")

    assert "[switch]$NoCredentialPrompt" in launcher
    assert 'failure_reason -eq "proxy_auth_failed"' in launcher
    assert "Read-Host" in launcher
    assert "-AsSecureString" in launcher
    assert "Using the saved school portal proxy username." in launcher
    assert "School portal proxy username [" not in launcher
    assert "Export-Clixml" in launcher
    assert '"credential_backups\\portal_proxy_"' in launcher
    assert 'Invoke-Manager -Command "stop"' in launcher
    assert 'Invoke-Manager -Command "start"' in launcher
    assert "Do not enter the job SSH password." in launcher
    assert "127.0.0.1:7890" not in launcher.split("try {", 1)[0]


def test_browser_proxy_has_separate_state_and_uses_dpapi_credential() -> None:
    manager = (ROOT / "scripts" / "manage_hpc_browser_proxy.ps1").read_text(encoding="utf-8")

    assert '"hpc_browser_proxy.pid"' in manager
    assert '"hpc_browser_proxy.out.log"' in manager
    assert '"hpc_browser_proxy.err.log"' in manager
    assert '"hpc_socks_credential.xml"' in manager
    assert "Import-Clixml" in manager
    assert "Export-Clixml" in manager
    assert "Get-Credential" in manager
    assert '"refresh-credential"' in manager
    assert '"proxy_auth_failed"' in manager
    assert '"credential_backups"' in manager
    assert "training_ssh_proxy_modified = $false" in manager
    assert "--socks5-hostname" in manager
    assert "hpc_socks_bridge.py" in manager


def test_browser_launcher_does_not_embed_credentials() -> None:
    launcher = (ROOT / "scripts" / "open_hpc_browser.ps1").read_text(encoding="utf-8")
    manager = (ROOT / "scripts" / "manage_hpc_browser_proxy.ps1").read_text(encoding="utf-8")

    forbidden = ("ProxyPassword", "-AsPlainText", "GPU_SSH_PASSWORD")
    for token in forbidden:
        assert token not in launcher
        assert token not in manager
