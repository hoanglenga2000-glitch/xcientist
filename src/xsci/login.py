"""`xsci login` — securely configure LLM and Kaggle credentials.

Secrets are written ONLY to ~/.xsci/secrets.toml (0600); non-secret choices
(which provider, base_url) go to ~/.xsci/config.toml. Prompting is separated
from persistence so the save logic is unit-testable without a TTY. Supports a
non-interactive path (flags / env) for CI and can import an existing
kaggle.json.
"""
from __future__ import annotations

import getpass
import json
from pathlib import Path
from typing import Optional

from .config import set_global, write_secret

_LLM_PROVIDERS = ("anthropic", "deepseek")


def save_llm_credentials(provider: str, api_key: str, *, base_url: Optional[str] = None,
                         model: Optional[str] = None, brand: Optional[str] = None) -> None:
    """Persist an LLM provider's key (secret) and choice/base_url/model (config).

    ``provider`` is the ENGINE FAMILY the request is sent as (``anthropic`` = native
    tool-use, ``deepseek`` = OpenAI-compatible). ``model`` is the user's chosen
    default; it is exported to the family's ``*_MODEL`` env var by
    :func:`config.inject_engine_env` so both the chat client and the deep agent use it.
    ``brand`` is the human name of the picked provider (e.g. "OpenAI (GPT)") kept only
    for honest display in the status card — several brands share one engine family.
    """
    provider = provider.lower().strip()
    if provider not in _LLM_PROVIDERS:
        raise ValueError(f"unknown LLM provider {provider!r}; expected one of {_LLM_PROVIDERS}")
    if not api_key.strip():
        raise ValueError("empty API key")
    write_secret(f"{provider}_api_key", api_key.strip())
    set_global("llm", "provider", provider)
    if base_url:
        set_global("llm", f"{provider}_base_url", base_url.strip())
    if model:
        set_global("llm", "model", model.strip())
    if brand:
        set_global("llm", "brand", brand.strip())


def save_kaggle_credentials(username: str, key: str) -> None:
    if not username.strip() or not key.strip():
        raise ValueError("kaggle username and key are both required")
    write_secret("kaggle_username", username.strip())
    write_secret("kaggle_key", key.strip())


def save_kaggle_api_token(token: str) -> None:
    token = token.strip()
    if not token:
        raise ValueError("empty Kaggle API token")
    write_secret("kaggle_api_token", token)


def import_kaggle_json(path: Path) -> None:
    """Import credentials from a standard kaggle.json ({username, key})."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    save_kaggle_credentials(data["username"], data["key"])


def _prompt_llm() -> bool:
    print("LLM provider setup")
    print(f"  available: {', '.join(_LLM_PROVIDERS)} (default: anthropic)")
    provider = input("  provider> ").strip().lower() or "anthropic"
    if provider not in _LLM_PROVIDERS:
        print(f"  skipped: unknown provider {provider!r}")
        return False
    api_key = getpass.getpass("  API key (hidden)> ").strip()
    if not api_key:
        print("  skipped: no key entered")
        return False
    base_url = input("  custom base_url (blank = default)> ").strip() or None
    save_llm_credentials(provider, api_key, base_url=base_url)
    print(f"  saved LLM key for {provider} (~/.xsci/secrets.toml, 0600)")
    return True


def _prompt_kaggle() -> bool:
    print("\nKaggle setup (needed to fetch competitions)")
    print("  tip: paste the path to an existing kaggle.json, or leave blank to type creds")
    jpath = input("  kaggle.json path (blank to type)> ").strip()
    if jpath:
        import_kaggle_json(Path(jpath).expanduser())
        print("  imported Kaggle credentials from kaggle.json")
        return True
    username = input("  kaggle username> ").strip()
    if not username:
        print("  skipped Kaggle setup")
        return False
    key = getpass.getpass("  kaggle key (hidden)> ").strip()
    save_kaggle_credentials(username, key)
    print("  saved Kaggle credentials (~/.xsci/secrets.toml, 0600)")
    return True


def run_login(
    *,
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    kaggle_username: Optional[str] = None,
    kaggle_key: Optional[str] = None,
    kaggle_json: Optional[str] = None,
    non_interactive: bool = False,
) -> int:
    """Interactive by default; falls back to flags for CI (non_interactive)."""
    did_something = False

    if non_interactive:
        if provider and api_key:
            save_llm_credentials(provider, api_key, base_url=base_url)
            print(f"saved LLM key for {provider}")
            did_something = True
        if kaggle_json:
            import_kaggle_json(Path(kaggle_json).expanduser())
            print("imported Kaggle credentials")
            did_something = True
        elif kaggle_username and kaggle_key:
            save_kaggle_credentials(kaggle_username, kaggle_key)
            print("saved Kaggle credentials")
            did_something = True
        if not did_something:
            print("nothing to do: pass --provider/--api-key and/or kaggle flags")
            return 1
        return 0

    print("xsci login - keys are stored only in ~/.xsci (0600), never in your project.\n")
    try:
        did_something |= _prompt_llm()
        did_something |= _prompt_kaggle()
    except (KeyboardInterrupt, EOFError):
        print("\naborted.")
        return 1
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"error: {exc}")
        return 1

    print("\ndone." if did_something else "\nno credentials changed.")
    print("run `xsci doctor` to confirm.")
    return 0
