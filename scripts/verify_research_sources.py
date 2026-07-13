from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "research_sources.yaml"
FINAL_AUDIT_PATH = ROOT / "docs" / "科研Agent工作站最终完成审计.md"


def fail(message: str) -> None:
    raise SystemExit(f"RESEARCH_SOURCE_VALIDATION_FAILED: {message}")


def check_url(url: str) -> dict[str, Any]:
    last_error = ""
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 research-agent-workstation/1.0"})
            with urllib.request.urlopen(request, timeout=15) as response:
                response.read(512)
                return {
                    "url": url,
                    "status": response.status,
                    "content_type": response.headers.get("Content-Type", ""),
                    "attempt": attempt + 1,
                }
        except Exception as exc:
            last_error = repr(exc)
            time.sleep(1)
    raise RuntimeError(f"source URL not reachable after retries: {url}; last_error={last_error}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify research-source metadata and optionally check source URLs online.")
    parser.add_argument("--strict-network", action="store_true", help="Fail when source URLs cannot be reached online.")
    args = parser.parse_args()

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    sources = config.get("sources", [])
    if len(sources) < 6:
        fail("expected at least six primary sources and open-source references")
    if not FINAL_AUDIT_PATH.exists():
        fail(f"final audit document is missing: {FINAL_AUDIT_PATH.relative_to(ROOT)}")

    audit_text = FINAL_AUDIT_PATH.read_text(encoding="utf-8")
    required_ids = {"autokaggle", "agent_k", "autoresearch_ai", "autosota", "nanoresearch", "kaggle_cli"}
    ids = {source.get("id") for source in sources}
    missing = sorted(required_ids - ids)
    if missing:
        fail(f"missing source ids: {missing}")

    url_results = []
    network_warnings = []
    for source in sources:
        if not source.get("url"):
            fail(f"source missing URL: {source.get('id')}")
        if not source.get("core_insights"):
            fail(f"source missing core insights: {source.get('id')}")
        if not source.get("local_mappings"):
            fail(f"source missing local mappings: {source.get('id')}")

        urls_to_try = [source["url"], *source.get("verification_urls", [])]
        checked = None
        errors = []
        for url in urls_to_try:
            try:
                checked = check_url(url)
                break
            except Exception as exc:
                errors.append(str(exc))
        if checked is None:
            warning = {
                "source_id": source.get("id"),
                "display_url": source.get("url"),
                "errors": errors,
                "mode": "warning" if not args.strict_network else "strict_failure",
            }
            network_warnings.append(warning)
            if args.strict_network:
                fail(f"source URL not reachable through primary or verification URLs: {source.get('id')}; errors={errors}")
            continue
        checked["source_id"] = source["id"]
        checked["display_url"] = source["url"]
        url_results.append(checked)

    for required_text in ["AutoKaggle", "Agent K", "AutoResearch AI", "AutoSOTA", "NanoResearch", "Kaggle"]:
        if required_text not in audit_text:
            fail(f"final audit does not mention {required_text}")

    print(
        json.dumps(
            {
                "status": "passed",
                "sources": [source["id"] for source in sources],
                "last_verified": config.get("last_verified"),
                "url_results": url_results,
                "network_warnings": network_warnings,
                "network_mode": "strict" if args.strict_network else "best_effort",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
