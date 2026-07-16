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
REQUIRED_SOURCE_NAMES = {
    "autokaggle": "AutoKaggle",
    "agent_k": "Agent K",
    "autoresearch_ai": "AutoResearch AI",
    "autosota": "AutoSOTA",
    "nanoresearch": "NanoResearch",
    "kaggle_cli": "Kaggle CLI",
}


def fail(message: str) -> None:
    raise SystemExit(f"RESEARCH_SOURCE_VALIDATION_FAILED: {message}")


def validate_source_metadata(sources: list[dict[str, Any]]) -> None:
    if len(sources) < len(REQUIRED_SOURCE_NAMES):
        fail("expected at least six primary sources and open-source references")
    by_id = {source.get("id"): source for source in sources}
    missing = sorted(set(REQUIRED_SOURCE_NAMES) - set(by_id))
    if missing:
        fail(f"missing source ids: {missing}")
    for source_id, expected_name in REQUIRED_SOURCE_NAMES.items():
        source = by_id[source_id]
        if source.get("name") != expected_name:
            fail(f"source name mismatch for {source_id}: expected {expected_name}")
    for source in sources:
        if not source.get("url"):
            fail(f"source missing URL: {source.get('id')}")
        if not source.get("core_insights"):
            fail(f"source missing core insights: {source.get('id')}")
        if not source.get("local_mappings"):
            fail(f"source missing local mappings: {source.get('id')}")


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
    parser = argparse.ArgumentParser(
        description="Verify research-source metadata and optionally check source URLs online."
    )
    parser.add_argument("--strict-network", action="store_true", help="Fail when source URLs cannot be reached online.")
    args = parser.parse_args()

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    sources = config.get("sources", [])
    validate_source_metadata(sources)

    url_results = []
    network_warnings = []
    for source in sources:
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
                fail(
                    f"source URL not reachable through primary or verification URLs: {source.get('id')}; errors={errors}"
                )
            continue
        checked["source_id"] = source["id"]
        checked["display_url"] = source["url"]
        url_results.append(checked)

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
