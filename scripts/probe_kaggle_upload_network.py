from __future__ import annotations

import json
import time

import requests


def probe(url: str) -> dict[str, object]:
    started = time.perf_counter()
    try:
        response = requests.get(url, timeout=12, stream=True)
        return {
            "url": url,
            "ok": True,
            "status": response.status_code,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
    except Exception as exc:
        return {
            "url": url,
            "ok": False,
            "error": repr(exc),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }


def main() -> None:
    urls = [
        "https://www.kaggle.com/",
        "https://www.googleapis.com/",
        "https://storage.googleapis.com/",
    ]
    print(json.dumps({"probes": [probe(url) for url in urls]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
