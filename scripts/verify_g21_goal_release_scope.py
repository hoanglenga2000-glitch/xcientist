from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from evomind_runtime.goal_release_scope import load_and_verify_release_scope


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify that a runtime manifest contains the G21 Goal scope.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args(argv)
    result = load_and_verify_release_scope(args.manifest, source_root=args.source_root)
    print(json.dumps(result.to_dict(), ensure_ascii=False, separators=(",", ":")))
    return 0 if result.valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
