from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any


MAX_FILES = 20_000
MAX_SOURCE_BYTES = 512 * 1024
MAX_SAMPLE_BYTES = 128 * 1024
ENV_PATTERN = re.compile(r"\b[A-Za-z][A-Za-z0-9_-]*-v\d+\b")
ACTION_PATTERN = re.compile(r"\[(?:Player|player)\s+[A-Za-z0-9_-]+\]")
SENSITIVE_PATTERN = re.compile(
    r"(?i)(password|passwd|pwd|credential|token|secret|api[ _-]*key|authorization|bearer|"
    r"密码|口令|凭据|账号|登录)"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def safe_relative(path: Path, root: Path) -> str:
    value = path.relative_to(root).as_posix()
    if not value or value.startswith("/") or ".." in Path(value).parts:
        raise ValueError("unsafe relative path")
    return value


def write_json(path: Path, value: Any) -> None:
    data = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def source_contract(path: Path, root: Path) -> dict[str, Any]:
    raw = path.read_bytes()[: MAX_SOURCE_BYTES + 1]
    truncated = len(raw) > MAX_SOURCE_BYTES
    text = raw[:MAX_SOURCE_BYTES].decode("utf-8", "replace")
    if SENSITIVE_PATTERN.search(text):
        text = "\n".join(
            "[REDACTED SENSITIVE LINE]" if SENSITIVE_PATTERN.search(line) else line
            for line in text.splitlines()
        )
    result: dict[str, Any] = {
        "path": safe_relative(path, root),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "truncated": truncated,
        "environment_ids": sorted(set(ENV_PATTERN.findall(text))),
        "action_examples": sorted(set(ACTION_PATTERN.findall(text)))[:50],
        "trueskill_mentions": text.casefold().count("trueskill"),
        "reward_mentions": text.casefold().count("reward"),
        "score_mentions": text.casefold().count("score"),
    }
    try:
        tree = ast.parse(text)
    except SyntaxError:
        result["ast_parsed"] = False
        return result
    result["ast_parsed"] = True
    result["imports"] = sorted(
        {
            node.names[0].name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import) and node.names
        }
        | {
            str(node.module or "").split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
    )[:100]
    result["classes"] = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)][:100]
    result["functions"] = [node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))][:200]
    result["agent_classes"] = [
        name for name in result["classes"] if any(marker in name.casefold() for marker in ("agent", "player", "strategy"))
    ]
    result["environment_classes"] = [
        name for name in result["classes"] if any(marker in name.casefold() for marker in ("env", "game", "arena"))
    ]
    result["action_functions"] = [
        name
        for name in result["functions"]
        if name.casefold() in {"act", "step", "reset", "observe", "vote", "generate_action", "get_action"}
    ]
    result["scoring_functions"] = [
        name
        for name in result["functions"]
        if any(marker in name.casefold() for marker in ("score", "reward", "rating", "trueskill", "rank"))
    ]
    return result


def sample_contract(path: Path, root: Path) -> dict[str, Any]:
    raw = path.read_bytes()[: MAX_SAMPLE_BYTES + 1]
    text = raw[:MAX_SAMPLE_BYTES].decode("utf-8", "replace")
    if SENSITIVE_PATTERN.search(text):
        text = "\n".join(
            "[REDACTED SENSITIVE LINE]" if SENSITIVE_PATTERN.search(line) else line
            for line in text.splitlines()
        )
    result: dict[str, Any] = {
        "path": safe_relative(path, root),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "preview_truncated": len(raw) > MAX_SAMPLE_BYTES,
        "environment_ids": sorted(set(ENV_PATTERN.findall(text))),
        "action_examples": sorted(set(ACTION_PATTERN.findall(text)))[:50],
    }
    nonempty = [line.strip() for line in text.splitlines() if line.strip()][:50]
    parsed: list[Any] = []
    for line in nonempty:
        try:
            parsed.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    result["json_lines_parsed"] = len(parsed)
    keys: Counter[str] = Counter()
    for value in parsed:
        if isinstance(value, dict):
            keys.update(map(str, value))
    result["json_keys"] = sorted(keys)[:200]
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    root = Path(args.data_dir).resolve(strict=True)
    output = Path(args.out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    file_count = 0
    total_bytes = 0

    for current_root, directories, names in os.walk(root, followlinks=False):
        directories[:] = sorted(
            name
            for name in directories
            if name not in {".git", "__pycache__", ".evomind"}
            and not (Path(current_root) / name).is_symlink()
        )
        for name in sorted(names):
            path = Path(current_root) / name
            if path.is_symlink() or not path.is_file():
                continue
            file_count += 1
            total_bytes += path.stat().st_size
            if file_count > MAX_FILES:
                raise ValueError("MindGames probe file cap exceeded")
            suffix = path.suffix.casefold()
            folded = name.casefold()
            if suffix == ".py" and any(
                marker in folded
                for marker in ("agent", "env", "arena", "game", "score", "rating", "repository", "action")
            ):
                sources.append(source_contract(path, root))
            elif suffix == ".sample":
                samples.append(sample_contract(path, root))
            elif suffix in {".toml", ".yaml", ".yml"} or "readme" in folded:
                manifests.append(
                    {
                        "path": safe_relative(path, root),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )

    environment_ids = sorted(
        {
            value
            for item in [*sources, *samples]
            for value in item.get("environment_ids", [])
        }
    )
    agent_entrypoints = sorted(
        {
            item["path"]
            for item in sources
            if item.get("agent_classes") or item.get("action_functions")
        }
    )
    scoring_entrypoints = sorted(
        {
            item["path"]
            for item in sources
            if item.get("scoring_functions") or item.get("trueskill_mentions")
        }
    )
    payload = {
        "schema": "evomind.mindgames_protocol_probe.v1",
        "read_only": True,
        "source_execution": False,
        "llm_calls": 0,
        "training_performed": False,
        "regular_files": file_count,
        "total_bytes": total_bytes,
        "environment_ids": environment_ids,
        "agent_entrypoints": agent_entrypoints,
        "scoring_entrypoints": scoring_entrypoints,
        "source_contracts": sources,
        "sample_contracts": samples,
        "manifests": manifests,
    }
    write_json(output / "mindgames-protocol.json", payload)
    write_json(
        output / "mindgames-protocol-receipt.json",
        {
            "schema": "evomind.mindgames_protocol_probe_receipt.v1",
            "artifact": "mindgames-protocol.json",
            "bytes": (output / "mindgames-protocol.json").stat().st_size,
            "sha256": sha256_file(output / "mindgames-protocol.json"),
            "environment_count": len(environment_ids),
            "agent_entrypoint_count": len(agent_entrypoints),
            "scoring_entrypoint_count": len(scoring_entrypoints),
            "source_execution": False,
            "llm_calls": 0,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
