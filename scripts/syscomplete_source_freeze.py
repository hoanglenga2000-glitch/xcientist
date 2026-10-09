"""Freeze only the reviewed system-completeness web delta on the 3800faca base.

This is an OFFLINE SOURCE preparation tool, not a deployment tool.  It never
copies the entire dirty worktree and deliberately does not produce a standard
build_invitation_release source receipt: the candidate still needs the frozen
runtime, shared scheduler, report dependencies, and release acceptance gates.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import stat
import sys


ROOT = Path(__file__).resolve().parents[1]
BASE_BUILD = "overlay-invitation-beta-3800faca9fc5-ack1"
BASE_TREE = "3800faca9fc507c17152904af09629ccc98a74769723e4f7d1ae02ed6f1086be"
BASE_WEB = "bf129bce03aabde3d16c65a136c4630eaa727ba9a132ce233f83fc26e20995fc"
BASE_RUNTIME = "a5b22dbff4161a5624c780ba36b07ab2ab1e7684c76d655637527ebebd8a74f8"
BASE_SOURCE_MANIFEST = "e3d80e0abd54dd73b4777dbe7b2b4f62bf488f9e4355119d297eb7931dab9507"
BASE_RECEIPT = "db48e7d5a924946d2b7979ef6b84646cba89af9640299f726f7579107b11698b"
BACKUP_UI = (
    "src/app/globals.css",
    "src/components/workstation/AppShell.tsx",
    "src/components/workstation/screens/AssistantReportPanel.tsx",
    "src/components/workstation/screens/AssistantScreen.tsx",
    "src/components/workstation/screens/ReportStudioScreen.tsx",
    "src/components/workstation/screens/run-progress-presentation.ts",
    "src/components/workstation/screens/RunProgressPanel.tsx",
    "src/lib/assistant-report.test.ts",
    "src/lib/assistant-report.ts",
)
ASSISTANT_SCREEN = "src/components/workstation/screens/AssistantScreen.tsx"
ASSISTANT_BACKUP = "e60a4e64e03920bd6854ac90ae0cad3865a3345da1e3cad67c66ebbba14feaa9"
# These two files had no step-1 backup. Their complete small diffs/new contents
# were reviewed against the frozen base, so changes need another review/hash.
REVIEWED_DIRECT = {
    "src/app/api/assistant/runs/[runId]/actions/route.ts":
        "57efdc60c489f976947c8e9485913996b5ea369098ccec0bc82c6c1355f99a40",
    "src/app/api/assistant/runs/[runId]/reports/route.ts":
        "8097980979945bca90b0618f1fbe660e5bf7f9caec5f726f9517cb11b3af13d1",
}
BUILD_INPUTS = frozenset("web/" + name for name in (
    "package.json", "package-lock.json", "next.config.mjs", "postcss.config.mjs",
    "tailwind.config.ts", "tsconfig.json", "prisma/schema.prisma",
))
TEMPLATES = ("runtime-build-manifest.json", "operational-overlay-manifest.json")
HOLD_GATES = (
    "fresh_production_base_identity_confirmation",
    "frozen_gpt55_runtime_integration",
    "shared_scheduler_delivery_and_import_binding",
    "report_dependency_delivery_and_import_binding",
    "offline_build_and_server_isolation_acceptance",
    "real_chrome_end_to_end_acceptance",
    "qualified_model_and_bounded_training_acceptance",
)


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def safe_relative(name: str) -> str:
    path = PurePosixPath(name)
    if (not name or "\\" in name or path.is_absolute() or ".." in path.parts
            or path.as_posix() != name or any(":" in part for part in path.parts)):
        raise ValueError("unsafe_source_path")
    return name


def checked_path(root: Path, relative: str, *, must_exist: bool = True) -> Path:
    """Reject aliases including Windows directory junctions at every component."""
    root = root.absolute()
    parts = PurePosixPath(safe_relative(relative)).parts
    current = root
    for part in (None, *parts):
        if part is not None:
            current = current / part
        if current.exists() or current.is_symlink():
            info = current.lstat()
            if (stat.S_ISLNK(info.st_mode)
                    or getattr(info, "st_file_attributes", 0) & 0x400):
                raise ValueError("source_path_alias_rejected")
    if current.resolve(strict=False) != current:
        raise ValueError("source_parent_alias_rejected")
    if must_exist and not current.is_file():
        raise ValueError("source_file_missing:" + relative)
    return current


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def canonical_json(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=True, indent=2) + "\n").encode()


def row_for(name: str, payload: bytes) -> dict:
    return {"path": safe_relative(name), "bytes": len(payload), "sha256": digest(payload)}


def tree_sha(rows: list[dict]) -> str:
    text = "\n".join(f"{row['path']}|{row['bytes']}|{row['sha256']}"
                     for row in sorted(rows, key=lambda row: row["path"]))
    return digest(text.encode())


def exact_delta(base: bytes, before: bytes, after: bytes) -> tuple[bytes, int]:
    """Transplant only before->after hunks; no fuzzy or whitespace matching.

    The start-of-task snapshot can contain unrelated unshipped code. Returning
    the whole edited file in that situation would accidentally release it.
    Every three-line-context hunk must have one unique occurrence in the live
    base; overlaps or changed context fail closed for manual reconciliation.
    """
    if before == base:
        return after, 0
    old_lines = before.splitlines(keepends=True)
    new_lines = after.splitlines(keepends=True)
    matcher = difflib.SequenceMatcher(a=old_lines, b=new_lines, autojunk=False)
    edits = []
    for group in matcher.get_grouped_opcodes(3):
        original = b"".join(old_lines[group[0][1]:group[-1][2]])
        replacement = b"".join(new_lines[group[0][3]:group[-1][4]])
        start = base.find(original)
        if not original or start < 0 or base.find(original, start + 1) >= 0:
            raise ValueError("delta_context_not_unique")
        edits.append((start, start + len(original), replacement))
    previous_end = -1
    for start, end, _replacement in edits:
        if start < previous_end:
            raise ValueError("delta_context_out_of_order_or_overlapping")
        previous_end = end
    result = base
    for start, end, replacement in reversed(edits):
        result = result[:start] + replacement + result[end:]
    return result, len(edits)


def load_builder():
    path = ROOT / "scripts/build_invitation_release.py"
    spec = importlib.util.spec_from_file_location("syscomplete_existing_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_parent(parent: Path) -> dict:
    for name, expected in (("web.zip", BASE_WEB), ("runtime.zip", BASE_RUNTIME),
                           ("source-receipt.json", BASE_RECEIPT),
                           ("release/release-source-manifest.json", BASE_SOURCE_MANIFEST)):
        if sha(checked_path(parent, name)) != expected:
            raise ValueError("pinned_parent_hash_mismatch:" + name)
    source = read_json(parent / "source-receipt.json")
    result = read_json(checked_path(parent, "build-result.json"))
    if (source.get("build_id") != BASE_BUILD or source.get("source_tree_sha256") != BASE_TREE
            or source.get("schema") != "evomind.invitation_source.v1"
            or result.get("status") != "built" or result.get("source_receipt_sha256") != BASE_RECEIPT
            or result.get("source_manifest_sha256") != BASE_SOURCE_MANIFEST
            or result.get("web", {}).get("sha256") != BASE_WEB
            or result.get("runtime", {}).get("sha256") != BASE_RUNTIME):
        raise ValueError("pinned_parent_identity_mismatch")
    names = [safe_relative(row["path"]) for row in source["files"]]
    if len(names) != len({name.casefold() for name in names}) or tree_sha(source["files"]) != BASE_TREE:
        raise ValueError("pinned_source_tree_mismatch")
    load_builder().validate_release_identity(parent / "release", parent / "runtime", source)
    return source


def source_payloads(parent: Path, source: dict, input_roots: list[Path]) -> tuple[dict, list[dict]]:
    payloads, restored = {}, []
    for row in source["files"]:
        name = safe_relative(row["path"])
        if not name.startswith(("web/", "runtime/evomind_runtime/")):
            raise ValueError("source_scope_rejected")
        relative = "release/" + name[4:] if name.startswith("web/") else name
        candidates = [checked_path(parent, relative, must_exist=False)]
        if name in BUILD_INPUTS:
            candidates += [checked_path(root, name[4:], must_exist=False) for root in input_roots]
        selected = None
        for path in candidates:
            if path.is_file() and path.stat().st_size == row["bytes"] and sha(path) == row["sha256"]:
                selected = path
                break
        if selected is None:
            raise ValueError("source_bytes_not_recovered:" + name)
        payload = selected.read_bytes()
        if row_for(name, payload) != {key: row[key] for key in ("path", "bytes", "sha256")}:
            raise ValueError("source_changed_during_read")
        payloads[name] = payload
        if selected != candidates[0]:
            restored.append({**row, "recovered_from": str(selected)})
    return payloads, restored


def overlay_web(payloads: dict[str, bytes], workspace: Path, backup: Path) -> list[dict]:
    changes = []
    for relative in (*BACKUP_UI, *REVIEWED_DIRECT):
        name = "web/" + relative
        current = checked_path(workspace, "web/research-agent-workstation/" + relative).read_bytes()
        base = payloads.get(name)
        before = None
        hunks = 0
        if relative in BACKUP_UI:
            before = checked_path(backup, "web/research-agent-workstation/" + relative).read_bytes()
            if base is None:
                raise ValueError("backed_up_source_missing_from_parent")
            if before != base and (relative != ASSISTANT_SCREEN or digest(before) != ASSISTANT_BACKUP):
                raise ValueError("unreviewed_starting_snapshot_drift:" + relative)
            result, hunks = exact_delta(base, before, current)
            operation = "exact_snapshot_delta" if before != base else "snapshot_bound_edit"
        else:
            if digest(current) != REVIEWED_DIRECT[relative]:
                raise ValueError("reviewed_direct_source_changed:" + relative)
            if relative.endswith("/reports/route.ts") and base is not None:
                raise ValueError("reviewed_new_route_already_exists")
            result = current
            operation = "reviewed_new_file" if base is None else "reviewed_complete_diff"
        if result == base:
            continue
        payloads[name] = result
        changes.append({**row_for(name, result), "before_sha256": digest(base) if base is not None else None,
                        "workspace_sha256": digest(current), "snapshot_sha256": digest(before) if before is not None else None,
                        "operation": operation, "transplanted_hunks": hunks})
    return changes


def prepare(parent: Path, workspace: Path, backup: Path, input_roots: list[Path]) -> tuple[dict, dict]:
    source = verify_parent(parent)
    payloads, restored = source_payloads(parent, source, input_roots)
    changes = overlay_web(payloads, workspace, backup)
    rows = [row_for(name, payload) for name, payload in sorted(payloads.items())]
    plan = {
        "schema": "evomind.syscomplete_source_freeze.v1", "status": "source_prepared_release_hold",
        "base_build_id": BASE_BUILD, "base_source_tree_sha256": BASE_TREE,
        "base_web_sha256": BASE_WEB, "base_runtime_sha256": BASE_RUNTIME,
        "base_source_manifest_sha256": BASE_SOURCE_MANIFEST, "recipe_sha256": sha(Path(__file__)),
        "parent_root": str(parent.absolute()), "workspace_root": str(workspace.absolute()),
        "snapshot_root": str(backup.absolute()), "build_input_roots": [str(root.absolute()) for root in input_roots],
        "source_tree_sha256": tree_sha(rows), "file_count": len(rows), "files": rows,
        "web_changes": changes, "restored_build_inputs": restored,
        "templates": [row_for("web/" + name, checked_path(parent, "release/" + name).read_bytes()) for name in TEMPLATES],
        "runtime_payload_changed": False, "dirty_tree_wholesale_copy": False,
        "build_ready": False, "production_deployed": False, "pending_gates": list(HOLD_GATES),
    }
    return plan, payloads


def write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)


def freeze(plan_path: Path, expected_sha: str, destination: Path) -> dict:
    if sha(plan_path) != expected_sha:
        raise ValueError("freeze_plan_hash_mismatch")
    expected = read_json(plan_path)
    if expected.get("schema") != "evomind.syscomplete_source_freeze.v1":
        raise ValueError("freeze_plan_schema_rejected")
    parent = Path(expected["parent_root"])
    plan, payloads = prepare(parent, Path(expected["workspace_root"]), Path(expected["snapshot_root"]),
                             [Path(value) for value in expected["build_input_roots"]])
    if plan != expected:
        raise ValueError("freeze_inputs_changed_since_plan")
    destination = destination.absolute()
    if destination.exists() or destination.is_symlink() or destination.resolve(strict=False) != destination:
        raise ValueError("freeze_destination_exists_or_aliased")
    destination.mkdir(parents=True, exist_ok=False)
    for name, payload in payloads.items():
        write_new(destination / name, payload)
    for template in plan["templates"]:
        name = template["path"][4:]
        payload = checked_path(parent, "release/" + name).read_bytes()
        if row_for(template["path"], payload) != template:
            raise ValueError("freeze_template_changed")
        write_new(destination / template["path"], payload)
    frozen = {**plan, "frozen": True, "plan_sha256": expected_sha, "destination_root": str(destination)}
    write_new(destination / "syscomplete-source-manifest.json", canonical_json(frozen))
    # Source-only schema deliberately cannot be consumed as build_invitation_release's
    # standard source-receipt. Integration and all HOLD gates stay explicit.
    return {"status": frozen["status"], "destination_root": str(destination), "file_count": len(payloads),
            "web_changes": len(plan["web_changes"]), "source_tree_sha256": plan["source_tree_sha256"],
            "manifest_sha256": sha(destination / "syscomplete-source-manifest.json"), "build_ready": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--parent-root", type=Path, default=Path("D:/EV12/ack07-11727a59"))
    audit.add_argument("--workspace", type=Path, default=ROOT)
    audit.add_argument("--snapshot", type=Path, default=ROOT / "artifacts/system-completeness-20260907/step1-baseline")
    audit.add_argument("--build-input-root", type=Path, action="append", required=True)
    audit.add_argument("--plan", type=Path, required=True)
    staging = sub.add_parser("freeze")
    staging.add_argument("--plan", type=Path, required=True)
    staging.add_argument("--plan-sha256", required=True)
    staging.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "audit":
        plan, _payloads = prepare(args.parent_root, args.workspace, args.snapshot, args.build_input_root)
        write_new(args.plan, canonical_json(plan))
        print(json.dumps({"status": plan["status"], "plan": str(args.plan), "plan_sha256": sha(args.plan),
                          "base_source_tree_sha256": BASE_TREE, "source_file_count": plan["file_count"],
                          "web_change_count": len(plan["web_changes"]), "restored_build_inputs": len(plan["restored_build_inputs"]),
                          "build_ready": False}))
    else:
        print(json.dumps(freeze(args.plan, args.plan_sha256, args.destination)))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        code = str(error) if re.fullmatch(r"[A-Za-z0-9_:/.[\]-]{1,240}", str(error)) else "diagnostic_withheld"
        print(json.dumps({"status": "failed", "error_class": type(error).__name__, "error_code": code,
                          "production_deployed": False}))
        sys.exit(1)
