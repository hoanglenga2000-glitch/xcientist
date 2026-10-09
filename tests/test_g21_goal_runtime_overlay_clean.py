from __future__ import annotations

from pathlib import Path
import shutil
import json
import zipfile

import pytest

from scripts.build_g21_goal_runtime_overlay_clean import (
    DEPENDENT_FILES,
    EXPECTED_R115_PRODUCTION_TREE_SHA256,
    NEW_FILES,
    REJECTED_CANDIDATE_ZIP_SHA256,
    apply_goal_patches,
    extract_block,
    load_bootstrap_artifact,
    patch_diff_evidence,
    runtime_tree_sha256,
    smoke_candidate,
    write_json,
)


def _write(root: Path, relative: str, text: str, *, source: bool = True) -> None:
    path = root / ("src" if source else "") / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _fixture_roots(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source"
    baseline = tmp_path / "baseline"
    _write(
        source,
        "evomind_runtime/__init__.py",
        "from .runtime import AgentRuntime\n",
    )
    _write(
        source,
        "evomind_runtime/runtime.py",
        """class AgentRuntime:
    def ensure_fixed_goal(self):
        return None

    def get_goal(self):
        return None

    def update_fixed_goal(self):
        return None

    def tools(self):
        return []
""",
    )
    _write(
        source,
        "evomind_runtime/store.py",
        """class RuntimeStore:
    @staticmethod
    def _decode_goal_row(row):
        return row

    def get_goal_record(self):
        return None

    def get_goal_for_run(self):
        return None

    def list_goal_records(self):
        return []

    def goal_records_schema_issues(self):
        return ()

    def ensure_goal_record(self):
        return None

    def update_goal_record(self):
        return None

    def add_turn(self):
        return None
""",
    )
    _write(source, "evomind_runtime/assistant_runs.py", "current assistant\n")
    _write(source, "evomind_runtime/http_server.py", "current http\n")
    _write(source, "evomind_runtime/competition_goal.py", "GOAL = True\n")
    _write(source, "evomind_runtime/goal_board.py", "BOARD = True\n")

    _write(
        baseline,
        "evomind_runtime/__init__.py",
        """from .runtime import AgentRuntime
from .super_agent_runtime import SuperAgentRuntime

__all__ = [
    "AgentRuntime",
    "SuperAgentRuntime",
]
""",
        source=False,
    )
    _write(
        baseline,
        "evomind_runtime/runtime.py",
        """class AgentRuntime:
    def list_sessions(self, limit: int = 100) -> list[dict[str, Any]]:
        return self.store.list_sessions(limit)

    def tools(self):
        return []
""",
        source=False,
    )
    _write(
        baseline,
        "evomind_runtime/store.py",
        """SCHEMA = """ + '"""' + """CREATE TABLE IF NOT EXISTS uploads (
CREATE INDEX IF NOT EXISTS idx_attachments_run ON attachments(run_id, created_at);
""" + '"""' + """

class RuntimeStore:
    def add_turn(self, session_id: str, role: str, content: Any) -> str:
        return "turn"
""",
        source=False,
    )
    _write(
        baseline,
        "evomind_runtime/assistant_runs.py",
        """PUBLIC_EVENT_TYPES = {
    "run_failed",
    "model.response": "model_observed",
}

class AssistantRunService:
    def snapshot(self, run_id):
        if not isinstance(attachment_context, dict):
            attachment_context = {"mode": "explicit" if attachments else "none", "source_run_id": ""}
        return {
            "approvals": approvals,
        }

    def aliases(self):
        return {
            "model.response": "model_observed",
        }
""",
        source=False,
    )
    _write(
        baseline,
        "evomind_runtime/http_server.py",
        """from .assistant_runs import CHUNK_SIZE

def handler():
                    if parsed.path == "/v1/tools":
                        return self._json(200, {"tools": runtime.tools()})
                    if len(parts) == 3 and parts[:2] == ["v1", "runs"]:
                        return self._json(200, runtime.assistant.snapshot(parts[2]))
                    if parts == ["v1", "sessions"]:
                        return self._json(201, runtime.create_session(**body))
            try:
                if len(parts) == 5 and parts[:2] == ["v1", "uploads"] and parts[3] == "chunks":
                    return None
""",
        source=False,
    )
    return source, baseline


def test_clean_patch_uses_only_goal_allowlist_and_literal_anchors(tmp_path: Path) -> None:
    source, baseline = _fixture_roots(tmp_path)
    changed, hashes = apply_goal_patches(source, baseline)
    assert changed == set((*DEPENDENT_FILES, *NEW_FILES))
    assert set(hashes) == set((*DEPENDENT_FILES, *NEW_FILES))
    assert (baseline / "evomind_runtime/competition_goal.py").read_text(encoding="utf-8") == "GOAL = True\n"
    assert (baseline / "evomind_runtime/goal_board.py").read_text(encoding="utf-8") == "BOARD = True\n"
    assert "def ensure_fixed_goal" in (baseline / "evomind_runtime/runtime.py").read_text(encoding="utf-8")
    assert "goal_records" in (baseline / "evomind_runtime/store.py").read_text(encoding="utf-8")
    assert 'status=str(body.get("status") or "blocked")' in (
        baseline / "evomind_runtime/http_server.py"
    ).read_text(encoding="utf-8")


def test_clean_patch_rejects_ambiguous_method_anchors() -> None:
    text = "    def ensure_fixed_goal(self):\n        return None\n" * 2 + "    def tools(self):\n        return None\n"
    with pytest.raises(RuntimeError, match="GOAL_CLEAN_RUNTIME_METHOD_BLOCK:anchor_count"):
        extract_block(
            text,
            "    def ensure_fixed_goal(",
            "    def tools(",
            "GOAL_CLEAN_RUNTIME_METHOD_BLOCK",
            ("ensure_fixed_goal", "get_goal", "update_fixed_goal"),
        )


def test_clean_patch_rejects_missing_baseline_anchor(tmp_path: Path) -> None:
    source, baseline = _fixture_roots(tmp_path)
    path = baseline / "evomind_runtime/runtime.py"
    path.write_text("class AgentRuntime:\n    def tools(self):\n        return []\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="GOAL_CLEAN_RUNTIME_ANCHOR:count=0"):
        apply_goal_patches(source, baseline)


def test_candidate_smoke_uses_extracted_runtime_and_real_goal_http_api(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    shutil.copytree(Path("src/evomind_runtime"), candidate / "evomind_runtime")
    bootstrap_path = tmp_path / "bootstrap.json"
    bootstrap = load_bootstrap_artifact(
        Path(".").resolve(),
        Path("configs/g21_five_competition_goal_board_bootstrap.json").resolve(),
    )
    write_json(bootstrap_path, bootstrap)

    result = smoke_candidate(candidate, bootstrap_path, 7)

    assert result["round"] == 7
    assert result["status"] == "passed"
    assert result["http"]["candidate_import"].startswith("evomind_runtime")
    assert result["http"]["bind_host"] == "127.0.0.1"
    assert result["http"]["port_ephemeral"] is True
    assert result["http"]["methods"] == ["POST", "GET", "PUT"]
    assert result["http"]["post_status"] == 201
    assert result["http"]["replay_status"] == 200
    assert result["http"]["get_status"] == 200
    assert result["http"]["put_status"] == 200
    assert result["http"]["session_status"] == 200
    assert "goal_human_baseline_sha256" in result["http"]["session_metadata_fields"]
    assert result["http"]["created_events"] == 1
    assert result["http"]["updated_events"] == 1


def test_patch_diff_evidence_binds_all_five_patches_and_two_new_modules(tmp_path: Path) -> None:
    source, baseline = _fixture_roots(tmp_path)
    before = {relative: (baseline / relative).read_bytes() for relative in DEPENDENT_FILES}
    apply_goal_patches(source, baseline)

    records, combined_sha = patch_diff_evidence(before, baseline)

    assert len(records) == 7
    assert {item["path"] for item in records} == set((*DEPENDENT_FILES, *NEW_FILES))
    assert sum(bool(item["new_file"]) for item in records) == 2
    assert all(item["diff_bytes"] > 0 for item in records)
    assert all(len(item["diff_sha256"]) == 64 for item in records)
    assert len(combined_sha) == 64


def test_runtime_tree_sha_matches_r115_production_target_root_contract() -> None:
    baseline_zip = Path(
        "D:/AI-Outputs/EvoMind-Cloud-Deploy/artifacts/"
        "evomind-runtime-r115-progress-parser-r114-base-20260829.zip"
    )
    if not baseline_zip.is_file():
        pytest.skip("frozen R115 baseline is unavailable")
    with zipfile.ZipFile(baseline_zip) as archive:
        manifest = json.loads(archive.read("runtime-hotfix-manifest.json"))

    assert runtime_tree_sha256(manifest["files"]) == EXPECTED_R115_PRODUCTION_TREE_SHA256


def test_runtime_tree_sha_rejects_paths_outside_runtime_target_root() -> None:
    with pytest.raises(RuntimeError, match="GOAL_CLEAN_RUNTIME_TREE_PATH_REJECTED"):
        runtime_tree_sha256([{"path": "other/file.py", "bytes": 1, "sha256": "0" * 64}])


def test_builder_rejects_superseded_tree_prefix_candidate_sha() -> None:
    assert REJECTED_CANDIDATE_ZIP_SHA256 == {
        "37929a1c9baf682e4f946450f834f10fe3b7a78323084ae96948bd0e92f8de8d"
    }
