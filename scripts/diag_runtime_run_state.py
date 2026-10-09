import hashlib
import json
import pathlib
import sqlite3

RUN_ID = "run_7b1efb878afb40f396db431e91f093a5"
DB = "C:/ProgramData/EvoMind/data/workspace/runtime/runtime.sqlite3"

connection = sqlite3.connect(DB)
connection.row_factory = sqlite3.Row
tables = [row[0] for row in connection.execute("select name from sqlite_master where type='table'")]
output = {"tables": tables}
if "sessions" in tables:
    row = connection.execute("select metadata_json from sessions where id=?", (RUN_ID,)).fetchone()
    if row:
        try:
            metadata = json.loads(row[0] or "{}")
            refs = metadata.get("run_secret_refs") if isinstance(metadata, dict) else []
            output["secret_ref_count"] = len(refs) if isinstance(refs, list) else 0
            claimed_root = pathlib.Path("C:/ProgramData/EvoMind/data/workspace/runtime/run-secrets/claimed") / RUN_ID
            claimed_hashes = {path.stem for path in claimed_root.glob("*.json")}
            consumed_root = pathlib.Path("C:/ProgramData/EvoMind/data/workspace/runtime/run-secrets/consumed") / RUN_ID
            consumed_hashes = {path.stem for path in consumed_root.glob("*.json")}
            output["secret_ref_hashes_match_claimed"] = [
                hashlib.sha256(str(ref).encode("utf-8")).hexdigest() in claimed_hashes
                for ref in refs
            ] if isinstance(refs, list) else []
            output["secret_ref_hashes_match_consumed"] = [
                hashlib.sha256(str(ref).encode("utf-8")).hexdigest() in consumed_hashes
                for ref in refs
            ] if isinstance(refs, list) else []
        except Exception:
            output["secret_ref_count"] = "unparsed"

for table in ("assistant_runs", "tool_calls", "run_events", "events"):
    if table not in tables:
        continue
    columns = [row[1] for row in connection.execute(f"pragma table_info({table})")]
    key = "session_id" if "session_id" in columns else "run_id" if "run_id" in columns else ""
    if not key:
        continue
    rows = [dict(row) for row in connection.execute(f"select * from {table} where {key}=? order by rowid desc limit 30", (RUN_ID,))]
    keep = {
        "assistant_runs": {"id", "status", "retry_count", "error_class", "error_message", "updated_at"},
        "tool_calls": {"id", "tool_name", "status", "created_at", "started_at", "completed_at"},
        "events": {"seq", "event_type", "created_at"},
    }.get(table)
    if keep:
        projected = []
        for row in rows:
            item = {field: row.get(field) for field in keep if field in row}
            if table == "tool_calls" and row.get("result_json"):
                try:
                    item["error"] = json.loads(row["result_json"]).get("error", "")
                except Exception:
                    item["error"] = "unparsed"
            projected.append(item)
        rows = projected
    if table == "tool_calls":
        for item, raw in zip(rows, [row for row in connection.execute(f"select result_json from {table} where {key}=? order by rowid desc limit 30", (RUN_ID,))]):
            try:
                result = json.loads(raw[0] or "{}")
                content = result.get("content") if isinstance(result, dict) else {}
                receipt = content.get("receipt") if isinstance(content, dict) else {}
                if isinstance(receipt, dict):
                    item["receipt"] = {
                        field: receipt.get(field)
                        for field in (
                            "status", "files", "bytes", "manifest_sha256",
                            "archive_sha256", "worker_alive", "worker_exit_code",
                            "partial_archive_bytes", "loader_smoke",
                        )
                        if field in receipt
                    }
                    item["credential_source"] = content.get("credential_source")
                    item["hpc_identity_gate"] = content.get("hpc_identity_gate")
            except Exception:
                pass
    output[table] = rows

print(json.dumps(output, ensure_ascii=True))
