"""Read only action state fields; never dump arguments, outputs, or secrets."""
import json
from pathlib import Path
import sqlite3

path=Path('C:/ProgramData/EvoMind/data/workspace/runtime/runtime.sqlite3')
with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as connection:
    connection.row_factory=sqlite3.Row
    connection.execute('PRAGMA query_only=ON')
    rows=[dict(row) for row in connection.execute("""SELECT t.id,t.session_id,t.tool_name,t.status,
        t.started_at,t.completed_at,a.status AS assistant_status,s.status AS session_status,p.status AS approval_status
        FROM tool_calls t LEFT JOIN assistant_runs a ON a.session_id=t.session_id
        LEFT JOIN sessions s ON s.id=t.session_id LEFT JOIN approvals p ON p.id=t.approval_id
        WHERE t.status NOT IN ('completed','failed') ORDER BY t.created_at""")]
print(json.dumps({'scope':'metadata_only_no_execution','unsettled':rows},ensure_ascii=True))
