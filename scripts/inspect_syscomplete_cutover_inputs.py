"""Read-only, structurally redacted inventory. Never decrypt/connect/train."""
import json
from pathlib import Path
import re
import sqlite3

ROOT = Path('C:/ProgramData/EvoMind')
RUN = 'run_0c57bbde44e94a988843c2f6981f3c65'


def main():
    database = ROOT/'data/workspace/runtime/runtime.sqlite3'
    with sqlite3.connect(database.as_uri()+'?mode=ro', uri=True) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute('PRAGMA query_only=ON')
        run = dict(connection.execute('SELECT id,session_id,status,model,model_provider FROM assistant_runs WHERE id=?',(RUN,)).fetchone())
        calls=[]
        for raw in connection.execute('SELECT id,tool_name,status,result_json FROM tool_calls WHERE session_id=? ORDER BY created_at',(run['session_id'],)):
            row=dict(raw)
            result=json.loads(row.pop('result_json') or '{}')
            content=result.get('content') if isinstance(result.get('content'),dict) else {}
            error=result.get('error','')
            row.update(result_ok=result.get('ok'),content_fields=sorted(content),
                       error_code=error if isinstance(error,str) and re.fullmatch('[A-Za-z0-9_-]{0,120}',error) else 'withheld')
            if row['tool_name'] in {'hpc_execute_solution','managed_model_prepare'}:
                row['exit_code']=content.get('exit_code')
                row['execution_status']=content.get('status')
                row['solution_id']=content.get('solution_id')
                row['settlement_uncertain_marker']='remote_settlement_unconfirmed' in str(content.get('error',''))
                row['local_artifact_names']=[str(item.get('relative_path') or Path(item.get('path','')).name)
                    for item in content.get('local_artifacts',[])][:30]
            calls.append(row)
        other=[dict(row) for row in connection.execute("SELECT id,status FROM assistant_runs WHERE id!=? AND status NOT IN ('completed','failed','cancelled','blocked')",(RUN,))]
    print(json.dumps({'scope':'read_only_local_node_metadata_no_hpc_action','run':run,'calls':calls,'other_nonterminal_runs':other},ensure_ascii=True))


if __name__=='__main__': main()
