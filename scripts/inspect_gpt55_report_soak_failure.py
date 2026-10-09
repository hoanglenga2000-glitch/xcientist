"""Bounded read-only report provenance diagnostic for the owned v9 fixture."""
import json
from pathlib import Path
import sqlite3


def main():
    root = Path('C:/EMQA/sys0907-gpt55-stability-v9')
    receipt = json.loads((root / 'endurance.json').read_text(encoding='utf-8'))
    run_id = receipt['cases'][-1]['run_id']
    database = root / 'isolated-workspace/workspace/runtime/runtime.sqlite3'
    with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as connection:
        rows = connection.execute('SELECT tool_name,status,arguments_json,result_json,started_at,completed_at FROM tool_calls WHERE session_id=? ORDER BY rowid', (run_id,)).fetchall()
        calls = []
        for name, status, args, result, started, ended in rows:
            args, result = json.loads(args), json.loads(result)
            content = result.get('content') or {}
            job = content.get('report_job') or {}
            calls.append({'name':name,'status':status,'started_at':started,'completed_at':ended,
                'arguments':{key:args.get(key) for key in ['path','name','title','report_kind','report_id','formats','language'] if key in args},
                'report_job':{key:job.get(key) for key in ['id','status','document_sha256','run_id'] if key in job},
                'result_content_keys':list(content)})
        result = {'run_id':run_id,'calls':calls,
            'deliverables':connection.execute('SELECT id,name,sha256,source_tool_call FROM deliverables WHERE run_id=? ORDER BY rowid',(run_id,)).fetchall()}
    import sys
    sys.path.insert(0,'C:/ProgramData/EvoMind/staging/system-completeness-20260907/endurance-candidate-v9')
    from evomind_runtime.runtime import AgentRuntime
    # Do not open a live runtime. Its durable report jobs are private SQLite files.
    result['report_files'] = [str(path) for path in (root/'isolated-workspace/workspace/runtime').glob('**/*report*') if path.is_file() and path.suffix in {'.json','.sqlite3'}][:30]
    print(json.dumps(result,ensure_ascii=True),flush=True)


if __name__ == '__main__': main()
